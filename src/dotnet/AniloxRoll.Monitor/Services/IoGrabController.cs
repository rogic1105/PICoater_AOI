using System;
using System.Threading;
using System.Threading.Tasks;
using IoBridge.Core;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// IO ↔ PC 交握控制器：自動偵測 IO module 連線，連線時以 DI START 信號控制 Grab 啟停。
    /// 未連線時退回 UI 按鈕控制。
    ///
    /// 背景監控 loop（Task.Run，不依賴 message pump），啟動時不論連線成功失敗都跑：
    ///   IsConnected → PollTick + Task.Delay(PollIntervalMs)
    ///   !IsConnected → ReconnectTick + Task.Delay(ReconnectIntervalMs)
    /// 場景：程式先開啟、IO 後接 → loop 持續重連直到 IO module 上線，無需重啟程式。
    ///
    /// IO Mapping (ET-7044):
    ///   DI-0: PLC ALIVE   (PLC → PC)
    ///   DI-1: START        (PLC → PC, 上升/下降緣觸發)
    ///   DO-0: PC ALIVE     (PC → PLC, Form 開啟 = High)
    ///   DO-1: MURA         (PC → PLC, 檢測到瑕疵 = High)
    ///   DO-2: PC BUSY      (PC → PLC, Grab 中 = High)
    /// </summary>
    public class IoGrabController : IDisposable
    {
        private const int DI_NAKAN_ALIVE   = 0;
        private const int DI_INSPECT_START = 1;
        private const int DO_PC_ALIVE      = 0;
        private const int DO_MURA_DETECTED = 1;
        private const int DO_PC_INSPECT    = 2;

        private readonly IModbusTcpClient _plc;
        private CancellationTokenSource _bgCts;
        private Task _bgTask;

        private bool _lastDiStart;
        private bool _isPcAlive;
        private IoState _currentState = IoState.Disconnected;
        private string _plcIp = "192.168.255.1";
        private int _plcPort = 502;

        // DO 狀態追蹤（供 IO 快照使用）
        private bool _doPcAlive;
        private bool _doMura;
        private bool _doPcBusy;

        /// <summary>IO module 是否已連線。</summary>
        public bool IsConnected => _plc.IsConnected;

        /// <summary>目前 FSM 狀態。</summary>
        public IoState CurrentState => _currentState;

        /// <summary>硬體型號。</summary>
        public string Model => "ET-7044";

        /// <summary>Poll 週期（ms）。</summary>
        public int PollIntervalMs { get; set; } = 500;

        /// <summary>重連週期（ms）。</summary>
        public int ReconnectIntervalMs { get; set; } = 5000;

        /// <summary>讀寫逾時（ms）。</summary>
        public int ReadWriteTimeoutMs => _plc.ReadWriteTimeoutMs;

        /// <summary>目前連線 IP。</summary>
        public string IoIp => _plcIp;

        /// <summary>目前連線 Port。</summary>
        public int IoPort => _plcPort;

        /// <summary>PLC START 上升緣 → 要求開始 Grab。</summary>
        public event Action OnStartRequested;

        /// <summary>PLC START 下降緣 → 要求停止 Grab。</summary>
        public event Action OnStopRequested;

        /// <summary>狀態變更通知（UI 更新用）。</summary>
        public event Action<IoState> OnStateChanged;

        /// <summary>連線狀態變更（connected / disconnected）。</summary>
        public event Action<bool> OnConnectionChanged;

        /// <summary>每次 PollTick 結束時發送所有 IO 快照。</summary>
        public event Action<IoSnapshot> OnIoUpdated;

        /// <summary>
        /// 測試用：設 false 跳過 StartAsync 內的 BackgroundLoop 啟動，
        /// test 可以手動呼 PollTick / ReconnectTick 而不會跟背景 loop race。
        /// 生產代碼預設 true，不需碰。
        /// </summary>
        internal bool AutoBackgroundLoop { get; set; } = true;

        public IoGrabController() : this(new IcpDasModbusTcpClient()) { }

        internal IoGrabController(IModbusTcpClient plcClient)
        {
            _plc = plcClient;
            _plc.ReadWriteTimeoutMs = 2000;
        }

        /// <summary>啟動：嘗試連線 IO module，不論成功失敗都啟動背景 loop（自動重連 + poll）。</summary>
        public async Task StartAsync(string ip, int port = 502)
        {
            _plcIp = ip;
            _plcPort = port;

            bool ok = await _plc.ConnectAsync(ip, port, 3000);
            if (ok)
            {
                try { await EnterIdle(); }
                catch (Exception ex) { IoLogger.Error("EnterIdle on initial connect failed", ex); }
                OnConnectionChanged?.Invoke(true);
                IoLogger.Info($"IO module connected: {ip}:{port}");
            }
            else
            {
                IoLogger.Warn($"IO module initial connect failed ({ip}:{port}), background loop will retry every {ReconnectIntervalMs}ms.");
            }

            // 啟動背景 loop —— 不依賴 message pump，跑在 thread pool 上。
            // IsConnected=true → PollTick；IsConnected=false → ReconnectTick。
            if (AutoBackgroundLoop)
            {
                _bgCts = new CancellationTokenSource();
                _bgTask = Task.Run(() => BackgroundLoop(_bgCts.Token));
            }
        }

        private async Task BackgroundLoop(CancellationToken ct)
        {
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    if (_plc.IsConnected)
                    {
                        await PollTick();
                        try { await Task.Delay(PollIntervalMs, ct); } catch (OperationCanceledException) { break; }
                    }
                    else
                    {
                        await ReconnectTick();
                        if (!_plc.IsConnected)
                        {
                            try { await Task.Delay(ReconnectIntervalMs, ct); } catch (OperationCanceledException) { break; }
                        }
                    }
                }
                catch (OperationCanceledException) { break; }
                catch (Exception ex)
                {
                    IoLogger.Error("BackgroundLoop unexpected error", ex);
                    try { await Task.Delay(1000, ct); } catch { break; }
                }
            }
        }

        /// <summary>停止：清除所有 DO、斷線、停止背景 loop。</summary>
        public async Task StopAsync()
        {
            _bgCts?.Cancel();
            if (_bgTask != null)
            {
                try { await _bgTask; } catch { /* swallow — task cancelled or already faulted */ }
            }

            if (_plc.IsConnected)
            {
                try
                {
                    await _plc.WriteDo(DO_PC_ALIVE, false);
                    await _plc.WriteDo(DO_MURA_DETECTED, false);
                    await _plc.WriteDo(DO_PC_INSPECT, false);
                }
                catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[IoGrabController.Close] {ex.GetType().Name}: {ex.Message}"); }
            }
            _isPcAlive = false;
            _doPcAlive = false;
            _doMura = false;
            _doPcBusy = false;
            SetState(IoState.Closed);
            _plc.Dispose();
            OnConnectionChanged?.Invoke(false);
        }

        /// <summary>通知 IO：Grab 已開始（PC BUSY = High）。</summary>
        public async Task NotifyGrabStarted()
        {
            if (!_plc.IsConnected) return;
            try
            {
                await _plc.WriteDo(DO_PC_INSPECT, true);
                _doPcBusy = true;
            }
            catch (Exception ex) { IoLogger.Error("WriteDo PC_BUSY=true failed", ex); }
        }

        /// <summary>通知 IO：Grab 已停止（PC BUSY = Low）。</summary>
        public async Task NotifyGrabStopped()
        {
            if (!_plc.IsConnected) return;
            try
            {
                await _plc.WriteDo(DO_PC_INSPECT, false);
                _doPcBusy = false;
            }
            catch (Exception ex) { IoLogger.Error("WriteDo PC_BUSY=false failed", ex); }
        }

        /// <summary>通知 IO：檢測到 MURA（MURA = High）。</summary>
        public async Task NotifyMuraDetected()
        {
            if (!_plc.IsConnected) return;
            try
            {
                await _plc.WriteDo(DO_MURA_DETECTED, true);
                _doMura = true;
            }
            catch (Exception ex) { IoLogger.Error("WriteDo MURA=true failed", ex); }
        }

        /// <summary>通知 IO：清除 MURA 信號（MURA = Low）。</summary>
        public async Task ClearMura()
        {
            if (!_plc.IsConnected) return;
            try
            {
                await _plc.WriteDo(DO_MURA_DETECTED, false);
                _doMura = false;
            }
            catch (Exception ex) { IoLogger.Error("WriteDo MURA=false failed", ex); }
        }

        // ── 內部 FSM ──────────────────────────────────────────────────

        private async Task EnterIdle()
        {
            await _plc.WriteDo(DO_PC_ALIVE, true);
            _isPcAlive = true;
            _doPcAlive = true;
            await _plc.WriteDo(DO_MURA_DETECTED, false);
            _doMura = false;
            await _plc.WriteDo(DO_PC_INSPECT, false);
            _doPcBusy = false;
            _lastDiStart = false;
            SetState(IoState.Idle);
        }

        private void SetState(IoState state)
        {
            if (_currentState == state) return;
            IoLogger.Info($"IO State: {_currentState} -> {state}");
            _currentState = state;
            OnStateChanged?.Invoke(state);
        }

        private void FireIoSnapshot(bool diPlcAlive, bool diStart)
        {
            OnIoUpdated?.Invoke(new IoSnapshot
            {
                DiNakanAlive = diPlcAlive,
                DiInspectStart = diStart,
                DoPcAlive  = _doPcAlive,
                DoMuraDetected = _doMura,
                DoPcInspect = _doPcBusy
            });
        }

        internal async Task PollTick()
        {
            try
            {
                // ReadDiStatuses 產生 Modbus 流量，同時餵 ET-7044 Host Watchdog
                var diStates = await _plc.ReadDiStatuses();
                if (diStates == null || diStates.Length < 2) return;

                bool plcAlive = diStates[DI_NAKAN_ALIVE];
                bool diStart  = diStates[DI_INSPECT_START];

                // PLC ALIVE 消失 → Faulted
                if (_isPcAlive && !plcAlive && _currentState != IoState.Faulted && _currentState != IoState.CommLost)
                {
                    IoLogger.Warn("PLC ALIVE lost → Faulted");
                    SetState(IoState.Faulted);
                    try
                    {
                        await _plc.WriteDo(DO_MURA_DETECTED, false);
                        _doMura = false;
                        await _plc.WriteDo(DO_PC_INSPECT, false);
                        _doPcBusy = false;
                    }
                    catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[IoGrabController.Poll.StopCleanup] {ex.GetType().Name}: {ex.Message}"); }
                    OnStopRequested?.Invoke();
                    FireIoSnapshot(plcAlive, diStart);
                    return;
                }

                // PLC ALIVE 恢復
                if (_currentState == IoState.Faulted && plcAlive && _isPcAlive)
                {
                    IoLogger.Info("PLC ALIVE restored → Idle");
                    SetState(IoState.Idle);
                    _lastDiStart = diStart;
                    FireIoSnapshot(plcAlive, diStart);
                    return;
                }

                if (_currentState == IoState.Faulted || _currentState == IoState.CommLost)
                {
                    FireIoSnapshot(plcAlive, diStart);
                    return;
                }

                // START 上升緣 → 開始 Grab
                if (!_lastDiStart && diStart && _currentState == IoState.Idle)
                {
                    IoLogger.Info("START rising edge → Start Grab");
                    SetState(IoState.Running);
                    OnStartRequested?.Invoke();
                }

                // START 下降緣 → 停止 Grab
                if (_lastDiStart && !diStart && (_currentState == IoState.Running || _currentState == IoState.Faulted))
                {
                    IoLogger.Info("START falling edge → Stop Grab");
                    SetState(IoState.Stopping);
                    OnStopRequested?.Invoke();
                    await ClearMura();
                    await NotifyGrabStopped();
                    // Keepalive：多次 WriteDo 後補一次讀取，確保 1.5s Watchdog 不逾時
                    await _plc.ReadDiStatuses();
                    SetState(IoState.Idle);
                }

                _lastDiStart = diStart;
                FireIoSnapshot(plcAlive, diStart);
            }
            catch (Exception ex)
            {
                IoLogger.Error("IO polling error → CommLost", ex);
                SetState(IoState.CommLost);
                _isPcAlive = false;
                _doPcAlive = false;
                _doMura = false;
                _doPcBusy = false;
                OnConnectionChanged?.Invoke(false);
                OnStopRequested?.Invoke();
                FireIoSnapshot(false, false);
                _plc.Dispose();
                // BackgroundLoop 偵測 !IsConnected 後會自動走 ReconnectTick 路徑，不需手動排程。
            }
        }

        internal async Task ReconnectTick()
        {
            bool ok = await _plc.ConnectAsync(_plcIp, _plcPort, 3000);
            if (ok)
            {
                IoLogger.Info("IO module reconnected successfully.");
                try { await EnterIdle(); }
                catch (Exception ex) { IoLogger.Error("EnterIdle after reconnect failed", ex); }
                OnConnectionChanged?.Invoke(true);
            }
            // 失敗時不做任何事，BackgroundLoop 會 Task.Delay(ReconnectIntervalMs) 後再次呼叫。
        }

        public void Dispose()
        {
            _bgCts?.Cancel();
            _bgCts?.Dispose();
            _plc.Dispose();
        }
    }
}
