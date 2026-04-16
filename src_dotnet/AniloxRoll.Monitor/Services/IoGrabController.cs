using System;
using System.Threading.Tasks;
using System.Windows.Forms;
using PlcBridge.Core;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// PLC-PC 交握控制器：自動偵測 PLC 連線，連線時以 DI START 信號控制 Grab 啟停。
    /// 未連線時退回 UI 按鈕控制。
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
        private readonly Timer _pollTimer;
        private readonly Timer _reconnectTimer;

        private bool _lastDiStart;
        private bool _isPcAlive;
        private IoState _currentState = IoState.Disconnected;
        private string _plcIp = "192.168.255.1";
        private int _plcPort = 502;

        // DO 狀態追蹤（供 IO 快照使用）
        private bool _doPcAlive;
        private bool _doMura;
        private bool _doPcBusy;

        /// <summary>PLC 是否已連線。</summary>
        public bool IsConnected => _plc.IsConnected;

        /// <summary>目前 FSM 狀態。</summary>
        public IoState CurrentState => _currentState;

        /// <summary>硬體型號。</summary>
        public string Model => "ET-7044";

        /// <summary>Poll 週期（ms）。</summary>
        public int PollIntervalMs => _pollTimer.Interval;

        /// <summary>重連週期（ms）。</summary>
        public int ReconnectIntervalMs => _reconnectTimer.Interval;

        /// <summary>讀寫逾時（ms）。</summary>
        public int ReadWriteTimeoutMs => _plc.ReadWriteTimeoutMs;

        /// <summary>目前連線 IP。</summary>
        public string PlcIp => _plcIp;

        /// <summary>目前連線 Port。</summary>
        public int PlcPort => _plcPort;

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

        public IoGrabController() : this(new IcpDasModbusTcpClient()) { }

        internal IoGrabController(IModbusTcpClient plcClient)
        {
            _plc = plcClient;
            _plc.ReadWriteTimeoutMs = 2000;

            _pollTimer = new Timer { Interval = 500 };
            _pollTimer.Tick += async (s, e) => await PollTick();

            _reconnectTimer = new Timer { Interval = 5000 };
            _reconnectTimer.Tick += async (s, e) => await ReconnectTick();
        }

        /// <summary>啟動：嘗試連線 PLC，成功則進入 Idle 狀態，失敗則啟動背景重連。</summary>
        public async Task StartAsync(string ip, int port = 502)
        {
            _plcIp = ip;
            _plcPort = port;

            bool ok = await _plc.ConnectAsync(ip, port, 3000);
            if (ok)
            {
                await EnterIdle();
                _pollTimer.Start();
                OnConnectionChanged?.Invoke(true);
                PlcLogger.Info($"PLC connected: {ip}:{port}");
            }
            else
            {
                PlcLogger.Warn($"PLC initial connect failed ({ip}:{port}), will retry in background.");
                _reconnectTimer.Start();
            }
        }

        /// <summary>停止：清除所有 DO、斷線。</summary>
        public async Task StopAsync()
        {
            _pollTimer.Stop();
            _reconnectTimer.Stop();
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

        /// <summary>通知 PLC：Grab 已開始（PC BUSY = High）。</summary>
        public async Task NotifyGrabStarted()
        {
            if (!_plc.IsConnected) return;
            try
            {
                await _plc.WriteDo(DO_PC_INSPECT, true);
                _doPcBusy = true;
            }
            catch (Exception ex) { PlcLogger.Error("WriteDo PC_BUSY=true failed", ex); }
        }

        /// <summary>通知 PLC：Grab 已停止（PC BUSY = Low）。</summary>
        public async Task NotifyGrabStopped()
        {
            if (!_plc.IsConnected) return;
            try
            {
                await _plc.WriteDo(DO_PC_INSPECT, false);
                _doPcBusy = false;
            }
            catch (Exception ex) { PlcLogger.Error("WriteDo PC_BUSY=false failed", ex); }
        }

        /// <summary>通知 PLC：檢測到 MURA（MURA = High）。</summary>
        public async Task NotifyMuraDetected()
        {
            if (!_plc.IsConnected) return;
            try
            {
                await _plc.WriteDo(DO_MURA_DETECTED, true);
                _doMura = true;
            }
            catch (Exception ex) { PlcLogger.Error("WriteDo MURA=true failed", ex); }
        }

        /// <summary>通知 PLC：清除 MURA 信號（MURA = Low）。</summary>
        public async Task ClearMura()
        {
            if (!_plc.IsConnected) return;
            try
            {
                await _plc.WriteDo(DO_MURA_DETECTED, false);
                _doMura = false;
            }
            catch (Exception ex) { PlcLogger.Error("WriteDo MURA=false failed", ex); }
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
            PlcLogger.Info($"PLC State: {_currentState} -> {state}");
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
            _pollTimer.Stop();
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
                    PlcLogger.Warn("PLC ALIVE lost → Faulted");
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
                    _pollTimer.Start();
                    return;
                }

                // PLC ALIVE 恢復
                if (_currentState == IoState.Faulted && plcAlive && _isPcAlive)
                {
                    PlcLogger.Info("PLC ALIVE restored → Idle");
                    SetState(IoState.Idle);
                    _lastDiStart = diStart;
                    FireIoSnapshot(plcAlive, diStart);
                    _pollTimer.Start();
                    return;
                }

                if (_currentState == IoState.Faulted || _currentState == IoState.CommLost)
                {
                    FireIoSnapshot(plcAlive, diStart);
                    _pollTimer.Start();
                    return;
                }

                // START 上升緣 → 開始 Grab
                if (!_lastDiStart && diStart && _currentState == IoState.Idle)
                {
                    PlcLogger.Info("START rising edge → Start Grab");
                    SetState(IoState.Running);
                    OnStartRequested?.Invoke();
                }

                // START 下降緣 → 停止 Grab
                if (_lastDiStart && !diStart && (_currentState == IoState.Running || _currentState == IoState.Faulted))
                {
                    PlcLogger.Info("START falling edge → Stop Grab");
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
                _pollTimer.Start();
            }
            catch (Exception ex)
            {
                PlcLogger.Error("PLC polling error → CommLost", ex);
                SetState(IoState.CommLost);
                _isPcAlive = false;
                _doPcAlive = false;
                _doMura = false;
                _doPcBusy = false;
                OnConnectionChanged?.Invoke(false);
                OnStopRequested?.Invoke();
                FireIoSnapshot(false, false);
                _plc.Dispose();
                _reconnectTimer.Start();
            }
        }

        internal async Task ReconnectTick()
        {
            _reconnectTimer.Stop();
            bool ok = await _plc.ConnectAsync(_plcIp, _plcPort, 3000);
            if (ok)
            {
                PlcLogger.Info("PLC reconnected successfully.");
                await EnterIdle();
                _pollTimer.Start();
                OnConnectionChanged?.Invoke(true);
            }
            else
            {
                _reconnectTimer.Start();
            }
        }

        public void Dispose()
        {
            _pollTimer.Stop();
            _pollTimer.Dispose();
            _reconnectTimer.Stop();
            _reconnectTimer.Dispose();
            _plc.Dispose();
        }
    }
}
