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
        private int _connectionAccepted;
        private int _reconnectAttemptCount;

        /// <summary>IO module 已完成 TCP + safe-output + DI handshake。</summary>
        public bool IsConnected => Volatile.Read(ref _connectionAccepted) == 1 && _plc.IsConnected;

        /// <summary>目前 FSM 狀態。</summary>
        public IoState CurrentState => _currentState;

        /// <summary>硬體型號。</summary>
        public string Model => "ET-7044";

        /// <summary>Poll 週期（ms）。</summary>
        public int PollIntervalMs { get; set; } = 500;

        /// <summary>
        /// True uses START Low as the capture stop request. False keeps the current capture
        /// running until the app reports that its configured time/height target was reached.
        /// </summary>
        public bool StopCaptureOnStartLow { get; set; } = true;

        /// <summary>重連週期（ms）。</summary>
        public int ReconnectIntervalMs { get; set; } = 5000;

        // 下次自動重連的預定時刻（UTC ticks，0=未排程）。供 UI 顯示「重連中 Ns」倒數，秒數源自此處（單一真實來源）。
        private long _nextReconnectAtTicksUtc;

        /// <summary>下次自動重連預定時刻（UTC）；已連線或未排程時回 null。供 UI 倒數顯示。</summary>
        public DateTime? NextReconnectAtUtc
        {
            get
            {
                long t = System.Threading.Interlocked.Read(ref _nextReconnectAtTicksUtc);
                return t == 0 ? (DateTime?)null : new DateTime(t, DateTimeKind.Utc);
            }
        }

        internal static int CalculateReconnectDelayMs(int intervalMs, int elapsedMs)
        {
            return Math.Max(0, intervalMs - Math.Max(0, elapsedMs));
        }

        /// <summary>讀寫逾時（ms）。調小 → 斷線偵測更快（健康設備回應 &lt;100ms，故可安全縮短）。</summary>
        public int ReadWriteTimeoutMs
        {
            get => _plc.ReadWriteTimeoutMs;
            set => _plc.ReadWriteTimeoutMs = value;
        }

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

        public IoGrabController(string model = "ET-7044") : this(IoModuleFactory.Create(model)) { }

        internal IoGrabController(IModbusTcpClient plcClient)
        {
            _plc = plcClient;
            _plc.ReadWriteTimeoutMs = 2000;
        }

        /// <summary>啟動：嘗試連線 IO module，不論成功失敗都啟動背景 loop（自動重連 + poll）。</summary>
        public async Task StartAsync(string ip, int port = 502)
        {
            Volatile.Write(ref _connectionAccepted, 0);
            Interlocked.Exchange(ref _reconnectAttemptCount, 0);
            _plcIp = ip;
            _plcPort = port;

            bool tcpConnected = await _plc.ConnectAsync(ip, port, 3000);
            bool accepted = tcpConnected && await TryAcceptConnectedModule("initial", IoState.Disconnected);
            if (accepted)
            {
                OnConnectionChanged?.Invoke(true);
                IoLogger.Info($"IO module connected: {ip}:{port}");
            }
            else if (!tcpConnected)
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
                        System.Threading.Interlocked.Exchange(ref _nextReconnectAtTicksUtc, 0); // 已連線 → 清除倒數
                        await PollTick();
                        try { await Task.Delay(PollIntervalMs, ct); } catch (OperationCanceledException) { break; }
                    }
                    else
                    {
                        DateTime attemptStartedUtc = DateTime.UtcNow;
                        await ReconnectTick();
                        if (!_plc.IsConnected)
                        {
                            // ReconnectIntervalMs 是「兩次嘗試起點」的週期。ConnectAsync 本身可能
                            // 已等待數秒，不可在它後面再完整 delay 一次，否則 3s 設定會變成最差 6s。
                            int elapsedMs = (int)(DateTime.UtcNow - attemptStartedUtc).TotalMilliseconds;
                            int delayMs = CalculateReconnectDelayMs(ReconnectIntervalMs, elapsedMs);
                            System.Threading.Interlocked.Exchange(ref _nextReconnectAtTicksUtc,
                                DateTime.UtcNow.AddMilliseconds(delayMs).Ticks);
                            try { await Task.Delay(delayMs, ct); } catch (OperationCanceledException) { break; }
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
            Volatile.Write(ref _connectionAccepted, 0);
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
            if (!IsConnected) return;
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
            if (!IsConnected) return;
            try
            {
                await _plc.WriteDo(DO_PC_INSPECT, false);
                _doPcBusy = false;
            }
            catch (Exception ex) { IoLogger.Error("WriteDo PC_BUSY=false failed", ex); }
        }

        public async Task NotifyFixedGrabCompleted()
        {
            await NotifyGrabStopped();
            if (_currentState != IoState.Running &&
                _currentState != IoState.AwaitingStartLow)
                return;

            SetState(_lastDiStart
                ? IoState.AwaitingStartLow
                : IoState.Idle);
        }

        /// <summary>
        /// App 暫時無法開始 Grab：BUSY 保持 Low，FSM 回 Idle。
        /// 若 START 仍為 High，下一次 PollTick 會重試；成功進入 Running 後同一段 High 不再重複。
        /// </summary>
        public async Task NotifyGrabStartRejected()
        {
            await NotifyGrabStopped();
            if (_currentState == IoState.Running)
                SetState(IoState.Idle);
        }

        /// <summary>通知 IO：檢測到 MURA（MURA = High）。</summary>
        public async Task NotifyMuraDetected()
        {
            if (!IsConnected) return;
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
            if (!IsConnected) return;
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
            // Clear business outputs before publishing PC ALIVE. A reconnect may
            // inherit stale remote coils from the previous transport session.
            await _plc.WriteDo(DO_MURA_DETECTED, false);
            _doMura = false;
            await _plc.WriteDo(DO_PC_INSPECT, false);
            _doPcBusy = false;
            await _plc.WriteDo(DO_PC_ALIVE, true);
            _isPcAlive = true;
            _doPcAlive = true;
            _lastDiStart = false;
        }

        /// <summary>
        /// TCP connected is not enough: publish connected only after safe DO initialization
        /// and one valid Modbus DI response. DI values are not consumed here, so a held-high
        /// START still becomes a rising edge on the next PollTick.
        /// </summary>
        private async Task<bool> TryAcceptConnectedModule(string phase, IoState failureState)
        {
            try
            {
                await EnterIdle();
                bool[] di = await _plc.ReadDiStatuses();
                if (di == null || di.Length < 2)
                    throw new InvalidOperationException("Handshake returned an invalid DI response");
                Volatile.Write(ref _connectionAccepted, 1);
                SetState(IoState.Idle);
                return true;
            }
            catch (Exception ex)
            {
                IoLogger.Error($"IO {phase} handshake failed; connection rejected", ex);
                _isPcAlive = false;
                _doPcAlive = false;
                _doMura = false;
                _doPcBusy = false;
                Volatile.Write(ref _connectionAccepted, 0);
                _plc.Dispose();
                SetState(failureState);
                return false;
            }
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

                // START is level-sensitive while Idle. A rejected request returns to Idle, so a
                // held HIGH is retried after the previous tail or camera preparation finishes.
                // Running consumes the current HIGH and LOW rearms the next capture.
                if (diStart && _currentState == IoState.Idle)
                {
                    IoLogger.Info(
                        _lastDiStart
                            ? "START held high -> Retry Start Grab"
                            : "START rising/high level -> Start Grab");
                    SetState(IoState.Running);
                    OnStartRequested?.Invoke();
                }

                if (_lastDiStart && !diStart &&
                    _currentState == IoState.AwaitingStartLow)
                {
                    IoLogger.Info("START low -> Fixed capture rearmed");
                    SetState(IoState.Idle);
                }
                else if (_lastDiStart && !diStart &&
                    (_currentState == IoState.Running || _currentState == IoState.Faulted) &&
                    !StopCaptureOnStartLow)
                {
                    IoLogger.Info("START falling edge -> Capture continues to fixed target");
                }
                // START 下降緣 → 停止 Grab
                else if (_lastDiStart && !diStart && (_currentState == IoState.Running || _currentState == IoState.Faulted))
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
                Volatile.Write(ref _connectionAccepted, 0);
                OnConnectionChanged?.Invoke(false);
                OnStopRequested?.Invoke();
                FireIoSnapshot(false, false);
                _plc.Dispose();
                // BackgroundLoop 偵測 !IsConnected 後會自動走 ReconnectTick 路徑，不需手動排程。
            }
        }

        internal async Task ReconnectTick()
        {
            Volatile.Write(ref _connectionAccepted, 0);
            int attempt = Interlocked.Increment(ref _reconnectAttemptCount);
            bool tcpConnected = await _plc.ConnectAsync(_plcIp, _plcPort, 3000);
            bool accepted = false;
            if (tcpConnected)
            {
                IoState failureState = _currentState == IoState.Disconnected
                    ? IoState.Disconnected
                    : IoState.CommLost;
                accepted = await TryAcceptConnectedModule("reconnect", failureState);
                if (accepted)
                {
                    Interlocked.Exchange(ref _nextReconnectAtTicksUtc, 0);
                    Interlocked.Exchange(ref _reconnectAttemptCount, 0);
                    IoLogger.Info($"IO module reconnected and handshake verified (attempt {attempt}).");
                    OnConnectionChanged?.Invoke(true);
                }
            }

            if (!accepted && (attempt == 1 || attempt % 10 == 0))
            {
                string stage = tcpConnected ? "handshake rejected" : "TCP unavailable";
                IoLogger.Warn($"IO reconnect pending: attempt {attempt}, {stage} ({_plcIp}:{_plcPort}).");
            }
            // 失敗時不做任何事，BackgroundLoop 會 Task.Delay(ReconnectIntervalMs) 後再次呼叫。
        }

        public void Dispose()
        {
            Volatile.Write(ref _connectionAccepted, 0);
            _bgCts?.Cancel();
            _bgCts?.Dispose();
            _plc.Dispose();
        }
    }
}
