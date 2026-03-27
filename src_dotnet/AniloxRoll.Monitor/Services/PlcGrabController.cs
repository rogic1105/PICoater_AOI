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
    public class PlcGrabController : IDisposable
    {
        private const int DI_PLC_ALIVE = 0;
        private const int DI_START     = 1;
        private const int DO_PC_ALIVE  = 0;
        private const int DO_MURA      = 1;
        private const int DO_PC_BUSY   = 2;

        private readonly IcpDasModbusTcpClient _plc = new IcpDasModbusTcpClient();
        private readonly Timer _pollTimer;
        private readonly Timer _reconnectTimer;

        private bool _lastDiStart;
        private bool _isPcAlive;
        private string _currentState = "Initial";
        private string _plcIp = "192.168.1.1";
        private int _plcPort = 502;

        /// <summary>PLC 是否已連線。</summary>
        public bool IsConnected => _plc.IsConnected;

        /// <summary>目前 FSM 狀態。</summary>
        public string CurrentState => _currentState;

        /// <summary>PLC START 上升緣 → 要求開始 Grab。</summary>
        public event Action OnStartRequested;

        /// <summary>PLC START 下降緣 → 要求停止 Grab。</summary>
        public event Action OnStopRequested;

        /// <summary>狀態變更通知（UI 更新用）。</summary>
        public event Action<string> OnStateChanged;

        /// <summary>連線狀態變更（connected / disconnected）。</summary>
        public event Action<bool> OnConnectionChanged;

        public PlcGrabController()
        {
            _plc.ReadWriteTimeoutMs = 2000;

            _pollTimer = new Timer { Interval = 500 };
            _pollTimer.Tick += async (s, e) => await PollTick();

            _reconnectTimer = new Timer { Interval = 5000 };
            _reconnectTimer.Tick += async (s, e) => await ReconnectTick();
        }

        /// <summary>啟動：嘗試連線 PLC，成功則進入 Ready 狀態，失敗則啟動背景重連。</summary>
        public async Task StartAsync(string ip, int port = 502)
        {
            _plcIp = ip;
            _plcPort = port;

            bool ok = await _plc.ConnectAsync(ip, port, 3000);
            if (ok)
            {
                await EnterReady();
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
                    await _plc.WriteDo(DO_MURA, false);
                    await _plc.WriteDo(DO_PC_BUSY, false);
                }
                catch { }
            }
            _isPcAlive = false;
            SetState("Closed");
            _plc.Dispose();
            OnConnectionChanged?.Invoke(false);
        }

        /// <summary>通知 PLC：Grab 已開始（PC BUSY = High）。</summary>
        public async Task NotifyGrabStarted()
        {
            if (!_plc.IsConnected) return;
            try { await _plc.WriteDo(DO_PC_BUSY, true); }
            catch (Exception ex) { PlcLogger.Error("WriteDo PC_BUSY=true failed", ex); }
        }

        /// <summary>通知 PLC：Grab 已停止（PC BUSY = Low）。</summary>
        public async Task NotifyGrabStopped()
        {
            if (!_plc.IsConnected) return;
            try { await _plc.WriteDo(DO_PC_BUSY, false); }
            catch (Exception ex) { PlcLogger.Error("WriteDo PC_BUSY=false failed", ex); }
        }

        /// <summary>通知 PLC：檢測到 MURA（MURA = High）。</summary>
        public async Task NotifyMuraDetected()
        {
            if (!_plc.IsConnected) return;
            try { await _plc.WriteDo(DO_MURA, true); }
            catch (Exception ex) { PlcLogger.Error("WriteDo MURA=true failed", ex); }
        }

        /// <summary>通知 PLC：清除 MURA 信號（MURA = Low）。</summary>
        public async Task ClearMura()
        {
            if (!_plc.IsConnected) return;
            try { await _plc.WriteDo(DO_MURA, false); }
            catch (Exception ex) { PlcLogger.Error("WriteDo MURA=false failed", ex); }
        }

        // ── 內部 FSM ──────────────────────────────────────────────────

        private async Task EnterReady()
        {
            await _plc.WriteDo(DO_PC_ALIVE, true);
            _isPcAlive = true;
            await _plc.WriteDo(DO_MURA, false);
            await _plc.WriteDo(DO_PC_BUSY, false);
            _lastDiStart = false;
            SetState("Ready");
        }

        private void SetState(string state)
        {
            if (_currentState == state) return;
            PlcLogger.Info($"PLC State: {_currentState} -> {state}");
            _currentState = state;
            OnStateChanged?.Invoke(state);
        }

        private async Task PollTick()
        {
            _pollTimer.Stop();
            try
            {
                var diStates = await _plc.ReadDiStatuses();
                if (diStates == null || diStates.Length < 2) return;

                bool plcAlive = diStates[DI_PLC_ALIVE];
                bool diStart  = diStates[DI_START];

                // PLC ALIVE 消失 → PLCErr
                if (_isPcAlive && !plcAlive && _currentState != "PLCErr" && _currentState != "PCErr")
                {
                    PlcLogger.Warn("PLC ALIVE lost → PLCErr");
                    SetState("PLCErr");
                    try
                    {
                        await _plc.WriteDo(DO_MURA, false);
                        await _plc.WriteDo(DO_PC_BUSY, false);
                    }
                    catch { }
                    OnStopRequested?.Invoke();
                    _pollTimer.Start();
                    return;
                }

                // PLC ALIVE 恢復
                if (_currentState == "PLCErr" && plcAlive && _isPcAlive)
                {
                    PlcLogger.Info("PLC ALIVE restored → Ready");
                    SetState("Ready");
                    _lastDiStart = diStart;
                    _pollTimer.Start();
                    return;
                }

                if (_currentState == "PLCErr" || _currentState == "PCErr")
                {
                    _pollTimer.Start();
                    return;
                }

                // START 上升緣 → 開始 Grab
                if (!_lastDiStart && diStart && (_currentState == "Ready"))
                {
                    PlcLogger.Info("START rising edge → Start Grab");
                    SetState("Run");
                    OnStartRequested?.Invoke();
                }

                // START 下降緣 → 停止 Grab
                if (_lastDiStart && !diStart && (_currentState == "Run" || _currentState == "MURA"))
                {
                    PlcLogger.Info("START falling edge → Stop Grab");
                    SetState("Stop");
                    OnStopRequested?.Invoke();
                    await ClearMura();
                    await NotifyGrabStopped();
                    SetState("Ready");
                }

                _lastDiStart = diStart;
                _pollTimer.Start();
            }
            catch (Exception ex)
            {
                PlcLogger.Error("PLC polling error → PCErr", ex);
                SetState("PCErr");
                _isPcAlive = false;
                OnConnectionChanged?.Invoke(false);
                OnStopRequested?.Invoke();
                _plc.Dispose();
                _reconnectTimer.Start();
            }
        }

        private async Task ReconnectTick()
        {
            _reconnectTimer.Stop();
            bool ok = await _plc.ConnectAsync(_plcIp, _plcPort, 3000);
            if (ok)
            {
                PlcLogger.Info("PLC reconnected successfully.");
                await EnterReady();
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
