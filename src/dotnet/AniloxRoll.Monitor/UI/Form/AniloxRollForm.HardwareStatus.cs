using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Management;
using System.Windows.Forms;
using StorageBridge.Core;
using LightBridge.Core;
using MilGrabber.Core;
using TanukiCv.Controls;
using TanukiCv.Utils;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Interop;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.State;
using AniloxRoll.Monitor.UI.Managers;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Presenters;
using AniloxRoll.Monitor.UI.Widgets;

namespace AniloxRoll.Monitor.Forms
{
    /// <summary>AniloxRollForm IO / 光源 / 儲存硬體狀態（初始化 + 連線標籤 + LED）相關方法 — 由主檔拆出的 partial。</summary>
    public partial class AniloxRollForm
    {
        /// <summary>初始化 IO 連動：自動偵測連線，連上後以 DI START 控制 Grab。</summary>
        private void InitIoController()
        {
            if (!_settings.IoEnabled) return;

            _ioGrabController = new IoGrabController(_settings.IoModel);
            _ioGrabController.ReconnectIntervalMs = 3000;   // 重連週期 5s→3s（TCP 連線嘗試很輕量）
            _ioGrabController.ReadWriteTimeoutMs = 500;     // 讀寫逾時 2000→500ms：斷線偵測更快（健康設備回應 <100ms，零資源代價）

            // 背景 Modbus 輪詢執行緒回 UI 更新；關閉時 Handle 已銷毀 → SafeBeginInvoke 守 guard 防 InvalidOperationException
            _ioGrabController.OnStartRequested += () => SafeBeginInvoke(IoStartGrab);

            _ioGrabController.OnStopRequested += () => SafeBeginInvoke(IoStopGrab);

            _ioGrabController.OnStateChanged += state => SafeBeginInvoke(() => UpdateIoStateLabel(state));

            _ioGrabController.OnConnectionChanged += connected => SafeBeginInvoke(() => UpdateIoConnectionUi(connected));

            _ioGrabController.OnIoUpdated += snapshot => SafeBeginInvoke(() => UpdateIoLeds(snapshot));

            // 背景嘗試連線（不阻塞 Form 顯示）
            _ = _ioGrabController.StartAsync(_settings.IoIp, _settings.IoPort);
        }

        private void InitLightController()
        {
            if (!_settings.LightEnabled) return;

            // AutoDetect 會先試設定 COM，失敗則掃描全部 port —— 這是阻塞數秒的序列埠 IO。
            // 過去直接跑在 UI 執行緒（InitServiceLayer）→ 啟動期間整個視窗拖不動（光源是真正瓶頸，不是相機）。
            // 改為背景偵測，完成後 marshal 回 UI 接管 _lightController + 更新 COM 設定 + 刷狀態燈。
            string preferredPort = _settings.LightComPort;
            int channel = _settings.LightChannel;
            var controller = new LightController();
            _lightProbeInFlight = true;   // 佔住，避免 telemetry 重連 probe 同時也跑 AutoDetect 搶 COM
            Task.Run(() =>
            {
                string found;
                try { found = controller.AutoDetect(preferredPort, channel); }
                finally { _lightProbeInFlight = false; }
                SafeBeginInvoke(() =>
                {
                    _lightProbedOnce = true;   // 初次偵測完成 → label 從「初始化中」切到實際狀態
                    if (_settings == null || !_settings.LightEnabled || IsDisposed || Disposing)
                    {
                        controller.Dispose();
                        return;
                    }
                    if (found == null)
                    {
                        System.Diagnostics.Trace.WriteLine("[Light] 光源控制器: NA（設定 " + preferredPort + " + 全 port 掃描均無回應）");
                        controller.Dispose();
                        _lightController = null;
                        UpdateLightConnLabel();
                        return;
                    }
                    _lightController?.Dispose();
                    _lightController = controller;
                    // 掃描找到但非原設定 → 更新記錄的 COM（下次啟動直接命中）。SetBatch（save only no event）避免遞迴。
                    if (!string.Equals(found, preferredPort, StringComparison.OrdinalIgnoreCase))
                    {
                        _settingsHub.SetBatch(s => s.LightComPort = found);
                        RefreshGridItem(nameof(InspectionSettings.LightComPort));
                    }
                    UpdateLightConnLabel();
                });
            });
        }

        private void IoStartGrab()
        {
            if (_isIoSuspended) return;
            if (_liveCameraManager == null || _liveCameraManager.IsLiveGrabbing) return;
            if (IsStandardBgSubEnabled && !IsBgBinReady())
            {
                System.Diagnostics.Trace.TraceWarning("[IoStartGrab] StandardBgSub 無背景 bin，自動取得背景後接續 grab");
                _autoStartGrabAfterBg = true;
                btnLiveGetBackground_Click(null, null);
                return;
            }
            btnLiveGrab_Click(null, null);
            _ = _ioGrabController?.NotifyGrabStarted();
        }

        private void IoStopGrab()
        {
            if (_isIoSuspended) return;
            if (_liveCameraManager == null || !_liveCameraManager.IsLiveGrabbing) return;
            btnLiveGrab_Click(null, null);
            _ = _ioGrabController?.NotifyGrabStopped();
        }

        private void LightTurnOn()
        {
            if (_lightController == null || !_lightController.IsConnected) return;
            _lightController.TurnOn(_settings.LightChannel, _settings.LightBrightness);
        }

        private void LightTurnOff()
        {
            if (_lightController == null || !_lightController.IsConnected) return;
            _lightController.TurnOff(_settings.LightChannel);
        }

        /// <summary>
        /// 光源 PropertyGrid 變更 → 立即生效：
        /// - LightEnabled false→true：啟動偵測；true→false：關閉連線
        /// - COM Port / 通道變更：重新偵測
        /// - 亮度變更：立即套用到硬體（若正在點燈，連同 TurnOn 更新輸出）
        /// </summary>
        private void HandleLightSettingsChanged(string changedPropertyName)
        {
            switch (changedPropertyName)
            {
                case nameof(InspectionSettings.LightEnabled):
                    if (_settings.LightEnabled)
                    {
                        if (_lightController == null) InitLightController();
                    }
                    else
                    {
                        _lightController?.Dispose();
                        _lightController = null;
                    }
                    break;

                case nameof(InspectionSettings.LightComPort):
                case nameof(InspectionSettings.LightChannel):
                    if (_settings.LightEnabled)
                    {
                        _lightController?.Dispose();
                        _lightController = null;
                        InitLightController();
                    }
                    break;

                case nameof(InspectionSettings.LightBrightness):
                    if (_lightController != null && _lightController.IsConnected)
                        _lightController.SetBrightness(_settings.LightChannel, _settings.LightBrightness);
                    UpdateLightConnLabel();
                    break;
            }
        }

        /// <summary>IO 設定變更（IP/Port/型號/啟用）→ 重啟 IO controller，改完數值立即生效（不用重開程式）。</summary>
        private void HandleIoSettingsChanged(string changedPropertyName)
        {
            switch (changedPropertyName)
            {
                case nameof(InspectionSettings.IoEnabled):
                case nameof(InspectionSettings.IoIp):
                case nameof(InspectionSettings.IoPort):
                case nameof(InspectionSettings.IoModel):
                    RestartIoController();
                    break;
            }
        }

        /// <summary>停掉舊 IO controller 並依目前設定重建（IoEnabled=false 時 InitIoController 會 early-return＝關閉）。
        /// async void：StopAsync 在背景跑、不阻塞 dispatcher；重建期間 _ioGrabController 短暫為 null（讀取端皆 null-safe）。</summary>
        private async void RestartIoController()
        {
            if (_ioGrabController != null)
            {
                try { await _ioGrabController.StopAsync(); } catch { }
                _ioGrabController.Dispose();
                _ioGrabController = null;
            }
            UpdateIoConnectionUi(false);   // 立即顯示斷線（重連中），避免殘留舊 IP 的「已連線」狀態
            InitIoController();            // 用新設定（IP/Port/型號）重建並背景連線
        }

        private void UpdateIoStateLabel(IoState state)
        {
            if (_isIoSuspended) return;
            string text;
            Color bgColor;
            switch (state)
            {
                case IoState.Idle:      text = "待機";   bgColor = IecGreen;  break;
                case IoState.Running:   text = "取像";   bgColor = IecBlue;   break;
                case IoState.Stopping:  text = "停止";   bgColor = IecYellow; break;
                case IoState.Faulted:   text = "故障";   bgColor = IecRed;    break;
                case IoState.CommLost:  text = "斷線";   bgColor = IecRed;    break;
                case IoState.Closed:    text = "關閉";   bgColor = IecGray;   break;
                default:                text = "未連線"; bgColor = IecGray;   break;  // Disconnected
            }
            lblIoState.Text = text;
            lblIoState.BackColor = bgColor;
        }

        /// <summary>啟動時硬體狀態列先顯示「初始化中」（灰）—— 各硬體連線/偵測完成後由各自 Update*Label 接手。
        /// 只對「已啟用/已設定」的硬體顯示，避免停用項目卡在「初始化中」。</summary>
        private void ShowHardwareStatusInitializing()
        {
            if (_settings == null) return;
            if (_settings.IoEnabled)
            {
                lblIoConn.Text = "● IO: 初始化中…";  lblIoConn.BackColor = IecGray;
            }
            if (_settings.LightEnabled)
            {
                lblLightConn.Text = "● 光源: 初始化中…";  lblLightConn.BackColor = IecGray;
            }
            if (!string.IsNullOrWhiteSpace(_settings.RemotePath))
            {
                lblStorageConn.Text = "● 儲存電腦: 初始化中…";  lblStorageConn.BackColor = IecGray;
            }
        }

        private void UpdateIoConnectionUi(bool connected)
        {
            if (_isIoSuspended) return;
            // 「按鈕是否變 IO 控制中」交給 RefreshGrabButtonState 統一決定（相機未就緒前不被 IO 搶顯示）。
            lblIoConn.ForeColor = Color.White;   // 統一白字
            if (connected)
            {
                lblIoConn.Text = "● IO 已連線";
                lblIoConn.BackColor = IecGreen;
            }
            else
            {
                // 不顯示靜止「IO 離線」：斷線一律走重連倒數（RefreshIoConnLabel 每 tick 補上秒數）
                lblIoConn.Text = "● IO 重連中…";
                lblIoConn.BackColor = IecRed;
            }
            RefreshGrabButtonState();
        }

        /// <summary>由 TelemetryTimer 每 tick 呼叫：IO 斷線時顯示重連倒數（秒數源自 IoGrabController）。
        /// 手動暫停（_isIoSuspended）不覆蓋；初始連線中（尚未排程重連）維持「初始化中」。</summary>
        // ── H 系列：硬體連線邊緣留痕（IO/光源/儲存；狀態轉變才記一行，斷線/恢復現場排障關鍵）──
        private bool? _lastFlowIoConn, _lastFlowLightConn, _lastFlowStorageConn;

        private void FlowHardwareEdges()
        {
            void Edge(ref bool? last, bool now, string name)
            {
                if (last == now) return;
                // 首次觀測（null→值）只記基線斷線（開機就連不上也值得留痕）；恢復/斷線轉變一律記
                if (last.HasValue)
                    FlowTrace.Log(now ? $"{name} 恢復連線" : $"⚠ {name} 斷線");
                else if (!now)
                    FlowTrace.Log($"⚠ {name} 未連線（開機基線）");
                last = now;
            }
            Edge(ref _lastFlowIoConn, _ioGrabController?.IsConnected == true, "IO");
            if (_settings?.LightEnabled == true)
                Edge(ref _lastFlowLightConn, _lightController?.IsConnected == true, "光源");
            if (!string.IsNullOrWhiteSpace(_settings?.RemotePath))
                Edge(ref _lastFlowStorageConn, _storageLastConnected == true, "儲存電腦");
        }

        private void RefreshIoConnLabel()
        {
            if (_ioGrabController == null || _isIoSuspended) return;  // 手動暫停 → 保留「IO 暫停 ⏸」
            if (_ioGrabController.IsConnected)
            {
                if (lblIoConn.BackColor != IecGreen) UpdateIoConnectionUi(true);  // 連上 → 綠（idempotent）
                return;
            }
            var next = _ioGrabController.NextReconnectAtUtc;
            if (!next.HasValue) return;  // 尚未排程重連（初始連線進行中）→ 維持「初始化中」
            int sec = (int)Math.Ceiling((next.Value - DateTime.UtcNow).TotalSeconds);
            sec = Math.Max(1, Math.Min(sec, _ioGrabController.ReconnectIntervalMs / 1000));
            lblIoConn.Text = $"● IO 重連中 {sec}s…";
            lblIoConn.BackColor = IecRed;
            lblIoConn.ForeColor = Color.White;
        }

        private void UpdateLightConnLabel()
        {
            if (_settings == null || !_settings.LightEnabled)
            {
                lblLightConn.Text = "● 光源 停用";
                lblLightConn.BackColor = IecGray;
                return;
            }
            if (_lightController != null && _lightController.IsConnected)
            {
                lblLightConn.Text = $"● 光源 已連線 ({_settings.LightBrightness})";
                lblLightConn.BackColor = IecGreen;
            }
            else if (!_lightProbedOnce)
            {
                // 初次偵測還沒回來 → 維持「初始化中」（與 IO/儲存一致）
                lblLightConn.Text = "● 光源: 初始化中…";
                lblLightConn.BackColor = IecGray;
            }
            else
            {
                // 斷線 → 顯示重連倒數（探測進行中時顯示「探測中」）。秒數源自 LightProbeIntervalTicks。
                lblLightConn.Text = _lightProbeInFlight
                    ? "● 光源 探測中…"
                    : $"● 光源 重連中 {CountdownSec(_lightProbeTickCounter, LightProbeIntervalTicks)}s…";
                lblLightConn.BackColor = IecRed;
            }

            UpdateStandardBgSubLockState();
        }

        private int _storageProbeTickCounter;
        private volatile bool _storageProbeInFlight;
        private int _lightProbeTickCounter;
        private volatile bool _lightProbeInFlight;
        private bool? _storageLastConnected;   // 最後一次 probe 結果（供每 tick 刷新倒數用）
        private bool _lightProbedOnce;         // 光源初次偵測是否完成（未完成前 label 維持「初始化中」）

        // ── 重連倒數 / probe 排程的單一真實來源（秒數由此推導，不寫死）──────────────
        internal const int TelemetryTickMs = 500;          // = SettingsTabs 的 _telemetryTimer.Interval
        private const int LightProbeIntervalTicks = 4;     // 光源每 4 tick(2s) 背景 probe / 重連一次
        private const int StorageProbeIntervalTicks = 4;   // 儲存每 4 tick(2s) 背景 probe 一次

        /// <summary>倒數剩餘秒數 = (intervalTicks − 已過 ticks) × tick 間隔，至少 1。</summary>
        private static int CountdownSec(int elapsedTicks, int intervalTicks)
            => Math.Max(1, (int)Math.Ceiling((intervalTicks - elapsedTicks) * TelemetryTickMs / 1000.0));

        private void UpdateStorageConnLabel(bool? connected)
        {
            // connected 有值 = probe 回報新結果；null = 每 tick 刷新倒數（沿用上次結果）
            if (connected.HasValue) _storageLastConnected = connected;

            string path = _settings?.RemotePath ?? string.Empty;
            if (string.IsNullOrWhiteSpace(path))
            {
                lblStorageConn.Text = "● 儲存電腦 停用";
                lblStorageConn.BackColor = IecGray;
                return;
            }
            if (_storageLastConnected == true)
            {
                lblStorageConn.Text = "● 儲存電腦 已連線";
                lblStorageConn.BackColor = IecGreen;
            }
            else if (_storageLastConnected == false)
            {
                // 斷線 → 重連倒數（探測進行中顯示「探測中」）。秒數源自 StorageProbeIntervalTicks。
                lblStorageConn.Text = _storageProbeInFlight
                    ? "● 儲存電腦 探測中…"
                    : $"● 儲存電腦 重連中 {CountdownSec(_storageProbeTickCounter, StorageProbeIntervalTicks)}s…";
                lblStorageConn.BackColor = IecRed;
            }
            // _storageLastConnected == null（還沒 probe 過）：維持「初始化中」不覆蓋
        }

        /// <summary>
        /// 由 TelemetryTimer_Tick 每 500ms 呼叫。光源每 5 秒背景 probe 一次（SerialPort.IsOpen 偵測不到拔線，
        /// 必須實際送命令驗證）；儲存機每 5 秒背景 probe 一次（UNC Directory.Exists 可能阻塞，不可在 UI thread）。
        /// </summary>
        private void UpdateConnectionStatusLabels()
        {
            if (_appMode?.Role == MachineRole.Storage)
            {
                if (_storageDiskFreeRow != null)
                {
                    try
                    {
                        string root = GetStorageRetentionRoot();
                        if (!string.IsNullOrWhiteSpace(root))
                        {
                            var di = new System.IO.DriveInfo(
                                System.IO.Path.GetPathRoot(System.IO.Path.GetFullPath(root)));
                            double freeGb  = di.AvailableFreeSpace / (1024.0 * 1024 * 1024);
                            double totalGb = di.TotalSize           / (1024.0 * 1024 * 1024);
                            _storageDiskFreeRow.SubItems[1].Text = $"{freeGb:F1} / {totalGb:F1} GB";
                        }
                    }
                    catch { }
                }
                return;
            }

            // Grab watchdog：取像中超過 30 秒沒有 result callback → 觸發循環儲存
            if (_liveCameraManager?.IsLiveGrabbing == true &&
                _lastGrabEventTime != DateTime.MinValue &&
                (DateTime.UtcNow - _lastGrabEventTime).TotalSeconds > 30)
            {
                _lastGrabEventTime = DateTime.UtcNow;
                Task.Run(() => _retentionService?.RunCleanup());
            }

            // 光源：先同步更新（用 IsConnected 快取結果），再 2 秒背景實測 / 重連
            // （Probe 用 TryEnter，與取像時 SendCommand 不會競爭，可放心高頻）
            UpdateLightConnLabel();
            RefreshIoConnLabel();          // IO 重連倒數每 tick 刷新（源自 IoGrabController）
            UpdateStorageConnLabel(null);  // 儲存重連倒數每 tick 刷新（用上次 probe 結果）
            FlowHardwareEdges();           // H 系列：IO/光源/儲存 斷線/恢復 邊緣留痕
            if (++_lightProbeTickCounter >= LightProbeIntervalTicks)
            {
                _lightProbeTickCounter = 0;
                if (_settings != null && _settings.LightEnabled && !_lightProbeInFlight)
                {
                    _lightProbeInFlight = true;
                    int channel = _settings.LightChannel;
                    string preferredPort = _settings.LightComPort;
                    var lc = _lightController;
                    System.Threading.Tasks.Task.Run(() =>
                    {
                        try
                        {
                            if (lc != null && lc.IsConnected)
                            {
                                // 已連線 → 實測（拔線會被 Probe 偵測，內部關 port）
                                lc.Probe(channel);
                            }
                            else
                            {
                                // 未連線 → 嘗試重連（背景 AutoDetect，成功才接管欄位）
                                var fresh = new LightController();
                                string found = fresh.AutoDetect(preferredPort, channel);
                                if (found != null && !IsDisposed && !Disposing)
                                {
                                    try
                                    {
                                        BeginInvoke(new Action(() =>
                                        {
                                            if (_settings != null && _settings.LightEnabled)
                                            {
                                                _lightController?.Dispose();
                                                _lightController = fresh;
                                                if (!string.Equals(found, _settings.LightComPort, StringComparison.OrdinalIgnoreCase))
                                                {
                                                    _settingsHub.SetBatch(s => s.LightComPort = found);
                                                    RefreshGridItem(nameof(InspectionSettings.LightComPort));
                                                }
                                            }
                                            else
                                            {
                                                fresh.Dispose();
                                            }
                                        }));
                                    }
                                    catch (InvalidOperationException) { fresh.Dispose(); }
                                }
                                else
                                {
                                    fresh.Dispose();
                                }
                            }
                        }
                        catch { /* Probe/AutoDetect 內已處理例外，這裡保險 */ }
                        finally { _lightProbeInFlight = false; }

                        if (IsDisposed || Disposing) return;
                        try { BeginInvoke(new Action(UpdateLightConnLabel)); }
                        catch (InvalidOperationException) { }
                    });
                }
            }

            // 儲存機：每 5 秒背景 probe UNC 路徑
            if (++_storageProbeTickCounter < StorageProbeIntervalTicks) return;
            _storageProbeTickCounter = 0;

            string path = _settings?.RemotePath ?? string.Empty;
            if (string.IsNullOrWhiteSpace(path))
            {
                UpdateStorageConnLabel(null);
                return;
            }
            if (_storageProbeInFlight) return;
            _storageProbeInFlight = true;

            System.Threading.Tasks.Task.Run(() =>
            {
                bool ok;
                try { ok = ProbeStorageReachable(path); }
                catch { ok = false; }
                finally { _storageProbeInFlight = false; }

                if (IsDisposed || Disposing) return;
                try { BeginInvoke(new Action<bool?>(UpdateStorageConnLabel), (bool?)ok); }
                catch (InvalidOperationException) { }
            });
        }

        /// <summary>本機網路介面變動（拔/插網路線）→ 立即觸發儲存重探（下一個 telemetry tick ≤500ms），
        /// 不必等整個探測週期。事件驅動、零輪詢成本。
        /// 注意：遠端 PC 自己關機/更新時本機網卡不變、此事件不觸發，那種情況仍靠週期探測。</summary>
        private void OnNetworkAddressChanged(object sender, EventArgs e)
        {
            // 事件在 thread pool 觸發 → marshal 回 UI 執行緒改計數器（與 telemetry tick 同執行緒，無 race）
            SafeBeginInvoke(() => { _storageProbeTickCounter = StorageProbeIntervalTicks; });
        }

        /// <summary>儲存機可達性探測：解析 UNC host 後 TCP 連 445(SMB) port，2s 逾時。
        /// 不用 Directory.Exists —— 拔網路線後 Windows SMB redirector 會把該 server 快取為不可達，
        /// 重插網路線甚至重開程式都恢復不了（"找不到"）。直接 TCP 探測繞過 SMB session 快取，
        /// 網路一通就立刻恢復綠燈。實際檔案複製仍由 RemoteCopyService 處理。</summary>
        private static bool ProbeStorageReachable(string uncPath)
        {
            string host = ParseUncHost(uncPath);
            if (string.IsNullOrEmpty(host)) return false;
            System.Net.Sockets.Socket sock = null;
            try
            {
                sock = new System.Net.Sockets.Socket(
                    System.Net.Sockets.AddressFamily.InterNetwork,
                    System.Net.Sockets.SocketType.Stream,
                    System.Net.Sockets.ProtocolType.Tcp);
                var ar = sock.BeginConnect(host, 445, null, null);
                // 逾時就直接 Close（不呼叫 EndConnect）—— 避免 TcpClient.ConnectAsync 在 dispose 後
                // 仍呼叫 EndConnect 於已釋放 socket 而拋 NullReferenceException（並非儲存機端問題）。
                // 1s 逾時（LAN 連線 <50ms，足夠）→ 斷線偵測更快。
                if (ar.AsyncWaitHandle.WaitOne(1000) && sock.Connected)
                {
                    sock.EndConnect(ar);
                    return true;
                }
                return false;
            }
            catch { return false; }
            finally { try { sock?.Close(); } catch { } }
        }

        /// <summary>從 UNC 路徑（\\host\share\...）解析出 host。</summary>
        private static string ParseUncHost(string uncPath)
        {
            if (string.IsNullOrWhiteSpace(uncPath)) return null;
            string p = uncPath.TrimStart('\\', '/');
            int slash = p.IndexOfAny(new[] { '\\', '/' });
            return slash > 0 ? p.Substring(0, slash) : p;
        }

        private void UpdateIoLeds(IoSnapshot io)
        {
            if (_isIoSuspended) return;
            SetIoLed(lblIoDiAlive,   io.DiNakanAlive);
            SetIoLed(lblIoDiStart,   io.DiInspectStart);
            SetIoLed(lblIoDoPcAlive, io.DoPcAlive);
            UpdateMuraLed(io.DoMuraDetected);
            SetIoLed(lblIoDoPcBusy,  io.DoPcInspect);
        }

        private static void SetIoLed(Label lbl, bool on)
        {
            string[] parts = lbl.Text.Split(new[] { "\r\n" }, StringSplitOptions.None);
            string id   = parts[0].TrimStart('◎', '×', ' ');
            string name = parts.Length > 1 ? parts[1] : "";
            lbl.Text = (on ? "◎ " : "× ") + id + "\r\n" + name;
            lbl.BackColor = on ? IecGreen : IecDarkGray;
        }

        /// <summary>檢測暫停 setting 變更 → LED 先顯示 ×（下次 IO snapshot 由 UpdateIoLeds 更新真實 ◎/×）。
        /// （Wave3 選項1：從 OnSettingChanged dispatcher 搬入。）</summary>
        private void HandleMuraPauseSettingsChanged(string name)
        {
            if (name == nameof(InspectionSettings.MuraDetectPaused))
                UpdateMuraLed(false);
        }

        private void UpdateMuraLed(bool doMuraOn)
        {
            if (_settings.MuraDetectPaused)
            {
                lblIoDoMura.BackColor = IecYellow;
                lblIoDoMura.ForeColor = Color.Black;
                lblIoDoMura.Text = "⏸ DO1\r\nMURA_DET";
            }
            else
            {
                lblIoDoMura.BackColor = doMuraOn ? IecGreen : IecDarkGray;
                lblIoDoMura.ForeColor = Color.White;
                lblIoDoMura.Text = (doMuraOn ? "◎ " : "× ") + "DO1\r\nMURA_DET";
            }
        }

        private void TriggerRetentionAndFlagAsync()
        {
            Task.Run(() => _retentionService?.RunCleanup());
            WriteFlagToRemoteAsync();
        }

        private void WriteFlagToRemoteAsync()
        {
            // JSON 有設定就用，否則從 RemotePath 推算（同 IP，固定 AniloxConfig share）
            string configPath = _settings?.RemoteConfigPath ?? string.Empty;
            if (string.IsNullOrWhiteSpace(configPath))
                configPath = DeriveFlagSharePath(_settings?.RemotePath);
            if (string.IsNullOrWhiteSpace(configPath)) return;

            Task.Run(() =>
            {
                try
                {
                    string flagPath = Path.Combine(configPath, "cleanup-request.flag");
                    File.WriteAllText(flagPath, DateTime.UtcNow.ToString("O"),
                        System.Text.Encoding.UTF8);
                }
                catch (Exception ex)
                {
                    Trace.TraceWarning($"[RetentionFlag] 寫旗標失敗: {ex.Message}");
                }
            });
        }

        private void ApplyStorageModeUi()
        {
            if (_appMode?.Role != MachineRole.Storage) return;

            tabMain.TabPages.Remove(tabPageLiveView);
            tabControlRight.TabPages.Remove(tabPageCamera);

            // PropertyGrid：隱藏 IO / 相機 / 光源三個大類
            TypeDescriptor.AddProvider(
                new StorageModeSettingsFilter(TypeDescriptor.GetProvider(_settings)), _settings);
            propertyGridSettings.Refresh();

            lblCamCount.Visible      = false;
            lblStorageConn.Visible   = false;

            lblIoState.Visible    = false;
            lblIoConn.Visible     = false;
            lblLightConn.Visible   = false;
            lblIoDiAlive.Visible   = false;
            lblIoDiStart.Visible   = false;
            lblIoDoPcAlive.Visible = false;
            lblIoDoMura.Visible    = false;
            lblIoDoPcBusy.Visible  = false;
        }

        // \\server\share → \\server\AniloxConfig（cleanup-request.flag 目標）
        private static string DeriveFlagSharePath(string remotePath)
        {
            if (string.IsNullOrWhiteSpace(remotePath)) return "";
            var parts = remotePath.TrimStart('\\').Split('\\');
            return parts.Length < 1 || string.IsNullOrEmpty(parts[0])
                ? "" : $@"\\{parts[0]}\AniloxConfig";
        }

        private string GetStorageRetentionRoot()
        {
            if (_appMode?.Role == MachineRole.Storage &&
                !string.IsNullOrWhiteSpace(_appMode.StorageMachineDataPath))
                return _appMode.StorageMachineDataPath;
            return _settings?.CaptureRootPath ?? string.Empty;
        }

        private void lblIoDoMura_Click(object sender, EventArgs e)
        {
            // intent 行：此鈕走 Hub（程式化通道）→ 後續 set:[MuraDetectPaused] 記程式來源，
            // 這行先蓋「使用者親手做的」章（孤兒判讀規則的主人）。
            FlowTrace.Log("ui:【暫停Mura檢測】鈕");
            // 走 SettingsHub → Changed event → OnSettingChanged 接管 UpdateMuraLed
            _settingsHub.Set(s => s.MuraDetectPaused, !_settings.MuraDetectPaused);
        }

        private void lblIoConn_Click(object sender, EventArgs e)
        {
            if (_ioGrabController == null) return;
            _isIoSuspended = !_isIoSuspended;
            FlowTrace.Log($"ui:【IO暫停】鈕 → {(_isIoSuspended ? "暫停" : "恢復")}");   // intent 行（原本完全無痕＝盲區）
            if (_isIoSuspended)
            {
                lblIoConn.BackColor = IecYellow;
                lblIoConn.ForeColor = Color.White;   // 統一白字（原黑字）
                lblIoConn.Text = "● IO 暫停 ⏸";
                btnLiveGrab.Enabled = true;
                UpdateGrabButton(_liveCameraManager?.IsLiveGrabbing ?? false);
                btnLiveGrab.BackColor = SystemColors.Control;
                btnLiveGrab.ForeColor = SystemColors.ControlText;
                // 暫停 = 等同 IO 離線：重置狀態燈和所有 IO 燈號
                lblIoState.Text = "關閉";
                lblIoState.BackColor = IecGray;
                SetIoLed(lblIoDiAlive,   false);
                SetIoLed(lblIoDiStart,   false);
                SetIoLed(lblIoDoPcAlive, false);
                SetIoLed(lblIoDoPcBusy,  false);
                UpdateMuraLed(false);
            }
            else
            {
                UpdateIoConnectionUi(_ioGrabController.IsConnected);
            }
        }
    }
}
