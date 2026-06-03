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
            Task.Run(() =>
            {
                string found = controller.AutoDetect(preferredPort, channel);
                SafeBeginInvoke(() =>
                {
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

        private void UpdateIoStateLabel(IoState state)
        {
            if (_isIoSuspended) return;
            string text;
            Color bgColor;
            switch (state)
            {
                case IoState.Idle:      text = "Idle 待機"; bgColor = IecGreen;  break;
                case IoState.Running:   text = "取像中";   bgColor = IecBlue;   break;
                case IoState.Stopping:  text = "停止中";   bgColor = IecYellow; break;
                case IoState.Faulted:   text = "設備離線"; bgColor = IecRed;    break;
                case IoState.CommLost:  text = "通訊中斷"; bgColor = IecRed;    break;
                case IoState.Closed:    text = "已關閉";   bgColor = IecGray;   break;
                default:                text = "未連線";   bgColor = IecGray;   break;  // Disconnected
            }
            lblIoState.Text = $"〔{text}〕";
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
            // IO 連線標籤照實顯示（IO 確實先連上是事實）；但「按鈕是否變 IO 控制中」交給
            // RefreshGrabButtonState 統一決定 —— 相機未就緒前按鈕維持「初始化中」，不被 IO 搶顯示，
            // 避免使用者誤以為系統已可操作。
            lblIoConn.Text = connected ? "● IO 已連線" : "● IO 離線";
            lblIoConn.BackColor = connected ? IecGreen : IecGray;
            RefreshGrabButtonState();
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
            else
            {
                lblLightConn.Text = "● 光源 離線";
                lblLightConn.BackColor = IecGray;
            }

            UpdateStandardBgSubLockState();
        }

        private int _storageProbeTickCounter;
        private volatile bool _storageProbeInFlight;
        private int _lightProbeTickCounter;
        private volatile bool _lightProbeInFlight;

        private void UpdateStorageConnLabel(bool? connected)
        {
            string path = _settings?.RemotePath ?? string.Empty;
            if (string.IsNullOrWhiteSpace(path))
            {
                lblStorageConn.Text = "● 儲存電腦 停用";
                lblStorageConn.BackColor = IecGray;
                return;
            }
            if (connected == true)
            {
                lblStorageConn.Text = "● 儲存電腦 已連線";
                lblStorageConn.BackColor = IecGreen;
            }
            else if (connected == false)
            {
                lblStorageConn.Text = "● 儲存電腦 離線";
                lblStorageConn.BackColor = IecRed;
            }
            // connected == null：保留上次結果（probe 還沒回來）
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
            if (++_lightProbeTickCounter >= 4)
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
            if (++_storageProbeTickCounter < 10) return;
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
                try { ok = System.IO.Directory.Exists(path); }
                catch { ok = false; }
                finally { _storageProbeInFlight = false; }

                if (IsDisposed || Disposing) return;
                try { BeginInvoke(new Action<bool?>(UpdateStorageConnLabel), (bool?)ok); }
                catch (InvalidOperationException) { }
            });
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
            // 走 SettingsHub → Changed event → OnSettingChanged 接管 UpdateMuraLed
            _settingsHub.Set(s => s.MuraDetectPaused, !_settings.MuraDetectPaused);
        }

        private void lblIoConn_Click(object sender, EventArgs e)
        {
            if (_ioGrabController == null) return;
            _isIoSuspended = !_isIoSuspended;
            if (_isIoSuspended)
            {
                lblIoConn.BackColor = IecYellow;
                lblIoConn.ForeColor = Color.Black;
                lblIoConn.Text = "● IO 暫停 ⏸";
                btnLiveGrab.Enabled = true;
                UpdateGrabButton(_liveCameraManager?.IsLiveGrabbing ?? false);
                btnLiveGrab.BackColor = SystemColors.Control;
                btnLiveGrab.ForeColor = SystemColors.ControlText;
                // 暫停 = 等同 IO 離線：重置狀態燈和所有 IO 燈號
                lblIoState.Text = "〔已關閉〕";
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
