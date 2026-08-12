using System;
using System.ComponentModel;
using System.IO;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Management;
using System.Windows.Forms;
using StorageBridge.Core;
using MilGrabber.Core;
using TanukiCv.Controls;
using TanukiCv.Utils;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Interop;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.State;
using AniloxRoll.Monitor.UI.Coordinators;
using AniloxRoll.Monitor.UI.Managers;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Presenters;
using AniloxRoll.Monitor.UI.Widgets;

namespace AniloxRoll.Monitor.Forms
{
    /// <summary>Light connection and shared hardware status aggregation.</summary>
    public partial class AniloxRollForm
    {










        private void InitLightController()
        {
            if (!_settings.LightEnabled) return;
            if (_lightConnectionCoordinator == null)
            {
                _lightConnectionCoordinator =
                    new LightConnectionCoordinator(TelemetryTickMs);
                _lightConnectionCoordinator.StateChanged += () =>
                    SafeBeginInvoke(UpdateLightConnLabel);
                _lightConnectionCoordinator.ActivePortChanged += found =>
                    SafeBeginInvoke(() =>
                    {
                        if (_settings == null ||
                            !_settings.LightEnabled ||
                            string.Equals(
                                found,
                                _settings.LightComPort,
                                StringComparison.OrdinalIgnoreCase))
                            return;

                        _settingsHub.SetBatch(s => s.LightComPort = found);
                        RefreshGridItem(nameof(InspectionSettings.LightComPort));
                    });
            }

            _lightConnectionCoordinator.Start(
                _settings.LightComPort,
                _settings.LightChannel);
        }




        private void LightTurnOn()
        {
            bool sent = _lightConnectionCoordinator != null &&
                _lightConnectionCoordinator.TurnOn(
                    _settings.LightChannel,
                    _settings.LightBrightness);
            FlowTrace.Log(
                $"light turn on result={(sent ? "sent" : "failed")} " +
                $"channel={_settings.LightChannel} brightness={_settings.LightBrightness}");
        }

        private void LightTurnOff()
        {
            _lightConnectionCoordinator?.TurnOff(_settings.LightChannel);
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
                        InitLightController();
                    }
                    else
                    {
                        _lightConnectionCoordinator?.Disable();
                    }
                    break;

                case nameof(InspectionSettings.LightComPort):
                case nameof(InspectionSettings.LightChannel):
                    if (_settings.LightEnabled)
                        InitLightController();
                    break;

                case nameof(InspectionSettings.LightBrightness):
                    _lightConnectionCoordinator?.SetBrightness(
                        _settings.LightChannel,
                        _settings.LightBrightness);
                    ArmLiveInspectionStimulusProbe(_settings.LightBrightness);
                    UpdateLightConnLabel();
                    break;
            }
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


        /// <summary>由 TelemetryTimer 每 tick 呼叫：IO 斷線時顯示重連倒數（秒數源自 IoGrabController）。
        /// 手動暫停（_isIoSuspended）不覆蓋；初始連線中（尚未排程重連）維持「初始化中」。</summary>
        // ── H 系列：硬體連線邊緣留痕（IO/光源/儲存；狀態轉變才記一行，斷線/恢復現場排障關鍵）──
        private bool? _lastFlowIoConn, _lastFlowLightConn, _lastFlowStorageShareConn;

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
            IoGrabController ioController = CurrentIoController;
            Edge(ref _lastFlowIoConn, ioController?.IsConnected == true, "IO");
            if (_settings?.LightEnabled == true)
                Edge(
                    ref _lastFlowLightConn,
                    _lightConnectionCoordinator?.Snapshot.Connected == true,
                    "光源");
            if (!string.IsNullOrWhiteSpace(_settings?.RemotePath))
                Edge(
                    ref _lastFlowStorageShareConn,
                    _storageHealthCoordinator?.Snapshot.RemoteShareConnected == true,
                    "儲存分享");

            if (_settings?.IoEnabled == true && ioController != null)
            {
                if (ioController.IsConnected)
                    _outputHealthService?.Resolve("IoConnection");
                else
                    _outputHealthService?.Report(
                        "IoConnection", OutputHealthSeverity.Critical, "IO 未連線");
            }
            else
            {
                _outputHealthService?.Resolve("IoConnection");
            }

            LightConnectionSnapshot light =
                _lightConnectionCoordinator?.Snapshot;
            if (_settings?.LightEnabled == true && light?.HasProbed == true)
            {
                if (light.Connected)
                    _outputHealthService?.Resolve("LightConnection");
                else
                    _outputHealthService?.Report(
                        "LightConnection", OutputHealthSeverity.Critical, "光源未連線");
            }
            else if (_settings?.LightEnabled != true)
            {
                _outputHealthService?.Resolve("LightConnection");
            }
        }


        private void UpdateLightConnLabel()
        {
            if (_settings == null || !_settings.LightEnabled)
            {
                lblLightConn.Text = "● 光源 停用";
                lblLightConn.BackColor = IecGray;
                return;
            }

            LightConnectionSnapshot light =
                _lightConnectionCoordinator?.Snapshot;
            if (light?.Connected == true)
            {
                lblLightConn.Text = $"● 光源 已連線 ({_settings.LightBrightness})";
                lblLightConn.BackColor = IecGreen;
            }
            else if (light == null || !light.HasProbed)
            {
                // 初次偵測還沒回來 → 維持「初始化中」（與 IO/儲存一致）
                lblLightConn.Text = "● 光源: 初始化中…";
                lblLightConn.BackColor = IecGray;
            }
            else
            {
                // 斷線 → 顯示 coordinator 提供的 probe 狀態與倒數。
                lblLightConn.Text = light.ProbeInFlight
                    ? "● 光源 探測中…"
                    : $"● 光源 重連中 {light.ReconnectSeconds}s…";
                lblLightConn.BackColor = IecRed;
            }

            UpdateStandardBgSubLockState();
        }

        internal const int TelemetryTickMs = 500;          // = SettingsTabs 的 _telemetryTimer.Interval









        /// <summary>
        /// 由 TelemetryTimer_Tick 每 500ms 呼叫。各 coordinator 推進自己的
        /// 連線生命週期，Form 只依快照更新控制項與產品告警。
        /// </summary>
        private void UpdateConnectionStatusLabels()
        {
            _storageHealthCoordinator?.Tick();
            StorageHealthSnapshot storage =
                _storageHealthCoordinator?.Snapshot;

            if (_appMode?.Role == MachineRole.Storage)
            {
                if (storage?.LocalFreeBytes >= 0 &&
                    storage.LocalTotalBytes > 0)
                {
                    if (_storageDiskFreeRow != null)
                    {
                        double freeGb =
                            storage.LocalFreeBytes /
                            (1024.0 * 1024 * 1024);
                        double totalGb =
                            storage.LocalTotalBytes /
                            (1024.0 * 1024 * 1024);
                        _storageDiskFreeRow.SubItems[1].Text = $"{freeGb:F1} / {totalGb:F1} GB";
                    }
                }
                RefreshCapacityInfoLabel();
                return;
            }

            RefreshCapacityInfoLabel();

            // Grab watchdog：取像中超過 30 秒沒有 result callback → 觸發循環儲存
            if (_liveCameraManager?.IsLiveGrabbing == true &&
                _lastGrabEventTime != DateTime.MinValue &&
                (DateTime.UtcNow - _lastGrabEventTime).TotalSeconds > 30)
            {
                _lastGrabEventTime = DateTime.UtcNow;
                Task.Run(() => _retentionService?.RunCleanup());
            }

            // 光源連線生命週期由 coordinator 管理；Form 只推進 timer 並畫快照。
            _lightConnectionCoordinator?.Tick();
            UpdateLightConnLabel();
            RefreshIoConnLabel();          // IO 重連倒數每 tick 刷新（源自 IoGrabController）
            UpdateStorageConnLabel();
            FlowHardwareEdges();           // H 系列：IO/光源/儲存 斷線/恢復 邊緣留痕
        }











    }
}
