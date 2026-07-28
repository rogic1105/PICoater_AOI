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
    /// <summary>Storage capacity, remote health, retention, and storage-mode presentation.</summary>
    public partial class AniloxRollForm
    {
        private const long RemoteBacklogWarningBytes = 20L * 1024 * 1024 * 1024;

        private void RefreshOutputCapacityHealth()
        {
            if (_outputHealthService == null) return;

            StorageHealthSnapshot storage =
                _storageHealthCoordinator?.Snapshot;
            long minFreeBytes = GetStorageMinFreeBytes();
            if (storage?.LocalFreeBytes >= 0 &&
                storage.LocalTotalBytes > 0)
            {
                if (minFreeBytes >= storage.LocalTotalBytes)
                {
                    _outputHealthService.Report(
                        "StorageThresholdInvalid",
                        OutputHealthSeverity.OutputFault,
                        "預留空間設定超過磁碟容量，已停止自動清理");
                    _outputHealthService.Resolve("LocalLowSpace");
                }
                else
                {
                    _outputHealthService.Resolve("StorageThresholdInvalid");
                    if (storage.LocalFreeBytes < minFreeBytes)
                    {
                        _outputHealthService.Report(
                            "LocalLowSpace",
                            OutputHealthSeverity.Notice,
                            (_appMode?.Role == MachineRole.Storage ? "儲存電腦" : "檢測電腦") +
                            "空間低於預留值，正在清理最舊資料");
                    }
                    else
                    {
                        _outputHealthService.Resolve("LocalLowSpace");
                    }
                }
            }

            long pendingBytes = _remoteCopyService?.PendingBytes ?? 0;
            if (pendingBytes >= RemoteBacklogWarningBytes)
            {
                _outputHealthService.Report(
                    "RemoteBacklog",
                    OutputHealthSeverity.Notice,
                    "遠端待傳已超過 20 GB");
            }
            else
            {
                _outputHealthService.Resolve("RemoteBacklog");
            }
        }

        private void HandleSettingsStoreIssue(SettingsStoreIssue issue)
        {
            if (issue == null || _outputHealthService == null) return;

            string file = Path.GetFileName(issue.Path);
            if (issue.Kind == SettingsStoreIssueKind.RebuiltDefaults)
            {
                _outputHealthService.Report(
                    "ConfigRebuilt." + file,
                    OutputHealthSeverity.OutputFault,
                    $"{file} 損壞，已用預設值重建");
                _outputHealthService.Resolve("ConfigRebuilt." + file);
                return;
            }

            _outputHealthService.Report(
                "ConfigSaveFailed." + file,
                OutputHealthSeverity.OutputFault,
                $"{file} 寫入失敗：{issue.Reason}");
        }

        private void HandleStorageSettingsChanged(string changedPropertyName)
        {
            if (changedPropertyName != nameof(InspectionSettings.LocalMinFreeGB)) return;

            StorageHealthSnapshot storage =
                _storageHealthCoordinator?.RefreshLocalCapacity();
            long totalBytes = storage?.LocalTotalBytes ?? 0;

            int requestedGb = _settings.LocalMinFreeGB;
            int maxGb = totalBytes > 0
                ? Math.Max(1, (int)(totalBytes / (1024L * 1024L * 1024L)) - 1)
                : requestedGb;
            if (totalBytes > 0 && requestedGb > maxGb)
            {
                _settingsHub.SetBatch(s => s.LocalMinFreeGB = maxGb);
                requestedGb = maxGb;
                RefreshGridItem(nameof(InspectionSettings.LocalMinFreeGB));
                _outputHealthService?.Report(
                    "StorageThresholdAdjusted",
                    OutputHealthSeverity.OutputFault,
                    $"預留空間超過磁碟容量，已調整為 {maxGb} GB");
                _outputHealthService?.Resolve("StorageThresholdAdjusted");
            }

            if (_appMode?.Role == MachineRole.Storage &&
                _appMode.StorageMinFreeGB != requestedGb)
            {
                _appMode.StorageMinFreeGB = requestedGb;
                _appMode.Save();
            }

            RefreshOutputCapacityHealth();
            Task.Run(() => _retentionService?.RunCleanup());
        }

        private int CancelRemoteCopyForDay(string dayDirectory)
        {
            if (_remoteCopyService == null || string.IsNullOrWhiteSpace(dayDirectory)) return 0;

            int canceled = _remoteCopyService.CancelPendingFilesUnder(dayDirectory);
            string monthDirectory = Path.GetDirectoryName(dayDirectory);
            string dailyCsv = monthDirectory == null
                ? null
                : Path.Combine(monthDirectory, Path.GetFileName(dayDirectory) + ".csv");
            if (_remoteCopyService.CancelPendingFile(dailyCsv)) canceled++;
            return canceled;
        }

        private void HandleRetentionCleanupCompleted(RetentionCleanupResult result)
        {
            if (result == null) return;
            _storageHeartbeatService?.RecordCleanup(result.FreedBytes);
            if (result.DeletedDayFolders > 0 &&
                _appMode?.Role != MachineRole.Storage)
            {
                CleanupInactiveBackgroundVersions();
            }

            if (result.CanceledPendingFiles > 0)
            {
                const string code = "RetentionDiscardedPending";
                _outputHealthService?.Report(
                    code,
                    OutputHealthSeverity.OutputFault,
                    $"空間不足，已刪除最舊資料（含 {result.CanceledPendingFiles} 個未傳檔案）");
                _outputHealthService?.Resolve(code);
            }
            else if (result.DeletedDayFolders > 0)
            {
                const string code = "RetentionCleanup";
                _outputHealthService?.Report(
                    code,
                    OutputHealthSeverity.Notice,
                    $"空間不足，已清理最舊 {result.DeletedDayFolders} 天資料");
                _outputHealthService?.Resolve(code);
            }
        }

        private static string FormatCapacity(string computerName, long freeBytes, long totalBytes)
        {
            if (freeBytes < 0 || totalBytes <= 0)
                return computerName + "：無法讀取";

            double freeGb = freeBytes / (1024.0 * 1024 * 1024);
            double totalGb = totalBytes / (1024.0 * 1024 * 1024);
            return $"{computerName}：剩餘 {freeGb:N1} / {totalGb:N1} GB";
        }

        private void RefreshCapacityInfoLabel()
        {
            if (lblInfo == null) return;

            RefreshOutputCapacityHealth();

            StorageHealthSnapshot storage =
                _storageHealthCoordinator?.Snapshot;
            string capacityText = _appMode?.Role == MachineRole.Storage
                ? FormatCapacity(
                    "儲存電腦",
                    storage?.LocalFreeBytes ?? -1,
                    storage?.LocalTotalBytes ?? 0)
                : FormatCapacity(
                    "檢測電腦",
                    storage?.LocalFreeBytes ?? -1,
                    storage?.LocalTotalBytes ?? 0) +
                  " ｜ " + FormatCapacity(
                    "儲存電腦",
                    storage?.RemoteFreeBytes ?? -1,
                    storage?.RemoteTotalBytes ?? 0);

            if (_appMode?.Role != MachineRole.Storage && _remoteCopyService != null)
            {
                capacityText += $" ｜ 待傳：{_remoteCopyService.PendingBytes / (1024.0 * 1024 * 1024):N1} GB" +
                    $"（{_remoteCopyService.QueueCount} 檔）";
                long localTicks = System.Threading.Interlocked.Read(
                    ref _lastLocalSaveUtcTicks);
                DateTime? remoteUtc = _remoteCopyService.LastSuccessfulCopyUtc;
                capacityText += " ｜ 最近存檔：" +
                    (localTicks > 0
                        ? new DateTime(localTicks, DateTimeKind.Utc).ToLocalTime().ToString("HH:mm:ss")
                        : "--");
                capacityText += " ｜ 最近遠傳：" +
                    (remoteUtc.HasValue
                        ? remoteUtc.Value.ToLocalTime().ToString("HH:mm:ss")
                        : "--");
            }

            if (!string.Equals(lblInfo.Text, capacityText, StringComparison.Ordinal))
                lblInfo.Text = capacityText;
        }

        private void UpdateStorageConnLabel()
        {
            string path = _settings?.RemotePath ?? string.Empty;
            if (string.IsNullOrWhiteSpace(path))
            {
                lblStorageConn.Text = "● 儲存電腦 停用";
                lblStorageConn.BackColor = IecGray;
                _outputHealthService?.Resolve("StorageConnection");
                _outputHealthService?.Resolve("StorageHeartbeat");
                return;
            }

            StorageHealthSnapshot storage =
                _storageHealthCoordinator?.Snapshot;
            if (storage?.RemoteShareConnected == true)
            {
                _outputHealthService?.Resolve("StorageConnection");
                if (storage.RemoteAppAlive == true)
                {
                    lblStorageConn.Text = "● 儲存電腦 已連線";
                    lblStorageConn.BackColor = IecGreen;
                    _outputHealthService?.Resolve("StorageHeartbeat");
                }
                else
                {
                    lblStorageConn.Text = "● 儲存分享可用 / 程式未回報";
                    lblStorageConn.BackColor = IecYellow;
                    _outputHealthService?.Report(
                        "StorageHeartbeat",
                        OutputHealthSeverity.Critical,
                        "儲存電腦程式未回報");
                }
            }
            else if (storage?.RemoteShareConnected == false)
            {
                _outputHealthService?.Resolve("StorageHeartbeat");
                _outputHealthService?.Report(
                    "StorageConnection",
                    OutputHealthSeverity.Critical,
                    "儲存電腦連線中斷，本機持續存檔");
                lblStorageConn.Text = storage.RemoteProbeInFlight
                    ? "● 儲存電腦 探測中…"
                    : $"● 儲存電腦 重連中 {storage.ReconnectSeconds}s…";
                lblStorageConn.BackColor = IecRed;
            }
            // 尚未 probe 過時維持「初始化中」。
        }


        /// <summary>本機網路介面變動（拔/插網路線）→ 立即觸發儲存重探（下一個 telemetry tick ≤500ms），
        /// 不必等整個探測週期。事件驅動、零輪詢成本。
        /// 注意：遠端 PC 自己關機/更新時本機網卡不變、此事件不觸發，那種情況仍靠週期探測。</summary>
        private void OnNetworkAddressChanged(object sender, EventArgs e)
        {
            _storageHealthCoordinator?.ForceRemoteProbe();
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

        private bool _storageModeLayoutApplied;

        private void ApplyStorageModeUi()
        {
            if (_appMode?.Role != MachineRole.Storage || _storageModeLayoutApplied) return;
            _storageModeLayoutApplied = true;

            tabMain.TabPages.Remove(tabPageLiveView);
            tabControlRight.TabPages.Remove(tabPageCamera);

            // PropertyGrid：隱藏 IO / 相機 / 光源三個大類
            TypeDescriptor.AddProvider(
                new StorageModeSettingsFilter(TypeDescriptor.GetProvider(_settings)), _settings);
            propertyGridSettings.Refresh();

            // The remaining panes are anchored at a fixed Y instead of Dock.Fill, so
            // hiding the parent alone does not reclaim its row. Compact both panes
            // before ProportionalScaler captures the storage-mode baseline.
            int releasedHeight = panelStatusBar.Height;
            panelStatusBar.Visible = false;
            tabMain.SetBounds(
                tabMain.Left,
                tabMain.Top - releasedHeight,
                tabMain.Width,
                tabMain.Height + releasedHeight);
            tabControlRight.SetBounds(
                tabControlRight.Left,
                tabControlRight.Top - releasedHeight,
                tabControlRight.Width,
                tabControlRight.Height + releasedHeight);
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

        private long GetStorageMinFreeBytes()
        {
            int minFreeGb = _settings?.LocalMinFreeGB ?? InspectionDefaults.LocalMinFreeGB;
            if (_appMode?.Role == MachineRole.Storage)
            {
                minFreeGb = _appMode.StorageMinFreeGB > 0
                    ? _appMode.StorageMinFreeGB
                    : AppModeDefaults.StorageMinFreeGB;
            }
            return (long)minFreeGb * 1024L * 1024L * 1024L;
        }
    }
}
