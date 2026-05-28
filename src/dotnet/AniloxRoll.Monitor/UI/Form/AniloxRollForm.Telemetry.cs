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
    /// <summary>AniloxRollForm Telemetry / 資源監控 timer 相關方法 — 由主檔拆出的 partial。</summary>
    public partial class AniloxRollForm
    {
        // ==========================================
        // --- Telemetry Timer ---
        // ==========================================

        private bool _telemetryFitDone;

        private void TelemetryTimer_Tick(object sender, EventArgs e)
        {
            // 連線狀態不受相機釋放影響，先於 gate 更新
            UpdateConnectionStatusLabels();

            if (_liveCameraManager == null || _liveCameraManager.IsReleasing) return;

            _telemetryPresenter?.Update(_liveCameraManager.Cameras);

            if (!_telemetryFitDone)
            {
                AutoFitListViewColumns(listViewCameras);
                _telemetryFitDone = true;
            }

            if (_liveCameraManager.IsAllocated)
            {
                SyncCameraParamsFromHardware();

                // 動態調整 Live Overview Timer：跟隨最大 FPS，下限 50ms（20Hz），上限 500ms（2Hz）
                double maxFps = 0;
                foreach (var cam in _liveCameraManager.Cameras)
                {
                    double fps = cam.CurrentFps;
                    if (fps > maxFps) maxFps = fps;
                }
                if (maxFps > 0.1 && _liveOverviewTimer != null)
                {
                    int interval = Math.Max(50, Math.Min(500, (int)(1000.0 / maxFps)));
                    if (_liveOverviewTimer.Interval != interval)
                        _liveOverviewTimer.Interval = interval;
                }
            }

            // ── Resource Monitor 更新 ──
            UpdateResourceMonitor();
        }

        private ListViewItem AddResMonItem(string key, string value)
        {
            var item = new ListViewItem(new[] { key, value });
            listViewHardware.Items.Add(item);
            return item;
        }

        private void UpdateResourceMonitor()
        {
            try
            {
                var cameras = _liveCameraManager?.Cameras;
                if (cameras == null || cameras.Count == 0) return;

                // 取第一台有效相機的 frame size
                int w = 0, h = 0;
                long maxGpuMs = 0;
                long totalSaveBytes = 0;
                long totalFrames = 0;
                long lastSaveBytes = 0;

                foreach (var cam in cameras)
                {
                    if (cam == null) continue;
                    if (cam.FrameWidth > 0 && w == 0) { w = cam.FrameWidth; h = cam.FrameHeight; }
                    if (cam.LastGpuTimeMs > maxGpuMs) maxGpuMs = cam.LastGpuTimeMs;
                    if (cam.LastSaveBytesTotal > lastSaveBytes) lastSaveBytes = cam.LastSaveBytesTotal;
                    totalSaveBytes += cam.SessionSaveBytes;
                    totalFrames += cam.SessionFrameCount;
                }

                long rawBytes = (long)w * h;
                double rawMB = rawBytes / (1024.0 * 1024);

                if (_resMonRawSize == null) return;
                _resMonRawSize.SubItems[1].Text = w > 0 ? $"{w}×{h} = {rawMB:F1} MB" : "—";
                _resMonGpuTime.SubItems[1].Text = maxGpuMs > 0 ? $"{maxGpuMs} ms" : "—";
                _resMonSaveSize.SubItems[1].Text = lastSaveBytes > 0 ? $"{lastSaveBytes / 1024.0:F0} KB" : "—";
                _resMonDiskWrite.SubItems[1].Text = totalSaveBytes > 0
                    ? $"{totalSaveBytes / (1024.0 * 1024 * 1024):F2} GB ({totalFrames} frames)"
                    : "—";
                _resMonFrames.SubItems[1].Text = totalFrames > 0 ? $"{totalFrames}" : "—";

                // RAM: process working set
                long ramBytes = System.Diagnostics.Process.GetCurrentProcess().WorkingSet64;
                _resMonRamUsed.SubItems[1].Text = $"{ramBytes / (1024.0 * 1024):F0} MB";

                // VRAM: 根據演算法計算（6×W×H + Gaussian workspace 3×W×H×4）
                if (w > 0)
                {
                    long pixels = (long)w * h;
                    long fixedBuf = pixels * 6;                      // 6 個 uint8 buffer
                    long workspace = pixels * 4 * 3;                 // Gaussian: 3 個 float buffer
                    long vramTotal = fixedBuf + workspace + 200L * 1024 * 1024; // + CUDA runtime ~200MB
                    _resMonVramEst.SubItems[1].Text = $"~{vramTotal / (1024.0 * 1024):F0} MB (est.)";
                }
            }
            catch { /* 非關鍵，忽略 */ }
        }

        private void LiveOverviewTimer_Tick(object sender, EventArgs e)
        {
            if (_liveCameraManager == null || _liveCameraManager.IsReleasing) return;
            if (!_liveOverviewDirty || _liveOverviewHelper == null || _settings == null) return;
            _liveOverviewDirty = false;
            CurveMergeHelper.UpdateOverviewChart(_liveCurveMean, _liveCurveMax,
                _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray(),
                _settings.ErrorValueMeanV, _settings.ErrorValueMaxV,
                _liveOverviewHelper, CameraCount, _settings.StitchMode, LiveViewRangeProvider);
        }
    }
}
