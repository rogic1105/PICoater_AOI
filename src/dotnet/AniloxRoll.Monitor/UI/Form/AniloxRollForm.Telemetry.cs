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
        private volatile bool _telemetryCaptureInFlight;   // 背景 telemetry MIL 查詢進行中，避免重疊堆積

        private void TelemetryTimer_Tick(object sender, EventArgs e)
        {
            // 連線狀態不受相機釋放影響，先於 gate 更新
            UpdateConnectionStatusLabels();

            if (_liveCameraManager == null || _liveCameraManager.IsReleasing) return;

            // CLProtocol 背景初始化期間（!AreCamerasHwReady）UI 執行緒不可碰 MIL 查詢：telemetry 的
            // MdigInquire/MsysInquire（presenter.Update 16 欄 + 下方 maxFps）會與背景 CLProtocol enable
            // 持有的 MIL 內部鎖競爭 → UI 執行緒卡死在查詢裡數秒（視窗拖不動、燈號一次 flush）。
            // 與 CameraStatusTimer_Tick / SyncCameraParamsFromHardware 同一條規則（初始化期間跳過 UI 端 MIL 查詢）。
            bool hwReady = _liveCameraManager.AreCamerasHwReady;

            // Telemetry 的 MIL 查詢（16 欄 MdigInquire/MsysInquire ≈ 195ms/tick）移到背景執行緒做，
            // UI 執行緒只 Apply 字串快照（極快）→ 不再每 500ms 卡 UI ~195ms。maxFps 也由快照算，不另查 MIL。
            if (hwReady && _telemetryPresenter != null && !_telemetryCaptureInFlight)
            {
                _telemetryCaptureInFlight = true;
                var cams = _liveCameraManager.Cameras;
                System.Threading.Tasks.Task.Run(() =>
                {
                    System.Collections.Generic.List<LiveTelemetryPresenter.CamSnapshot> snaps = null;
                    try { snaps = _telemetryPresenter.Capture(cams); }
                    catch { }
                    finally { _telemetryCaptureInFlight = false; }
                    if (snaps == null || IsDisposed || Disposing) return;
                    try
                    {
                        BeginInvoke(new Action(() =>
                        {
                            if (IsDisposed || Disposing) return;
                            _telemetryPresenter.Apply(snaps);
                            if (!_telemetryFitDone)
                            {
                                AutoFitListViewColumns(listViewCameras);
                                _telemetryFitDone = true;
                            }
                            // 動態調整 Live Overview Timer：跟隨最大 FPS（由快照算，不再查 MIL）
                            double maxFps = 0;
                            foreach (var s in snaps) if (s.Fps > maxFps) maxFps = s.Fps;
                            if (maxFps > 0.1 && _liveOverviewTimer != null)
                            {
                                int interval = Math.Max(50, Math.Min(500, (int)(1000.0 / maxFps)));
                                if (_liveOverviewTimer.Interval != interval)
                                    _liveOverviewTimer.Interval = interval;
                            }
                        }));
                    }
                    catch (InvalidOperationException) { }
                });
            }

            if (hwReady && _liveCameraManager.IsAllocated)
                SyncCameraParamsFromHardware();

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
