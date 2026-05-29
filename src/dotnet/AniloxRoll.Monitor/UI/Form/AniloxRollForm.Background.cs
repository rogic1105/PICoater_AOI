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
    /// <summary>AniloxRollForm 背景（取得/載入/預覽）相關方法 — 由主檔拆出的 partial。</summary>
    public partial class AniloxRollForm
    {
        /// <summary>
        /// 取得背景：啟動 grab → 採集 N 秒 → 多幀平均 column mean → 存 MCBF bin。
        /// </summary>
        private async void btnLiveGetBackground_Click(object sender, EventArgs e)
        {
            if (!IsStandardBgSubEnabled)
            {
                MessageBox.Show("請先將去背演算法切換為「標準去背」。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            // 先清除舊的背景預覽（釋放 overlay + 恢復 MIL display）
            if (_bgPreviewActive) ClearBackgroundPreview();

            // 確保相機已 allocate
            if (!_liveCameraManager.IsAllocated)
            {
                try
                {
                    _liveCameraManager.EnsureAllocatedAndToggleGrab(false); // 不需影像處理
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"相機配置失敗: {ex.Message}", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
            }

            // 確保 grab 中，先開燈等穩定再開始
            if (!_liveCameraManager.IsLiveGrabbing)
            {
                LightTurnOn();
                int warmup = _settings?.LightWarmupMs ?? 0;
                if (warmup > 0) await Task.Delay(warmup);
                _liveCameraManager.ToggleGrab();
                UpdateGrabButton(true);
            }

            btnLiveGetBackground.Enabled = false;
            btnLiveGrab.Enabled = false;

            int sampleSeconds = Math.Max(1, _settings.Recipe.BackgroundSampleSeconds);
            string bgDir = _settings.Storage.BackgroundPath;

            try
            {
                if (!Directory.Exists(bgDir))
                    Directory.CreateDirectory(bgDir);

                var cameras = _liveCameraManager.Cameras;
                int camCount = cameras.Count;

                double[][] accum = new double[camCount][];
                int[] frameCount = new int[camCount];

                // 採集 sampleSeconds 秒，按鈕顯示倒數
                var sw = Stopwatch.StartNew();
                int lastShown = -1;
                while (sw.Elapsed.TotalSeconds < sampleSeconds)
                {
                    int remaining = sampleSeconds - (int)sw.Elapsed.TotalSeconds;
                    if (remaining != lastShown)
                    {
                        lastShown = remaining;
                        btnLiveGetBackground.Text = $"採集中 {remaining}s";
                    }

                    await Task.Delay(100);

                    for (int i = 0; i < camCount; i++)
                    {
                        var cam = cameras[i];
                        if (!cam.IsConnected || cam.FrameWidth <= 0) continue;

                        if (accum[i] == null)
                            accum[i] = new double[cam.FrameWidth];

                        float[] colMean = new float[cam.FrameWidth];
                        if (cam.TryComputeColumnMean(colMean))
                        {
                            for (int c = 0; c < cam.FrameWidth; c++)
                                accum[i][c] += colMean[c];
                            frameCount[i]++;
                        }
                    }
                }

                // 平均並存檔
                for (int i = 0; i < camCount; i++)
                {
                    if (frameCount[i] == 0 || accum[i] == null) continue;

                    var cam = cameras[i];
                    float[] avgColMean = new float[cam.FrameWidth];
                    double invN = 1.0 / frameCount[i];
                    for (int c = 0; c < cam.FrameWidth; c++)
                        avgColMean[c] = (float)(accum[i][c] * invN);

                    string binPath = Path.Combine(bgDir, CaptureFileNaming.BgBin(cam.FrameWidth, cam.CameraId));
                    SaveBackgroundBin(avgColMean, binPath, _settings.LightBrightness, (float)cam.CameraExposureTimeUs); // LightBrightness = light controller level (0-255)
                }

                // 載入到各相機
                LoadBackgroundBins();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"背景採集失敗: {ex.Message}", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                btnLiveGetBackground.Text = "取得背景";
                btnLiveGetBackground.Enabled = true;

                // 採集完成後一律停止 grab
                if (_liveCameraManager.IsLiveGrabbing)
                {
                    _liveCameraManager.ToggleGrab();
                    LightTurnOff();
                    UpdateGrabButton(false);
                }

                UpdateStandardBgSubLockState();
            }

            if (_autoStartGrabAfterBg)
            {
                _autoStartGrabAfterBg = false;
                _liveCameraManager.FreeCameras();
                btnLiveGrab_Click(null, null);
                _ = _ioGrabController?.NotifyGrabStarted();
                return;
            }

            // 採集完成後直接預覽（先清除舊預覽，確保每次都重新開啟）
            if (_bgPreviewActive) ClearBackgroundPreview();
            btnLiveViewBackground_Click(btnLiveViewBackground, EventArgs.Empty);
        }

        /// <summary>MCBF v2 格式存 background column mean（含光源等級與曝光時間）。</summary>
        private static void SaveBackgroundBin(float[] data, string path, int lightLevel, float exposureUs)
        {
            using (var bw = new BinaryWriter(File.Open(path, FileMode.Create, FileAccess.Write)))
            {
                bw.Write(new byte[] { (byte)'M', (byte)'C', (byte)'B', (byte)'F' });
                bw.Write(2);                    // version 2
                bw.Write(1.0f);                 // scale_factor (1 = 全解析度)
                bw.Write(lightLevel);           // light controller level (0-255)
                bw.Write(exposureUs);           // camera exposure (µs)
                bw.Write(data.Length);          // array_length
                foreach (float v in data) bw.Write(v);
            }
        }

        /// <summary>
        /// 從 BackgroundPath 載入各相機的 bg bin → pinned buffer → 設定到 AniloxCamera.PrecomputedColMean。
        /// </summary>
        private void LoadBackgroundBins()
        {
            if (!IsStandardBgSubEnabled)
            {
                // 非 StandardBgSub 模式：清除所有預算背景
                foreach (var cam in _liveCameraManager.Cameras)
                    cam.PrecomputedColMean = IntPtr.Zero;
                return;
            }

            string bgDir = _settings.Storage.BackgroundPath;
            if (!Directory.Exists(bgDir)) return;

            foreach (var cam in _liveCameraManager.Cameras)
            {
                if (cam.FrameWidth <= 0) continue;

                string binPath = Path.Combine(bgDir, CaptureFileNaming.BgBin(cam.FrameWidth, cam.CameraId));
                float[] colMean = InspectionEngine.LoadCurveBin(binPath);
                if (colMean != null && colMean.Length == cam.FrameWidth)
                {
                    // 分配 pinned memory 並複製
                    IntPtr pinned = NativeMethods.CoreCV_AllocPinned((ulong)(cam.FrameWidth * sizeof(float)));
                    if (pinned != IntPtr.Zero)
                    {
                        Marshal.Copy(colMean, 0, pinned, colMean.Length);

                        // 釋放舊的（如果有）
                        if (cam.PrecomputedColMean != IntPtr.Zero)
                            NativeMethods.CoreCV_FreePinned(cam.PrecomputedColMean);

                        cam.PrecomputedColMean = pinned;
                    }
                }
            }

            UpdateViewBackgroundButtonText();
        }

        private void UpdateViewBackgroundButtonText()
        {
            string bgDir = _settings.Storage.BackgroundPath;
            string[] bins = Directory.Exists(bgDir) ? Directory.GetFiles(bgDir, CaptureFileNaming.BgGlob) : Array.Empty<string>();
            if (bins.Length == 0) { lblBgBinInfo.Text = ""; return; }
            var meta = InspectionEngine.ReadBgBinMeta(bins[0]);
            lblBgBinInfo.Text = meta.HasValue
                ? $"光源{meta.Value.Light} 曝光{(int)meta.Value.ExposureUs}us"
                : "";
        }

        /// <summary>釋放所有相機的 PrecomputedColMean pinned buffer。</summary>
        private void FreePrecomputedColMeanBuffers()
        {
            if (_liveCameraManager == null) return;
            foreach (var cam in _liveCameraManager.Cameras)
            {
                if (cam.PrecomputedColMean != IntPtr.Zero)
                {
                    NativeMethods.CoreCV_FreePinned(cam.PrecomputedColMean);
                    cam.PrecomputedColMean = IntPtr.Zero;
                }
            }
        }

        /// <summary>
        /// StandardBgSub 時檢查是否有 bin → 控制按鈕鎖定狀態。
        /// </summary>
        private void UpdateStandardBgSubLockState()
        {
            // IO 已連線且未暫停：btnLiveGrab 由 IO 連線邏輯控制，不覆寫
            if (_ioGrabController?.IsConnected == true && !_isIoSuspended) return;
            // IO 暫停模式：交由使用者手動控制，不受 StandardBgSub bin 限制
            if (_isIoSuspended) { btnLiveGrab.Enabled = true; return; }

            if (!IsStandardBgSubEnabled)
            {
                // 非 StandardBgSub：正常解鎖（仍需光源就緒）
                btnLiveGrab.Enabled = true;
                btnLiveGetBackground.Enabled = IsLightReadyForBg;
                return;
            }

            btnLiveGetBackground.Enabled = IsLightReadyForBg;
            btnLiveGrab.Enabled = IsBgBinReady();
        }

        // --- 背景預覽狀態 ---
        private Bitmap[] _bgPreviewBitmaps;
        private bool _bgPreviewActive;
        private SmartCanvas[] _bgPreviewBoxes;      // camLive 上的 overlay（SmartCanvas with ClampPan）
        private SmartCanvas _bgPreviewMainCanvas;  // camLiveMain 上的 SmartCanvas（支援縮放/拖曳）
        private int _bgPreviewSelectedCamIndex;    // 目前預覽中的相機 index (0-based)

        /// <summary>
        /// 預覽背景：讀取各相機的 bg bin → 擴展為 width × grabHeight 灰階影像。
        /// 用 PictureBox 疊在 camLive 上方，SmartCanvas 疊在 camLiveMain 上方（支援縮放拖曳）。
        /// 點選 camLive 可切換 camLiveMain。再按一次清除預覽。
        /// </summary>
        private void btnLiveViewBackground_Click(object sender, EventArgs e)
        {
            // 先清除舊預覽（釋放 overlay + 恢復 MIL display），再重新載入
            if (_bgPreviewActive)
                ClearBackgroundPreview();

            string bgDir = _settings.Storage.BackgroundPath;
            if (!Directory.Exists(bgDir))
            {
                MessageBox.Show("背景目錄不存在。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            // 先卸載 MIL primary + secondary display，避免 native window 殘影
            if (_liveCameraManager.IsAllocated)
            {
                foreach (var cam in _liveCameraManager.Cameras)
                {
                    cam.SetPrimaryDisplayVisible(false);
                    cam.SetSecondaryDisplay(IntPtr.Zero);
                }
            }

            // 清除殘留的最後一幀（MIL native window detach 後面板不會自動重繪）
            camLiveMain.Invalidate();
            camLiveMain.Update();

            Panel[] livePanels = GetLivePanels();
            foreach (var p in livePanels) { p.Invalidate(); p.Update(); }
            int[] grabHeights = _settings.Acquisition.CameraGrabHeight;
            _bgPreviewBitmaps = new Bitmap[livePanels.Length];
            _bgPreviewBoxes = new SmartCanvas[livePanels.Length];
            int firstValid = -1;

            for (int i = 0; i < livePanels.Length; i++)
            {
                int camId = i + 1;
                string[] matches = Directory.GetFiles(bgDir, CaptureFileNaming.BgGlobForCam(camId));
                if (matches.Length == 0) continue;

                float[] colMean = InspectionEngine.LoadCurveBin(matches[0]);
                if (colMean == null || colMean.Length == 0) continue;

                int height = (i < grabHeights.Length && grabHeights[i] > 0) ? grabHeights[i] : 3000;
                Bitmap bmp = ExpandColMeanToBitmap(colMean, colMean.Length, height);
                _bgPreviewBitmaps[i] = bmp;

                // SmartCanvas 疊在 panel 最上層（ClampPan 模式，同 grab 的 MIL 顯示行為）
                var sc = new SmartCanvas
                {
                    Dock = DockStyle.Fill,
                    ClampPan = true,
                    Tag = i,
                    BackColor = Color.Black
                };
                sc.Image = bmp;
                livePanels[i].Controls.Add(sc);
                sc.BringToFront();
                sc.FitToScreen();
                sc.Click += BgPreviewPanel_Click;
                _bgPreviewBoxes[i] = sc;

                if (firstValid < 0) firstValid = i;
            }

            if (firstValid >= 0)
            {
                // SmartCanvas 覆蓋 camLiveMain：支援滑鼠滾輪縮放 + 左鍵拖曳
                _bgPreviewMainCanvas = new SmartCanvas { Dock = DockStyle.Fill, ClampPan = true };
                _bgPreviewMainCanvas.Image = _bgPreviewBitmaps[firstValid];
                _bgPreviewMainCanvas.StatusChanged += BgPreviewCanvas_StatusChanged;
                camLiveMain.Controls.Add(_bgPreviewMainCanvas);
                _bgPreviewMainCanvas.BringToFront();
                _bgPreviewMainCanvas.FitToScreen();
                _bgPreviewActive = true;
                _bgPreviewSelectedCamIndex = firstValid;
            }
            else
            {
                _bgPreviewBitmaps = null;
                _bgPreviewBoxes = null;
                MessageBox.Show("未找到背景 bin 檔。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }

        /// <summary>點選 camLive 上的 PictureBox → 切換 camLiveMain 顯示該相機背景。</summary>
        private void BgPreviewPanel_Click(object sender, EventArgs e)
        {
            if (!_bgPreviewActive || _bgPreviewBitmaps == null || _bgPreviewMainCanvas == null) return;

            var sc = sender as SmartCanvas;
            if (sc?.Tag is int idx && idx >= 0 && idx < _bgPreviewBitmaps.Length && _bgPreviewBitmaps[idx] != null)
            {
                _bgPreviewMainCanvas.Image = _bgPreviewBitmaps[idx];
                _bgPreviewMainCanvas.FitToScreen();
                _bgPreviewSelectedCamIndex = idx;
            }
        }

        /// <summary>SmartCanvas 滑鼠移動時更新 lblPixelInfo（與 camReviewMain 同格式：mm 座標 + 範圍 + 倍率）。</summary>
        private void BgPreviewCanvas_StatusChanged(CanvasInfo info)
        {
            if (lblPixelInfo == null) return;

            int camIdx = _bgPreviewSelectedCamIndex;
            int camId = camIdx + 1;
            string text;
            if (info.ImageX < 0 || info.ImageY < 0 ||
                _bgPreviewMainCanvas?.Image == null ||
                info.ImageX >= _bgPreviewMainCanvas.Image.Width ||
                info.ImageY >= _bgPreviewMainCanvas.Image.Height)
            {
                text = $"背景預覽 [CAM {camId}] | 游標超出影像範圍";
            }
            else if (_settings == null)
            {
                int gray = info.PixelColor.R;
                text = $"背景預覽 [CAM {camId}] | 座標: ({info.ImageX}, {info.ImageY}) | 亮度: {gray} | 縮放: {info.Zoom:F2}x";
            }
            else
            {
                double[] opsUmArr  = _settings.GetCameraOpsUmArray();
                double[] startMmArr = _settings.GetCameraStartPositionMmArray();
                if (camIdx < 0 || camIdx >= opsUmArr.Length)
                {
                    int gray = info.PixelColor.R;
                    text = $"背景預覽 [CAM {camId}] | 座標: ({info.ImageX}, {info.ImageY}) | 亮度: {gray} | 縮放: {info.Zoom:F2}x";
                }
                else
                {
                    double opsInMm    = opsUmArr[camIdx] / 1000.0;
                    double startPosMm = startMmArr[camIdx];

                    // 背景圖為全解析度，scaleFactor = 1
                    double physicalX = startPosMm + info.ImageX * opsInMm;
                    double rowPitchMm = _interactionHelper?.RowPitchMm ?? 0;
                    double physicalY = info.ImageY * rowPitchMm;

                    // 視野範圍
                    double viewLeftMm = 0, viewRightMm = 0;
                    double viewTopMm = 0, viewBotMm = 0;
                    if (info.Zoom > 0)
                    {
                        double pixelLeft  = (0                            - info.PanOffset.X) / info.Zoom;
                        double pixelRight = (_bgPreviewMainCanvas.Width   - info.PanOffset.X) / info.Zoom;
                        viewLeftMm  = startPosMm + pixelLeft  * opsInMm;
                        viewRightMm = startPosMm + pixelRight * opsInMm;

                        if (rowPitchMm > 0)
                        {
                            double pixelTop = (0                            - info.PanOffset.Y) / info.Zoom;
                            double pixelBot = (_bgPreviewMainCanvas.Height  - info.PanOffset.Y) / info.Zoom;
                            viewTopMm = pixelTop * rowPitchMm;
                            viewBotMm = pixelBot * rowPitchMm;
                        }
                    }

                    // 實體倍率
                    double screenMmPerPx = _interactionHelper?.ScreenMmPerPixel ?? 0;
                    string magStr = "-";
                    if (info.Zoom > 0 && screenMmPerPx > 0 && opsInMm > 0)
                    {
                        double physicalMag = (info.Zoom * screenMmPerPx) / opsInMm;
                        magStr = $"{physicalMag:F2}x";
                    }

                    text = $"背景預覽 [CAM {camId}] | " +
                           $"位置:({physicalX:F2}, {physicalY:F2}) mm | " +
                           $"X範圍:{viewLeftMm:F1}~{viewRightMm:F1} mm | " +
                           $"Y範圍:{viewTopMm:F1}~{viewBotMm:F1} mm | " +
                           $"座標: ({info.ImageX}, {info.ImageY}) | " +
                           $"亮度: {info.PixelColor.R} | " +
                           $"實體倍率:{magStr}";
                }
            }

            SafeBeginInvoke(() => lblPixelInfo.Text = text);
        }

        /// <summary>背景預覽模式：將 camLiveMain 上的 SmartCanvas 設為實體倍率 1x（畫面中心不動）。</summary>
        private void SetBgPreviewPhysicalMag1x()
        {
            if (_bgPreviewMainCanvas?.Image == null || _settings == null) return;

            int camIdx = _bgPreviewSelectedCamIndex;
            double[] opsUmArr = _settings.GetCameraOpsUmArray();
            if (camIdx < 0 || camIdx >= opsUmArr.Length) return;

            double opsInMm = opsUmArr[camIdx] / 1000.0;
            double screenMmPerPx = _interactionHelper?.ScreenMmPerPixel ?? 0;
            if (opsInMm <= 0 || screenMmPerPx <= 0) return;

            // scaleFactor=1 for background preview
            float zoom1x = (float)(opsInMm / screenMmPerPx);

            // keep center of current view stable
            float oldZoom = _bgPreviewMainCanvas.Zoom;
            PointF oldPan = _bgPreviewMainCanvas.PanOffset;
            float cx = _bgPreviewMainCanvas.Width / 2f;
            float cy = _bgPreviewMainCanvas.Height / 2f;
            float imgCx = (cx - oldPan.X) / oldZoom;
            float imgCy = (cy - oldPan.Y) / oldZoom;
            float newPanX = cx - imgCx * zoom1x;
            float newPanY = cy - imgCy * zoom1x;

            _bgPreviewMainCanvas.SetView(zoom1x, new PointF(newPanX, newPanY));
        }

        /// <summary>
        /// 清除所有面板的背景預覽。
        /// restoreMilDisplay=true 時恢復 MIL display（用於 btnLiveGrab 等需要回到即時畫面的場景）。
        /// 預設 false，避免在即將重新進入預覽時產生殘影。
        /// </summary>
        private void ClearBackgroundPreview(bool restoreMilDisplay = false)
        {
            // 移除 camLive 上的 overlay SmartCanvas
            Panel[] livePanels = GetLivePanels();
            if (_bgPreviewBoxes != null)
            {
                for (int i = 0; i < _bgPreviewBoxes.Length; i++)
                {
                    var sc = _bgPreviewBoxes[i];
                    if (sc == null) continue;
                    sc.Click -= BgPreviewPanel_Click;
                    sc.Image = null;
                    livePanels[i].Controls.Remove(sc);
                    sc.Dispose();
                }
                _bgPreviewBoxes = null;
            }

            // 移除 camLiveMain 上的 SmartCanvas
            if (_bgPreviewMainCanvas != null)
            {
                _bgPreviewMainCanvas.StatusChanged -= BgPreviewCanvas_StatusChanged;
                _bgPreviewMainCanvas.Image = null;
                camLiveMain.Controls.Remove(_bgPreviewMainCanvas);
                _bgPreviewMainCanvas.Dispose();
                _bgPreviewMainCanvas = null;
            }

            // Dispose bitmaps
            if (_bgPreviewBitmaps != null)
            {
                foreach (var bmp in _bgPreviewBitmaps)
                    bmp?.Dispose();
                _bgPreviewBitmaps = null;
            }

            _bgPreviewActive = false;

            if (restoreMilDisplay && _liveCameraManager?.IsAllocated == true)
            {
                // 恢復 primary display（camLive）+ secondary display（camLiveMain）
                foreach (var cam in _liveCameraManager.Cameras)
                    cam.SetPrimaryDisplayVisible(true);
                _liveCameraManager.RefreshMainDisplay();
            }
        }

        private bool IsStandardBgSubEnabled =>
            _settings?.Recipe?.Algorithm == BackgroundAlgorithm.StandardBgSub;

        private bool IsLightReadyForBg =>
            !(_settings?.LightEnabled == true) || (_lightController != null && _lightController.IsConnected);

        private bool _autoStartGrabAfterBg;

        private bool IsBgBinReady()
        {
            if (!IsStandardBgSubEnabled) return true;
            string bgDir = _settings.Storage.BackgroundPath;
            if (_liveCameraManager?.IsAllocated == true)
            {
                foreach (var cam in _liveCameraManager.Cameras)
                {
                    if (!cam.IsConnected) continue;
                    if (cam.FrameWidth <= 0) continue;
                    string binPath = Path.Combine(bgDir, CaptureFileNaming.BgBin(cam.FrameWidth, cam.CameraId));
                    if (!File.Exists(binPath)) return false;
                }
                return true;
            }
            return Directory.Exists(bgDir) && Directory.GetFiles(bgDir, CaptureFileNaming.BgGlob).Length > 0;
        }
    }
}
