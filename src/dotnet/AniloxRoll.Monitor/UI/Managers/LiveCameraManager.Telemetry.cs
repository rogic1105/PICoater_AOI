using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;
using MilGrabber.Core;
using TanukiCv.Core; // PixelMmMapper（已收進 sdk 唯一來源）
using TanukiCv.Controls; // LiveDisplayView（共用多相機監控顯示元件）
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.Core.Interop; // NativeMethods（LOD GPU resize；P/Invoke 宣告唯一點）
    // partial：游標座標 + 狀態 timer（lblPixelInfo / hw-ready）→ 未來 LiveDisplayCoordinator

namespace AniloxRoll.Monitor.UI.Managers
{
    public partial class LiveCameraManager
    {
        // ==================== Stall 偵測（縮圖 FPS 判據）====================
        // IsLive 但 FPS≈0 持續數 tick＝stall（改線掃/高度最常見）。此時 IsLive 仍 true → 無法自救。
        // 實測：停/開（甚至停止抓取→開始抓取）都救不回，只有重開程式 → 這是硬體層 CL 失鎖，
        // 不做無效的自動停/開（純 thrash），只用縮圖紅「STALL」明確標示；救援交給深度 re-init（後續）。
        // FPS 可低到 0.1＝合法慢速（一幀 10s），故 floor 設遠低於 0.1，只認真正的 0。
        private const double StallFpsFloor = 0.05; // FPS 低於此＝真的沒回傳（0.1 仍算活著）
        private const int StallTicks = 4;          // 連續 N 個 500ms tick（=2s）才判 stall（避開重啟瞬間暫態）
        private readonly Dictionary<int, int> _stallTicks = new Dictionary<int, int>();

        // ==================== Mouse Data ====================

        private void HandleMouseDataChanged(int camId, int x, int y, int pixelValue)
        {
            // MIL display hook 執行緒回 UI；關閉/釋放期 form 已 dispose → 守 guard 防 InvalidOperationException
            if (IsReleasing || _mainForm == null || _mainForm.IsDisposed || !_mainForm.IsHandleCreated) return;
            if (_mainForm.InvokeRequired)
            {
                try { _mainForm.BeginInvoke(new Action(() => HandleMouseDataChanged(camId, x, y, pixelValue))); }
                catch (InvalidOperationException) { /* ObjectDisposedException 亦繼承自此 */ }
                return;
            }

            string infoText;
            if (pixelValue == -1)
            {
                infoText = $"即時影像 [CAM {camId}] | 游標超出影像範圍";
            }
            else
            {
                int camIdx = camId - 1;
                var s = _inspectionSettings;
                double[] opsUmArr  = s?.GetCameraOpsUmArray();
                double[] startMmArr = s?.GetCameraStartPositionMmArray();

                if (opsUmArr == null || camIdx < 0 || camIdx >= opsUmArr.Length)
                {
                    infoText = $"即時影像 [CAM {camId}] | 座標: ({x}, {y}) | 亮度: {pixelValue}";
                }
                else
                {
                    double opsInMm    = opsUmArr[camIdx] / 1000.0;
                    double startPosMm = startMmArr[camIdx];
                    double physicalX  = PixelMmMapper.PixelToMm(x, startPosMm, opsInMm);
                    double lineRateHz = (camIdx < _cameraLineRateHz.Length) ? _cameraLineRateHz[camIdx] : 0;
                    double speedMPerMin = s.AniloxRollSpeedMPerMin;
                    double rowPitchMm = (speedMPerMin > 0 && lineRateHz > 0)
                        ? (speedMPerMin / 60.0 * 1000.0) / lineRateHz : 0;
                    double physicalY  = y * rowPitchMm;

                    // MIL display zoom/pan → 視野範圍
                    string rangeStr = "";
                    string magStr = "-";
                    var cam = _cameras.Find(c => c.CameraId == camId);
                    if (cam != null && cam.TryGetSecondaryDisplayGeometry(
                            out double zoomX, out _, out double panOffX, out double panOffY))
                    {
                        double panelW = _mainDisplayPanel.Width;
                        double panelH = _mainDisplayPanel.Height;
                        double viewLeftMm  = PixelMmMapper.PixelToMm(panOffX, startPosMm, opsInMm);
                        double viewRightMm = PixelMmMapper.PixelToMm(panOffX + panelW / zoomX, startPosMm, opsInMm);
                        rangeStr = $"X範圍:{viewLeftMm:F1}~{viewRightMm:F1} mm | ";

                        if (rowPitchMm > 0)
                        {
                            double viewTopMm = panOffY * rowPitchMm;
                            double viewBotMm = (panOffY + panelH / zoomX) * rowPitchMm;
                            rangeStr += $"Y範圍:{viewTopMm:F1}~{viewBotMm:F1} mm | ";
                        }

                        if (_screenMmPerPx > 0 && opsInMm > 0)
                        {
                            double physicalMag = PixelMmMapper.PhysicalMagnification(zoomX, _screenMmPerPx, opsInMm);
                            magStr = $"{physicalMag:F2}x";
                        }
                    }

                    infoText = $"即時影像 [CAM {camId}] | " +
                               $"位置:({physicalX:F2}, {physicalY:F2}) mm | " +
                               rangeStr +
                               $"座標: ({x}, {y}) | " +
                               $"亮度: {pixelValue} | " +
                               $"實體倍率:{magStr}";
                }
            }

            _updatePixelInfoCallback?.Invoke(infoText);
        }

        // ==================== Status Timer ====================

        /// <summary>
        /// 每 500ms 輪詢相機連線狀態並自動重啟抓圖，同 CameraSession.UpdatePresence()。
        /// IsReleasing = true 時提早返回，防止存取已釋放的相機資源。
        /// 使用快照（ToArray）避免 background FreeCameras 呼叫 _cameras.Clear() 時導致 InvalidOperationException。
        /// </summary>
        private void CameraStatusTimer_Tick(object sender, EventArgs e)
        {
            if (IsReleasing) return;

            // CLProtocol 背景初始化期間（分配後 ~2-10s）：UI 執行緒不可呼叫 CheckPresence（MdigInquire），
            // 否則與背景 CLProtocol enable（MdigControl）搶 MIL 內部鎖 → UI 執行緒卡在 tick 裡 →
            // 整個視窗凍結（拖不動）+ 顯示誤導的暫態連線數。就緒前完全跳過輪詢，UI 維持「初始化中」。
            // AreCamerasHwReady 只讀 _clProtocolInitDone 旗標（非 MIL 呼叫），不造成競爭。
            if (!AreCamerasHwReady) return;

            // 先拍快照：防止 ReleaseAsync 在 background thread 執行 _cameras.Clear() 時，
            // foreach 拋出 InvalidOperationException 或存取已釋放的相機物件。
            AniloxCamera[] snapshot;
            try { snapshot = _cameras.ToArray(); }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.CameraStatusTimer] {ex.GetType().Name}: {ex.Message}"); return; }

            foreach (var cam in snapshot)
            {
                if (IsReleasing) return; // 釋放流程已開始，立即中止

                bool isConnected = cam.CheckPresence();

                // 連線恢復且使用者希望抓圖時，自動重啟（同 CameraSession.UpdatePresence）
                if (isConnected && cam.UserWantsGrab && !cam.IsLive)
                    cam.ApplyGrabState();

                string statusText;
                Color color;
                if (!isConnected)
                {
                    statusText = "斷線"; color = Color.Pink;
                    _stallTicks[cam.CameraId] = 0;
                }
                else if (!cam.IsLive)
                {
                    statusText = "就緒"; color = Color.Yellow;
                    _stallTicks[cam.CameraId] = 0;
                }
                else
                {
                    double fps = cam.CurrentFps;
                    if (fps < StallFpsFloor)   // 真正沒回傳（0.1 等慢速不算）
                    {
                        int t = (_stallTicks.TryGetValue(cam.CameraId, out var v) ? v : 0) + 1;
                        _stallTicks[cam.CameraId] = t;
                        if (t >= StallTicks) { statusText = "STALL"; color = Color.Red; }   // 標示即可，停/開救不回
                        else { statusText = $"FPS: {fps:F1}"; color = Color.LightGreen; }
                    }
                    else
                    {
                        _stallTicks[cam.CameraId] = 0;
                        statusText = $"FPS: {fps:F1}"; color = Color.LightGreen;
                    }
                }

                UpdateSingleCameraStatus(cam.CameraId, statusText, color);
            }

            // 彙總連線數，變化時通知 UI
            int connected = 0;
            foreach (var cam in snapshot)
                if (cam.IsConnected) connected++;
            if (connected != ConnectedCameraCount)
            {
                ConnectedCameraCount = connected;
                OnCameraCountChanged?.Invoke(connected, ExpectedCameraCount);
            }

            // CLProtocol 全就緒 → 一次性通知 UI 解鎖「開始抓取」鈕
            if (!_hwReadyRaised && AreCamerasHwReady)
            {
                _hwReadyRaised = true;
                OnHwReady?.Invoke();
            }
        }
    }
}
