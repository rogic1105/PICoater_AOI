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
        // Stall 偵測職責已提取到 CameraStallDetector（純邏輯、可單獨測；判據＝M_PROCESS_FRAME_COUNT 前進、非 FPS 門檻）。
        private readonly CameraStallDetector _stallDetector = new CameraStallDetector();
        private readonly Dictionary<int, bool> _lastPresence = new Dictionary<int, bool>();    // 上次 tick 在線狀態（斷線→連線邊緣偵測 → 重跑 CLProtocol）
        private bool _wasHwReady;                                                              // 上次 tick AreCamerasHwReady（偵 false→true 轉變 → 強制刷 count label/解鎖鈕）

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
            // 重連重跑 CLProtocol 時會暫時 false（gate-off 防搶 MIL 鎖）→ 恢復 true 時需強制刷 count label
            // 與重發 OnHwReady（否則連線數在 gate-off 前就已更新、恢復後 == 舊值 → 不觸發事件 → lblCamCount 卡灰色「初始化中」）。
            bool hwReady = AreCamerasHwReady;
            if (!hwReady) { _wasHwReady = false; _hwReadyRaised = false; return; }
            bool hwJustReady = !_wasHwReady;
            _wasHwReady = true;

            // 先拍快照：防止 ReleaseAsync 在 background thread 執行 _cameras.Clear() 時，
            // foreach 拋出 InvalidOperationException 或存取已釋放的相機物件。
            AniloxCamera[] snapshot;
            try { snapshot = _cameras.ToArray(); }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.CameraStatusTimer] {ex.GetType().Name}: {ex.Message}"); return; }

            foreach (var cam in snapshot)
            {
                if (IsReleasing) return; // 釋放流程已開始，立即中止

                bool isConnected = cam.CheckPresence();

                // 斷線→連線「邊緣」自動重跑 CLProtocol：啟動時相機不在線（CheckPresence=false 跳過 CLProtocol 啟用）、
                // 之後才連上的相機 → 此時重新啟用 CLProtocol，否則曝光/線掃等 GenICam 參數一律讀不到（回 0）。
                // **邊緣觸發**（非每輪）：避免 CL 握手失敗時每 500ms 一直重試 + 反覆 gate-off；下次拔插再觸發。
                // RetryCLProtocolOnReconnect 內含守門：已啟用/不在線/grab 中/in-flight 皆跳過（grab 中 enable 會掉幀）。
                bool wasConnected = _lastPresence.TryGetValue(cam.CameraId, out var pv) && pv;
                _lastPresence[cam.CameraId] = isConnected;
                if (isConnected && !wasConnected) cam.RetryCLProtocolOnReconnect();

                // 連線恢復且使用者希望抓圖時，自動重啟（同 CameraSession.UpdatePresence）
                if (isConnected && cam.UserWantsGrab && !cam.IsLive)
                    cam.ApplyGrabState();

                string statusText;
                Color color;
                if (!isConnected)
                {
                    statusText = "斷線"; color = Color.Pink;
                    _stallDetector.Reset(cam.CameraId);
                }
                else if (!cam.IsLive)
                {
                    statusText = "就緒"; color = Color.Yellow;
                    _stallDetector.Reset(cam.CameraId);
                }
                else
                {
                    // stall 偵測委派 CameraStallDetector（純邏輯）：餵累計幀數 + 預期 FPS（=線掃/高度）→ 判是否卡死。
                    double expFps = (cam.FrameHeight > 0 && cam.AppliedLineRateHz > 0)
                        ? cam.AppliedLineRateHz / cam.FrameHeight : 0;
                    bool stalled = _stallDetector.Update(cam.CameraId, cam.GetFrameCount(), expFps);
                    if (stalled) { statusText = "STALL"; color = Color.Red; }            // 幀數凍住超過窗＝真卡死
                    else { statusText = $"FPS: {cam.CurrentFps:F1}"; color = Color.LightGreen; }
                }

                UpdateSingleCameraStatus(cam.CameraId, statusText, color);
            }

            // 彙總連線數，變化時通知 UI
            int connected = 0;
            foreach (var cam in snapshot)
                if (cam.IsConnected) connected++;
            // 連線數變化、或剛從重連 CLProtocol gate-off 恢復就緒（hwJustReady）→ 通知 UI 刷 lblCamCount
            // （後者把 label 從灰色「初始化中」切回正確連線數/顏色）。
            if (connected != ConnectedCameraCount || hwJustReady)
            {
                ConnectedCameraCount = connected;
                OnCameraCountChanged?.Invoke(connected, ExpectedCameraCount);
            }

            // CLProtocol 全就緒 → 通知 UI 解鎖「開始抓取」鈕（_hwReadyRaised 於 gate-off 時重置 → 重連恢復後會重發）。
            if (!_hwReadyRaised && AreCamerasHwReady)
            {
                _hwReadyRaised = true;
                OnHwReady?.Invoke();
            }
        }
    }
}
