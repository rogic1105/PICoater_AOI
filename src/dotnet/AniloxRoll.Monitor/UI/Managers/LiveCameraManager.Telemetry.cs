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
        // ==================== Stall 偵測（幀數前進判據，**非** FPS 門檻）====================
        // IsLive 但 grab 卡死＝stall（改線掃/高度最常見）。此時 IsLive 仍 true → 無法自救。
        // 實測：停/開（甚至停止抓取→開始抓取）都救不回，只有重開程式 → 硬體層 CL 失鎖，
        // 不做無效的自動停/開（純 thrash），只用縮圖紅「STALL」明確標示；救援交給深度 re-init（後續）。
        //
        // ★ 判據＝「M_PROCESS_FRAME_COUNT 有沒有前進」，不是 FPS 門檻（2026-06-24 修誤判）：
        //   低線掃 + 高高度時**合法** FPS 本來就極低（100Hz/12000＝0.0083 fps、一幀 120s），固定 FPS 門檻
        //   會把「慢但正常」誤判成 stall。改看幀數：真 stall＝幀數凍住不動；慢速 grab＝幀數仍慢慢加（不誤判）。
        //   偵測窗依「預期幀週期＝高度/線掃」自動拉長：高速 ~2s 偵到、低速自動等久一點（仍會偵到真卡死，只是慢）。
        private const int StallBaseTicks = 4;        // 基準窗（4×500ms＝2s，避開重啟暫態）
        private const double StallPeriodFactor = 1.5; // 額外等「預期幀週期 × 此倍數」才判（容忍合法慢速抖動）
        private const int StatusTickMs = 500;        // CameraStatusTimer 間隔（與 new Timer{Interval=500} 一致）
        private readonly Dictionary<int, int> _stallTicks = new Dictionary<int, int>();
        private readonly Dictionary<int, long> _lastFrameCount = new Dictionary<int, long>(); // 上次 tick 的 M_PROCESS_FRAME_COUNT

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
                    long fc = cam.GetFrameCount();   // M_PROCESS_FRAME_COUNT（累計處理幀數）
                    long lastFc = _lastFrameCount.TryGetValue(cam.CameraId, out var lv) ? lv : -1;
                    _lastFrameCount[cam.CameraId] = fc;

                    // 幀數有任何變化（前進，或重啟後歸零＝減少）＝grab 活著 → 重置。只有「凍住不動」才累計。
                    bool advanced = (lastFc < 0) || (fc != lastFc);
                    if (advanced)
                    {
                        _stallTicks[cam.CameraId] = 0;
                        statusText = $"FPS: {fps:F1}"; color = Color.LightGreen;
                    }
                    else
                    {
                        int t = (_stallTicks.TryGetValue(cam.CameraId, out var v) ? v : 0) + 1;
                        _stallTicks[cam.CameraId] = t;

                        // 偵測窗＝基準 + 預期幀週期×倍數（低線掃/高高度合法慢→窗自動拉長，不誤判）。
                        int needed = StallBaseTicks;
                        double expFps = (cam.FrameHeight > 0 && cam.AppliedLineRateHz > 0)
                            ? cam.AppliedLineRateHz / cam.FrameHeight : 0;
                        if (expFps > 0)
                        {
                            double framePeriodMs = 1000.0 / expFps;
                            needed = StallBaseTicks + (int)Math.Ceiling(framePeriodMs * StallPeriodFactor / StatusTickMs);
                        }
                        if (t >= needed) { statusText = "STALL"; color = Color.Red; }   // 幀數凍住超過窗＝真卡死
                        else { statusText = $"FPS: {fps:F1}"; color = Color.LightGreen; }
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
