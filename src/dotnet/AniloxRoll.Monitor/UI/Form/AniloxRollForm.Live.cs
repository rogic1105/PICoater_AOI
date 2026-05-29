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
    /// <summary>AniloxRollForm 即時監控（grab 流程 / 即時曲線 / Mura 判定 / 強化切換）相關方法 — 由主檔拆出的 partial。</summary>
    public partial class AniloxRollForm
    {
        // ==========================================
        // --- 相機按鈕事件 ---
        // ==========================================

        private async void btnLiveGrab_Click(object sender, EventArgs e)
        {
            // 背景預覽中按 Grab → 先清除預覽並 Free，讓 MIL 能重新初始化
            if (_bgPreviewActive)
            {
                ClearBackgroundPreview(restoreMilDisplay: true);
                _liveCameraManager.FreeCameras();
                _telemetryPresenter?.ResetAll();
            }

            bool wasGrabbing = _liveCameraManager.IsLiveGrabbing;

            // 啟動路徑：先亮燈 → 等光源穩定 → 再開始 grab
            if (!wasGrabbing)
            {
                LightTurnOn();
                int warmup = _settings?.LightWarmupMs ?? 0;
                if (warmup > 0) await Task.Delay(warmup);
            }

            if (!_liveCameraManager.IsAllocated)
            {
                try
                {
                    _liveCameraManager.EnsureAllocatedAndToggleGrab(_settings.EnableMuraEnhance);
                    LoadBackgroundBins();

                    // 初次分配即為 Global 模式 → 立即啟用即時合圖
                    if (_settings.StitchMode == StitchMode.Global)
                        _liveCameraManager.EnableGlobalMerge(
                            _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray());
                }
                catch (Exception ex)
                {
                    LightTurnOff();
                    MessageBox.Show($"相機配置失敗: {ex.Message}", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
            }
            else
            {
                _liveCameraManager.ToggleGrab();
            }

            // 剛從「未抓取」→「抓取中」：分配新的抓圖編號（燈已在上方開啟）
            if (!wasGrabbing && _liveCameraManager.IsLiveGrabbing)
            {
                _currentGrabId = _inspectionLogService.NextGrabId();
            }

            // 剛從「抓取中」→「停止」：關燈 + 觸發循環儲存 + 通知儲存機清理
            if (wasGrabbing && !_liveCameraManager.IsLiveGrabbing)
            {
                LightTurnOff();
                TriggerRetentionAndFlagAsync();
            }

            UpdateGrabButton(_liveCameraManager.IsLiveGrabbing);
        }

        /// <summary>
        /// 相機存檔後回呼（MIL 執行緒，非 UI 執行緒）。
        /// EnableAutoCapture=true 且抓取中時才會觸發。
        /// </summary>
        private void OnCameraInspectionResult(int camId, string fileNameNoExt, float meanPeak, float maxPeak)
        {
            if (string.IsNullOrEmpty(_currentGrabId)) return;
            int idx = camId - 1;
            if (_inspectionLogService != null)
            {
                // OnCameraInspectionResult 的 meanPeak/maxPeak 為 V 方向（pipeline 主處理方向），用 V 閾值記錄
                _inspectionLogService.AppendRecord(
                    _currentGrabId,
                    fileNameNoExt,
                    meanPeak,
                    maxPeak,
                    _settings.ErrorValueMeanV,
                    _settings.ErrorValueMaxV,
                    idx >= 0 && idx < _settings.Acquisition.CameraGrabHeight.Length
                        ? _settings.Acquisition.CameraGrabHeight[idx] : 0,
                    idx >= 0 && idx < _settings.Acquisition.CameraLineRateHz.Length
                        ? _settings.Acquisition.CameraLineRateHz[idx] : 0,
                    idx >= 0 && idx < _settings.Acquisition.CameraExposureTimeUs.Length
                        ? _settings.Acquisition.CameraExposureTimeUs[idx] : 0,
                    CsvConfigSnapshot.FromSettings(_settings));

                // CSV 寫完後排入遠端複製佇列（CSV 在 month 目錄，不在 OnFilesSaved 的 day 目錄）
                string csvPath = _inspectionLogService.LastCsvPath;
                if (!string.IsNullOrEmpty(csvPath))
                    _remoteCopyService?.EnqueueFile(csvPath);
            }

            // IO MURA 信號：任一相機超過閾值即通知
            if (_ioGrabController?.IsConnected == true)
            {
                // meanPeak/maxPeak 為 V 方向，按 V 閾值判定
                bool isMura = meanPeak > _settings.ErrorValueMeanV || maxPeak > _settings.ErrorValueMaxV;
                if (isMura) _ = _ioGrabController.NotifyMuraDetected();
            }

            // 抓圖計數器 + watchdog 時間戳（Inspection 模式）
            if (_appMode?.Role != MachineRole.Storage)
            {
                _lastGrabEventTime = DateTime.UtcNow;
                int count = System.Threading.Interlocked.Increment(ref _completedGrabCount);
                if (count % 10 == 0)
                    TriggerRetentionAndFlagAsync();
            }
        }


        /// <summary>
        /// Live 曲線閾值判斷（callback 執行緒呼叫）。
        /// direction: "v"=垂直, "h"=水平；依 CheckLiveMura 設定的「檢測方向」決定是否觸發 DO1。
        /// 陣列為 0-255，閾值為 0-1，取陣列 max 後除以 255 比較。
        /// </summary>
        private void CheckLiveMura(float[] meanArr, float[] maxArr, string direction)
        {
            if (_isMuraDetectPaused) return;
            if (_ioGrabController?.IsConnected != true) return;
            if (_settings == null) return;
            if (!_liveCameraManager.IsLiveGrabbing) return;

            var ridgeDir = _settings.RidgeDir;
            if (direction == "v" && ridgeDir == RidgeDirection.Horizontal) return;
            if (direction == "h" && ridgeDir == RidgeDirection.Vertical)   return;

            float meanPeak = 0f, maxPeak = 0f;
            if (meanArr != null) { for (int i = 0; i < meanArr.Length; i++) if (meanArr[i] > meanPeak) meanPeak = meanArr[i]; }
            if (maxArr  != null) { for (int i = 0; i < maxArr.Length;  i++) if (maxArr[i]  > maxPeak)  maxPeak  = maxArr[i];  }
            meanPeak /= 255f;
            maxPeak  /= 255f;

            // 依 direction 用對應方向閾值
            float thMean = direction == "h" ? _settings.ErrorValueMeanH : _settings.ErrorValueMeanV;
            float thMax  = direction == "h" ? _settings.ErrorValueMaxH  : _settings.ErrorValueMaxV;

            if (meanPeak > thMean || maxPeak > thMax)
            {
                // fire-and-forget; 寫入失敗不應影響取像流程
                _ = _ioGrabController.NotifyMuraDetected().ContinueWith(
                    t => { /* swallow — PollTick 會偵測真正的 CommLost */ },
                    TaskContinuationOptions.OnlyOnFaulted);
            }
        }

        private void OnLiveCurveData(int camId, float[] meanArr, float[] maxArr)
        {
            // 快取每台相機最新曲線（callback 執行緒，只是 ref 賦值）
            int cameraIndex = camId - 1;
            if (cameraIndex >= 0 && cameraIndex < CameraCount)
            {
                _liveCurveMean[cameraIndex] = meanArr;
                _liveCurveMax[cameraIndex]  = maxArr;
                // M8: memory barrier 確保 UI thread 透過 volatile _liveOverviewDirty 讀到 dirty=true 時，
                // array reference 寫入已完成（避免讀到舊指標）
                System.Threading.Interlocked.MemoryBarrier();
                _liveOverviewDirty = true;
            }

            // Live Mura 判斷（callback 執行緒，所有相機都檢查）
            CheckLiveMura(meanArr, maxArr, "v");

            // Global 模式不更新 Live mura 垂直圖（單台資料無意義）
            if (_settings.StitchMode == StitchMode.Global) return;

            // 只有選中相機才 marshal 到 UI 執行緒更新 muraChartLive
            if (camId != _liveCameraManager.SelectedMainCameraId) return;

            if (InvokeRequired)
            {
                if (!IsHandleCreated || IsDisposed || Disposing) return;
                BeginInvoke(new Action<int, float[], float[]>(OnLiveCurveData), camId, meanArr, maxArr);
                return;
            }

            if (_liveColumnChartHelper == null || _settings == null) return;

            double[] opsUmArr       = _settings.GetCameraOpsUmArray();
            double[] startPositions = _settings.GetCameraStartPositionMmArray();

            double opsUm = (cameraIndex >= 0 && cameraIndex < opsUmArr.Length)
                ? opsUmArr[cameraIndex] : _settings.Cam1_Ops;
            double opsInMm  = opsUm / 1000.0;
            double startPos = (cameraIndex >= 0 && cameraIndex < startPositions.Length)
                ? startPositions[cameraIndex] : 0;

            _liveColumnChartHelper.SetOps(opsUm);

            // 查詢 MIL 副顯示器的實際 zoom/pan（隨使用者滾輪操作即時變化）
            // panOffsetX = 面板左邊緣對應的 buffer pixel X
            // rightPixel = panOffsetX + panelWidth / zoomX
            double viewLeftMm = double.NaN, viewRightMm = double.NaN;

            var liveCam = FindCameraById(camId);

            if (liveCam != null && opsInMm > 0 &&
                liveCam.TryGetSecondaryDisplayGeometry(
                    out double milZoomX, out double milZoomY, out double milPanX, out double milPanY))
            {
                double panelW = camLiveMain.Width;
                double leftPixel  = milPanX;
                double rightPixel = milPanX + panelW / milZoomX;
                viewLeftMm  = startPos + leftPixel  * opsInMm;
                viewRightMm = startPos + rightPixel * opsInMm;
            }

            _liveColumnChartHelper.UpdateDataAndView(meanArr, maxArr,
                startPos, viewLeftMm, viewRightMm);
        }

        private void OnLiveRowCurveData(int camId, float[] meanArr, float[] maxArr)
        {
            // Live Mura 判斷（水平方向）
            CheckLiveMura(meanArr, maxArr, "h");

            if (InvokeRequired)
            {
                if (!IsHandleCreated || IsDisposed || Disposing) return;
                BeginInvoke(new Action<int, float[], float[]>(OnLiveRowCurveData), camId, meanArr, maxArr);
                return;
            }

            if (_liveRowChartHelper == null) return;

            bool isGlobal = _liveCameraManager?.IsGlobalMergeActive == true;

            if (isGlobal)
            {
                // 全域模式：快取每台相機資料，合併後更新（mean 取 mean, max 取 max）
                _liveRowMeanCache[camId] = meanArr;
                _liveRowMaxCache[camId]  = maxArr;
                MergeAndUpdateLiveRowChart();

                // 同步 Y 軸視野：查詢 _mergedDisplay 的 zoom/pan
                double rowPitch = _liveRowChartHelper.RowPitchMm;
                if (rowPitch > 0 && _liveCameraManager.TryGetMergedViewRangeY(
                    out double topPixel, out double botPixel))
                {
                    _liveRowChartHelper.UpdateViewRange(topPixel * rowPitch, botPixel * rowPitch);
                }
            }
            else
            {
                // 垂直模式：只顯示選中相機
                if (camId != _liveCameraManager.SelectedMainCameraId) return;
                _liveRowChartHelper.UpdateData(meanArr, maxArr);

                // 同步 Y 軸視野：查詢 MIL 副顯示器 zoom/pan
                var liveCam = FindCameraById(camId);
                double rowPitch = _liveRowChartHelper.RowPitchMm;
                if (liveCam != null && rowPitch > 0 &&
                    liveCam.TryGetSecondaryDisplayGeometry(
                        out double milZoomX, out double milZoomY, out double milPanX, out double milPanY))
                {
                    double panelH  = camLiveMain.Height;
                    double topPixel = milPanY;
                    double botPixel = milPanY + panelH / milZoomY;
                    _liveRowChartHelper.UpdateViewRange(topPixel * rowPitch, botPixel * rowPitch);
                }
            }
        }

        /// <summary>合併所有快取的 row curve 資料：mean 取平均、max 取最大值。</summary>
        private void MergeAndUpdateLiveRowChart()
        {
            if (_liveRowMeanCache.Count == 0) return;

            // 取最短長度對齊
            int minLen = int.MaxValue;
            foreach (var arr in _liveRowMeanCache.Values)
                if (arr.Length < minLen) minLen = arr.Length;
            if (minLen <= 0 || minLen == int.MaxValue) return;

            float[] mergedMean = new float[minLen];
            float[] mergedMax  = new float[minLen];

            int camCount = _liveRowMeanCache.Count;
            foreach (var arr in _liveRowMeanCache.Values)
                for (int i = 0; i < minLen; i++)
                    mergedMean[i] += arr[i];
            for (int i = 0; i < minLen; i++)
                mergedMean[i] /= camCount;

            foreach (var arr in _liveRowMaxCache.Values)
                for (int i = 0; i < minLen; i++)
                    if (arr[i] > mergedMax[i]) mergedMax[i] = arr[i];

            _liveRowChartHelper.UpdateData(mergedMean, mergedMax);
        }

        /// <summary>用 A輪速度 和選中相機的取樣頻率（Line Rate）更新法向圖表座標。</summary>
        private void UpdateRowChartPitch()
        {
            if (_settings == null) return;
            double lineRateHz = _settings.Acquisition.CameraLineRateHz[0]; // CAM1 master
            _liveRowChartHelper?.SetRowPitchFromSpeed(
                _settings.AniloxRollSpeedMPerMin, lineRateHz);
            _reviewRowChartHelper?.SetRowPitchFromSpeed(
                _settings.AniloxRollSpeedMPerMin, lineRateHz);
        }


        private Panel[] GetLivePanels() => new[]
        {
            camLive1, camLive2, camLive3,
            camLive4, camLive5, camLive6, camLive7
        };

        /// <summary>
        /// 將 float[] column mean 擴展為 width×height 的 8bpp 灰階 Bitmap。
        /// 每列（row）相同：pixel[x] = clamp(colMean[x], 0, 255)。
        /// </summary>
        private static Bitmap ExpandColMeanToBitmap(float[] colMean, int width, int height)
        {
            byte[] row = new byte[width];
            for (int x = 0; x < width; x++)
            {
                float v = colMean[x];
                row[x] = v <= 0 ? (byte)0 : v >= 255 ? (byte)255 : (byte)(v + 0.5f);
            }

            byte[] pixels = new byte[width * height];
            for (int y = 0; y < height; y++)
                Buffer.BlockCopy(row, 0, pixels, y * width, width);

            return ImageUtils.Create8bppBitmap(pixels, width, height);
        }

        private void UpdateGrabButton(bool isGrabbing)
        {
            btnLiveGrab.Text = isGrabbing ? "停止抓取" : "開始抓取";
            // 抓取中：凍結取得背景/預覽背景；停止後解鎖
            btnLiveGetBackground.Enabled = !isGrabbing;
            btnLiveViewBackground.Enabled = !isGrabbing;
            if (!isGrabbing)
            {
                UpdateStandardBgSubLockState(); // 停止後依 bin 狀態重新檢查
            }
        }

        private void ApplyMuraEnhance(bool enabled)
        {
            _liveCameraManager?.SetImageProcessingEnabled(enabled);
            _liveCameraManager?.SetLiveDisplayDirection(_liveDisplayDirection);
            UpdateLiveDirectionVisual();
        }

        /// <summary>
        /// 安全序列化：Live chart 點選切 StitchMode 時，若同時要關掉強化，
        /// 必須先把 callback thread 的 chart 更新訂閱斷開（C），避免轉場期間 callback
        /// BeginInvoke 到 chart handle 不穩定的視窗。
        /// 並把 UpdateLiveDirectionVisual 延後到 OnStitchModeChangedAsync 之後一次性執行（D），
        /// 減少 Border 屬性變更引起的 paint storm。
        ///
        /// **DEBUG**：每步驟 Trace.WriteLine + 寫 D:\Anilox\stitch-debug.log；
        /// 任何 exception 抓到後彈 MessageBox 顯示完整 stack trace，並寫 log 檔。
        /// </summary>
        private static void LogClick(string msg, MouseEventArgs e = null)
        {
            string suffix = e != null ? $" (Button={e.Button} Loc={e.X},{e.Y})" : "";
            string line = $"[{DateTime.Now:HH:mm:ss.fff}] [Click] {msg}{suffix}";
            try { System.IO.File.AppendAllText(@"D:\Anilox\stitch-debug.log", line + Environment.NewLine); } catch { }
        }

        /// <summary>
        /// 全域 IMessageFilter：log 每次 WM_LBUTTONDOWN 命中的控制項 + 螢幕座標。
        /// 用來診斷 Live chart click 為什麼沒觸發 MouseClick（panel/MIL native window 截獲？
        /// chart 內部吞掉？bounds 重疊？）。試一次就能看出 click 去了哪裡。
        /// </summary>
        private sealed class GlobalMouseLogger : IMessageFilter
        {
            private const int WM_LBUTTONDOWN = 0x0201;
            public bool PreFilterMessage(ref Message m)
            {
                if (m.Msg == WM_LBUTTONDOWN)
                {
                    try
                    {
                        var c = Control.FromHandle(m.HWnd);
                        var pt = Cursor.Position;
                        string name = c?.Name ?? "(null)";
                        string type = c?.GetType().Name ?? "(no-type)";
                        string line = $"[{DateTime.Now:HH:mm:ss.fff}] [MsgFilter] WM_LBUTTONDOWN hwnd=0x{m.HWnd.ToInt64():X} ctl={name}({type}) screen=({pt.X},{pt.Y})";
                        System.IO.File.AppendAllText(@"D:\Anilox\stitch-debug.log", line + Environment.NewLine);
                    }
                    catch { }
                }
                return false; // 不攔截，繼續傳遞
            }
        }

        /// <summary>
        /// Live chart 點選切 StitchMode 時，若同時要關掉強化，必須先把 callback thread 的 chart 更新訂閱斷開，
        /// 避免轉場期間 callback BeginInvoke 到 chart handle 不穩定的視窗。
        /// L2：setting 變更走 Hub.SetBatch 統一 save；副作用 transition 仍 inline await（避免 event race）。
        /// </summary>
        private async Task SwitchStitchModeWithEnhanceSequence(StitchMode newMode)
        {
            if (_settings == null) return;
            bool wasEnhanced = _settings.EnableMuraEnhance;
            try
            {
                _liveCameraManager.OnLiveCurveData    -= OnLiveCurveData;
                _liveCameraManager.OnLiveRowCurveData -= OnLiveRowCurveData;

                _settingsHub.SetBatch(s =>
                {
                    if (wasEnhanced) s.EnableMuraEnhance = false;
                    s.hb_StitchMode = newMode;
                });
                if (wasEnhanced) _liveCameraManager?.SetImageProcessingEnabled(false);
                if (wasEnhanced) RefreshGridItem(nameof(InspectionSettings.hc_EnableMuraEnhance));
                RefreshGridItem(nameof(InspectionSettings.hb_StitchMode));
                await OnStitchModeChangedAsync();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"切換 StitchMode 異常:\n{ex}", "StitchMode", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                _liveCameraManager.OnLiveCurveData    += OnLiveCurveData;
                _liveCameraManager.OnLiveRowCurveData += OnLiveRowCurveData;
                try { UpdateLiveDirectionVisual(); } catch (Exception ex) { Trace.WriteLine(ex); }
            }
        }
    }
}
