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

            // CLProtocol 尚未就緒不可開始抓取（grab 進行中才 enable + 重套線掃會掉幀，cam1 最明顯）。
            // 手動鈕在就緒前已是灰色；此處主要擋 IO 觸發路徑（IoStartGrab 直接呼叫本方法繞過按鈕狀態）。
            if (!wasGrabbing && _liveCameraManager.IsAllocated && !_liveCameraManager.AreCamerasHwReady)
            {
                Trace.WriteLine("[Grab] CLProtocol 尚未就緒，忽略開始抓取請求。");
                return;
            }

            // 啟動路徑：先亮燈 → 等光源穩定 → 再開始 grab
            if (!wasGrabbing)
            {
                LightTurnOn();
                int warmup = _settings?.LightWarmupMs ?? 0;
                if (warmup > 0) await Task.Delay(warmup);
                ResetLiveWaterfallRowChart();
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
        /// direction: "v"=欄, "h"=列；依 CheckLiveMura 設定的「檢測方向」決定是否觸發 DO1。
        /// 陣列為 0-255，閾值為 0-1，取陣列 max 後除以 255 比較。
        /// </summary>
        private void CheckLiveMura(float[] meanArr, float[] maxArr, string direction)
        {
            if (_settings.MuraDetectPaused) return;
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

        /// <summary>監控主畫面（ImageDisplayView）縮放/平移 → live 曲線圖 zoom 連動（bin↔主畫面對齊）。
        /// 欄(X)/overview(X) 用左右範圍、列(Y) 用上下範圍。UI 執行緒（ViewRangeMmChanged 來）。</summary>
        // 主畫面即時 X 可見範圍（mm）：ApplyLiveViewRange 存 → LiveViewRangeProvider 給 overview 的 500ms 更新沿用同值
        // （overview 立即跟隨 + 500ms 重畫沿用同範圍 → 不閃回原點）。NaN=非 ImageCanvas 即時狀態。
        private double _liveViewLeftMm = double.NaN, _liveViewRightMm = double.NaN;
        private double _liveViewTopMm = double.NaN, _liveViewBotMm = double.NaN;
        private bool _liveRowRangeSuspended;

        private bool ShouldFlipDisplayVertical()
            => GetVerticalDisplayDirection() == VerticalDisplayDirection.BottomToTop;

        private VerticalDisplayDirection GetVerticalDisplayDirection()
            => _settings?.VerticalDirection ?? InspectionDefaults.VerticalDirection;

        private void ApplyLiveViewRange(double leftMm, double rightMm, double topMm, double botMm)
        {
            if (IsDisposed) return;
            _liveViewLeftMm = leftMm; _liveViewRightMm = rightMm;     // 供 overview provider 沿用（不閃）
            _liveViewTopMm = topMm; _liveViewBotMm = botMm;
            if (!_liveRowRangeSuspended)
                _liveRowDisplay?.UpdateViewRange(topMm, botMm);            // 列(Y)
            _liveOverviewHelper?.UpdateViewRange(leftMm, rightMm);     // overview 立即跟隨（500ms 重畫用同值不閃）
        }

        private bool TryApplyLiveImageCanvasRowViewRange()
        {
            var mode = _settings?.he_MainDisplay;
            if (mode != MainDisplayMode.ImageCanvas && mode != MainDisplayMode.Waterfall) return false;
            if (double.IsNaN(_liveViewTopMm) || double.IsNaN(_liveViewBotMm)) return true;
            _liveRowDisplay?.UpdateViewRange(_liveViewTopMm, _liveViewBotMm);
            return true;
        }

        private void SuspendLiveRowRangeUntilNextData()
        {
            _liveRowRangeSuspended = true;
        }

        private bool ResumeLiveRowRangeAfterDataUpdate()
        {
            _liveRowRangeSuspended = false;
            return TryApplyLiveImageCanvasRowViewRange();
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
            // 單台欄 chart（chartLiveColumn 舊版）已刪除：全覽圖（接位後的 chartLiveColumn）
            // 由 _liveOverviewDirty + UpdateOverviewChart 路徑更新（boundary 唯一歸屬、與影像對齊）。
        }

        private void OnLiveRowCurveData(int camId, float[] meanArr, float[] maxArr)
        {
            // Live Mura 判斷（列方向）
            CheckLiveMura(meanArr, maxArr, "h");

            if (InvokeRequired)
            {
                if (!IsHandleCreated || IsDisposed || Disposing) return;
                BeginInvoke(new Action<int, float[], float[]>(OnLiveRowCurveData), camId, meanArr, maxArr);
                return;
            }

            if (_liveRowDisplay == null) return;

            if (_settings?.he_MainDisplay == MainDisplayMode.Waterfall)
            {
                UpdateLiveWaterfallRowChart(camId, meanArr, maxArr);
                return;
            }

            bool isGlobal = _liveCameraManager?.IsGlobalMergeActive == true;

            if (isGlobal)
            {
                // 全域模式：快取每台相機資料，合併後更新（mean 取 mean, max 取 max）
                _liveRowMeanCache[camId] = meanArr;
                _liveRowMaxCache[camId]  = maxArr;
                MergeAndUpdateLiveRowChart();
                if (ResumeLiveRowRangeAfterDataUpdate()) return;

                // 同步 Y 軸視野：查詢 _mergedDisplay 的 zoom/pan
                double rowPitch = _liveRowDisplay.RowPitchMm;
                if (rowPitch > 0 && _liveCameraManager.TryGetMergedViewRangeY(
                    out double topPixel, out double botPixel))
                {
                    _liveRowDisplay.UpdateViewRange(topPixel * rowPitch, botPixel * rowPitch);
                }
            }
            else
            {
                // 垂直模式：只顯示選中相機
                if (camId != _liveCameraManager.SelectedMainCameraId) return;
                _liveRowDisplay.UpdateData(meanArr, maxArr);
                if (ResumeLiveRowRangeAfterDataUpdate()) return;

                // 同步 Y 軸視野：查詢 MIL 副顯示器 zoom/pan
                var liveCam = FindCameraById(camId);
                double rowPitch = _liveRowDisplay.RowPitchMm;
                if (liveCam != null && rowPitch > 0 &&
                    liveCam.TryGetSecondaryDisplayGeometry(
                        out double milZoomX, out double milZoomY, out double milPanX, out double milPanY))
                {
                    double panelH  = camLiveMain.Height;
                    double topPixel = milPanY;
                    double botPixel = milPanY + panelH / milZoomY;
                    _liveRowDisplay.UpdateViewRange(topPixel * rowPitch, botPixel * rowPitch);
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

            _liveRowDisplay?.UpdateData(mergedMean, mergedMax);
        }

        private void ResetLiveWaterfallRowChart()
        {
            SuspendLiveRowRangeUntilNextData();
            _waterfallRowMeanPending.Clear();
            _waterfallRowMaxPending.Clear();
            _waterfallRowMean = null;
            _waterfallRowMax = null;
            _waterfallRowWrite = 0;
        }

        private void UpdateLiveWaterfallRowChart(int camId, float[] meanArr, float[] maxArr)
        {
            if (meanArr == null || meanArr.Length == 0 || _liveRowDisplay == null) return;

            bool isGlobal = _liveCameraManager?.IsGlobalMergeActive == true;
            if (!isGlobal)
            {
                if (camId != _liveCameraManager.SelectedMainCameraId) return;
                AppendLiveWaterfallRowBand(meanArr, maxArr);
                return;
            }

            _waterfallRowMeanPending[camId] = meanArr;
            _waterfallRowMaxPending[camId] = maxArr;

            int expected = Math.Max(1, _liveCameraManager?.ConnectedCameraCount ?? CameraCount);
            if (_waterfallRowMeanPending.Count < expected) return;

            var rowMean = new float[CameraCount][];
            var rowMax = new float[CameraCount][];
            foreach (var kv in _waterfallRowMeanPending)
                if (kv.Key >= 1 && kv.Key <= CameraCount) rowMean[kv.Key - 1] = kv.Value;
            foreach (var kv in _waterfallRowMaxPending)
                if (kv.Key >= 1 && kv.Key <= CameraCount) rowMax[kv.Key - 1] = kv.Value;

            CurveMergeHelper.MergeRowCurvesOverlap(rowMean, rowMax, CameraCount,
                out float[] mergedMean, out float[] mergedMax);
            if (mergedMean == null) return;

            _waterfallRowMeanPending.Clear();
            _waterfallRowMaxPending.Clear();
            AppendLiveWaterfallRowBand(mergedMean, mergedMax);
        }

        private void AppendLiveWaterfallRowBand(float[] meanBand, float[] maxBand)
        {
            int capacity = _settings?.ImageView?.WaterfallTotalHeight ?? InspectionDefaults.WaterfallTotalHeight;
            capacity = Math.Max(1000, capacity);
            if (_waterfallRowMean == null || _waterfallRowMean.Length != capacity)
            {
                _waterfallRowMean = new float[capacity];
                _waterfallRowMax = new float[capacity];
                _waterfallRowWrite = 0;
            }

            int bandLen = Math.Min(meanBand.Length, capacity);
            bool ring = (_settings?.ImageView?.WaterfallFullMode ?? InspectionDefaults.WaterfallFullMode) == WaterfallFullMode.Ring;
            if (!ring && _waterfallRowWrite + bandLen > capacity)
            {
                Array.Clear(_waterfallRowMean, 0, _waterfallRowMean.Length);
                Array.Clear(_waterfallRowMax, 0, _waterfallRowMax.Length);
                _waterfallRowWrite = 0;
            }

            for (int i = 0; i < bandLen; i++)
            {
                int dst = ring ? (_waterfallRowWrite + i) % capacity : _waterfallRowWrite + i;
                if (dst < 0 || dst >= capacity) break;
                _waterfallRowMean[dst] = meanBand[i];
                _waterfallRowMax[dst] = maxBand != null && i < maxBand.Length ? maxBand[i] : 0;
            }

            _waterfallRowWrite = ring
                ? (_waterfallRowWrite + bandLen) % capacity
                : Math.Min(capacity, _waterfallRowWrite + bandLen);

            _liveRowDisplay.UpdateData(_waterfallRowMean, _waterfallRowMax);
            if (ResumeLiveRowRangeAfterDataUpdate()) return;
        }

        /// <summary>用 A輪速度 和選中相機的取樣頻率（Line Rate）更新列圖表座標。</summary>
        private void UpdateRowChartPitch()
        {
            if (_settings == null) return;
            double lineRateHz = _settings.Acquisition.CameraLineRateHz[0]; // CAM1 master
            _liveRowDisplay?.SetRowPitchFromSpeed(
                _settings.AniloxRollSpeedMPerMin, lineRateHz);
            _reviewRowDisplay?.SetRowPitchFromSpeed(
                _settings.AniloxRollSpeedMPerMin, lineRateHz);
            // 把 row pitch 餵給主畫面顯示 → SetLayout → 列曲線圖 Y 對齊（否則 ImageDisplayView 用 X ops 比例錯）
            if (_liveCameraManager != null)
                _liveCameraManager.RowPitchMm = _liveRowDisplay?.RowPitchMm ?? 0;
        }

        private void ApplyDisplayDirectionSetting()
        {
            _liveCameraManager?.ApplyDisplayDirection();
            _reviewDisplayManager?.SetFlipVertical(ShouldFlipDisplayVertical());
            _stitchCoordinator?.RefreshCurrentCameraChartsForSettingsChange();
            if (_stitchCoordinator?.IsStitchMode != true)
                _stitchCoordinator?.UpdateRowChartFromRepository();
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

        /// <summary>Live 顯示相關 setting（動態 LOD + OPS/Start 合圖佈局）→ 即時套到 LiveCameraManager。
        /// （Wave3 選項1：從 OnSettingChanged dispatcher 搬入；dispatcher 只 fan-out。）</summary>
        private void HandleLiveLayoutSettingsChanged(string name)
        {
            if (name == nameof(InspectionSettings.hf_LiveLod))
                _liveCameraManager?.SetLodMode(_settings.LiveLod);
            if (name == nameof(InspectionSettings.he_MainDisplay))
            {
                ResetLiveWaterfallRowChart();
                _liveCameraManager?.ApplyMainDisplayMode();   // 即時 / 瀑布 即時切換
            }
            if (name == nameof(InspectionSettings.hee_VerticalDirection))
                ApplyDisplayDirectionSetting();
            if (name == nameof(InspectionSettings.hg_WaterfallTotalHeight) || name == nameof(InspectionSettings.hh_WaterfallFullMode))
            {
                ResetLiveWaterfallRowChart();
                _liveCameraManager?.RefreshWaterfallDisplay(); // 瀑布總高/滿了行為變更 → 重建套新值
            }
            if (OpsStartSettingNames.Contains(name) && _liveCameraManager?.IsGlobalMergeActive == true)
                _liveCameraManager.RefreshGlobalMergeLayout(
                    _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray());
        }

        /// <summary>強化 setting（監控 hc / 回顧 hd）→ 套用對應強化。（Wave3 選項1：從 dispatcher 搬入。）</summary>
        private async Task HandleEnhanceSettingsChanged(string name)
        {
            if (name == nameof(InspectionSettings.hc_EnableMuraEnhance))
                ApplyMuraEnhance(_settings.EnableMuraEnhance);
            if (name == nameof(InspectionSettings.hd_EnableReviewEnhance))
                await ApplyReviewEnhance(_settings.EnableReviewEnhance);
        }

        private void ApplyMuraEnhance(bool enabled)
        {
            _liveCameraManager?.SetImageProcessingEnabled(enabled);
            _liveCameraManager?.SetLiveDisplayDirection(_liveDisplayDirection);
            UpdateLiveDirectionVisual();
        }
    }
}
