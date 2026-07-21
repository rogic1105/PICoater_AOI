using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using TanukiCv.Controls;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Binders;
using AniloxRoll.Monitor.UI.Coordinators;
using AniloxRoll.Monitor.UI.Managers;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Services;
using AniloxRoll.Monitor.UI.State;
using AniloxRoll.Monitor.UI.Widgets;

namespace AniloxRoll.Monitor.UI.Presenters
{
    /// <summary>
    /// Context 物件：傳遞 UI 控制項與服務參考給 <see cref="ReviewStitchCoordinator"/>。
    /// </summary>
    public class ReviewStitchContext
    {
        public Chart ChartReviewPatch { get; set; }
        public Chart ChartReviewVertical { get; set; }
        public Chart ChartReviewHorizontal { get; set; }
        public BusyUiBinder BusyUi { get; set; }
        public ReviewRuntimeState ReviewState { get; set; }
        public ColumnCurveChartHelper ColumnChartHelper { get; set; }
        public RowCurveChartHelper RowChartHelper { get; set; }
        public RowCurveDisplayAdapter RowChartDisplay { get; set; }
        public RowCurveSyncCoordinator RowChartSync { get; set; }
        public ColumnCurveChartHelper OverviewHelper { get; set; }

        public BatchInspectionService InspectionService { get; set; }
        public ImageRepository ImageRepository { get; set; }
        public DataStatisticsPresenter DataStatsPresenter { get; set; }

        public InspectionSettings Settings { get; set; }
        public DateTimeNavigator DateTimeNavigator { get; set; }
        public int CameraCount { get; set; }
    }

    /// <summary>
    /// Review tab 的 Stitch 模式管理：LoadGrabStitchedViewAsync、合圖、overview chart 聯動。
    /// 持有 _stitchedImages、_globalMergedImage、_periodMergedImage 等生命週期。
    /// </summary>
    public class ReviewStitchCoordinator
    {
        private readonly ReviewStitchContext _ctx;

        // ── State ──
        private Bitmap[] _stitchedImages;
        private float[][] _stitchedCurveMean;
        private float[][] _stitchedCurveMax;
        private float[][] _stitchedRowCurveMean;
        private float[][] _stitchedRowCurveMax;
        private float[] _stitchedMergedRowCurveMean;
        private float[] _stitchedMergedRowCurveMax;
        private Bitmap _globalMergedImage;
        private Bitmap _periodMergedImage;

        // ── Public State ──
        public bool IsStitchMode => _stitchedImages != null;
        public bool IsGlobalMerged => _globalMergedImage != null;
        public bool IsPeriodMerged => _periodMergedImage != null;
        public CsvConfigSnapshot CurrentGrabConfig => _ctx.ReviewState.Config;

        /// <summary>上一次 Review 頁面的處理模式旗標。</summary>
        public bool LastReviewProcessedMode { get; set; }

        // 曲線與圖片是兩條獨立資料流：曲線 latest-only，圖片由 Form 的 250ms debounce 觸發。
        // 不共用 token，避免圖片開始載入時把同一序號仍在讀取的曲線誤判為 stale。
        private readonly LatestCurveLoadCoordinator _curveLoads;
        private readonly SingleGrabCurveDataLoader _curveDataLoader;
        private readonly ReviewImageDataLoader _imageDataLoader;
        private readonly ReviewPeriodDataLoader _periodDataLoader;
        private readonly ReviewImageLoadGate _imageLoads = new ReviewImageLoadGate();

        /// <summary>快路：只載曲線（欄+列，.bin 數十 KB + tick 對齊 csv）+CFG → 更新欄/列 chart。
        /// 滾動掃描用——chart 即時跟著序號跑（使用者快速找異常），影像（重：JPEG 解碼+拼接）由
        /// debounce 後的完整載入跟上（硬體限制的分層載入）。</summary>
        public Task LoadGrabCurvesOnlyAsync(string grabId, DateTime hintFrom, DateTime hintTo)
            => _curveLoads.Enqueue(grabId, hintFrom, hintTo);

        /// <summary>使正在解碼的舊圖片結果失效；每次使用者改序號時呼叫，不等待 250ms。</summary>
        public void InvalidateImageLoad()
        {
            if (!_imageLoads.Invalidate()) return;
            _ctx.BusyUi.SetBusy(false);
            Core.Services.FlowTrace.Log("RV loadGrab busy off reason=invalidated");
        }

        private async Task LoadGrabCurvesCoreAsync(SingleGrabCurveLoadRequest request)
        {
            string grabId = request.GrabId;
            DateTime hintFrom = request.HintFrom;
            DateTime hintTo = request.HintTo;
            string root = !string.IsNullOrWhiteSpace(UI.State.UserSessionState.LastDataPath)
                          ? UI.State.UserSessionState.LastDataPath : _ctx.DataStatsPresenter.StatsDataRootPath;
            if (string.IsNullOrWhiteSpace(root)) return;

            var sw = Stopwatch.StartNew();
            int camCount = _ctx.CameraCount;
            try
            {
                SingleGrabCurveData loaded = await Task.Run(() =>
                {
                    SingleGrabCurveData data = _curveDataLoader.Load(
                        root, grabId, hintFrom, hintTo, camCount);
                    Core.Services.FlowTrace.Log($"RV curves paths {grabId} root={root} images={data.ImageCount} cams={data.MatchedCameraCount} cfg={(data.Config != null ? "yes" : "no")} align={data.AlignmentMode} source={data.StorageSource}");
                    return data;
                });
                if (!_curveLoads.IsCurrent(request))
                {
                    Core.Services.FlowTrace.Log($"RV curves stale-drop {grabId}");
                    return;
                }
                _stitchedCurveMean = loaded.ColumnMean;
                _stitchedCurveMax = loaded.ColumnMax;
                _stitchedRowCurveMean = loaded.RowMean;
                _stitchedRowCurveMax = loaded.RowMax;
                _stitchedMergedRowCurveMean = loaded.MergedRowMean;
                _stitchedMergedRowCurveMax = loaded.MergedRowMax;
                _ctx.ReviewState.Config = loaded.Config;
                UpdateStitchedOverviewChart();
                UpdateGlobalRowChart();
                Core.Services.FlowTrace.Log($"RV curves {grabId}（{sw.ElapsedMilliseconds}ms）");
            }
            catch (Exception ex) { Trace.WriteLine($"[CurvesOnly] {grabId}: {ex.GetType().Name}: {ex.Message}"); }
        }

        /// <summary>目前的 ridge 方向（"v" 或 "h"）。</summary>
        public string ActiveRidgeDirection { get; set; } = "v";

        /// <summary>
        /// UpdateStitchedOverviewChart 完成後觸發，傳遞與 chartReviewColumn 相同的曲線資料，
        /// 供外部（AniloxRollForm）同步 chartDataColumn。
        /// 參數：(mean[][], max[][], opsUm[], startPosMm[], errMean, errMax)
        /// </summary>
        public event Action<float[][], float[][], double[], double[], float, float> StitchedCurveUpdated;

        public ReviewStitchCoordinator(ReviewStitchContext ctx)
        {
            _ctx = ctx;
            _curveDataLoader = new SingleGrabCurveDataLoader();
            _imageDataLoader = new ReviewImageDataLoader();
            _periodDataLoader = new ReviewPeriodDataLoader();
            _curveLoads = new LatestCurveLoadCoordinator(LoadGrabCurvesCoreAsync);
        }

        /// <summary>延遲注入 DataStatsPresenter（初始化順序：coordinator 先於 presenter 建立）。</summary>
        public void SetDataStatsPresenter(DataStatisticsPresenter presenter)
        {
            _ctx.DataStatsPresenter = presenter;
        }

        /// <summary>
        /// 載入 GrabId 的拼接影像（使用上次的 processed 模式）。
        /// </summary>
        /// <summary>#13 同源新路徑：一組 grab 影像載好（7 台拼接圖 + CFG 有效 ops/pos + 是否 Global）。
        /// Form 訂閱 → ReviewDisplayManager.PushImages（ImageDisplayView 顯示）；舊 canvas 路徑照跑（平行建新）。</summary>
        public event Action<byte[][], int[], int[], double[], double[], bool> StitchedImagesReady; // gray bytes(不可變快照), w, h, ops, pos, isGlobal —— Bitmap 不出此類（GDI+ 單執行緒物件）

        public Task LoadGrabStitchedViewAsync(string grabId, DateTime hintFrom, DateTime hintTo)
            => LoadGrabStitchedViewAsync(grabId, hintFrom, hintTo, LastReviewProcessedMode);

        /// <summary>
        /// 載入 GrabId 的拼接影像。背景執行拼接後更新 UI。
        /// </summary>
        public async Task LoadGrabStitchedViewAsync(string grabId, DateTime hintFrom, DateTime hintTo,
            bool enableProcess)
        {
            string root = !string.IsNullOrWhiteSpace(UI.State.UserSessionState.LastDataPath)
                          ? UI.State.UserSessionState.LastDataPath : _ctx.DataStatsPresenter.StatsDataRootPath;
            if (string.IsNullOrWhiteSpace(root)) return;

            // 圖片自己的最後贏 token；序號 intent 會先 InvalidateImageLoad，防舊圖片在 debounce 前上畫面。
            int myLoad = _imageLoads.Begin();
            _ctx.BusyUi.SetBusy(true);
            Core.Services.FlowTrace.Log($"RV loadGrab begin {grabId}（proc={enableProcess}）");
            LastReviewProcessedMode = enableProcess;
            // L2 SSoT：setting 由 caller 透過 SettingsHub 設置，coordinator 不再 bypass Hub 直接寫 memory。
            // caller 路徑：PropertyGrid → OnSettingChanged → ApplyReviewEnhance → ReloadCurrentStitchedView；
            //              chart click → Hub.Set(hd_EnableReviewEnhance) → 同上。
            var swTotal = Stopwatch.StartNew();
            try
            {
                string ridgeDir = ActiveRidgeDirection;
                int camCount = _ctx.CameraCount;
                ReviewImageData loaded = await Task.Run(() => _imageDataLoader.Load(
                    root, grabId, hintFrom, hintTo, camCount, enableProcess, ridgeDir));
                var newImages = loaded.Images;

                // token 閘門：背景載入期間已有更新的選取 → 本結果作廢（不上畫面、不動 chart）
                if (!_imageLoads.IsCurrent(myLoad))
                {
                    Core.Services.FlowTrace.Log($"RV loadGrab stale-drop {grabId}（{swTotal.ElapsedMilliseconds}ms）");
                    loaded.DisposeImages();
                    return;
                }

                ClearStitchedMode();
                _stitchedImages       = newImages;
                _stitchedCurveMean    = loaded.ColumnMean;
                _stitchedCurveMax     = loaded.ColumnMax;
                _stitchedRowCurveMean = loaded.RowMean;
                _stitchedRowCurveMax  = loaded.RowMax;
                _stitchedMergedRowCurveMean = null;
                _stitchedMergedRowCurveMax = null;
                _ctx.ReviewState.Config = loaded.Config;
                _ctx.DataStatsPresenter?.SetReviewGroupBoxes(true);

                var opsEff = loaded.Config?.CamOps ?? _ctx.Settings.GetCameraOpsUmArray();
                var posEff = loaded.Config?.CamPos ?? _ctx.Settings.GetCameraStartPositionMmArray();
                StitchedImagesReady?.Invoke(
                    loaded.GrayFrames, loaded.GrayWidths, loaded.GrayHeights, opsEff, posEff,
                    _ctx.Settings.StitchMode == StitchMode.Global);

                UpdateGlobalRowChart();   // 畫布顯示走 ImageDisplayView（同源）；row 曲線照合併更新
                UpdateStitchedOverviewChart();

                Trace.WriteLine($"[StitchView] {grabId} proc={enableProcess} | CSV={loaded.ConfigMs}ms | Stitch={loaded.StitchMs}ms | Merge(bg)=0ms | UIapply={swTotal.ElapsedMilliseconds - loaded.ConfigMs - loaded.StitchMs}ms | Total={swTotal.ElapsedMilliseconds}ms");
                Core.Services.FlowTrace.Log($"RV loadGrab done {grabId}（{swTotal.ElapsedMilliseconds}ms）");

                // Resource log
                int loadedCams = 0, finalW = 0, finalH = 0;
                for (int i = 0; i < newImages.Length; i++)
                {
                    if (newImages[i] != null)
                    {
                        loadedCams++;
                        if (finalW == 0) { finalW = newImages[i].Width; finalH = newImages[i].Height; }
                    }
                }
                string mode;
                if (_globalMergedImage != null)
                {
                    mode = "Global";
                    finalW = _globalMergedImage.Width;
                    finalH = _globalMergedImage.Height;
                }
                else
                {
                    mode = (loaded.TotalImageCount > loadedCams) ? "Stitch" : "Single";
                }
                Core.Camera.CameraFrameSaver.AppendReviewResourceLog(mode, loadedCams, loaded.TotalImageCount,
                    finalW, finalH, swTotal.ElapsedMilliseconds);
            }
            finally
            {
                // 只釋放自己持有的 busy lease；新的載入開始後，舊 finally 不得關掉新游標。
                if (_imageLoads.Complete(myLoad))
                    _ctx.BusyUi.SetBusy(false);
            }
        }

        public void EnableMergedOverviewSync(double[] opsArr, double[] posArr)
        {
            double globalMinMm = double.MaxValue;
            double refOpsUm = opsArr[0];
            for (int i = 0; i < opsArr.Length && i < _ctx.CameraCount; i++)
                if (posArr[i] < globalMinMm) globalMinMm = posArr[i];
            if (globalMinMm == double.MaxValue) globalMinMm = 0;

            // 2b-ii：合圖座標覆寫原餵 CanvasInteractionHelper 顯示路徑（已砍）；overview X 視野連動
            //   現由 ImageDisplayView.ViewRangeMmChanged → _reviewOverviewHelper.UpdateViewRange 承接。
            if (_ctx.ChartReviewPatch.ChartAreas.Count > 0)
                _ctx.ChartReviewPatch.ChartAreas[0].AxisX.ScaleView.Zoomable = true;
        }

        /// <summary>Form 關閉時控制項已 disposed → 這幾條 cleanup 的 UI 操作應 no-op：
        /// Form 自身會清控制項與資源，且關程式時 fire-and-forget 的 StitchMode 切換 async
        /// 可能續跑碰到已 disposed 的 chartReviewColumn/canvas → NullReferenceException。</summary>
        private bool UiDisposed =>
            (_ctx?.ChartReviewPatch?.IsDisposed ?? true);

        /// <summary>離開合圖模式：清除座標覆寫、停用互動 zoom。
        /// 不重設 ScaleView（ZoomReset）：避免 await 期間 message pump 渲染出全範圍閃爍，
        /// 由後續 UpdateDataAndView 原子性地取代資料與 zoom。</summary>
        public void DisableMergedOverviewSync()
        {
            if (UiDisposed) return;
            // 2b-ii：原 ClearMergedMode 清的是 CanvasInteractionHelper 座標覆寫（已砍）。
            if (_ctx.ChartReviewPatch.ChartAreas.Count > 0)
            {
                _ctx.ChartReviewPatch.ChartAreas[0].AxisX.ScaleView.Zoomable = false;
            }
        }

        /// <summary>切到 Vertical 時清掉殘留的 Global/Period 合圖 bitmap（保留 _stitchedImages 與曲線）。</summary>
        public void DisposeGlobalMergedImage()
        {
            if (UiDisposed) return;
            DisableMergedOverviewSync();
            // 2b-ii-B：合圖 bitmap 不再貼到 canvas（顯示走 ImageDisplayView）→ 只還池。
            if (_globalMergedImage != null)
            {
                BitmapPool.Return(_globalMergedImage);
                _globalMergedImage = null;
            }
            if (_periodMergedImage != null)
            {
                BitmapPool.Return(_periodMergedImage);
                _periodMergedImage = null;
            }
        }

        public void ClearStitchedMode()
        {
            if (UiDisposed) return;
            DisposeGlobalMergedImage();
            if (_stitchedImages == null) return;
            // 2b-ii-B：canvas/縮圖 PictureBox 已退場（ImageDisplayView 接管）→ 不再清它們的 Image。
            foreach (var bmp in _stitchedImages) BitmapPool.Return(bmp);
            _stitchedImages = null;
            _stitchedCurveMean    = null;
            _stitchedCurveMax     = null;
            _stitchedRowCurveMean = null;
            _stitchedRowCurveMax  = null;
            _stitchedMergedRowCurveMean = null;
            _stitchedMergedRowCurveMax = null;
            _ctx.ReviewState.Config = null;
            _ctx.ColumnChartHelper?.SetOps(_ctx.Settings.Cam1_Ops);
            _ctx.ColumnChartHelper?.SetThresholds(_ctx.Settings.ErrorValueMeanV, _ctx.Settings.ErrorValueMaxV);
            _ctx.DataStatsPresenter?.SetReviewGroupBoxes(false);
        }

        /// <summary>Global 模式：7 台 row curves 重疊合併後更新列曲線圖。
        /// row chart 是列 (row) 曲線 → 用 (HM_V_capture / HM_H_current) ratio rescale，
        /// 讓 PropertyGrid 改列正規值時 H 曲線坡度立即變化。
        /// 公式：bin baked-in 的縮放是 HM_V_capture（native 只用單一 HM=V），
        /// view-time 想要的目標是 HM_H_current，所以 ratio = HM_V_capture / HM_H_current。</summary>
        private void UpdateGlobalRowChart()
        {
            if (_ctx.RowChartSync == null ||
                (_stitchedRowCurveMean == null && _stitchedMergedRowCurveMean == null)) return;
            var swRow = System.Diagnostics.Stopwatch.StartNew();   // [UiSlow] 卡頓歸因
            try { UpdateGlobalRowChartBody(); }
            finally
            {
                if (swRow.ElapsedMilliseconds > 50)
                    Core.Services.FlowTrace.Log($"[UiSlow] RvRowChart {swRow.ElapsedMilliseconds}ms");
            }
        }

        private void UpdateGlobalRowChartBody()
        {
            float[] mergedMean;
            float[] mergedMax;
            if (_stitchedMergedRowCurveMean != null)
            {
                mergedMean = (float[])_stitchedMergedRowCurveMean.Clone();
                mergedMax = _stitchedMergedRowCurveMax == null
                    ? null
                    : (float[])_stitchedMergedRowCurveMax.Clone();
            }
            else
            {
                CurveMergeHelper.MergeRowCurvesOverlap(
                    _stitchedRowCurveMean, _stitchedRowCurveMax,
                    _ctx.CameraCount, out mergedMean, out mergedMax);
            }
            if (mergedMean != null)
            {
                float captureHmV = _ctx.ReviewState.Config?.HessianMaxFactorV ?? _ctx.Settings.HessianMaxFactorV;
                HessianRescaleHelper.RescaleInPlace1D(mergedMean, captureHmV, _ctx.Settings.HessianMaxFactorH);
                HessianRescaleHelper.RescaleInPlace1D(mergedMax,  captureHmV, _ctx.Settings.HessianMaxFactorH);
                _ctx.RowChartSync.UpdateData(mergedMean, mergedMax, requireViewRange: true);
                // 舊 else RefreshRowChartRange（讀已砍 canvas，恆 no-op）移除；視野由 ImageDisplayView 連動
            }
        }

        public void Dispose()
        {
            _curveDataLoader.Dispose();
        }

        /// <summary>
        /// 合圖路徑：用 _stitchedCurveMean/Max 更新 chart1 全覽圖。
        /// 套用 view-time 正規值 rescale + 當前閾值：
        ///   - 曲線：(bin/255) × (HM_capture / HM_current) ← 改 PropertyGrid 正規值會立即反映坡度
        ///   - 閾值線：用 _ctx.Settings 的當前 ErrorValueMeanV/MaxV ← 改 PropertyGrid 閾值會立即移動門檻線
        ///   - OPS/Pos：用該 grab 的 #CFG 快照（與資料一起 baked-in，不可後驗調整）
        /// </summary>
        public void UpdateStitchedOverviewChart()
        {
            if (_stitchedCurveMean == null) return;
            var swOv = System.Diagnostics.Stopwatch.StartNew();   // [UiSlow] 卡頓歸因
            try { UpdateStitchedOverviewChartBody(); }
            finally
            {
                if (swOv.ElapsedMilliseconds > 50)
                    Core.Services.FlowTrace.Log($"[UiSlow] RvOverviewChart {swOv.ElapsedMilliseconds}ms");
            }
        }

        private void UpdateStitchedOverviewChartBody()
        {
            double[] opsArr, posArr;
            float captureHm;
            if (_ctx.ReviewState.Config != null)
            {
                opsArr    = _ctx.ReviewState.Config.CamOps;
                posArr    = _ctx.ReviewState.Config.CamPos;
                captureHm = _ctx.ReviewState.Config.HessianMaxFactorV;
            }
            else
            {
                opsArr    = _ctx.Settings.GetCameraOpsUmArray();
                posArr    = _ctx.Settings.GetCameraStartPositionMmArray();
                captureHm = _ctx.Settings.HessianMaxFactorV;
            }
            // 閾值固定用當前 Settings（view-time 可調），不再從 capture config 取
            float errMean = _ctx.Settings.ErrorValueMeanV;
            float errMax  = _ctx.Settings.ErrorValueMaxV;

            // chartReviewColumn 是欄 (column) 曲線 → 用 V 的 capture/current ratio
            var displayMean = HessianRescaleHelper.CloneAndRescale2D(_stitchedCurveMean, captureHm, _ctx.Settings.HessianMaxFactorV);
            var displayMax  = HessianRescaleHelper.CloneAndRescale2D(_stitchedCurveMax,  captureHm, _ctx.Settings.HessianMaxFactorV);

            CurveMergeHelper.UpdateOverviewChart(displayMean, displayMax,
                opsArr, posArr, errMean, errMax,
                _ctx.OverviewHelper, _ctx.CameraCount, _ctx.Settings.StitchMode,
                ViewRangeProvider);

            StitchedCurveUpdated?.Invoke(displayMean, displayMax, opsArr, posArr, errMean, errMax);
        }


        /// <summary>
        /// 更新單台相機的 chartReviewColumn（V）+ chartReviewRow（H）。
        /// 套用 view-time 正規值 rescale：
        ///   - V 曲線：(bin/255) × (HM_V_capture / HM_V_current) → 改 PropertyGrid 欄正規值生效
        ///   - H 曲線：(bin/255) × (HM_V_capture / HM_H_current) → 改 PropertyGrid 列正規值生效
        /// 閾值線用當前 Settings（view-time tunable）。
        /// </summary>
        public void UpdatePerCameraCharts(int idx)
        {
            if (_stitchedImages == null) return;

            // 欄 (Column / V)
            if (_ctx.Settings.StitchMode == StitchMode.Global)
            {
                // Global 模式：單台欄資料無意義，清空
                if (_ctx.ChartReviewVertical != null)
                {
                    _ctx.ChartReviewVertical.Series["Mean"].Points.Clear();
                    _ctx.ChartReviewVertical.Series["Max"].Points.Clear();
                }
            }
            else if (_ctx.ColumnChartHelper != null && _ctx.Settings != null)
            {
                float[] mean = (_stitchedCurveMean != null && idx >= 0 && idx < _stitchedCurveMean.Length)
                    ? _stitchedCurveMean[idx] : null;
                float[] max = (_stitchedCurveMax != null && idx >= 0 && idx < _stitchedCurveMax.Length)
                    ? _stitchedCurveMax[idx] : null;

                double[] posArr;
                float captureHmV;
                if (_ctx.ReviewState.Config != null)
                {
                    double opsUm = (idx >= 0 && idx < _ctx.ReviewState.Config.CamOps.Length)
                        ? _ctx.ReviewState.Config.CamOps[idx] : _ctx.Settings.Cam1_Ops;
                    _ctx.ColumnChartHelper.SetOps(opsUm);
                    posArr = _ctx.ReviewState.Config.CamPos;
                    captureHmV = _ctx.ReviewState.Config.HessianMaxFactorV;
                }
                else
                {
                    posArr = _ctx.Settings.GetCameraStartPositionMmArray();
                    captureHmV = _ctx.Settings.HessianMaxFactorV;
                }
                // 閾值固定用當前 Settings（view-time tunable）
                _ctx.ColumnChartHelper.SetThresholds(
                    _ctx.Settings.ErrorValueMeanV, _ctx.Settings.ErrorValueMaxV);

                var displayMean = HessianRescaleHelper.CloneAndRescale1D(mean, captureHmV, _ctx.Settings.HessianMaxFactorV);
                var displayMax  = HessianRescaleHelper.CloneAndRescale1D(max,  captureHmV, _ctx.Settings.HessianMaxFactorV);

                double startPos = (idx >= 0 && idx < posArr.Length) ? posArr[idx] : 0;
                // 2b-ii：當前 X 視野改取 ImageDisplayView 快取（原 TryComputeCurrentViewRange 讀已砍 canvas，恆回 0,0）
                var nv = SameSourceViewRange?.Invoke();
                double leftMm = nv?[0] ?? 0, rightMm = nv?[1] ?? 0;
                _ctx.ColumnChartHelper.UpdateDataAndView(displayMean, displayMax, startPos, leftMm, rightMm);
            }

            // 列 (Row / H)
            if (_ctx.RowChartSync != null)
            {
                if (_ctx.Settings.StitchMode == StitchMode.Global)
                {
                    UpdateGlobalRowChart();
                }
                else
                {
                    float[] rowMean = (_stitchedRowCurveMean != null && idx >= 0 && idx < _stitchedRowCurveMean.Length)
                        ? _stitchedRowCurveMean[idx] : null;
                    float[] rowMax = (_stitchedRowCurveMax != null && idx >= 0 && idx < _stitchedRowCurveMax.Length)
                        ? _stitchedRowCurveMax[idx] : null;
                    if (rowMean != null)
                    {
                        float captureHmV = _ctx.ReviewState.Config?.HessianMaxFactorV ?? _ctx.Settings.HessianMaxFactorV;
                        var displayMean = HessianRescaleHelper.CloneAndRescale1D(rowMean, captureHmV, _ctx.Settings.HessianMaxFactorH);
                        var displayMax  = HessianRescaleHelper.CloneAndRescale1D(rowMax,  captureHmV, _ctx.Settings.HessianMaxFactorH);
                        _ctx.RowChartSync.UpdateData(displayMean, displayMax, requireViewRange: true);
                    }
                }
            }
        }

        /// <summary>
        /// 由 PropertyGrid 變更觸發：重畫當前選中相機的 V/H per-camera charts。
        /// 不重設 canvas view（避免使用者 zoom/pan 被打斷）。
        /// </summary>
        public void RefreshCurrentCameraChartsForSettingsChange()
        {
            if (_stitchedImages == null) return;
            int idx = SelectedCamIndexProvider?.Invoke() ?? 0;
            if (idx < 0) idx = 0;
            UpdatePerCameraCharts(idx);
        }

        /// <summary>
        /// 原圖路徑（非 Stitch）：合併全域圖（Period 切換用）。
        /// </summary>
        public void ApplyGlobalMergeIfNeeded()
            => ApplyGlobalMergeCore(null);

        public void ApplyGlobalMergeForPeriod(DateTime period)
            => ApplyGlobalMergeCore(period);

        private void ApplyGlobalMergeCore(DateTime? period)
        {
            if (_ctx.Settings.StitchMode != StitchMode.Global) return;

            var cfg = _ctx.ReviewState.Config;
            double[] opsArr = cfg?.CamOps ?? _ctx.Settings.GetCameraOpsUmArray();
            double[] posArr = cfg?.CamPos ?? _ctx.Settings.GetCameraStartPositionMmArray();
            int scale = InspectionEngineConfig.DefaultSaveResizeScale;

            var filesMap = GetPeriodImages(period);
            if (filesMap == null || filesMap.Count == 0) return;
            Core.Services.FlowTrace.Log($"RV period load {PeriodLabel(period)} images={filesMap.Count}/{_ctx.CameraCount} proc={LastReviewProcessedMode} cfg={(cfg != null ? "yes" : "no")}");

            Func<string, Bitmap> bmpLoader = _ctx.InspectionService != null
                ? (Func<string, Bitmap>)(p => _ctx.InspectionService.LoadBmpAtScale(p, scale))
                : null;

            ReviewPeriodFrames frames = _periodDataLoader.LoadFrames(
                filesMap, _ctx.CameraCount, scale, bmpLoader,
                LastReviewProcessedMode, ActiveRidgeDirection);
            StitchedImagesReady?.Invoke(
                frames.GrayFrames, frames.Widths, frames.Heights, opsArr, posArr, true);
        }

        /// <summary>
        /// 原圖路徑：從當前 Repository 時間點讀取 .bin 曲線更新 chartReviewColumn 全覽圖。
        /// </summary>
        public void UpdateOverviewChartFromRepository()
            => UpdateOverviewChartCore(null);

        public void UpdateOverviewChartForPeriod(DateTime period)
            => UpdateOverviewChartCore(period);

        private void UpdateOverviewChartCore(DateTime? period)
        {
            if (_ctx.OverviewHelper == null || _stitchedImages != null) return;

            var images = GetPeriodImages(period);

            if (images == null || images.Count == 0)
            {
                _ctx.ChartReviewPatch.Series["Mean"].Points.Clear();
                _ctx.ChartReviewPatch.Series["Max"].Points.Clear();
                if (_ctx.ChartReviewPatch.ChartAreas.Count > 0)
                    _ctx.ChartReviewPatch.ChartAreas[0].AxisX.ScaleView.ZoomReset();
                return;
            }

            int camCount = _ctx.CameraCount;
            ReviewPeriodColumnCurves curves = _periodDataLoader.LoadColumnCurves(images, camCount);

            var reviewCfg = _ctx.ReviewState.Config;
            if (reviewCfg != null)
            {
                CurveMergeHelper.UpdateOverviewChart(curves.Mean, curves.Max,
                    reviewCfg.CamOps, reviewCfg.CamPos, reviewCfg.ErrorValueMeanV, reviewCfg.ErrorValueMaxV,
                    _ctx.OverviewHelper, camCount, _ctx.Settings.StitchMode, ViewRangeProvider);
            }
            else
            {
                CurveMergeHelper.UpdateOverviewChart(curves.Mean, curves.Max,
                    _ctx.Settings.GetCameraOpsUmArray(), _ctx.Settings.GetCameraStartPositionMmArray(),
                    _ctx.Settings.ErrorValueMeanV, _ctx.Settings.ErrorValueMaxV,
                    _ctx.OverviewHelper, camCount, _ctx.Settings.StitchMode, ViewRangeProvider);
            }
        }

        /// <summary>
        /// 時序（period）路徑：從當前 Repository 時間點讀 MeanR/MaxR bin 曲線 → 合併更新列曲線圖
        /// （chartReviewRow）。單片路徑走 <see cref="UpdateGlobalRowChart"/>（吃 _stitchedRowCurveMean）；
        /// period 不進 stitch 模式（_stitchedImages=null），故獨立從 repository 載 → 與欄 overview 對稱。
        /// </summary>
        public void UpdateRowChartFromRepository()
            => UpdateRowChartCore(null);

        public void UpdateRowChartForPeriod(DateTime period)
            => UpdateRowChartCore(period);

        private void UpdateRowChartCore(DateTime? period)
        {
            if (_ctx.RowChartSync == null || _stitchedImages != null) return;

            var images = GetPeriodImages(period);
            if (images == null || images.Count == 0) return;

            ReviewPeriodRowCurves curves = _periodDataLoader.LoadMergedRowCurves(
                images, _ctx.CameraCount);
            if (curves.Mean == null) return;

            // 與 UpdateGlobalRowChart 同公式：bin baked-in 縮放=HM_V_capture，view-time 目標=HM_H_current。
            float captureHmV = _ctx.ReviewState.Config?.HessianMaxFactorV ?? _ctx.Settings.HessianMaxFactorV;
            HessianRescaleHelper.RescaleInPlace1D(curves.Mean, captureHmV, _ctx.Settings.HessianMaxFactorH);
            HessianRescaleHelper.RescaleInPlace1D(curves.Max, captureHmV, _ctx.Settings.HessianMaxFactorH);
            _ctx.RowChartSync.UpdateData(curves.Mean, curves.Max, requireViewRange: true);
        }

        /// <summary>#13 同源新路徑的「當前視野」注入（form 快取 ImageDisplayView 視野；[l,r,top,bot]，null=無效）。
        /// chart 更新原子帶入此值 → 重載/強化切換不會先閃回預設再跟隨（同 Live 的 _liveViewLeftMm 解法）。</summary>
        public Func<double[]> SameSourceViewRange { get; set; }

        /// <summary>當前選中相機 index（0-based）來源＝ImageDisplayView（2b-ii-B 後取代舊 GalleryManager.SelectedIndex）。</summary>
        public Func<int> SelectedCamIndexProvider { get; set; }

        private Dictionary<int, string> GetPeriodImages(DateTime? period)
        {
            if (period.HasValue) return _ctx.ImageRepository.GetImages(period.Value);
            return _ctx.ImageRepository.GetImages(
                _ctx.DateTimeNavigator.GetCurrentYear(), _ctx.DateTimeNavigator.GetCurrentMonth(),
                _ctx.DateTimeNavigator.GetCurrentDay(), _ctx.DateTimeNavigator.GetCurrentHour(),
                _ctx.DateTimeNavigator.GetCurrentMin(), _ctx.DateTimeNavigator.GetCurrentSec());
        }

        private string PeriodLabel(DateTime? period)
            => period.HasValue
                ? period.Value.ToString("yyyy-MM-dd HH:mm:ss.fff")
                : $"{_ctx.DateTimeNavigator.GetCurrentYear()}-{_ctx.DateTimeNavigator.GetCurrentMonth()}-{_ctx.DateTimeNavigator.GetCurrentDay()} " +
                  $"{_ctx.DateTimeNavigator.GetCurrentHour()}:{_ctx.DateTimeNavigator.GetCurrentMin()}:{_ctx.DateTimeNavigator.GetCurrentSec()}";

        private double ViewRangeProvider(int cameraIndex, bool isLeft, double defaultValue)
        {
            // 2b-ii：視野唯一來源＝ImageDisplayView 快取（SameSourceViewRange）。
            //   舊 fallback TryComputeCurrentViewRange 讀已砍 canvas、恆失敗，移除。
            var nv = SameSourceViewRange?.Invoke();
            if (nv != null) return isLeft ? nv[0] : nv[1];
            return defaultValue;
        }
    }
}
