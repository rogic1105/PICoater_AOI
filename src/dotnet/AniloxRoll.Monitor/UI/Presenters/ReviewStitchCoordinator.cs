using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading.Tasks;
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
    public enum ReviewContentLoadMode
    {
        Full,
        ReuseSharedCurves,
        ImageVariantOnly
    }

    /// <summary>
    /// Context 物件：傳遞 UI 控制項與服務參考給 <see cref="ReviewStitchCoordinator"/>。
    /// </summary>
    public class ReviewStitchContext
    {
        public Chart ChartReviewPatch { get; set; }
        public BusyUiBinder BusyUi { get; set; }
        public ReviewRuntimeState ReviewState { get; set; }
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
    /// Review tab 的載入協調者：負責 latest-only、debounce 後圖片載入、共用曲線與事件發布。
    /// 顯示內容生命週期由 ReviewDisplayContent 管理；欄／列圖表套用由 ReviewChartPresenter 管理。
    /// </summary>
    public class ReviewStitchCoordinator
    {
        private readonly ReviewStitchContext _ctx;

        private readonly ReviewDisplayContent _content = new ReviewDisplayContent();
        private readonly ReviewChartPresenter _charts;

        // ── Public State ──
        public bool IsStitchMode => _content.HasImages;
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
        private readonly object _preparedPlanGate = new object();
        private ReviewImageLoadPlan _preparedPlan;
        private string _preparedPlanKey;
        private readonly object _sharedCurveGate = new object();
        private string _sharedCurveRoot;
        private string _sharedCurveGrabId;
        private SingleGrabCurveData _sharedCurveData;

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
            bool enableProcess = LastReviewProcessedMode;
            string ridgeDir = ActiveRidgeDirection;
            try
            {
                ReviewImageLoadPlan layoutPlan = null;
                SingleGrabCurveData loaded = await Task.Run(() =>
                {
                    // Geometry is intentionally prepared before curve presentation. The image
                    // decode remains debounced, but charts must never render the new record with
                    // the previous record's viewport.
                    layoutPlan = _imageDataLoader.Prepare(
                        root, grabId, hintFrom, hintTo, camCount,
                        enableProcess, ridgeDir, logPaths: false);
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
                CachePreparedPlan(root, grabId, enableProcess, ridgeDir, layoutPlan);
                PublishPreparedLayout(grabId, layoutPlan);
                Core.Services.FlowTrace.Dvt(
                    $"RV layout intent {grabId} images={layoutPlan.TotalImageCount} " +
                    $"cams={layoutPlan.GroupedPaths.Count} align={layoutPlan.Alignment.Mode} " +
                    "before=curves");
                _content.SetCurves(
                    loaded.ColumnMean, loaded.ColumnMax,
                    loaded.RowMean, loaded.RowMax,
                    loaded.MergedRowMean, loaded.MergedRowMax);
                _ctx.ReviewState.Config = loaded.Config;
                _charts.ApplyRowPhysicalScale(loaded.Config);
                UpdateStitchedOverviewChart();
                _charts.UpdateGlobalRowChart();
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
            _charts = new ReviewChartPresenter(
                new ReviewChartContext
                {
                    OverviewChart = ctx.ChartReviewPatch,
                    ReviewState = ctx.ReviewState,
                    RowChartSync = ctx.RowChartSync,
                    OverviewHelper = ctx.OverviewHelper,
                    Settings = ctx.Settings,
                    ImageRepository = ctx.ImageRepository,
                    DateTimeNavigator = ctx.DateTimeNavigator,
                    CameraCount = ctx.CameraCount
                },
                _content,
                _periodDataLoader);
            _charts.CurvesUpdated += (mean, max, ops, positions, errorMean, errorMax) =>
                StitchedCurveUpdated?.Invoke(
                    mean, max, ops, positions, errorMean, errorMax);
            _curveLoads = new LatestCurveLoadCoordinator(LoadGrabCurvesCoreAsync);
        }

        /// <summary>延遲注入 DataStatsPresenter（初始化順序：coordinator 先於 presenter 建立）。</summary>
        public void SetDataStatsPresenter(DataStatisticsPresenter presenter)
        {
            _ctx.DataStatsPresenter = presenter;
        }

        /// <summary>
        /// 保存報表已完成的原始欄／列曲線。切到回顧時可直接套用，不再讀取同一批 bin。
        /// </summary>
        internal void CacheDataCurveSnapshot(
            string root, string grabId, SingleGrabCurveData data)
        {
            if (string.IsNullOrWhiteSpace(root) ||
                string.IsNullOrWhiteSpace(grabId) || data == null) return;

            lock (_sharedCurveGate)
            {
                _sharedCurveRoot = root;
                _sharedCurveGrabId = grabId;
                _sharedCurveData = data;
            }
            Core.Services.FlowTrace.Log($"DT curve share {grabId} target=Review");
        }

        /// <summary>
        /// 載入 GrabId 的拼接影像（使用上次的 processed 模式）。
        /// </summary>
        /// <summary>一組 grab 影像載好（7 台拼接圖 + CFG 有效 ops/pos + 是否 Global）。
        /// Form 訂閱後交給 ReviewDisplayManager.PushImages，以 ImageDisplayView 顯示。</summary>
        public event Action<byte[][], int[], int[], double[], double[], bool, bool> StitchedImagesReady; // gray bytes, w, h, ops, pos, isGlobal, preserveChartView

        /// <summary>JPEG 表頭與 CFG 就緒後、完整解碼前發布預期合圖尺寸，供主畫面同源 fit 預算。</summary>
        public event Action<string, int[], int[], double[], double[], bool> StitchedLayoutReady;

        public Task LoadGrabStitchedViewAsync(string grabId, DateTime hintFrom, DateTime hintTo)
            => LoadGrabStitchedViewAsync(grabId, hintFrom, hintTo, LastReviewProcessedMode);

        /// <summary>
        /// 載入 GrabId 的拼接影像。背景執行拼接後更新 UI。
        /// </summary>
        public async Task LoadGrabStitchedViewAsync(string grabId, DateTime hintFrom, DateTime hintTo,
            bool enableProcess, ReviewContentLoadMode loadMode = ReviewContentLoadMode.Full)
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
                ReviewImageLoadPlan plan;
                if (TryGetPreparedPlan(root, grabId, enableProcess, ridgeDir, out plan))
                {
                    LogImagePlan(grabId, root, plan);
                    Core.Services.FlowTrace.Log($"RV loadGrab plan reuse {grabId}");
                }
                else
                {
                    plan = await Task.Run(() => _imageDataLoader.Prepare(
                        root, grabId, hintFrom, hintTo, camCount, enableProcess, ridgeDir));
                }
                if (!_imageLoads.IsCurrent(myLoad))
                {
                    Core.Services.FlowTrace.Log($"RV loadGrab stale-drop {grabId}（prefit {swTotal.ElapsedMilliseconds}ms）");
                    return;
                }

                bool keepDisplayedCurves =
                    loadMode == ReviewContentLoadMode.ImageVariantOnly && _content.HasImages;
                if (!keepDisplayedCurves)
                    PublishPreparedLayout(grabId, plan);
                var opsEff = plan.Config?.CamOps ?? _ctx.Settings.GetCameraOpsUmArray();
                var posEff = plan.Config?.CamPos ?? _ctx.Settings.GetCameraStartPositionMmArray();
                bool isGlobal = _ctx.Settings.StitchMode == StitchMode.Global;
                bool reuseSharedCurves = loadMode == ReviewContentLoadMode.ReuseSharedCurves &&
                    TryActivateSharedCurves(root, grabId);
                bool preserveCurves = keepDisplayedCurves || reuseSharedCurves;
                string curveSource = keepDisplayedCurves
                    ? "keep source=display"
                    : reuseSharedCurves ? "reuse source=Data" : "load source=bin";
                Core.Services.FlowTrace.Log(
                    $"RV loadGrab curves={curveSource} {grabId}");

                ReviewImageData loaded = await Task.Run(() => _imageDataLoader.Load(
                    plan, camCount, enableProcess, ridgeDir, includeCurves: !preserveCurves));
                var newImages = loaded.Images;

                // token 閘門：背景載入期間已有更新的選取 → 本結果作廢（不上畫面、不動 chart）
                if (!_imageLoads.IsCurrent(myLoad))
                {
                    Core.Services.FlowTrace.Log($"RV loadGrab stale-drop {grabId}（{swTotal.ElapsedMilliseconds}ms）");
                    loaded.DisposeImages();
                    return;
                }

                if (preserveCurves)
                    _content.ReplaceImages(newImages);
                else
                {
                    ClearStitchedMode();
                    _content.ReplaceImages(newImages);
                }
                if (!preserveCurves)
                {
                    _content.SetCurves(
                        loaded.ColumnMean, loaded.ColumnMax,
                        loaded.RowMean, loaded.RowMax,
                        null, null);
                }
                if (!keepDisplayedCurves)
                {
                    _ctx.ReviewState.Config = loaded.Config;
                    // The image layout and row chart must receive the same capture-time mm/row
                    // before either is presented. ImageDisplayView then publishes the fitted range.
                    _charts.ApplyRowPhysicalScale(loaded.Config);
                }
                _ctx.DataStatsPresenter?.SetReviewGroupBoxes(true);

                StitchedImagesReady?.Invoke(
                    loaded.GrayFrames, loaded.GrayWidths, loaded.GrayHeights, opsEff, posEff,
                    isGlobal, keepDisplayedCurves);

                if (!preserveCurves)
                {
                    _charts.UpdateGlobalRowChart();   // 畫布顯示走 ImageDisplayView（同源）；row 曲線照合併更新
                    UpdateStitchedOverviewChart();
                }

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
                string mode = (loaded.TotalImageCount > loadedCams) ? "Stitch" : "Single";
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

        public void ClearStitchedMode()
        {
            _content.ClearAll();
            _ctx.ReviewState.Config = null;
            if (_ctx.ChartReviewPatch?.IsDisposed ?? true) return;
            _ctx.DataStatsPresenter?.SetReviewGroupBoxes(false);
        }

        private bool TryActivateSharedCurves(string root, string grabId)
        {
            SingleGrabCurveData data;
            lock (_sharedCurveGate)
            {
                if (!string.Equals(_sharedCurveRoot, root, StringComparison.OrdinalIgnoreCase) ||
                    !string.Equals(_sharedCurveGrabId, grabId, StringComparison.Ordinal) ||
                    _sharedCurveData == null)
                    return false;
                data = _sharedCurveData;
            }

            _content.SetCurves(
                data.ColumnMean, data.ColumnMax,
                data.RowMean, data.RowMax,
                data.MergedRowMean, data.MergedRowMax);
            _ctx.ReviewState.Config = data.Config;
            _charts.ApplyRowPhysicalScale(data.Config);

            // 回顧與報表是兩個實體 chart，仍各需一次畫面套用；但這裡只吃記憶體快照，
            // 不再讀 bin、合併曲線，也不反向通知報表重畫。
            UpdateStitchedOverviewChart(notifyData: false);
            _charts.UpdateGlobalRowChart();
            return true;
        }

        private void PublishPreparedLayout(string grabId, ReviewImageLoadPlan plan)
        {
            if (plan == null) return;
            _ctx.ReviewState.Config = plan.Config;
            _charts.ApplyRowPhysicalScale(plan.Config);
            var ops = plan.Config?.CamOps ?? _ctx.Settings.GetCameraOpsUmArray();
            var positions = plan.Config?.CamPos ?? _ctx.Settings.GetCameraStartPositionMmArray();
            StitchedLayoutReady?.Invoke(
                grabId, plan.ExpectedWidths, plan.ExpectedHeights,
                ops, positions, _ctx.Settings.StitchMode == StitchMode.Global);
        }

        private void CachePreparedPlan(
            string root, string grabId, bool enableProcess, string ridgeDirection,
            ReviewImageLoadPlan plan)
        {
            lock (_preparedPlanGate)
            {
                _preparedPlanKey = BuildPreparedPlanKey(
                    root, grabId, enableProcess, ridgeDirection);
                _preparedPlan = plan;
            }
        }

        private bool TryGetPreparedPlan(
            string root, string grabId, bool enableProcess, string ridgeDirection,
            out ReviewImageLoadPlan plan)
        {
            string key = BuildPreparedPlanKey(root, grabId, enableProcess, ridgeDirection);
            lock (_preparedPlanGate)
            {
                if (string.Equals(_preparedPlanKey, key, StringComparison.Ordinal))
                {
                    plan = _preparedPlan;
                    return plan != null;
                }
            }
            plan = null;
            return false;
        }

        private static string BuildPreparedPlanKey(
            string root, string grabId, bool enableProcess, string ridgeDirection)
            => string.Concat(
                root ?? "", "|", grabId ?? "", "|",
                enableProcess ? "1" : "0", "|", ridgeDirection ?? "");

        private static void LogImagePlan(
            string grabId, string root, ReviewImageLoadPlan plan)
        {
            FlowTrace.Log(
                $"RV loadGrab paths {grabId} root={root} images={plan.TotalImageCount} " +
                $"cams={plan.GroupedPaths.Count} cfg={(plan.Config != null ? "yes" : "no")} " +
                $"align={plan.Alignment.Mode}");
        }

        public void Dispose()
        {
            _curveDataLoader.Dispose();
            _content.ClearAll();
            lock (_preparedPlanGate)
            {
                _preparedPlan = null;
                _preparedPlanKey = null;
            }
        }

        public void UpdateStitchedOverviewChart(bool notifyData = true)
            => _charts.UpdateStitchedOverviewChart(notifyData);

        public void RefreshChartsForSettingsChange()
            => _charts.RefreshChartsForSettingsChange();

        /// <summary>
        /// 原圖路徑（非 Stitch）：合併全域圖（Period 切換用）。
        /// </summary>
        public void ApplyGlobalMergeIfNeeded(bool preserveChartView = false)
            => ApplyGlobalMergeCore(null, preserveChartView);

        public void ApplyGlobalMergeForPeriod(DateTime period)
            => ApplyGlobalMergeCore(period, preserveChartView: false);

        private void ApplyGlobalMergeCore(DateTime? period, bool preserveChartView)
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
                frames.GrayFrames, frames.Widths, frames.Heights, opsArr, posArr, true,
                preserveChartView);
        }

        public void UpdateOverviewChartFromRepository()
            => _charts.UpdateOverviewChart(null);

        public void UpdateOverviewChartForPeriod(DateTime period)
            => _charts.UpdateOverviewChart(period);

        public void UpdateRowChartFromRepository()
            => _charts.UpdateRowChart(null);

        public void UpdateRowChartForPeriod(DateTime period)
            => _charts.UpdateRowChart(period);

        /// <summary>#13 同源新路徑的「當前視野」注入（form 快取 ImageDisplayView 視野；[l,r,top,bot]，null=無效）。
        /// chart 更新原子帶入此值 → 重載/強化切換不會先閃回預設再跟隨（同 Live 的 _liveViewLeftMm 解法）。</summary>
        public Func<double[]> SameSourceViewRange
        {
            get => _charts.SameSourceViewRange;
            set => _charts.SameSourceViewRange = value;
        }

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

    }
}
