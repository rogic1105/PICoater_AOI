using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Threading;
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
        private const int ReviewCurveMinimumCycleMs = 80;
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
        private readonly LatestGrabLoadCoordinator _curveLoads;
        private readonly LatestGrabLoadCoordinator _thumbnailLoads;
        private readonly SingleGrabCurveDataLoader _curveDataLoader;
        private readonly ReviewImageDataLoader _imageDataLoader;
        private readonly ReviewPeriodDataLoader _periodDataLoader;
        private readonly ReviewPeriodImagePresenter _periodImages;
        private readonly ReviewImageLoadGate _imageLoads = new ReviewImageLoadGate();
        private readonly ReviewAsyncLruCache<ReviewImageLoadPlan> _planCache =
            new ReviewAsyncLruCache<ReviewImageLoadPlan>(32, 32, plan => 1);
        private readonly ReviewAsyncLruCache<ReviewThumbnailSnapshot> _thumbnailCache =
            new ReviewAsyncLruCache<ReviewThumbnailSnapshot>(24, 96L * 1024 * 1024,
                thumbnail => thumbnail.EstimatedBytes);
        private int _prefetchGeneration;
        private int _disposed;
        private readonly object _sharedCurveGate = new object();
        private string _sharedCurveRoot;
        private string _sharedCurveGrabId;
        private SingleGrabCurveData _sharedCurveData;

        /// <summary>快路：只載曲線（欄+列，.bin 數十 KB + tick 對齊 csv）+CFG → 更新欄/列 chart。
        /// 滾動掃描用——chart 即時跟著序號跑（使用者快速找異常），影像（重：JPEG 解碼+拼接）由
        /// debounce 後的完整載入跟上（硬體限制的分層載入）。</summary>
        public Task LoadGrabCurvesOnlyAsync(string grabId, DateTime hintFrom, DateTime hintTo)
            => _curveLoads.Enqueue(grabId, hintFrom, hintTo);

        /// <summary>
        /// Invalidates only the debounced full-resolution load. The serialized thumbnail lane
        /// remains active so selections arriving while it is busy can coalesce to the latest one.
        /// </summary>
        public void InvalidateSettledImageLoad()
        {
            CancelAdjacentPrefetch();
            if (_imageLoads.Invalidate())
            {
                _ctx.BusyUi.SetBusy(false);
                Core.Services.FlowTrace.Log("RV loadGrab busy off reason=invalidated");
            }
        }

        /// <summary>
        /// Invalidates curve, preview, and full-resolution image lanes when leaving the current
        /// single-grab display context.
        /// </summary>
        public void InvalidateImageLoad()
        {
            _curveLoads.Invalidate();
            _thumbnailLoads.Invalidate();
            InvalidateSettledImageLoad();
        }

        public Task LoadGrabThumbnailAsync(
            string grabId, DateTime hintFrom, DateTime hintTo)
            => _thumbnailLoads.Enqueue(grabId, hintFrom, hintTo);

        public void CancelAdjacentPrefetch()
            => Interlocked.Increment(ref _prefetchGeneration);

        public void BeginAdjacentPrefetch(
            IList<GrabIdInfo> items, int currentIndex, int direction)
        {
            if (Volatile.Read(ref _disposed) != 0) return;
            GrabIdInfo[] neighbors = ReviewAdjacentPrefetchPolicy.Select(
                items, currentIndex, direction);
            if (neighbors.Length == 0) return;

            int generation = Interlocked.Increment(ref _prefetchGeneration);
            string center = items[currentIndex].GrabId;
            _ = PrefetchAdjacentCoreAsync(center, neighbors, generation);
        }

        private async Task PrefetchAdjacentCoreAsync(
            string centerGrabId, GrabIdInfo[] neighbors, int generation)
        {
            string root = !string.IsNullOrWhiteSpace(
                UI.State.UserSessionState.LastDataPath)
                ? UI.State.UserSessionState.LastDataPath
                : _ctx.DataStatsPresenter.StatsDataRootPath;
            if (string.IsNullOrWhiteSpace(root)) return;

            int cameraCount = _ctx.CameraCount;
            bool enableProcess = LastReviewProcessedMode;
            string ridgeDirection = ActiveRidgeDirection;
            FlowTrace.Dvt(
                $"RV prefetch begin center={centerGrabId} neighbors=" +
                string.Join("|", Array.ConvertAll(neighbors, item => item.GrabId)));

            for (int i = 0; i < neighbors.Length; i++)
            {
                if (Volatile.Read(ref _disposed) != 0) return;
                if (generation != Volatile.Read(ref _prefetchGeneration)) return;
                GrabIdInfo info = neighbors[i];
                var watch = Stopwatch.StartNew();
                try
                {
                    ReviewImageLoadPlan plan = await GetOrPreparePlanAsync(
                        root, info.GrabId, info.Earliest, info.Latest,
                        cameraCount, enableProcess, ridgeDirection, logPaths: false);
                    if (generation != Volatile.Read(ref _prefetchGeneration)) return;

                    await _curveDataLoader.PrefetchAsync(root, info, cameraCount);
                    if (generation != Volatile.Read(ref _prefetchGeneration)) return;

                    ReviewCacheAccess thumbnailAccess;
                    ReviewThumbnailSnapshot thumbnail = await GetOrLoadThumbnailAsync(
                        root, info.GrabId, plan, cameraCount,
                        enableProcess, ridgeDirection, out thumbnailAccess);
                    if (thumbnail == null)
                    {
                        FlowTrace.Dvt(
                            $"RV prefetch unavailable center={centerGrabId} " +
                            $"neighbor={info.GrabId} error=no-preview");
                        continue;
                    }
                    FlowTrace.Dvt(
                        $"RV prefetch ready center={centerGrabId} neighbor={info.GrabId} " +
                        $"thumbnail={CacheAccessText(thumbnailAccess)} " +
                        $"total={watch.ElapsedMilliseconds}ms");
                }
                catch (Exception ex)
                {
                    Trace.WriteLine(
                        $"[ReviewPrefetch] {info.GrabId}: " +
                        $"{ex.GetType().Name}: {ex.Message}");
                    FlowTrace.Dvt(
                        $"RV prefetch unavailable center={centerGrabId} " +
                        $"neighbor={info.GrabId} error={ex.GetType().Name}");
                }
            }
        }

        private async Task LoadGrabThumbnailCoreAsync(SingleGrabLoadRequest request)
        {
            string grabId = request.GrabId;
            DateTime hintFrom = request.HintFrom;
            DateTime hintTo = request.HintTo;
            string root = !string.IsNullOrWhiteSpace(
                UI.State.UserSessionState.LastDataPath)
                ? UI.State.UserSessionState.LastDataPath
                : _ctx.DataStatsPresenter.StatsDataRootPath;
            if (string.IsNullOrWhiteSpace(root)) return;

            var watch = Stopwatch.StartNew();
            if (request.CoalescedCount > 0)
            {
                Core.Services.FlowTrace.Log(
                    $"RV thumbnail coalesced {grabId} skipped={request.CoalescedCount} " +
                    "minCycleMs=33");
            }
            Core.Services.FlowTrace.Log($"RV thumbnail begin {grabId}");
            try
            {
                int cameraCount = _ctx.CameraCount;
                bool enableProcess = LastReviewProcessedMode;
                string ridgeDirection = ActiveRidgeDirection;
                ReviewImageLoadPlan plan = await GetOrPreparePlanAsync(
                    root, grabId, hintFrom, hintTo, cameraCount,
                    enableProcess, ridgeDirection, logPaths: false);
                ReviewCacheAccess cacheAccess;
                ReviewThumbnailSnapshot loaded = await GetOrLoadThumbnailAsync(
                    root, grabId, plan, cameraCount,
                    enableProcess, ridgeDirection, out cacheAccess);

                if (loaded == null)
                {
                    Core.Services.FlowTrace.Log(
                        $"RV thumbnail unavailable {grabId} ({watch.ElapsedMilliseconds}ms)");
                    return;
                }

                if (!_thumbnailLoads.CanApplyStarted(request))
                {
                    Core.Services.FlowTrace.Log(
                        $"RV thumbnail stale-drop {grabId} ({watch.ElapsedMilliseconds}ms)");
                    return;
                }

                int imageCount = loaded.ImageCount;

                double[] ops = plan.Config?.CamOps ??
                    _ctx.Settings.GetCameraOpsUmArray();
                double[] positions = plan.Config?.CamPos ??
                    _ctx.Settings.GetCameraStartPositionMmArray();
                double exactFeedScale =
                    InspectionEngineConfig.DefaultSaveResizeScale *
                    loaded.PixelScaleRatio;
                int feedScale = Math.Max(1, (int)Math.Round(exactFeedScale));
                double rowPitchCorrection = exactFeedScale / feedScale;
                StitchedImagesReady?.Invoke(
                    loaded.GrayFrames, loaded.GrayWidths, loaded.GrayHeights,
                    ops, positions, true, true,
                    feedScale, rowPitchCorrection);
                Core.Services.FlowTrace.Log(
                    $"RV thumbnail done {grabId} total={watch.ElapsedMilliseconds}ms " +
                    $"decode={loaded.DecodeMs}ms images={imageCount} " +
                    $"ratio={loaded.PixelScaleRatio:F2} source={loaded.PreviewSource} " +
                    $"cache={CacheAccessText(cacheAccess)} " +
                    $"atlas={(loaded.PreviewSource == "atlas" ? loaded.PreviewWidth + "x" + loaded.PreviewHeight : "none")}");
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[ReviewThumbnail] {grabId}: {ex.GetType().Name}: {ex.Message}");
                Core.Services.FlowTrace.Log(
                    $"RV thumbnail unavailable {grabId} " +
                    $"({watch.ElapsedMilliseconds}ms; {ex.GetType().Name})");
            }
        }

        private async Task LoadGrabCurvesCoreAsync(SingleGrabLoadRequest request)
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
                ReviewImageLoadPlan preparedPlan;
                bool planReady = TryGetPreparedPlan(
                    root, grabId, enableProcess, ridgeDir, out preparedPlan);
                Task<ReviewImageLoadPlan> layoutTask = GetOrPreparePlanAsync(
                    root, grabId, hintFrom, hintTo, camCount,
                    enableProcess, ridgeDir, logPaths: false);
                // Geometry and curve data may load in parallel, but neither is presented until
                // both are ready. The thumbnail lane shares the same geometry task.
                Task<SingleGrabCurveData> curveTask = Task.Run(() =>
                {
                    SingleGrabCurveData data = _curveDataLoader.Load(
                        root, grabId, hintFrom, hintTo, camCount,
                        planReady ? preparedPlan.Config : null);
                    Core.Services.FlowTrace.Log($"RV curves paths {grabId} root={root} images={data.ImageCount} cams={data.MatchedCameraCount} cfg={(data.Config != null ? "yes" : "no")} align={data.AlignmentMode} source={data.StorageSource} coalesced={request.CoalescedCount}");
                    return data;
                });
                await Task.WhenAll(layoutTask, curveTask);
                ReviewImageLoadPlan layoutPlan = layoutTask.Result;
                SingleGrabCurveData loaded = curveTask.Result;
                bool isLatest = _curveLoads.IsCurrent(request);
                if (!_curveLoads.CanApplyStarted(request))
                {
                    Core.Services.FlowTrace.Log($"RV curves stale-drop {grabId}");
                    return;
                }
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
                Core.Services.FlowTrace.Log(
                    $"RV curves {grabId}（{sw.ElapsedMilliseconds}ms） " +
                    $"presentation={(isLatest ? "latest" : "progressive")}");
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
            _periodImages = new ReviewPeriodImagePresenter(
                new ReviewPeriodImageContext
                {
                    ReviewState = ctx.ReviewState,
                    Settings = ctx.Settings,
                    ImageRepository = ctx.ImageRepository,
                    DateTimeNavigator = ctx.DateTimeNavigator,
                    InspectionService = ctx.InspectionService,
                    CameraCount = ctx.CameraCount,
                    PublishFrames = (
                        frames, widths, heights, ops, positions,
                        isGlobal, preserveView, feedScale, rowPitchScale) =>
                        StitchedImagesReady?.Invoke(
                            frames, widths, heights, ops, positions,
                            isGlobal, preserveView, feedScale, rowPitchScale)
                },
                _periodDataLoader);
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
            _curveLoads = new LatestGrabLoadCoordinator(
                LoadGrabCurvesCoreAsync,
                minimumCycleMs: ReviewCurveMinimumCycleMs);
            _thumbnailLoads = new LatestGrabLoadCoordinator(
                LoadGrabThumbnailCoreAsync, minimumCycleMs: 33);
            Core.Services.FlowTrace.Log(
                $"RV curve load policy latest-only minCycleMs={ReviewCurveMinimumCycleMs}");
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
        internal event ReviewFramesReady StitchedImagesReady;

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
            _thumbnailLoads.Invalidate();
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
                ReviewImageLoadPlan cachedPlan;
                bool planWasCached = TryGetPreparedPlan(
                    root, grabId, enableProcess, ridgeDir, out cachedPlan);
                ReviewImageLoadPlan plan = planWasCached
                    ? cachedPlan
                    : await GetOrPreparePlanAsync(
                        root, grabId, hintFrom, hintTo, camCount,
                        enableProcess, ridgeDir, logPaths: true);
                if (planWasCached)
                {
                    LogImagePlan(grabId, root, plan);
                    Core.Services.FlowTrace.Log($"RV loadGrab plan reuse {grabId}");
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
                    plan, camCount, enableProcess, ridgeDir,
                    includeCurves: !preserveCurves,
                    standardDisplayGain: ResolveStandardDisplayGain(
                        enableProcess, ridgeDir)));
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

                double exactFeedScale =
                    InspectionEngineConfig.DefaultSaveResizeScale *
                    loaded.PixelScaleRatio;
                int feedScale = Math.Max(1, (int)Math.Round(exactFeedScale));
                double rowPitchCorrection = exactFeedScale / feedScale;
                StitchedImagesReady?.Invoke(
                    loaded.GrayFrames, loaded.GrayWidths, loaded.GrayHeights, opsEff, posEff,
                    isGlobal, keepDisplayedCurves,
                    feedScale,
                    rowPitchCorrection);
                if (enableProcess && loaded.PixelScaleRatio > 1.0)
                {
                    string standardAxis = ridgeDir == "r" || ridgeDir == "h" ? "R" : "C";
                    byte sampleMin;
                    byte sampleMax;
                    double sampleMean;
                    TryMeasureGrayFrames(
                        loaded.GrayFrames, out sampleMin, out sampleMax, out sampleMean);
                    Core.Services.FlowTrace.Log(
                        $"RV hessian standard {grabId} dir={standardAxis} " +
                        $"gain={ResolveStandardDisplayGain(true, ridgeDir):0.######} " +
                        $"scale={feedScale} sampleMin={sampleMin} sampleMax={sampleMax} " +
                        $"sampleMean={sampleMean.ToString("0.000", CultureInfo.InvariantCulture)}");
                }

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

        private async Task<ReviewImageLoadPlan> GetOrPreparePlanAsync(
            string root, string grabId, DateTime hintFrom, DateTime hintTo,
            int cameraCount, bool enableProcess, string ridgeDirection,
            bool logPaths)
        {
            string key = BuildPreparedPlanKey(
                root, grabId, enableProcess, ridgeDirection);
            ReviewCacheAccess access;
            ReviewImageLoadPlan plan = await _planCache.GetOrLoadAsync(
                key,
                () => _imageDataLoader.Prepare(
                    root, grabId, hintFrom, hintTo, cameraCount,
                    enableProcess, ridgeDirection, logPaths),
                out access);
            if (access == ReviewCacheAccess.Joined)
                FlowTrace.Dvt($"RV plan prepare reuse-inflight {grabId}");
            else if (access == ReviewCacheAccess.Cold)
                FlowTrace.Dvt($"RV plan prepare begin {grabId}");
            return plan;
        }

        private bool TryGetPreparedPlan(
            string root, string grabId, bool enableProcess, string ridgeDirection,
            out ReviewImageLoadPlan plan)
        {
            string key = BuildPreparedPlanKey(root, grabId, enableProcess, ridgeDirection);
            return _planCache.TryGet(key, out plan);
        }

        private Task<ReviewThumbnailSnapshot> GetOrLoadThumbnailAsync(
            string root, string grabId, ReviewImageLoadPlan plan,
            int cameraCount, bool enableProcess, string ridgeDirection,
            out ReviewCacheAccess access)
        {
            string key = BuildPreparedPlanKey(
                root, grabId, enableProcess, ridgeDirection);
            return _thumbnailCache.GetOrLoadAsync(
                key,
                () =>
                {
                    ReviewImageData loaded = null;
                    try
                    {
                        loaded = _imageDataLoader.Load(
                            plan, cameraCount, enableProcess, ridgeDirection,
                            includeCurves: false, useThumbnail: true);
                        var snapshot = new ReviewThumbnailSnapshot
                        {
                            GrayFrames = loaded.GrayFrames,
                            GrayWidths = loaded.GrayWidths,
                            GrayHeights = loaded.GrayHeights,
                            DecodeMs = loaded.StitchMs,
                            PixelScaleRatio = loaded.PixelScaleRatio,
                            PreviewSource = loaded.PreviewSource,
                            PreviewWidth = loaded.PreviewWidth,
                            PreviewHeight = loaded.PreviewHeight
                        };
                        return snapshot.IsUsable ? snapshot : null;
                    }
                    finally
                    {
                        loaded?.DisposeImages();
                    }
                },
                out access);
        }

        private static string CacheAccessText(ReviewCacheAccess access)
        {
            switch (access)
            {
                case ReviewCacheAccess.Hit: return "hit";
                case ReviewCacheAccess.Joined: return "join";
                default: return "cold";
            }
        }

        private static string BuildPreparedPlanKey(
            string root, string grabId, bool enableProcess, string ridgeDirection)
            => string.Concat(
                root ?? "", "|", grabId ?? "", "|",
                enableProcess ? "1" : "0", "|", ridgeDirection ?? "");

        private float ResolveStandardDisplayGain(bool enableProcess, string ridgeDirection)
        {
            if (!enableProcess) return 0f;
            return ridgeDirection == "r" || ridgeDirection == "h"
                ? (float)_ctx.Settings.HessianMaxFactorH
                : (float)_ctx.Settings.HessianMaxFactorV;
        }

        private static bool TryMeasureGrayFrames(
            byte[][] frames, out byte minimum, out byte maximum, out double mean)
        {
            minimum = byte.MaxValue;
            maximum = byte.MinValue;
            mean = 0.0;
            if (frames == null || frames.Length == 0)
            {
                minimum = 0;
                return false;
            }

            long sum = 0;
            long count = 0;
            for (int frameIndex = 0; frameIndex < frames.Length; frameIndex++)
            {
                byte[] frame = frames[frameIndex];
                if (frame == null || frame.Length == 0) continue;

                int stride = Math.Max(1, frame.Length / 4096);
                for (int index = 0; index < frame.Length; index += stride)
                {
                    byte value = frame[index];
                    if (value < minimum) minimum = value;
                    if (value > maximum) maximum = value;
                    sum += value;
                    count++;
                }
            }

            if (count == 0)
            {
                minimum = 0;
                maximum = 0;
                return false;
            }

            mean = (double)sum / count;
            return true;
        }

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
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            CancelAdjacentPrefetch();
            _curveDataLoader.Dispose();
            _planCache.Dispose();
            _thumbnailCache.Dispose();
            _content.ClearAll();
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
            _periodImages.Apply(
                period, preserveChartView,
                LastReviewProcessedMode, ActiveRidgeDirection);
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

    }
}
