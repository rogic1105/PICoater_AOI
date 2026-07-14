using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Services;
using AniloxRoll.Monitor.UI.Widgets;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.Presenters
{
    /// <summary>
    /// Data tab「Mura 空間分布圖」（chartDataPatch / chartDataColumn）的繪圖職責 ——
    /// 從 DataStatisticsPresenter 提取（2026-06-30）。只管「這張缺陷分布圖怎麼畫」：
    /// 範圍/時間模式 aggregate 平均、單片模式用單一 grab .bin + #CFG（與 chartReviewColumn 對齊、套 view-time 正規值 rescale）、
    /// 設定變更重畫閾值線、從回顧同步資料、清空。
    ///
    /// 不擁有「現在哪個模式 / 有哪些 grab / 資料根目錄」這些核心狀態：透過注入的 Func 即時讀
    /// （DataStatisticsPresenter 仍是這些狀態的單一真相）。對外 public 門面（RefreshForSettingsChange /
    /// SyncFromReview）由 DataStatisticsPresenter 轉發，外部呼叫端零修改。
    /// </summary>
    public sealed class MuraProfileChartPresenter : IDisposable
    {
        private const int SingleGrabCacheEntries = 64;
        private const long SingleGrabCacheBytes = 64L * 1024 * 1024;
        private const int PrefetchLookAhead = 4;

        private readonly DataStatisticsContext _ctx;
        private readonly Func<System.Windows.Forms.GroupBox> _getActiveStatMode;
        private readonly Func<List<GrabIdInfo>> _getGrabIdInfos;
        private readonly Func<string> _getStatsRoot;
        private readonly SingleGrabCurveCache _singleGrabCache =
            new SingleGrabCurveCache(SingleGrabCacheEntries, SingleGrabCacheBytes);

        private ColumnCurveChartHelper _muraProfileHelper;
        private CancellationTokenSource _prefetchCancellation;
        private Task<SingleGrabCurveProfile> _prefetchTask;
        private string _prefetchKey;
        private int _lastSingleGrabIndex = -1;
        private int _lastScrollDirection = 1;

        public MuraProfileChartPresenter(
            DataStatisticsContext ctx,
            Func<System.Windows.Forms.GroupBox> getActiveStatMode,
            Func<List<GrabIdInfo>> getGrabIdInfos,
            Func<string> getStatsRoot)
        {
            _ctx = ctx ?? throw new ArgumentNullException(nameof(ctx));
            _getActiveStatMode = getActiveStatMode ?? throw new ArgumentNullException(nameof(getActiveStatMode));
            _getGrabIdInfos = getGrabIdInfos ?? throw new ArgumentNullException(nameof(getGrabIdInfos));
            _getStatsRoot = getStatsRoot ?? throw new ArgumentNullException(nameof(getStatsRoot));
        }

        public void Init()
        {
            if (_ctx.ChartDataPatch == null) return;
            _muraProfileHelper = new ColumnCurveChartHelper(_ctx.ChartDataPatch);
        }

        public void Update(IList<GrabIdInfo> grabIds, IList<GrabIdInfo> candidateRange = null)
        {
            if (_muraProfileHelper == null || _ctx.Settings == null) return;

            // 單片模式（GrpDataSingleSheet）：永遠用 cbDataId.SelectedIndex 對應 grab，不依賴 caller 傳入的 grabIds。
            // 原因：cbDataId 變更不連動 cbDataIdStart/End（範圍獨立），caller 仍可能用舊範圍呼這函式
            // → 若用 grabIds[0] 會顯示舊範圍的第一筆而非剛點的 grab。
            // view-time 正規值 rescale（HM_capture / HM_current）讓改 PropertyGrid 正規值時曲線坡度立即變化。
            var grabIdInfos = _getGrabIdInfos();
            if (_getActiveStatMode() == _ctx.GrpDataSingleSheet)
            {
                int singleIdx = _ctx.CbDataGrabId.SelectedIndex;
                if (singleIdx >= 0 && singleIdx < grabIdInfos.Count)
                    UpdateForSingleGrab(grabIdInfos[singleIdx], singleIdx);
                else
                    Clear();
                return;
            }

            if (grabIds == null || grabIds.Count == 0)
            {
                Clear();
                return;
            }
            // 範圍/時間模式：aggregate 多 grab 平均，當作歷史快照不做 view-time rescale

            // ── 範圍/時間模式：舊 aggregate 邏輯 ──
            Dictionary<int, float[]> meanDict;
            Dictionary<int, float[]> maxDict;
            if (candidateRange != null)
            {
                var profiles = InspectionMuraProfileRepository.LoadRange(
                    _getStatsRoot(), candidateRange, 50);
                meanDict = profiles.Mean;
                maxDict = profiles.Max;
                string method = profiles.RankedCams == 0 ? "even" :
                    profiles.RankedCams == profiles.TotalCams ? "top-maxcmean" : "mixed";
                FlowTrace.Log($"DT curve candidates meanRows={profiles.MeanRows} maxRows={profiles.MaxRows} " +
                    $"method={method} coverage={profiles.ScoredRows}/{profiles.TotalRows} " +
                    $"rankedCams={profiles.RankedCams}/{profiles.TotalCams} " +
                    $"index={profiles.IndexHits}/{profiles.IndexBuilds}");
            }
            else
            {
                var profiles = InspectionMuraProfileRepository.LoadAverage(
                    _getStatsRoot(), grabIds);
                meanDict = profiles.Mean;
                maxDict = profiles.Max;
            }
            int camCount = _ctx.CameraCount;
            ApplyAggregateProfiles(
                meanDict, maxDict, camCount,
                _ctx.Settings.GetCameraOpsUmArray(),
                _ctx.Settings.GetCameraStartPositionMmArray(),
                _ctx.Settings.ErrorValueMeanV, _ctx.Settings.ErrorValueMaxV);
        }

        public async Task UpdateRangePreviewAsync(
            IList<GrabIdInfo> candidateRange, int generation, CancellationToken cancellationToken)
        {
            if (_muraProfileHelper == null || _ctx.Settings == null ||
                candidateRange == null || candidateRange.Count == 0)
                return;

            string statsRoot = _getStatsRoot();
            var rangeSnapshot = new List<GrabIdInfo>(candidateRange);
            int camCount = _ctx.CameraCount;
            double[] ops = _ctx.Settings.GetCameraOpsUmArray();
            double[] positions = _ctx.Settings.GetCameraStartPositionMmArray();
            float errorMean = _ctx.Settings.ErrorValueMeanV;
            float errorMax = _ctx.Settings.ErrorValueMaxV;
            string range = rangeSnapshot[0].GrabId + "~" +
                rangeSnapshot[rangeSnapshot.Count - 1].GrabId;
            var sw = Stopwatch.StartNew();

            var profiles = await Task.Run(() =>
                InspectionMuraProfileRepository.LoadRange(
                    statsRoot, rangeSnapshot, 50, cancellationToken), cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            long loadMs = sw.ElapsedMilliseconds;

            ApplyAggregateProfiles(
                profiles.Mean, profiles.Max, camCount, ops, positions, errorMean, errorMax);
            string method = profiles.RankedCams == 0 ? "even" :
                profiles.RankedCams == profiles.TotalCams ? "top-maxcmean" : "mixed";
            FlowTrace.Log($"DT range preview apply gen={generation} range={range} " +
                $"loadMs={loadMs} drawMs={sw.ElapsedMilliseconds - loadMs} " +
                $"meanRows={profiles.MeanRows} maxRows={profiles.MaxRows} method={method} " +
                $"coverage={profiles.ScoredRows}/{profiles.TotalRows} " +
                $"rankedCams={profiles.RankedCams}/{profiles.TotalCams} " +
                $"index={profiles.IndexHits}/{profiles.IndexBuilds}");
        }

        private void ApplyAggregateProfiles(
            Dictionary<int, float[]> meanDict,
            Dictionary<int, float[]> maxDict,
            int camCount,
            double[] ops,
            double[] positions,
            float errorMean,
            float errorMax)
        {
            if (meanDict.Count == 0)
            {
                Clear();
                return;
            }

            var allMean = new float[camCount][];
            var allMax = new float[camCount][];
            for (int i = 0; i < camCount; i++)
            {
                meanDict.TryGetValue(i + 1, out allMean[i]);
                maxDict.TryGetValue(i + 1, out allMax[i]);
            }
            CurveMergeHelper.UpdateOverviewChart(
                allMean, allMax, ops, positions, errorMean, errorMax,
                _muraProfileHelper, camCount, StitchMode.Vertical, null);
        }

        /// <summary>
        /// 用單一 grab 的 .bin（MergeCurves 合多 capture）+ 該 grab 的 CSV #CFG OPS/Pos
        /// 更新 chartDataColumn，與 chartReviewColumn 完全對齊。不依賴 camReviewMain 是否載入。
        /// 套用 view-time 正規值 rescale：display = (bin/255) × (HM_capture / HM_current)；
        /// 改 PropertyGrid 正規值會立刻反映在曲線坡度上。
        /// </summary>
        private void UpdateForSingleGrab(GrabIdInfo info, int selectedIndex)
        {
            if (_muraProfileHelper == null || _ctx.Settings == null) return;
            string statsRoot = _getStatsRoot();
            if (string.IsNullOrWhiteSpace(statsRoot)) return;
            SingleGrabCurveSummaryStore.NotifyReadActivity();

            var sw = Stopwatch.StartNew();
            var grabCfg = InspectionConfigRepository.LoadForGrabId(
                statsRoot, info.GrabId, info.Earliest, info.Latest);
            long configMs = sw.ElapsedMilliseconds;
            int camCount = _ctx.CameraCount;
            string cacheKey = BuildCacheKey(statsRoot, info, camCount);

            bool cacheHit = _singleGrabCache.TryGet(cacheKey, out SingleGrabCurveProfile profile);
            bool joinedPrefetch = !cacheHit && string.Equals(
                cacheKey, _prefetchKey, StringComparison.OrdinalIgnoreCase);
            long waitMs = 0;
            if (!cacheHit)
            {
                if (!joinedPrefetch) CancelPrefetch();
                long waitStartMs = sw.ElapsedMilliseconds;
                profile = _singleGrabCache.GetOrLoadAsync(cacheKey,
                    () => LoadSingleGrabProfile(
                        statsRoot, info, camCount, CancellationToken.None))
                    .GetAwaiter().GetResult();
                waitMs = sw.ElapsedMilliseconds - waitStartMs;
            }

            if (profile == null) return;
            float[][] allMean = profile.CloneMean();
            float[][] allMax = profile.CloneMax();

            // view-time 正規值 rescale：chartDataColumn 是欄曲線，用 V 的 capture/current ratio
            float captureHm = grabCfg?.HessianMaxFactorV ?? _ctx.Settings.HessianMaxFactorV;
            HessianRescaleHelper.RescaleInPlace2D(allMean, allMax, captureHm, _ctx.Settings.HessianMaxFactorV);

            double[] ops = grabCfg?.CamOps  ?? _ctx.Settings.GetCameraOpsUmArray();
            double[] pos = grabCfg?.CamPos  ?? _ctx.Settings.GetCameraStartPositionMmArray();
            float errMean = _ctx.Settings.ErrorValueMeanV;  // view-time 閾值用當前 Settings
            float errMax  = _ctx.Settings.ErrorValueMaxV;

            long drawStartMs = sw.ElapsedMilliseconds;
            CurveMergeHelper.UpdateOverviewChart(
                allMean, allMax, ops, pos, errMean, errMax,
                _muraProfileHelper, camCount,
                _ctx.Settings.StitchMode, null);
            string source = cacheHit ? "cache" : joinedPrefetch ? "prefetch" : "disk";
            FlowTrace.Log($"DT curve load {info.GrabId} captures={profile.CaptureCount} " +
                $"source={source} storage={profile.StorageSource} configMs={configMs} waitMs={waitMs} " +
                $"pathMs={profile.LookupMs} mergeMs={profile.MergeMs} " +
                $"summaryMs={profile.SummaryMs} points={_muraProfileHelper.DisplayPointCount} " +
                $"drawMs={sw.ElapsedMilliseconds - drawStartMs} totalMs={sw.ElapsedMilliseconds}");

            int direction = _lastSingleGrabIndex < 0
                ? 1
                : Math.Sign(selectedIndex - _lastSingleGrabIndex);
            if (direction != 0) _lastScrollDirection = direction;
            _lastSingleGrabIndex = selectedIndex;
            ScheduleAdjacentPrefetch(
                statsRoot, _getGrabIdInfos(), selectedIndex, _lastScrollDirection, camCount, cacheKey);
        }

        /// <summary>
        /// 由 PropertyGrid 變更觸發：刷新 chartDataColumn 的閾值線 + view-time 正規值 rescale。
        /// 不重做 RefreshStats（避免重算統計）；只重畫 chart。
        /// </summary>
        private static SingleGrabCurveProfile LoadSingleGrabProfile(
            string statsRoot, GrabIdInfo info, int camCount, CancellationToken cancellationToken)
        {
            var sw = Stopwatch.StartNew();
            if (SingleGrabCurveSummaryStore.TryLoad(
                statsRoot, info, camCount, out SingleGrabCurveSummary summary))
            {
                return new SingleGrabCurveProfile(
                    summary.Mean, summary.Max, summary.CaptureCount,
                    "summary", 0, 0, sw.ElapsedMilliseconds);
            }

            var grouped = InspectionImagePathRepository.LoadForGrabId(
                statsRoot, info.GrabId, info.Earliest, info.Latest);
            long lookupMs = sw.ElapsedMilliseconds;
            int captureCount = 0;
            int mergedCaptureCount = 0;
            var allMean = new float[camCount][];
            var allMax = new float[camCount][];

            for (int i = 0; i < camCount; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int camId = i + 1;
                if (!grouped.TryGetValue(camId, out var paths) || paths.Count == 0) continue;
                captureCount += paths.Count;
                CurveMergeHelper.MergeCurves(
                    paths, out allMean[i], out allMax[i], out int mergedForCamera,
                    cancellationToken);
                mergedCaptureCount += mergedForCamera;
            }

            long mergeMs = sw.ElapsedMilliseconds - lookupMs;
            long summaryStartMs = sw.ElapsedMilliseconds;
            string summaryWrite;
            if (captureCount > 0 && mergedCaptureCount == captureCount)
            {
                bool queued = SingleGrabCurveSummaryStore.QueueSave(
                    statsRoot, info, camCount,
                    new SingleGrabCurveSummary(allMean, allMax, captureCount));
                summaryWrite = queued ? "queued" : "dropped";
            }
            else
            {
                summaryWrite = "skip-incomplete";
            }
            long summaryMs = sw.ElapsedMilliseconds - summaryStartMs;
            FlowTrace.Log($"DT curve summary {info.GrabId} write={summaryWrite} " +
                $"captures={captureCount} merged={mergedCaptureCount} ms={summaryMs}");
            return new SingleGrabCurveProfile(
                allMean, allMax, captureCount,
                "bins", lookupMs, mergeMs, summaryMs);
        }

        private void ScheduleAdjacentPrefetch(
            string statsRoot,
            IList<GrabIdInfo> grabIdInfos,
            int selectedIndex,
            int direction,
            int camCount,
            string selectedKey)
        {
            if (grabIdInfos == null || grabIdInfos.Count == 0)
            {
                CancelPrefetch();
                return;
            }

            int candidateIndex = -1;
            string candidateKey = null;
            SingleGrabCurveProfile ignored;
            for (int step = 1; step <= PrefetchLookAhead; step++)
            {
                int index = selectedIndex + direction * step;
                if (index < 0 || index >= grabIdInfos.Count) break;
                string key = BuildCacheKey(statsRoot, grabIdInfos[index], camCount);
                if (string.Equals(key, selectedKey, StringComparison.OrdinalIgnoreCase)) continue;
                if (_singleGrabCache.TryGet(key, out ignored)) continue;
                candidateIndex = index;
                candidateKey = key;
                break;
            }

            if (candidateIndex < 0)
            {
                CancelPrefetch();
                return;
            }
            if (string.Equals(candidateKey, _prefetchKey, StringComparison.OrdinalIgnoreCase) &&
                _prefetchTask != null && !_prefetchTask.IsCompleted)
                return;

            CancelPrefetch();
            GrabIdInfo candidate = grabIdInfos[candidateIndex];
            _prefetchCancellation = new CancellationTokenSource();
            CancellationToken token = _prefetchCancellation.Token;
            _prefetchKey = candidateKey;
            var prefetchWatch = Stopwatch.StartNew();
            _prefetchTask = _singleGrabCache.GetOrLoadAsync(candidateKey,
                () => LoadSingleGrabProfile(statsRoot, candidate, camCount, token));
            _prefetchTask.ContinueWith(task =>
            {
                if (task.Status != TaskStatus.RanToCompletion) return;
                FlowTrace.Log($"DT curve prefetch {candidate.GrabId} readyMs={prefetchWatch.ElapsedMilliseconds} " +
                    $"storage={task.Result.StorageSource} " +
                    $"cacheEntries={_singleGrabCache.Count} " +
                    $"cacheMB={_singleGrabCache.CachedBytes / (1024 * 1024)}");
            }, CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
        }

        private static string BuildCacheKey(string statsRoot, GrabIdInfo info, int camCount)
        {
            string root = Path.GetFullPath(statsRoot).TrimEnd(
                Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            return root + "|" + info.GrabId + "|" + info.Earliest.Ticks + "|" +
                info.Latest.Ticks + "|" + camCount;
        }

        public void ResetSingleGrabCache()
        {
            CancelPrefetch();
            _singleGrabCache.Clear();
            _lastSingleGrabIndex = -1;
            _lastScrollDirection = 1;
        }

        public void Dispose()
        {
            CancelPrefetch();
            _singleGrabCache.Dispose();
        }

        private void CancelPrefetch()
        {
            _prefetchCancellation?.Cancel();
            _prefetchCancellation?.Dispose();
            _prefetchCancellation = null;
            _prefetchTask = null;
            _prefetchKey = null;
        }

        public void RefreshForSettingsChange()
        {
            if (_muraProfileHelper == null) return;
            _muraProfileHelper.SetThresholds(_ctx.Settings.ErrorValueMeanV, _ctx.Settings.ErrorValueMaxV);
            // 單片模式才需要按 HM 重算曲線坡度；aggregate 模式維持快照
            var grabIdInfos = _getGrabIdInfos();
            if (_getActiveStatMode() == _ctx.GrpDataSingleSheet
                && _ctx.CbDataGrabId.SelectedIndex >= 0
                && _ctx.CbDataGrabId.SelectedIndex < grabIdInfos.Count)
            {
                int index = _ctx.CbDataGrabId.SelectedIndex;
                UpdateForSingleGrab(grabIdInfos[index], index);
            }
        }

        /// <summary>
        /// SingleSheet 模式：直接使用 Review tab 已載入的曲線資料（已套 view-time HM rescale），
        /// 確保 chartDataColumn 與 chartReviewColumn 完全一致（相同 OPS/Pos 與顯示值）。
        /// </summary>
        public void SyncFromReview(float[][] mean, float[][] max,
            double[] ops, double[] pos, float errMean, float errMax)
        {
            if (_muraProfileHelper == null) return;
            CurveMergeHelper.UpdateOverviewChart(mean, max, ops, pos, errMean, errMax,
                _muraProfileHelper, _ctx.CameraCount, StitchMode.Vertical, null);
        }

        public void Clear()
        {
            if (_ctx.ChartDataPatch == null) return;
            foreach (var s in _ctx.ChartDataPatch.Series)
                s.Points.Clear();
        }
    }
}
