using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Coordinators;
using AniloxRoll.Monitor.UI.Services;
using AniloxRoll.Monitor.UI.Widgets;
using AniloxRoll.Monitor.UI.Managers;
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
        private readonly DataStatisticsContext _ctx;
        private readonly Func<System.Windows.Forms.GroupBox> _getActiveStatMode;
        private readonly Func<List<GrabIdInfo>> _getGrabIdInfos;
        private readonly Func<string> _getStatsRoot;
        private readonly SingleGrabCurveDataLoader _singleGrabDataLoader;
        private readonly ReviewImageDataLoader _reviewImageDataLoader;
        private readonly LatestCurveLoadCoordinator _singleGrabLoads;

        private ColumnCurveChartHelper _muraProfileHelper;
        private RowCurveDisplayAdapter _rowDisplay;
        private bool _rowHasData;
        private string _lastColumnRangeState;
        private string _lastRowRangeState;

        internal event Action<string, string, SingleGrabCurveData> SingleGrabCurvePresented;

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
            _singleGrabDataLoader = new SingleGrabCurveDataLoader();
            _reviewImageDataLoader = new ReviewImageDataLoader();
            _singleGrabLoads = new LatestCurveLoadCoordinator(LoadSingleGrabCoreAsync);
        }

        public void Init()
        {
            if (_ctx.ChartDataPatch == null) return;
            _muraProfileHelper = new ColumnCurveChartHelper(_ctx.ChartDataPatch);
            if (_ctx.ChartDataRow != null)
            {
                _rowDisplay = new RowCurveDisplayAdapter(
                    new RowCurveChartHelper(_ctx.ChartDataRow),
                    () => _ctx.Settings.VerticalDirection)
                {
                    FlowName = "DT row"
                };
                _rowDisplay.SetThresholds(
                    _ctx.Settings.ErrorValueMeanH,
                    _ctx.Settings.ErrorValueMaxH);
            }
            _ctx.ChartDataPatch.PostPaint += OnColumnChartPostPaint;
            if (_ctx.ChartDataRow != null)
                _ctx.ChartDataRow.PostPaint += OnRowChartPostPaint;
            FlowTrace.Log("DT curve load policy latest-only shared-loader " +
                "entries=512 maxMB=256 scale=merged-only");
        }

        private void OnColumnChartPostPaint(
            object sender, System.Windows.Forms.DataVisualization.Charting.ChartPaintEventArgs e)
            => LogChartRange(isRow: false);

        private void OnRowChartPostPaint(
            object sender, System.Windows.Forms.DataVisualization.Charting.ChartPaintEventArgs e)
            => LogChartRange(isRow: true);

        private void LogChartRange(bool isRow)
        {
            var chart = isRow ? _ctx.ChartDataRow : _ctx.ChartDataPatch;
            if (chart == null || chart.IsDisposed || chart.ChartAreas.Count == 0) return;
            var axis = isRow
                ? chart.ChartAreas[0].AxisY
                : chart.ChartAreas[0].AxisX;
            string grabId = Convert.ToString(_ctx.CbDataGrabId?.SelectedItem);
            if (string.IsNullOrWhiteSpace(grabId)) grabId = "-";
            string state =
                $"axis={axis.Minimum:F2}~{axis.Maximum:F2}/" +
                $"view={axis.ScaleView.ViewMinimum:F2}~{axis.ScaleView.ViewMaximum:F2}";
            string stateKey = grabId + "|" + state;
            string previous = isRow ? _lastRowRangeState : _lastColumnRangeState;
            if (string.Equals(previous, stateKey, StringComparison.Ordinal)) return;
            if (isRow)
                _lastRowRangeState = stateKey;
            else
                _lastColumnRangeState = stateKey;

            FlowTrace.Dvt(
                $"DT chartRange {grabId} chart={(isRow ? "row" : "col")} {state}");
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
                    ScheduleSingleGrab(grabIdInfos[singleIdx]);
                else
                    Clear();
                return;
            }

            // Leaving single-sheet mode invalidates a still-running single-grab load.
            // Otherwise its late result could overwrite the range curve after the mode switch.
            _singleGrabLoads.Invalidate();

            if (grabIds == null || grabIds.Count == 0)
            {
                Clear();
                return;
            }
            // 範圍/時間模式：aggregate 多 grab 平均，當作歷史快照不做 view-time rescale
            ClearRow("range");

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
        private void ScheduleSingleGrab(GrabIdInfo info)
        {
            if (_muraProfileHelper == null || _ctx.Settings == null) return;
            _singleGrabLoads.Enqueue(info.GrabId, info.Earliest, info.Latest);
        }

        private async Task LoadSingleGrabCoreAsync(SingleGrabCurveLoadRequest request)
        {
            string statsRoot = _getStatsRoot();
            if (string.IsNullOrWhiteSpace(statsRoot)) return;

            var sw = Stopwatch.StartNew();
            try
            {
                int camCount = _ctx.CameraCount;
                ReviewImageLoadPlan layoutPlan = null;
                SingleGrabCurveData data = await Task.Run(() =>
                {
                    // Geometry preparation reads paths, CFG, and JPEG headers only. It deliberately
                    // avoids image decoding while following the same layout rules as Review.
                    layoutPlan = _reviewImageDataLoader.Prepare(
                        statsRoot, request.GrabId, request.HintFrom, request.HintTo,
                        camCount, enableProcess: false, ridgeDirection: "c", logPaths: false);
                    return _singleGrabDataLoader.Load(
                        statsRoot, request.GrabId, request.HintFrom, request.HintTo, camCount);
                });
                if (!_singleGrabLoads.IsCurrent(request))
                {
                    FlowTrace.Log($"DT curve stale-drop {request.GrabId}");
                    return;
                }

                CsvConfigSnapshot grabCfg = layoutPlan?.Config ?? data.Config;
                float captureHm = grabCfg?.HessianMaxFactorV ?? _ctx.Settings.HessianMaxFactorV;
                float valueScale = HessianRescaleHelper.Ratio(
                    captureHm, _ctx.Settings.HessianMaxFactorV);
                double[] ops = grabCfg?.CamOps ?? _ctx.Settings.GetCameraOpsUmArray();
                double[] pos = grabCfg?.CamPos ?? _ctx.Settings.GetCameraStartPositionMmArray();
                float errMean = _ctx.Settings.ErrorValueMeanV;
                float errMax = _ctx.Settings.ErrorValueMaxV;
                RowCurvePhysicalScale physicalScale = RowCurvePhysicalScaleResolver.Resolve(
                    grabCfg, _ctx.Settings);
                _rowDisplay?.SetRowPitchFromSpeed(
                    physicalScale.SpeedMPerMin, physicalScale.LineRateHz);
                double rowPitchMm = _rowDisplay?.RowPitchMm ?? 0.01;
                ImageViewRange? preparedView = layoutPlan == null
                    ? null
                    : _ctx.ReviewFitViewRangeProvider?.Invoke(
                        layoutPlan.ExpectedWidths, layoutPlan.ExpectedHeights,
                        ops, pos, _ctx.Settings.StitchMode == StitchMode.Global,
                        rowPitchMm);
                double[] view = preparedView.HasValue
                    ? new[]
                    {
                        preparedView.Value.LeftMm, preparedView.Value.RightMm,
                        preparedView.Value.TopMm, preparedView.Value.BottomMm
                    }
                    : null;
                if (preparedView.HasValue)
                {
                    ImageViewRange range = preparedView.Value;
                    FlowTrace.Dvt(
                        $"DT prefit {request.GrabId} content={range.ContentWidth}x{range.ContentHeight} " +
                        $"viewX={range.LeftMm:F0}~{range.RightMm:F0} " +
                        $"viewY={range.TopMm:F0}~{range.BottomMm:F0} source=main-geometry");
                }
                else
                {
                    FlowTrace.Dvt($"DT prefit unavailable {request.GrabId}");
                }
                Func<int, bool, double, double> fitViewRange = view != null && view.Length >= 4
                    ? (Func<int, bool, double, double>)((_, isLeft, __) =>
                        isLeft ? view[0] : view[1])
                    : null;

                long drawStartMs = sw.ElapsedMilliseconds;
                CurveMergeHelper.UpdateOverviewChart(
                    data.ColumnMean, data.ColumnMax, ops, pos, errMean, errMax,
                    _muraProfileHelper, camCount,
                    _ctx.Settings.StitchMode, fitViewRange, valueScale: valueScale);
                UpdateRowChart(data, grabCfg, request.GrabId, view, physicalScale);
                SingleGrabCurvePresented?.Invoke(statsRoot, request.GrabId, data);
                FlowTrace.Log($"DT curve load {request.GrabId} captures={data.ImageCount} " +
                    $"source=shared storage={data.StorageSource} configMs={data.ConfigMs} " +
                    $"waitMs={drawStartMs} pathMs={data.LookupMs} mergeMs={data.MergeMs} " +
                    $"summaryMs={data.SummaryMs} points={_muraProfileHelper.DisplayPointCount} " +
                    $"drawMs={sw.ElapsedMilliseconds - drawStartMs} totalMs={sw.ElapsedMilliseconds}");
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"[DataCurve] {request.GrabId}: {ex.GetType().Name}: {ex.Message}");
            }
        }

        private void UpdateRowChart(
            SingleGrabCurveData data, CsvConfigSnapshot grabCfg,
            string grabId, double[] view, RowCurvePhysicalScale scale)
        {
            if (_rowDisplay == null) return;
            if (data?.MergedRowMean == null || data.MergedRowMean.Length == 0)
            {
                ClearRow("missing");
                FlowTrace.Log($"DT row curve missing {grabId}");
                return;
            }

            float captureHmV = grabCfg?.HessianMaxFactorV ??
                _ctx.Settings.HessianMaxFactorV;
            float[] mean = HessianRescaleHelper.CloneAndRescale1D(
                data.MergedRowMean, captureHmV, _ctx.Settings.HessianMaxFactorH);
            float[] max = HessianRescaleHelper.CloneAndRescale1D(
                data.MergedRowMax, captureHmV, _ctx.Settings.HessianMaxFactorH);
            _rowDisplay.SetRowPitchFromSpeed(
                scale.SpeedMPerMin, scale.LineRateHz);
            _rowDisplay.SetThresholds(
                _ctx.Settings.ErrorValueMeanH, _ctx.Settings.ErrorValueMaxH);
            if (view != null && view.Length >= 4)
                _rowDisplay.UpdateDataAndViewRange(mean, max, view[2], view[3]);
            else
                _rowDisplay.UpdateData(mean, max);
            _rowHasData = true;
            FlowTrace.Log($"DT row curve load {grabId} source=shared storage={data.StorageSource} " +
                $"points={mean.Length} pitch={_rowDisplay.RowPitchMm:F6}mm");
        }

        public void ResetSingleGrabCache()
        {
            _singleGrabLoads.Invalidate();
            _singleGrabDataLoader.Clear();
        }

        public void Dispose()
        {
            if (_ctx.ChartDataPatch != null)
                _ctx.ChartDataPatch.PostPaint -= OnColumnChartPostPaint;
            if (_ctx.ChartDataRow != null)
                _ctx.ChartDataRow.PostPaint -= OnRowChartPostPaint;
            _singleGrabLoads.Invalidate();
            _singleGrabDataLoader.Dispose();
        }

        public void RefreshForSettingsChange()
        {
            if (_muraProfileHelper == null) return;
            _muraProfileHelper.SetThresholds(_ctx.Settings.ErrorValueMeanV, _ctx.Settings.ErrorValueMaxV);
            _rowDisplay?.SetThresholds(
                _ctx.Settings.ErrorValueMeanH, _ctx.Settings.ErrorValueMaxH);
            // 單片模式才需要按 HM 重算曲線坡度；aggregate 模式維持快照
            var grabIdInfos = _getGrabIdInfos();
            if (_getActiveStatMode() == _ctx.GrpDataSingleSheet
                && _ctx.CbDataGrabId.SelectedIndex >= 0
                && _ctx.CbDataGrabId.SelectedIndex < grabIdInfos.Count)
            {
                int index = _ctx.CbDataGrabId.SelectedIndex;
                ScheduleSingleGrab(grabIdInfos[index]);
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
            double[] view = _ctx.ReviewViewRangeProvider?.Invoke();
            Func<int, bool, double, double> viewRange = view != null && view.Length >= 4
                ? (Func<int, bool, double, double>)((_, isLeft, __) => isLeft ? view[0] : view[1])
                : null;
            CurveMergeHelper.UpdateOverviewChart(mean, max, ops, pos, errMean, errMax,
                _muraProfileHelper, _ctx.CameraCount, _ctx.Settings.StitchMode, viewRange);
            if (_rowHasData && view != null && view.Length >= 4)
                _rowDisplay?.UpdateViewRange(view[2], view[3]);
        }

        public void SetReviewViewRange(
            double leftMm, double rightMm, double topMm, double bottomMm)
        {
            _muraProfileHelper?.UpdateViewRange(leftMm, rightMm);
            if (_rowHasData)
                _rowDisplay?.UpdateViewRange(topMm, bottomMm);
        }

        public void SetPreparedReviewViewRange(
            double leftMm, double rightMm, double topMm, double bottomMm)
        {
            _muraProfileHelper?.UpdateViewRangeImmediate(leftMm, rightMm);
            if (_rowHasData)
                _rowDisplay?.UpdateViewRangeImmediate(topMm, bottomMm);
        }

        public void Clear()
        {
            if (_ctx.ChartDataPatch == null) return;
            _singleGrabLoads.Invalidate();
            foreach (var s in _ctx.ChartDataPatch.Series)
                s.Points.Clear();
            ClearRow("all");
        }

        public void ClearRow(string reason = "range")
        {
            if (string.Equals(reason, "range", StringComparison.Ordinal))
                _singleGrabLoads.Invalidate();
            _rowDisplay?.Clear();
            if (_rowHasData)
                FlowTrace.Log($"DT row curve clear mode={reason}");
            _rowHasData = false;
        }
    }
}
