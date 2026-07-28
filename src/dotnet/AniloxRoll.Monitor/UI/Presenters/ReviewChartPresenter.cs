using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Windows.Forms.DataVisualization.Charting;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Managers;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Services;
using AniloxRoll.Monitor.UI.State;
using AniloxRoll.Monitor.UI.Widgets;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.Presenters
{
    internal sealed class ReviewChartContext
    {
        public Chart OverviewChart { get; set; }
        public ReviewRuntimeState ReviewState { get; set; }
        public RowCurveSyncCoordinator RowChartSync { get; set; }
        public ColumnCurveChartHelper OverviewHelper { get; set; }
        public InspectionSettings Settings { get; set; }
        public ImageRepository ImageRepository { get; set; }
        public DateTimeNavigator DateTimeNavigator { get; set; }
        public int CameraCount { get; set; }
    }

    /// <summary>
    /// Applies the current Review content to column and row charts.
    /// It does not schedule loads, own busy state, or decode files.
    /// </summary>
    internal sealed class ReviewChartPresenter
    {
        private readonly ReviewChartContext _ctx;
        private readonly ReviewDisplayContent _content;
        private readonly ReviewPeriodDataLoader _periodDataLoader;

        public ReviewChartPresenter(
            ReviewChartContext context,
            ReviewDisplayContent content,
            ReviewPeriodDataLoader periodDataLoader)
        {
            _ctx = context;
            _content = content;
            _periodDataLoader = periodDataLoader;
        }

        public Func<double[]> SameSourceViewRange { get; set; }

        public event Action<float[][], float[][], double[], double[], float, float> CurvesUpdated;

        public void ApplyRowPhysicalScale(CsvConfigSnapshot config)
        {
            if (_ctx.RowChartSync == null) return;
            RowCurvePhysicalScale scale = RowCurvePhysicalScaleResolver.Resolve(config, _ctx.Settings);
            _ctx.RowChartSync.SetRowPitchFromSpeed(scale.SpeedMPerMin, scale.LineRateHz);
        }

        public void UpdateGlobalRowChart()
        {
            if (_ctx.RowChartSync == null ||
                (_content.RowMean == null && _content.MergedRowMean == null)) return;

            var stopwatch = Stopwatch.StartNew();
            try { UpdateGlobalRowChartBody(); }
            finally
            {
                if (stopwatch.ElapsedMilliseconds > 50)
                    FlowTrace.Log($"[UiSlow] RvRowChart {stopwatch.ElapsedMilliseconds}ms");
            }
        }

        private void UpdateGlobalRowChartBody()
        {
            float[] mergedMean;
            float[] mergedMax;
            if (_content.MergedRowMean != null)
            {
                mergedMean = (float[])_content.MergedRowMean.Clone();
                mergedMax = _content.MergedRowMax == null
                    ? null
                    : (float[])_content.MergedRowMax.Clone();
            }
            else
            {
                CurveMergeHelper.MergeRowCurvesOverlap(
                    _content.RowMean, _content.RowMax,
                    _ctx.CameraCount, out mergedMean, out mergedMax);
            }

            if (mergedMean == null) return;
            float captureHm = _ctx.ReviewState.Config?.HessianMaxFactorV ??
                              _ctx.Settings.HessianMaxFactorV;
            HessianRescaleHelper.RescaleInPlace1D(
                mergedMean, captureHm, _ctx.Settings.HessianMaxFactorH);
            HessianRescaleHelper.RescaleInPlace1D(
                mergedMax, captureHm, _ctx.Settings.HessianMaxFactorH);
            _ctx.RowChartSync.UpdateData(mergedMean, mergedMax, requireViewRange: true);
        }

        public void UpdateStitchedOverviewChart(bool notifyData)
        {
            if (_content.ColumnMean == null) return;

            var stopwatch = Stopwatch.StartNew();
            try { UpdateStitchedOverviewChartBody(notifyData); }
            finally
            {
                if (stopwatch.ElapsedMilliseconds > 50)
                    FlowTrace.Log($"[UiSlow] RvOverviewChart {stopwatch.ElapsedMilliseconds}ms");
            }
        }

        private void UpdateStitchedOverviewChartBody(bool notifyData)
        {
            double[] ops;
            double[] positions;
            float captureHm;
            CsvConfigSnapshot config = _ctx.ReviewState.Config;
            if (config != null)
            {
                ops = config.CamOps;
                positions = config.CamPos;
                captureHm = config.HessianMaxFactorV;
            }
            else
            {
                ops = _ctx.Settings.GetCameraOpsUmArray();
                positions = _ctx.Settings.GetCameraStartPositionMmArray();
                captureHm = _ctx.Settings.HessianMaxFactorV;
            }

            float errorMean = _ctx.Settings.ErrorValueMeanV;
            float errorMax = _ctx.Settings.ErrorValueMaxV;
            float[][] displayMean = HessianRescaleHelper.CloneAndRescale2D(
                _content.ColumnMean, captureHm, _ctx.Settings.HessianMaxFactorV);
            float[][] displayMax = HessianRescaleHelper.CloneAndRescale2D(
                _content.ColumnMax, captureHm, _ctx.Settings.HessianMaxFactorV);

            CurveMergeHelper.UpdateOverviewChart(
                displayMean, displayMax, ops, positions, errorMean, errorMax,
                _ctx.OverviewHelper, _ctx.CameraCount, _ctx.Settings.StitchMode,
                ViewRangeProvider,
                trimHeadMm: config?.TrimHeadMm ?? _ctx.Settings.TrimHeadMm,
                trimTailMm: config?.TrimTailMm ?? _ctx.Settings.TrimTailMm);

            if (notifyData)
                CurvesUpdated?.Invoke(
                    displayMean, displayMax, ops, positions, errorMean, errorMax);
        }

        public void RefreshChartsForSettingsChange()
        {
            if (!_content.HasImages) return;
            UpdateGlobalRowChart();
        }

        public void UpdateOverviewChart(DateTime? period)
        {
            if (_ctx.OverviewHelper == null || _content.HasImages) return;
            Dictionary<int, string> images = GetPeriodImages(period);
            if (images == null || images.Count == 0)
            {
                _ctx.OverviewChart.Series["Mean"].Points.Clear();
                _ctx.OverviewChart.Series["Max"].Points.Clear();
                if (_ctx.OverviewChart.ChartAreas.Count > 0)
                    _ctx.OverviewChart.ChartAreas[0].AxisX.ScaleView.ZoomReset();
                return;
            }

            ReviewPeriodColumnCurves curves = _periodDataLoader.LoadColumnCurves(
                images, _ctx.CameraCount);
            CsvConfigSnapshot config = _ctx.ReviewState.Config;
            double[] ops = config?.CamOps ?? _ctx.Settings.GetCameraOpsUmArray();
            double[] positions = config?.CamPos ??
                                 _ctx.Settings.GetCameraStartPositionMmArray();
            float errorMean = config?.ErrorValueMeanV ?? _ctx.Settings.ErrorValueMeanV;
            float errorMax = config?.ErrorValueMaxV ?? _ctx.Settings.ErrorValueMaxV;
            CurveMergeHelper.UpdateOverviewChart(
                curves.Mean, curves.Max, ops, positions, errorMean, errorMax,
                _ctx.OverviewHelper, _ctx.CameraCount, _ctx.Settings.StitchMode,
                ViewRangeProvider,
                trimHeadMm: config?.TrimHeadMm ?? _ctx.Settings.TrimHeadMm,
                trimTailMm: config?.TrimTailMm ?? _ctx.Settings.TrimTailMm);
        }

        public void UpdateRowChart(DateTime? period)
        {
            if (_ctx.RowChartSync == null || _content.HasImages) return;
            Dictionary<int, string> images = GetPeriodImages(period);
            if (images == null || images.Count == 0) return;

            ReviewPeriodRowCurves curves = _periodDataLoader.LoadMergedRowCurves(
                images, _ctx.CameraCount);
            if (curves.Mean == null) return;

            ApplyRowPhysicalScale(_ctx.ReviewState.Config);
            float captureHm = _ctx.ReviewState.Config?.HessianMaxFactorV ??
                              _ctx.Settings.HessianMaxFactorV;
            HessianRescaleHelper.RescaleInPlace1D(
                curves.Mean, captureHm, _ctx.Settings.HessianMaxFactorH);
            HessianRescaleHelper.RescaleInPlace1D(
                curves.Max, captureHm, _ctx.Settings.HessianMaxFactorH);
            _ctx.RowChartSync.UpdateData(
                curves.Mean, curves.Max, requireViewRange: true);
        }

        private Dictionary<int, string> GetPeriodImages(DateTime? period)
        {
            if (period.HasValue)
                return _ctx.ImageRepository.GetImages(period.Value);
            return _ctx.ImageRepository.GetImages(
                _ctx.DateTimeNavigator.GetCurrentYear(),
                _ctx.DateTimeNavigator.GetCurrentMonth(),
                _ctx.DateTimeNavigator.GetCurrentDay(),
                _ctx.DateTimeNavigator.GetCurrentHour(),
                _ctx.DateTimeNavigator.GetCurrentMin(),
                _ctx.DateTimeNavigator.GetCurrentSec());
        }

        private double ViewRangeProvider(int cameraIndex, bool isLeft, double defaultValue)
        {
            double[] view = SameSourceViewRange?.Invoke();
            return view == null ? defaultValue : isLeft ? view[0] : view[1];
        }
    }
}
