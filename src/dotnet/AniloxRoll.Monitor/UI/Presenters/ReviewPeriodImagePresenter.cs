using System;
using System.Collections.Generic;
using System.Drawing;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Services;
using AniloxRoll.Monitor.UI.State;
using AniloxRoll.Monitor.UI.Widgets;

namespace AniloxRoll.Monitor.UI.Presenters
{
    internal sealed class ReviewPeriodImageContext
    {
        public ReviewRuntimeState ReviewState { get; set; }
        public InspectionSettings Settings { get; set; }
        public ImageRepository ImageRepository { get; set; }
        public DateTimeNavigator DateTimeNavigator { get; set; }
        public BatchInspectionService InspectionService { get; set; }
        public int CameraCount { get; set; }
        public Action<byte[][], int[], int[], double[], double[], bool, bool> PublishFrames { get; set; }
    }

    /// <summary>
    /// Owns period-image lookup, decoding and publication. It deliberately does not update
    /// Review charts; chart presentation remains with ReviewChartPresenter.
    /// </summary>
    internal sealed class ReviewPeriodImagePresenter
    {
        private readonly ReviewPeriodImageContext _ctx;
        private readonly ReviewPeriodDataLoader _loader;

        public ReviewPeriodImagePresenter(
            ReviewPeriodImageContext context,
            ReviewPeriodDataLoader loader)
        {
            _ctx = context ?? throw new ArgumentNullException(nameof(context));
            _loader = loader ?? throw new ArgumentNullException(nameof(loader));
        }

        public void Apply(
            DateTime? period,
            bool preserveChartView,
            bool processedMode,
            string ridgeDirection)
        {
            if (_ctx.Settings.StitchMode != StitchMode.Global) return;

            CsvConfigSnapshot config = _ctx.ReviewState.Config;
            double[] cameraOps = config?.CamOps ?? _ctx.Settings.GetCameraOpsUmArray();
            double[] cameraPositions = config?.CamPos ??
                _ctx.Settings.GetCameraStartPositionMmArray();
            int scale = InspectionEngineConfig.DefaultSaveResizeScale;

            Dictionary<int, string> files = GetImages(period);
            if (files == null || files.Count == 0) return;
            FlowTrace.Log(
                $"RV period load {GetPeriodLabel(period)} images={files.Count}/{_ctx.CameraCount} " +
                $"proc={processedMode} cfg={(config != null ? "yes" : "no")}");

            Func<string, Bitmap> bitmapLoader = _ctx.InspectionService != null
                ? (Func<string, Bitmap>)(path =>
                    _ctx.InspectionService.LoadBmpAtScale(path, scale))
                : null;
            ReviewPeriodFrames frames = _loader.LoadFrames(
                files, _ctx.CameraCount, scale, bitmapLoader,
                processedMode, ridgeDirection);
            _ctx.PublishFrames?.Invoke(
                frames.GrayFrames, frames.Widths, frames.Heights,
                cameraOps, cameraPositions, true, preserveChartView);
        }

        private Dictionary<int, string> GetImages(DateTime? period)
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

        private string GetPeriodLabel(DateTime? period)
        {
            if (period.HasValue)
                return period.Value.ToString("yyyy-MM-dd HH:mm:ss.fff");

            return $"{_ctx.DateTimeNavigator.GetCurrentYear()}-" +
                $"{_ctx.DateTimeNavigator.GetCurrentMonth()}-" +
                $"{_ctx.DateTimeNavigator.GetCurrentDay()} " +
                $"{_ctx.DateTimeNavigator.GetCurrentHour()}:" +
                $"{_ctx.DateTimeNavigator.GetCurrentMin()}:" +
                $"{_ctx.DateTimeNavigator.GetCurrentSec()}";
        }
    }
}
