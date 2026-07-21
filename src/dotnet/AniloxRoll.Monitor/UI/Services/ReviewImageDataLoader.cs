using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Widgets;

namespace AniloxRoll.Monitor.UI.Services
{
    internal sealed class ReviewImageLoadPlan
    {
        public Dictionary<int, List<string>> GroupedPaths { get; set; }
        public FrameAlignmentResult Alignment { get; set; }
        public CsvConfigSnapshot Config { get; set; }
        public int[] ExpectedWidths { get; set; }
        public int[] ExpectedHeights { get; set; }
        public int TotalImageCount { get; set; }
        public long ConfigMs { get; set; }
    }

    internal sealed class ReviewImageData
    {
        public Bitmap[] Images { get; set; }
        public byte[][] GrayFrames { get; set; }
        public int[] GrayWidths { get; set; }
        public int[] GrayHeights { get; set; }
        public float[][] ColumnMean { get; set; }
        public float[][] ColumnMax { get; set; }
        public float[][] RowMean { get; set; }
        public float[][] RowMax { get; set; }
        public CsvConfigSnapshot Config { get; set; }
        public int TotalImageCount { get; set; }
        public long ConfigMs { get; set; }
        public long StitchMs { get; set; }

        public void DisposeImages()
        {
            if (Images == null) return;
            foreach (var image in Images) image?.Dispose();
            Images = null;
        }
    }

    /// <summary>
    /// Loads one Review grab into a background result. Owns filesystem lookup, frame alignment,
    /// image stitching, optional curve-bin reads, and grayscale conversion; never touches WinForms controls.
    /// </summary>
    internal sealed class ReviewImageDataLoader
    {
        public ReviewImageLoadPlan Prepare(
            string root,
            string grabId,
            DateTime hintFrom,
            DateTime hintTo,
            int cameraCount,
            bool enableProcess,
            string ridgeDirection,
            bool logPaths = true)
        {
            var configWatch = Stopwatch.StartNew();
            var grouped = InspectionImagePathRepository.LoadForGrabId(
                root, grabId, hintFrom, hintTo);
            var config = InspectionConfigRepository.LoadForGrabId(
                root, grabId, hintFrom, hintTo);
            long configMs = configWatch.ElapsedMilliseconds;

            int totalImageCount = 0;
            foreach (var camera in grouped) totalImageCount += camera.Value.Count;

            var alignment = FrameTickIndex.ResolveAlignment(grouped);
            var expectedWidths = new int[cameraCount];
            var expectedHeights = new int[cameraCount];
            for (int index = 0; index < cameraCount; index++)
            {
                int cameraId = index + 1;
                if (!alignment.ByCamera.TryGetValue(cameraId, out var aligned)) continue;
                GrabImageStitcher.TryGetStitchedSize(
                    aligned, enableProcess, ridgeDirection,
                    out expectedWidths[index], out expectedHeights[index]);
            }

            if (logPaths)
                FlowTrace.Log($"RV loadGrab paths {grabId} root={root} images={totalImageCount} cams={grouped.Count} cfg={(config != null ? "yes" : "no")} align={alignment.Mode}");
            return new ReviewImageLoadPlan
            {
                GroupedPaths = grouped,
                Alignment = alignment,
                Config = config,
                ExpectedWidths = expectedWidths,
                ExpectedHeights = expectedHeights,
                TotalImageCount = totalImageCount,
                ConfigMs = configMs
            };
        }

        public ReviewImageData Load(
            ReviewImageLoadPlan plan,
            int cameraCount,
            bool enableProcess,
            string ridgeDirection,
            bool includeCurves = true)
        {
            if (plan == null) throw new ArgumentNullException(nameof(plan));
            var grouped = plan.GroupedPaths;
            var config = plan.Config;

            var images = new Bitmap[cameraCount];
            var grayFrames = new byte[cameraCount][];
            var grayWidths = new int[cameraCount];
            var grayHeights = new int[cameraCount];
            var columnMean = includeCurves ? new float[cameraCount][] : null;
            var columnMax = includeCurves ? new float[cameraCount][] : null;
            var rowMean = includeCurves ? new float[cameraCount][] : null;
            var rowMax = includeCurves ? new float[cameraCount][] : null;

            var stitchWatch = Stopwatch.StartNew();
            int scale = InspectionEngineConfig.DefaultSaveResizeScale;
            var alignment = plan.Alignment;
            var alignedByCamera = alignment.ByCamera;

            Parallel.For(0, cameraCount, index =>
            {
                int cameraId = index + 1;
                if (!grouped.TryGetValue(cameraId, out var paths) || paths.Count == 0) return;
                try
                {
                    var aligned = alignedByCamera.TryGetValue(cameraId, out var value)
                        ? value
                        : paths;
                    images[index] = GrabImageStitcher.StitchCamera(
                        aligned, scale, null, enableProcess, ridgeDirection);
                    if (includeCurves)
                    {
                        CurveMergeHelper.MergeCurves(
                            paths, out columnMean[index], out columnMax[index]);
                        CurveMergeHelper.MergeRowCurves(
                            aligned, out rowMean[index], out rowMax[index]);
                    }
                    if (images[index] != null)
                    {
                        grayFrames[index] = BitmapGrayConverter.ToGray8(
                            images[index], out grayWidths[index], out grayHeights[index]);
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine(
                        $"[StitchView] CAM{cameraId}: {ex.GetType().Name}: {ex.Message}");
                }
            });

            return new ReviewImageData
            {
                Images = images,
                GrayFrames = grayFrames,
                GrayWidths = grayWidths,
                GrayHeights = grayHeights,
                ColumnMean = columnMean,
                ColumnMax = columnMax,
                RowMean = rowMean,
                RowMax = rowMax,
                Config = config,
                TotalImageCount = plan.TotalImageCount,
                ConfigMs = plan.ConfigMs,
                StitchMs = stitchWatch.ElapsedMilliseconds
            };
        }

        public ReviewImageData Load(
            string root,
            string grabId,
            DateTime hintFrom,
            DateTime hintTo,
            int cameraCount,
            bool enableProcess,
            string ridgeDirection,
            bool includeCurves = true)
        {
            ReviewImageLoadPlan plan = Prepare(
                root, grabId, hintFrom, hintTo, cameraCount, enableProcess, ridgeDirection);
            return Load(plan, cameraCount, enableProcess, ridgeDirection, includeCurves);
        }
    }
}
