using System;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Widgets;

namespace AniloxRoll.Monitor.UI.Services
{
    internal sealed class ReviewCurveData
    {
        public float[][] ColumnMean { get; set; }
        public float[][] ColumnMax { get; set; }
        public float[][] RowMean { get; set; }
        public float[][] RowMax { get; set; }
        public CsvConfigSnapshot Config { get; set; }
        public int ImageCount { get; set; }
        public int MatchedCameraCount { get; set; }
        public string AlignmentMode { get; set; }
    }

    /// <summary>
    /// Loads the persisted curve and capture-config data for one review grab.
    /// This service owns file lookup, frame alignment, and bin merging; it has no UI state.
    /// </summary>
    internal sealed class ReviewCurveDataLoader
    {
        public ReviewCurveData Load(
            string root,
            string grabId,
            DateTime hintFrom,
            DateTime hintTo,
            int cameraCount)
        {
            var grouped = InspectionImagePathRepository.LoadForGrabId(
                root, grabId, hintFrom, hintTo);
            var config = InspectionConfigRepository.LoadForGrabId(
                root, grabId, hintFrom, hintTo);
            var alignment = FrameTickIndex.ResolveAlignment(grouped);
            var columnMean = new float[cameraCount][];
            var columnMax = new float[cameraCount][];
            var rowMean = new float[cameraCount][];
            var rowMax = new float[cameraCount][];

            for (int i = 0; i < cameraCount; i++)
            {
                int cameraId = i + 1;
                if (!grouped.TryGetValue(cameraId, out var paths) || paths.Count == 0)
                    continue;

                CurveMergeHelper.MergeCurves(paths, out columnMean[i], out columnMax[i]);
                var aligned = alignment.ByCamera.TryGetValue(cameraId, out var alignedPaths)
                    ? alignedPaths
                    : paths;
                CurveMergeHelper.MergeRowCurves(aligned, out rowMean[i], out rowMax[i]);
            }

            return new ReviewCurveData
            {
                ColumnMean = columnMean,
                ColumnMax = columnMax,
                RowMean = rowMean,
                RowMax = rowMax,
                Config = config,
                ImageCount = alignment.AllPaths.Count,
                MatchedCameraCount = grouped.Count,
                AlignmentMode = alignment.Mode
            };
        }
    }
}
