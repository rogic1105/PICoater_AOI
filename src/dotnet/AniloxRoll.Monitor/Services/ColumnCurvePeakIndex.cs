using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using AniloxRoll.Monitor.UI.Widgets;
using TanukiCv.Core;

namespace AniloxRoll.Monitor.Core.Services
{
    internal sealed class ColumnCurvePeakRecord
    {
        public string GrabId { get; set; }
        public int CameraId { get; set; }
        public float CaptureHmV { get; set; }
        public float MeanPeak { get; set; }
        public float MaxPeak { get; set; }
    }

    internal sealed class ColumnCurvePeakIndexResult
    {
        public Dictionary<string, ColumnCurvePeakRecord[]> ByGrabId { get; } =
            new Dictionary<string, ColumnCurvePeakRecord[]>(StringComparer.Ordinal);

        public int RequestedGrabCount { get; set; }
        public int SummaryGrabCount { get; set; }
        public int BinFallbackGrabCount { get; set; }
        public int MissingGrabCount { get; set; }
        public int CameraCount { get; set; }
        public long ElapsedMilliseconds { get; set; }
    }

    /// <summary>
    /// Builds a compact view-time verdict index from persisted merged curves.
    /// The index stores only scalar peaks, so later threshold changes do not reread curves.
    /// </summary>
    internal static class ColumnCurvePeakIndex
    {
        public static ColumnCurvePeakIndexResult Build(
            string root,
            IList<GrabIdInfo> grabInfos,
            IDictionary<string, float> captureHmVByGrabId,
            IDictionary<string, CsvConfigSnapshot> configByGrabId,
            int cameraCount,
            CancellationToken cancellationToken)
        {
            var result = new ColumnCurvePeakIndexResult
            {
                RequestedGrabCount = grabInfos?.Count ?? 0
            };
            if (string.IsNullOrWhiteSpace(root) || grabInfos == null || cameraCount <= 0)
                return result;

            var watch = Stopwatch.StartNew();
            var missingSummaries = new List<GrabIdInfo>();
            foreach (GrabIdInfo info in grabInfos)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (info == null || string.IsNullOrWhiteSpace(info.GrabId)) continue;
                if (!SingleGrabCurveSummaryStore.TryLoad(
                    root, info, cameraCount, out SingleGrabCurveSummary summary))
                {
                    missingSummaries.Add(info);
                    continue;
                }

                if (AddRecords(result, info.GrabId, summary.Mean, summary.Max,
                    captureHmVByGrabId, configByGrabId, cameraCount) > 0)
                    result.SummaryGrabCount++;
                else
                    missingSummaries.Add(info);
            }

            if (missingSummaries.Count > 0)
            {
                Dictionary<string, Dictionary<int, List<string>>> pathsByGrab =
                    InspectionImagePathRepository.LoadForGrabIds(root, missingSummaries);
                foreach (GrabIdInfo info in missingSummaries)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (!pathsByGrab.TryGetValue(info.GrabId, out var grouped))
                    {
                        result.MissingGrabCount++;
                        continue;
                    }

                    var columnMean = new float[cameraCount][];
                    var columnMax = new float[cameraCount][];
                    for (int i = 0; i < cameraCount; i++)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        if (!grouped.TryGetValue(i + 1, out List<string> paths) || paths.Count == 0)
                            continue;
                        CurveMergeHelper.MergeCurves(
                            paths, out columnMean[i], out columnMax[i], cancellationToken);
                    }
                    if (AddRecords(result, info.GrabId, columnMean, columnMax,
                        captureHmVByGrabId, configByGrabId, cameraCount) > 0)
                        result.BinFallbackGrabCount++;
                    else
                        result.MissingGrabCount++;
                }
            }

            result.ElapsedMilliseconds = watch.ElapsedMilliseconds;
            return result;
        }

        private static int AddRecords(
            ColumnCurvePeakIndexResult result,
            string grabId,
            float[][] means,
            float[][] maxes,
            IDictionary<string, float> captureHmVByGrabId,
            IDictionary<string, CsvConfigSnapshot> configByGrabId,
            int cameraCount)
        {
            float captureHmV = 0f;
            captureHmVByGrabId?.TryGetValue(grabId, out captureHmV);
            CsvConfigSnapshot config = null;
            configByGrabId?.TryGetValue(grabId, out config);
            ColumnCurvePeakRecord[] projected = ProjectVisibleRecords(
                grabId, means, maxes, config, captureHmV, cameraCount);
            if (projected != null)
            {
                int projectedCount = 0;
                for (int i = 0; i < projected.Length; i++)
                    if (projected[i] != null) projectedCount++;
                if (projectedCount > 0)
                {
                    result.ByGrabId[grabId] = projected;
                    result.CameraCount += projectedCount;
                    return projectedCount;
                }
            }

            var records = new ColumnCurvePeakRecord[cameraCount];
            int matched = 0;
            for (int i = 0; i < cameraCount; i++)
            {
                float meanPeak = ThresholdContext.FindPeakNormalized(
                    means != null && i < means.Length ? means[i] : null);
                float maxPeak = ThresholdContext.FindPeakNormalized(
                    maxes != null && i < maxes.Length ? maxes[i] : null);
                if (float.IsNaN(meanPeak) && float.IsNaN(maxPeak)) continue;

                records[i] = new ColumnCurvePeakRecord
                {
                    GrabId = grabId,
                    CameraId = i + 1,
                    CaptureHmV = captureHmV,
                    MeanPeak = meanPeak,
                    MaxPeak = maxPeak
                };
                matched++;
            }

            if (matched == 0) return 0;
            result.ByGrabId[grabId] = records;
            result.CameraCount += matched;
            return matched;
        }

        internal static ColumnCurvePeakRecord[] ProjectVisibleRecords(
            string grabId,
            float[][] means,
            float[][] maxes,
            CsvConfigSnapshot config,
            float fallbackCaptureHmV,
            int cameraCount)
        {
            if (config == null || means == null || cameraCount <= 0 ||
                config.CamOps == null || config.CamPos == null ||
                config.CamOps.Length < cameraCount || config.CamPos.Length < cameraCount)
                return null;

            CurveOverviewMerger.Result merged = CurveOverviewMerger.Merge(
                means, maxes, config.CamOps, config.CamPos,
                cameraCount, MergeOverlap.Midline);
            if (!merged.Valid || merged.OwnerCameraIndices == null ||
                merged.Mean == null || merged.Max == null)
                return null;

            var meanPeaks = new float[cameraCount];
            var maxPeaks = new float[cameraCount];
            var hasData = new bool[cameraCount];
            int count = Math.Min(
                merged.OwnerCameraIndices.Length,
                Math.Min(merged.Mean.Length, merged.Max.Length));
            for (int i = 0; i < count; i++)
            {
                int cameraIndex = merged.OwnerCameraIndices[i];
                if (cameraIndex < 0 || cameraIndex >= cameraCount) continue;
                float mean = merged.Mean[i] / 255f;
                float max = merged.Max[i] / 255f;
                if (!hasData[cameraIndex] || mean > meanPeaks[cameraIndex])
                    meanPeaks[cameraIndex] = mean;
                if (!hasData[cameraIndex] || max > maxPeaks[cameraIndex])
                    maxPeaks[cameraIndex] = max;
                hasData[cameraIndex] = true;
            }

            float captureHmV = config.HessianMaxFactorV > 0f
                ? config.HessianMaxFactorV
                : fallbackCaptureHmV;
            var records = new ColumnCurvePeakRecord[cameraCount];
            for (int i = 0; i < cameraCount; i++)
            {
                if (!hasData[i]) continue;
                records[i] = new ColumnCurvePeakRecord
                {
                    GrabId = grabId,
                    CameraId = i + 1,
                    CaptureHmV = captureHmV,
                    MeanPeak = meanPeaks[i],
                    MaxPeak = maxPeaks[i]
                };
            }
            return records;
        }
    }
}
