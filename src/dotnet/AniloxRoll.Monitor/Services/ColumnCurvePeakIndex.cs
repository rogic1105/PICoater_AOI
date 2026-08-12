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
        /// <summary>Peak of the persisted MeanC curve divided by 255; capture HM is not applied.</summary>
        public float RawMeanPeak { get; set; }
        /// <summary>Peak of the persisted MaxC curve divided by 255; capture HM is not applied.</summary>
        public float RawMaxPeak { get; set; }
    }

    internal sealed class ColumnCurvePeakIndexResult
    {
        public Dictionary<string, ColumnCurvePeakRecord[]> ByGrabId { get; } =
            new Dictionary<string, ColumnCurvePeakRecord[]>(StringComparer.Ordinal);
        public Dictionary<string, RowCurvePeakRecord> RowByGrabId { get; } =
            new Dictionary<string, RowCurvePeakRecord>(StringComparer.Ordinal);

        public int RequestedGrabCount { get; set; }
        public int CacheGrabCount { get; set; }
        public int CacheDayCount { get; set; }
        public long CacheLoadMilliseconds { get; set; }
        public int SummaryGrabCount { get; set; }
        public int BinFallbackGrabCount { get; set; }
        public int MissingGrabCount { get; set; }
        public int CameraCount { get; set; }
        public long ElapsedMilliseconds { get; set; }
        public List<GrabIdInfo> PendingBinGrabInfos { get; } = new List<GrabIdInfo>();
    }

    internal sealed class RowCurvePeakRecord
    {
        public string GrabId { get; set; }
        public float CaptureHmV { get; set; }
        public float RawMeanPeak { get; set; }
        public float RawMaxPeak { get; set; }
    }

    /// <summary>
    /// Builds a compact view-time verdict index from persisted merged curves.
    /// The index stores only scalar peaks, so later threshold changes do not reread curves.
    /// </summary>
    internal static class ColumnCurvePeakIndex
    {
        public static ColumnCurvePeakIndexResult BuildAndStoreSummaryProjection(
            string root,
            GrabIdInfo info,
            CsvConfigSnapshot config,
            SingleGrabCurveSummary summary,
            int cameraCount)
        {
            var result = new ColumnCurvePeakIndexResult
            {
                RequestedGrabCount = info == null ? 0 : 1
            };
            if (string.IsNullOrWhiteSpace(root) || info == null ||
                string.IsNullOrWhiteSpace(info.GrabId) || summary == null ||
                cameraCount <= 0)
                return result;

            var watch = Stopwatch.StartNew();
            var captureHmVByGrabId = new Dictionary<string, float>(StringComparer.Ordinal)
            {
                [info.GrabId] = config?.HessianMaxFactorV ?? 0f
            };
            var configByGrabId = new Dictionary<string, CsvConfigSnapshot>(StringComparer.Ordinal);
            if (config != null)
                configByGrabId[info.GrabId] = config;

            int columnCount = AddRecords(
                result, info.GrabId, summary.Mean, summary.Max,
                captureHmVByGrabId, configByGrabId, cameraCount);
            bool hasRow = AddRowRecord(
                result, info.GrabId, summary.RowMean, summary.RowMax,
                captureHmVByGrabId, configByGrabId);
            if (columnCount > 0 || hasRow)
            {
                bool saved = CurvePeakProjectionIndexStore.MergeSave(
                    root, new[] { info }, configByGrabId, cameraCount, result);
                result.SummaryGrabCount = saved ? 1 : 0;
                result.MissingGrabCount = saved ? 0 : 1;
            }
            else
            {
                result.MissingGrabCount = 1;
            }

            result.ElapsedMilliseconds = watch.ElapsedMilliseconds;
            return result;
        }

        public static ColumnCurvePeakIndexResult Build(
            string root,
            IList<GrabIdInfo> grabInfos,
            IDictionary<string, float> captureHmVByGrabId,
            IDictionary<string, CsvConfigSnapshot> configByGrabId,
            int cameraCount,
            CancellationToken cancellationToken)
        {
            var watch = Stopwatch.StartNew();
            ColumnCurvePeakIndexResult summaries = BuildSummaries(
                root, grabInfos, captureHmVByGrabId, configByGrabId,
                cameraCount, cancellationToken);
            ColumnCurvePeakIndexResult bins = BuildBinFallback(
                root, summaries.PendingBinGrabInfos, captureHmVByGrabId,
                configByGrabId, cameraCount, cancellationToken);

            MergeInto(summaries, bins);
            summaries.PendingBinGrabInfos.Clear();
            summaries.ElapsedMilliseconds = watch.ElapsedMilliseconds;
            return summaries;
        }

        public static ColumnCurvePeakIndexResult BuildSummaries(
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
            var cacheWatch = Stopwatch.StartNew();
            Dictionary<string, CurvePeakProjectionEntry> cached =
                CurvePeakProjectionIndexStore.LoadForGrabIds(
                    root, grabInfos, cameraCount, out int cacheDayCount);
            cacheWatch.Stop();
            result.CacheDayCount = cacheDayCount;
            result.CacheLoadMilliseconds = cacheWatch.ElapsedMilliseconds;
            int newlyProjected = 0;
            foreach (GrabIdInfo info in grabInfos)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (info == null || string.IsNullOrWhiteSpace(info.GrabId)) continue;
                string configKey = CurvePeakProjectionIndexStore.GetConfigKey(
                    configByGrabId, info.GrabId);
                if (cached.TryGetValue(info.GrabId, out CurvePeakProjectionEntry cachedEntry) &&
                    cachedEntry.Matches(info, configKey, cameraCount))
                {
                    AddCachedRecords(result, cachedEntry);
                    result.CacheGrabCount++;
                    result.SummaryGrabCount++;
                    continue;
                }
                if (!SingleGrabCurveSummaryStore.TryLoad(
                    root, info, cameraCount, out SingleGrabCurveSummary summary))
                {
                    result.PendingBinGrabInfos.Add(info);
                    continue;
                }

                int columnCount = AddRecords(
                    result, info.GrabId, summary.Mean, summary.Max,
                    captureHmVByGrabId, configByGrabId, cameraCount);
                bool hasRow = AddRowRecord(
                    result, info.GrabId, summary.RowMean, summary.RowMax,
                    captureHmVByGrabId, configByGrabId);
                if (columnCount > 0 || hasRow)
                {
                    result.SummaryGrabCount++;
                    newlyProjected++;
                }
                else
                    result.PendingBinGrabInfos.Add(info);
            }

            if (newlyProjected > 0)
            {
                CurvePeakProjectionIndexStore.MergeSave(
                    root, grabInfos, configByGrabId, cameraCount, result);
            }

            result.ElapsedMilliseconds = watch.ElapsedMilliseconds;
            return result;
        }

        public static ColumnCurvePeakIndexResult BuildBinFallback(
            string root,
            IList<GrabIdInfo> grabInfos,
            IDictionary<string, float> captureHmVByGrabId,
            IDictionary<string, CsvConfigSnapshot> configByGrabId,
            int cameraCount,
            CancellationToken cancellationToken,
            Action<ColumnCurvePeakIndexResult> progress = null,
            int progressBatchSize = 8)
        {
            var result = new ColumnCurvePeakIndexResult
            {
                RequestedGrabCount = grabInfos?.Count ?? 0
            };
            if (string.IsNullOrWhiteSpace(root) || grabInfos == null || cameraCount <= 0)
                return result;

            var watch = Stopwatch.StartNew();
            ColumnCurvePeakIndexResult progressBatch = null;
            int progressGrabCount = 0;
            if (grabInfos.Count > 0)
            {
                Dictionary<string, Dictionary<int, List<string>>> pathsByGrab =
                    InspectionImagePathRepository.LoadForGrabIds(root, grabInfos);
                foreach (GrabIdInfo info in grabInfos)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (!pathsByGrab.TryGetValue(info.GrabId, out var grouped))
                    {
                        result.MissingGrabCount++;
                        continue;
                    }

                    var columnMean = new float[cameraCount][];
                    var columnMax = new float[cameraCount][];
                    var rowMean = new float[cameraCount][];
                    var rowMax = new float[cameraCount][];
                    var alignment = FrameTickIndex.ResolveAlignment(grouped);
                    for (int i = 0; i < cameraCount; i++)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        if (!grouped.TryGetValue(i + 1, out List<string> paths) || paths.Count == 0)
                            continue;
                        CurveMergeHelper.MergeCurves(
                            paths, out columnMean[i], out columnMax[i], cancellationToken);
                        List<string> aligned = alignment.ByCamera.TryGetValue(
                            i + 1, out List<string> alignedPaths)
                            ? alignedPaths
                            : paths;
                        CurveMergeHelper.MergeRowCurves(
                            aligned, out rowMean[i], out rowMax[i]);
                    }
                    CurveMergeHelper.MergeRowCurvesOverlap(
                        rowMean, rowMax, cameraCount,
                        out float[] mergedRowMean, out float[] mergedRowMax);
                    int columnCount = AddRecords(
                        result, info.GrabId, columnMean, columnMax,
                        captureHmVByGrabId, configByGrabId, cameraCount);
                    bool hasRow = AddRowRecord(
                        result, info.GrabId, mergedRowMean, mergedRowMax,
                        captureHmVByGrabId, configByGrabId);
                    if (columnCount > 0 || hasRow)
                    {
                        result.BinFallbackGrabCount++;
                        if (progress != null)
                        {
                            if (progressBatch == null)
                                progressBatch = new ColumnCurvePeakIndexResult();
                            if (result.ByGrabId.TryGetValue(
                                info.GrabId, out ColumnCurvePeakRecord[] records))
                            {
                                progressBatch.ByGrabId[info.GrabId] = records;
                            }
                            if (result.RowByGrabId.TryGetValue(
                                info.GrabId, out RowCurvePeakRecord rowRecord))
                            {
                                progressBatch.RowByGrabId[info.GrabId] = rowRecord;
                            }
                            progressBatch.BinFallbackGrabCount++;
                            progressBatch.CameraCount += columnCount;
                            progressGrabCount++;
                            if (progressGrabCount >= Math.Max(1, progressBatchSize))
                            {
                                progress(progressBatch);
                                progressBatch = null;
                                progressGrabCount = 0;
                            }
                        }
                    }
                    else
                        result.MissingGrabCount++;
                }
            }

            if (progressBatch != null)
                progress(progressBatch);

            if (result.BinFallbackGrabCount > 0)
            {
                CurvePeakProjectionIndexStore.MergeSave(
                    root, grabInfos, configByGrabId, cameraCount, result);
            }

            result.ElapsedMilliseconds = watch.ElapsedMilliseconds;
            return result;
        }

        private static void AddCachedRecords(
            ColumnCurvePeakIndexResult result,
            CurvePeakProjectionEntry entry)
        {
            result.ByGrabId[entry.GrabId] = entry.Columns;
            int count = 0;
            for (int i = 0; i < entry.Columns.Length; i++)
                if (entry.Columns[i] != null) count++;
            result.CameraCount += count;
            if (entry.Row != null)
                result.RowByGrabId[entry.GrabId] = entry.Row;
        }

        private static void MergeInto(
            ColumnCurvePeakIndexResult target,
            ColumnCurvePeakIndexResult source)
        {
            if (target == null || source == null) return;
            foreach (var entry in source.ByGrabId)
                target.ByGrabId[entry.Key] = entry.Value;
            foreach (var entry in source.RowByGrabId)
                target.RowByGrabId[entry.Key] = entry.Value;
            target.SummaryGrabCount += source.SummaryGrabCount;
            target.CacheGrabCount += source.CacheGrabCount;
            target.CacheDayCount += source.CacheDayCount;
            target.CacheLoadMilliseconds += source.CacheLoadMilliseconds;
            target.BinFallbackGrabCount += source.BinFallbackGrabCount;
            target.MissingGrabCount += source.MissingGrabCount;
            target.CameraCount += source.CameraCount;
        }

        private static bool AddRowRecord(
            ColumnCurvePeakIndexResult result,
            string grabId,
            float[] mean,
            float[] max,
            IDictionary<string, float> captureHmVByGrabId,
            IDictionary<string, CsvConfigSnapshot> configByGrabId)
        {
            float meanPeak = ThresholdContext.FindPeakNormalized(mean);
            float maxPeak = ThresholdContext.FindPeakNormalized(max);
            if (float.IsNaN(meanPeak) && float.IsNaN(maxPeak)) return false;

            float captureHmV = 0f;
            captureHmVByGrabId?.TryGetValue(grabId, out captureHmV);
            if (configByGrabId != null &&
                configByGrabId.TryGetValue(grabId, out CsvConfigSnapshot config) &&
                config?.HessianMaxFactorV > 0f)
            {
                captureHmV = config.HessianMaxFactorV;
            }
            result.RowByGrabId[grabId] = new RowCurvePeakRecord
            {
                GrabId = grabId,
                CaptureHmV = captureHmV,
                RawMeanPeak = meanPeak,
                RawMaxPeak = maxPeak
            };
            return true;
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
                    RawMeanPeak = meanPeak,
                    RawMaxPeak = maxPeak
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
                    RawMeanPeak = meanPeaks[i],
                    RawMaxPeak = maxPeaks[i]
                };
            }
            return records;
        }
    }
}
