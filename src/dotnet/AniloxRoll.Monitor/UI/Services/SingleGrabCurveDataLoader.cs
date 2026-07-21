using System;
using System.Threading;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Widgets;

namespace AniloxRoll.Monitor.UI.Services
{
    internal sealed class SingleGrabCurveData
    {
        public float[][] ColumnMean { get; set; }
        public float[][] ColumnMax { get; set; }
        public float[][] RowMean { get; set; }
        public float[][] RowMax { get; set; }
        public float[] MergedRowMean { get; set; }
        public float[] MergedRowMax { get; set; }
        public CsvConfigSnapshot Config { get; set; }
        public int ImageCount { get; set; }
        public int MatchedCameraCount { get; set; }
        public string AlignmentMode { get; set; }
        public string StorageSource { get; set; }
        public long ConfigMs { get; set; }
        public long LookupMs { get; set; }
        public long MergeMs { get; set; }
        public long SummaryMs { get; set; }
    }

    /// <summary>
    /// Loads the persisted curve and capture-config data for one grab.
    /// This service owns file lookup, frame alignment, and bin merging; it has no UI state.
    /// </summary>
    internal sealed class SingleGrabCurveDataLoader : IDisposable
    {
        private const int CacheEntries = 512;
        private const long CacheBytes = 256L * 1024 * 1024;
        private readonly SingleGrabCurveCache _cache =
            new SingleGrabCurveCache(CacheEntries, CacheBytes);

        public SingleGrabCurveData Load(
            string root,
            string grabId,
            DateTime hintFrom,
            DateTime hintTo,
            int cameraCount)
        {
            SingleGrabCurveSummaryStore.NotifyReadActivity();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var config = InspectionConfigRepository.LoadForGrabId(
                root, grabId, hintFrom, hintTo);
            long configMs = sw.ElapsedMilliseconds;
            var info = new GrabIdInfo
            {
                GrabId = grabId,
                Earliest = hintFrom,
                Latest = hintTo
            };
            string cacheKey = SingleGrabCurveCache.BuildKey(root, info, cameraCount);
            bool cacheHit = _cache.TryGet(cacheKey, out SingleGrabCurveProfile profile);
            if (!cacheHit)
            {
                profile = _cache.GetOrLoadAsync(cacheKey,
                    () => LoadProfile(root, info, cameraCount))
                    .GetAwaiter().GetResult();
            }

            return new SingleGrabCurveData
            {
                ColumnMean = profile.Mean,
                ColumnMax = profile.Max,
                MergedRowMean = profile.RowMean,
                MergedRowMax = profile.RowMax,
                Config = config,
                ImageCount = profile.CaptureCount,
                MatchedCameraCount = profile.MatchedCameraCount,
                AlignmentMode = profile.AlignmentMode,
                StorageSource = cacheHit ? "memory-" + profile.StorageSource : profile.StorageSource,
                ConfigMs = configMs,
                LookupMs = profile.LookupMs,
                MergeMs = profile.MergeMs,
                SummaryMs = profile.SummaryMs
            };
        }

        private static SingleGrabCurveProfile LoadProfile(
            string root, GrabIdInfo info, int cameraCount)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            if (SingleGrabCurveSummaryStore.TryLoad(
                root, info, cameraCount, out SingleGrabCurveSummary summary))
            {
                return new SingleGrabCurveProfile(
                    summary.Mean, summary.Max, summary.RowMean, summary.RowMax,
                    summary.CaptureCount, "summary", 0, 0, sw.ElapsedMilliseconds,
                    CountMatchedCameras(summary.Mean), "summary");
            }

            var grouped = InspectionImagePathRepository.LoadForGrabId(
                root, info.GrabId, info.Earliest, info.Latest);
            long lookupMs = sw.ElapsedMilliseconds;
            var alignment = FrameTickIndex.ResolveAlignment(grouped);
            var columnMean = new float[cameraCount][];
            var columnMax = new float[cameraCount][];
            var rowMean = new float[cameraCount][];
            var rowMax = new float[cameraCount][];
            int mergedCaptureCount = 0;

            for (int i = 0; i < cameraCount; i++)
            {
                int cameraId = i + 1;
                if (!grouped.TryGetValue(cameraId, out var paths) || paths.Count == 0)
                    continue;

                CurveMergeHelper.MergeCurves(
                    paths, out columnMean[i], out columnMax[i], out int mergedForCamera,
                    CancellationToken.None);
                mergedCaptureCount += mergedForCamera;
                var aligned = alignment.ByCamera.TryGetValue(cameraId, out var alignedPaths)
                    ? alignedPaths
                    : paths;
                CurveMergeHelper.MergeRowCurves(aligned, out rowMean[i], out rowMax[i]);
            }

            CurveMergeHelper.MergeRowCurvesOverlap(
                rowMean, rowMax, cameraCount,
                out float[] mergedRowMean, out float[] mergedRowMax);
            long mergeMs = sw.ElapsedMilliseconds - lookupMs;
            long summaryStartMs = sw.ElapsedMilliseconds;
            int captureCount = alignment.AllPaths.Count;
            if (captureCount > 0 && mergedCaptureCount == captureCount)
            {
                SingleGrabCurveSummaryStore.QueueSave(
                    root, info, cameraCount,
                    new SingleGrabCurveSummary(
                        columnMean, columnMax, mergedRowMean, mergedRowMax, captureCount));
            }
            return new SingleGrabCurveProfile(
                columnMean, columnMax, mergedRowMean, mergedRowMax,
                captureCount, "bins", lookupMs, mergeMs,
                sw.ElapsedMilliseconds - summaryStartMs,
                grouped.Count, alignment.Mode);
        }

        private static int CountMatchedCameras(float[][] curves)
        {
            int count = 0;
            if (curves == null) return count;
            for (int i = 0; i < curves.Length; i++)
                if (curves[i] != null && curves[i].Length > 0) count++;
            return count;
        }

        public void Clear() => _cache.Clear();

        public void Dispose() => _cache.Dispose();
    }
}
