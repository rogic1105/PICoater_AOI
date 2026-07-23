using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>Loads and aggregates persisted Mura column-curve profiles.</summary>
    public static class InspectionMuraProfileRepository
    {
        private const int DailyIndexDayCapacity = 1024;
        private const int DailyIndexRecordCapacity = 250000;
        private static readonly object DailyIndexLock = new object();
        private static readonly Dictionary<string, CachedDailyRecords> DailyIndex =
            new Dictionary<string, CachedDailyRecords>(StringComparer.OrdinalIgnoreCase);
        private static long _dailyIndexAccess;
        private static int _dailyIndexRecords;

        internal const int RangeCurveCacheEntryCapacity = 2048;
        internal const int RangeCurveCacheByteCapacityMb = 256;
        private const long RangeCurveCacheByteCapacity =
            (long)RangeCurveCacheByteCapacityMb * 1024 * 1024;
        private static readonly object RangeCurveCacheLock = new object();
        private static readonly Dictionary<string, CachedCurve> RangeCurveCache =
            new Dictionary<string, CachedCurve>(StringComparer.OrdinalIgnoreCase);
        private static readonly LinkedList<string> RangeCurveLru = new LinkedList<string>();
        private static long _rangeCurveCacheBytes;

        private sealed class MuraCurveRecord
        {
            public int CameraId;
            public string BasePath;
            public float MaxCMean;
        }

        private sealed class CachedDailyRecords
        {
            public long Length;
            public long LastWriteTicks;
            public long LastAccess;
            public int RecordCount;
            public Dictionary<string, List<MuraCurveRecord>> ByGrabId;
        }

        private sealed class CachedCurve
        {
            public float[] Values;
            public long Bytes;
            public LinkedListNode<string> Node;
        }

        public static (Dictionary<int, float[]> Mean, Dictionary<int, float[]> Max)
            LoadAverage(string rootPath, IList<GrabIdInfo> grabIds)
        {
            var accMean = new Dictionary<int, float[]>();
            var accMax = new Dictionary<int, float[]>();
            var counts = new Dictionary<int, int>();

            foreach (var info in grabIds)
            {
                string dateDir = CaptureStoragePaths.DateImageDir(rootPath, info.Earliest);
                if (!Directory.Exists(dateDir)) continue;

                string archivePath = CaptureStoragePaths.GrabArchive(
                    rootPath, info.Earliest, info.GrabId);
                if (File.Exists(archivePath))
                {
                    for (int camId = 1; camId <= 7; camId++)
                    {
                        List<string> archiveFrames = CaptureArchiveStore.ListVirtualRawPaths(
                            archivePath, camId);
                        if (archiveFrames.Count == 0) continue;
                        string archiveBase = CaptureFileNaming.BaseFromImagePath(archiveFrames[0]);
                        float[] mean = CurveBinFile.Load(
                            CaptureFileNaming.ResolveMeanC(archiveBase));
                        float[] max = CurveBinFile.Load(
                            CaptureFileNaming.ResolveMaxC(archiveBase));
                        AccumulateAverage(
                            camId, mean, max, accMean, accMax, counts);
                    }
                    continue;
                }

                string prefix = info.Earliest.ToString("yyyyMMdd_HHmmss");
                for (int camId = 1; camId <= 7; camId++)
                {
                    string[] meanFiles = FindCurveFiles(dateDir, prefix, camId,
                        CaptureFileNaming.MeanC, CaptureFileNaming.MeanCPrevious, CaptureFileNaming.MeanCLegacy);
                    string[] maxFiles = FindCurveFiles(dateDir, prefix, camId,
                        CaptureFileNaming.MaxC, CaptureFileNaming.MaxCPrevious, CaptureFileNaming.MaxCLegacy);
                    if (meanFiles.Length == 0) continue;

                    float[] mean = CurveBinFile.Load(meanFiles[0]);
                    if (mean == null || mean.Length == 0) continue;
                    float[] max = maxFiles.Length > 0 ? CurveBinFile.Load(maxFiles[0]) : null;

                    AccumulateAverage(camId, mean, max, accMean, accMax, counts);
                }
            }

            var resultMean = new Dictionary<int, float[]>();
            var resultMax = new Dictionary<int, float[]>();
            foreach (var camera in accMean)
            {
                int count = counts[camera.Key];
                if (count == 0) continue;
                float[] average = new float[camera.Value.Length];
                for (int i = 0; i < average.Length; i++)
                    average[i] = camera.Value[i] / count;
                resultMean[camera.Key] = average;
                resultMax[camera.Key] = accMax[camera.Key];
            }

            return (resultMean, resultMax);
        }

        private static void AccumulateAverage(
            int camId,
            float[] mean,
            float[] max,
            Dictionary<int, float[]> accMean,
            Dictionary<int, float[]> accMax,
            Dictionary<int, int> counts)
        {
            if (mean == null || mean.Length == 0) return;
            if (!accMean.TryGetValue(camId, out float[] accumulatedMean))
            {
                accMean[camId] = new float[mean.Length];
                accMax[camId] = new float[mean.Length];
                counts[camId] = 0;
            }
            else if (accumulatedMean.Length != mean.Length)
            {
                return;
            }

            float[] sumMean = accMean[camId];
            float[] maxValues = accMax[camId];
            for (int i = 0; i < mean.Length; i++)
            {
                sumMean[i] += mean[i];
                if (max != null && i < max.Length && max[i] > maxValues[i])
                    maxValues[i] = max[i];
            }
            counts[camId]++;
        }

        public static (
            Dictionary<int, float[]> Mean,
            Dictionary<int, float[]> Max,
            int MeanRows,
            int MaxRows,
            int ScoredRows,
            int TotalRows,
            int RankedCams,
            int TotalCams,
            int IndexHits,
            int IndexBuilds,
            int CurveCacheHits,
            int CurveCacheMisses)
            LoadRange(string rootPath, IList<GrabIdInfo> rangeInfos, int limit,
                CancellationToken cancellationToken = default(CancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var meanResult = new Dictionary<int, float[]>();
            var maxResult = new Dictionary<int, float[]>();
            int meanRows = 0, maxRows = 0, scoredRows = 0, totalRows = 0;
            int rankedCams = 0;
            if (string.IsNullOrWhiteSpace(rootPath) || rangeInfos == null ||
                rangeInfos.Count == 0 || limit <= 0)
                return (meanResult, maxResult, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

            var rangeIds = new HashSet<string>(StringComparer.Ordinal);
            var dates = new HashSet<DateTime>();
            foreach (var info in rangeInfos)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (info == null || string.IsNullOrEmpty(info.GrabId)) continue;
                rangeIds.Add(info.GrabId);
                dates.Add(info.Earliest.Date);
            }

            var recordsByCam = new Dictionary<int, List<MuraCurveRecord>>();
            int indexHits = 0, indexBuilds = 0;
            foreach (DateTime date in dates)
            {
                cancellationToken.ThrowIfCancellationRequested();
                string csvPath = CaptureStoragePaths.DailyCsv(rootPath, date);
                CachedDailyRecords daily = GetDailyRecords(
                    csvPath, rootPath, out bool cacheHit);
                if (daily == null) continue;
                if (cacheHit) indexHits++; else indexBuilds++;

                foreach (string grabId in rangeIds)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (!daily.ByGrabId.TryGetValue(grabId, out var indexedRecords)) continue;
                    foreach (MuraCurveRecord indexedRecord in indexedRecords)
                    {
                        if (!recordsByCam.TryGetValue(indexedRecord.CameraId, out var records))
                            recordsByCam[indexedRecord.CameraId] = records = new List<MuraCurveRecord>();
                        records.Add(indexedRecord);
                        totalRows++;
                        if (!float.IsNaN(indexedRecord.MaxCMean) &&
                            !float.IsInfinity(indexedRecord.MaxCMean))
                            scoredRows++;
                    }
                }
            }

            int curveCacheHits = 0, curveCacheMisses = 0;
            object resultLock = new object();
            var parallelOptions = new ParallelOptions
            {
                CancellationToken = cancellationToken,
                MaxDegreeOfParallelism = Math.Min(4, Math.Max(1, recordsByCam.Count))
            };
            Parallel.ForEach(recordsByCam, parallelOptions, camera =>
            {
                List<MuraCurveRecord> records = camera.Value;
                List<MuraCurveRecord> meanCandidates = EvenSample(records, limit);
                var scored = records.FindAll(record =>
                    !float.IsNaN(record.MaxCMean) && !float.IsInfinity(record.MaxCMean));
                List<MuraCurveRecord> maxCandidates;
                if (scored.Count == records.Count && scored.Count > 0)
                {
                    scored.Sort((left, right) => right.MaxCMean.CompareTo(left.MaxCMean));
                    maxCandidates = scored.GetRange(0, Math.Min(limit, scored.Count));
                }
                else
                {
                    maxCandidates = EvenSample(records, limit);
                }

                int localCacheHits = 0, localCacheMisses = 0;
                float[] mean = Aggregate(
                    meanCandidates, true, cancellationToken,
                    ref localCacheHits, ref localCacheMisses);
                float[] max = Aggregate(
                    maxCandidates, false, cancellationToken,
                    ref localCacheHits, ref localCacheMisses);
                lock (resultLock)
                {
                    if (mean != null) meanResult[camera.Key] = mean;
                    if (max != null) maxResult[camera.Key] = max;
                    meanRows += meanCandidates.Count;
                    maxRows += maxCandidates.Count;
                    if (scored.Count == records.Count && scored.Count > 0)
                        rankedCams++;
                    curveCacheHits += localCacheHits;
                    curveCacheMisses += localCacheMisses;
                }
            });

            return (meanResult, maxResult, meanRows, maxRows, scoredRows, totalRows,
                rankedCams, recordsByCam.Count, indexHits, indexBuilds,
                curveCacheHits, curveCacheMisses);
        }

        private static CachedDailyRecords GetDailyRecords(
            string csvPath,
            string rootPath,
            out bool cacheHit)
        {
            cacheHit = false;
            var file = new FileInfo(csvPath);
            if (!file.Exists) return null;
            long length = file.Length;
            long lastWriteTicks = file.LastWriteTimeUtc.Ticks;
            string key = file.FullName;

            lock (DailyIndexLock)
            {
                if (DailyIndex.TryGetValue(key, out var cached) &&
                    cached.Length == length && cached.LastWriteTicks == lastWriteTicks)
                {
                    cached.LastAccess = ++_dailyIndexAccess;
                    cacheHit = true;
                    return cached;
                }
            }

            var byGrabId = new Dictionary<string, List<MuraCurveRecord>>(StringComparer.Ordinal);
            int recordCount = 0;
            try
            {
                using (var reader = InspectionCsvReader.OpenShared(csvPath))
                {
                    string line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        if (!InspectionCsvReader.TryParseRecord(line, out var record) ||
                            !InspectionCsvReader.TryExtractCameraId(record.FileName, out int camId) ||
                            !InspectionCsvReader.TryParseTimestamp(record.FileName, out DateTime timestamp))
                            continue;

                        if (!byGrabId.TryGetValue(record.GrabId, out var records))
                            byGrabId[record.GrabId] = records = new List<MuraCurveRecord>();
                        string basePath = Path.Combine(
                            CaptureStoragePaths.DateImageDir(rootPath, timestamp), record.FileName);
                        string archivePath = CaptureStoragePaths.GrabArchive(
                            rootPath, timestamp, record.GrabId);
                        if (File.Exists(archivePath))
                        {
                            string virtualRaw = CaptureArchiveStore.CreateVirtualRawPath(
                                archivePath, record.FileName);
                            if (CaptureArchiveStore.Exists(virtualRaw))
                                basePath = CaptureFileNaming.BaseFromImagePath(virtualRaw);
                        }
                        records.Add(new MuraCurveRecord
                        {
                            CameraId = camId,
                            BasePath = basePath,
                            MaxCMean = record.MaxCMean
                        });
                        recordCount++;
                    }
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[InspectionMuraProfileRepository.LoadRange] {csvPath}: {ex.GetType().Name}: {ex.Message}");
                return null;
            }

            var built = new CachedDailyRecords
            {
                Length = length,
                LastWriteTicks = lastWriteTicks,
                RecordCount = recordCount,
                ByGrabId = byGrabId
            };
            lock (DailyIndexLock)
            {
                built.LastAccess = ++_dailyIndexAccess;
                if (DailyIndex.TryGetValue(key, out var replaced))
                    _dailyIndexRecords -= replaced.RecordCount;
                DailyIndex[key] = built;
                _dailyIndexRecords += built.RecordCount;
                TrimDailyIndex();
            }
            return built;
        }

        private static void TrimDailyIndex()
        {
            while (DailyIndex.Count > 1 &&
                (DailyIndex.Count > DailyIndexDayCapacity ||
                 _dailyIndexRecords > DailyIndexRecordCapacity))
            {
                string oldestKey = null;
                long oldestAccess = long.MaxValue;
                foreach (var item in DailyIndex)
                {
                    if (item.Value.LastAccess >= oldestAccess) continue;
                    oldestAccess = item.Value.LastAccess;
                    oldestKey = item.Key;
                }
                if (oldestKey == null) return;
                _dailyIndexRecords -= DailyIndex[oldestKey].RecordCount;
                DailyIndex.Remove(oldestKey);
            }
        }

        private static List<MuraCurveRecord> EvenSample(List<MuraCurveRecord> records, int limit)
        {
            if (records.Count <= limit) return new List<MuraCurveRecord>(records);
            if (limit == 1) return new List<MuraCurveRecord> { records[0] };
            var sampled = new List<MuraCurveRecord>(limit);
            for (int i = 0; i < limit; i++)
            {
                int index = (int)((long)i * (records.Count - 1) / (limit - 1));
                sampled.Add(records[index]);
            }
            return sampled;
        }

        private static string[] FindCurveFiles(
            string directory, string prefix, int camId,
            string current, string previous, string legacy)
        {
            string[] files = Directory.GetFiles(directory, $"{prefix}*-{camId}{current}");
            if (files.Length > 0) return files;
            files = Directory.GetFiles(directory, $"{prefix}*-{camId}{previous}");
            return files.Length > 0
                ? files
                : Directory.GetFiles(directory, $"{prefix}*-{camId}{legacy}");
        }

        private static float[] Aggregate(
            List<MuraCurveRecord> records, bool mean, CancellationToken cancellationToken,
            ref int curveCacheHits, ref int curveCacheMisses)
        {
            float[] result = null;
            int loaded = 0;
            foreach (var record in records)
            {
                cancellationToken.ThrowIfCancellationRequested();
                string curvePath = mean
                    ? CaptureFileNaming.ResolveMeanC(record.BasePath)
                    : CaptureFileNaming.ResolveMaxC(record.BasePath);
                float[] curve = LoadRangeCurve(
                    curvePath, ref curveCacheHits, ref curveCacheMisses);
                if (curve == null || curve.Length == 0) continue;
                if (result == null) result = new float[curve.Length];
                if (result.Length != curve.Length) continue;

                for (int i = 0; i < curve.Length; i++)
                {
                    if ((i & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
                    if (mean) result[i] += curve[i];
                    else if (curve[i] > result[i]) result[i] = curve[i];
                }
                loaded++;
            }

            if (mean && result != null && loaded > 0)
                for (int i = 0; i < result.Length; i++)
                {
                    if ((i & 1023) == 0) cancellationToken.ThrowIfCancellationRequested();
                    result[i] /= loaded;
                }
            return result;
        }

        private static float[] LoadRangeCurve(
            string path, ref int cacheHits, ref int cacheMisses)
        {
            if (string.IsNullOrWhiteSpace(path)) return null;
            string key;
            try
            {
                key = CaptureArchiveStore.IsVirtualPath(path) ? path : Path.GetFullPath(path);
            }
            catch { key = path; }

            lock (RangeCurveCacheLock)
            {
                if (RangeCurveCache.TryGetValue(key, out CachedCurve cached))
                {
                    RangeCurveLru.Remove(cached.Node);
                    RangeCurveLru.AddFirst(cached.Node);
                    cacheHits++;
                    return cached.Values;
                }
            }

            float[] loaded = CurveBinFile.Load(key);
            cacheMisses++;
            if (loaded == null || loaded.Length == 0) return loaded;
            long bytes = (long)loaded.Length * sizeof(float);
            if (bytes > RangeCurveCacheByteCapacity) return loaded;

            lock (RangeCurveCacheLock)
            {
                if (RangeCurveCache.TryGetValue(key, out CachedCurve existing))
                    return existing.Values;

                var node = new LinkedListNode<string>(key);
                RangeCurveLru.AddFirst(node);
                RangeCurveCache[key] = new CachedCurve
                {
                    Values = loaded,
                    Bytes = bytes,
                    Node = node
                };
                _rangeCurveCacheBytes += bytes;
                TrimRangeCurveCache();
            }
            return loaded;
        }

        private static void TrimRangeCurveCache()
        {
            while (RangeCurveCache.Count > RangeCurveCacheEntryCapacity ||
                   _rangeCurveCacheBytes > RangeCurveCacheByteCapacity)
            {
                LinkedListNode<string> oldest = RangeCurveLru.Last;
                if (oldest == null) return;
                CachedCurve removed = RangeCurveCache[oldest.Value];
                _rangeCurveCacheBytes -= removed.Bytes;
                RangeCurveCache.Remove(oldest.Value);
                RangeCurveLru.RemoveLast();
            }
        }

    }
}
