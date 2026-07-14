using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>Loads and aggregates persisted Mura column-curve profiles.</summary>
    public static class InspectionMuraProfileRepository
    {
        private sealed class MuraCurveRecord
        {
            public string MeanCPath;
            public string MaxCPath;
            public float MaxCMean;
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

                    if (!accMean.TryGetValue(camId, out float[] accumulatedMean))
                    {
                        accMean[camId] = new float[mean.Length];
                        accMax[camId] = new float[mean.Length];
                        counts[camId] = 0;
                    }
                    else if (accumulatedMean.Length != mean.Length)
                    {
                        continue;
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

        public static (
            Dictionary<int, float[]> Mean,
            Dictionary<int, float[]> Max,
            int MeanRows,
            int MaxRows,
            int ScoredRows,
            int TotalRows,
            int RankedCams,
            int TotalCams)
            LoadRange(string rootPath, IList<GrabIdInfo> rangeInfos, int limit)
        {
            var meanResult = new Dictionary<int, float[]>();
            var maxResult = new Dictionary<int, float[]>();
            int meanRows = 0, maxRows = 0, scoredRows = 0, totalRows = 0;
            int rankedCams = 0;
            if (string.IsNullOrWhiteSpace(rootPath) || rangeInfos == null ||
                rangeInfos.Count == 0 || limit <= 0)
                return (meanResult, maxResult, 0, 0, 0, 0, 0, 0);

            var rangeIds = new HashSet<string>(StringComparer.Ordinal);
            var dates = new HashSet<DateTime>();
            foreach (var info in rangeInfos)
            {
                if (info == null || string.IsNullOrEmpty(info.GrabId)) continue;
                rangeIds.Add(info.GrabId);
                dates.Add(info.Earliest.Date);
            }

            var recordsByCam = new Dictionary<int, List<MuraCurveRecord>>();
            foreach (DateTime date in dates)
            {
                string csvPath = CaptureStoragePaths.DailyCsv(rootPath, date);
                if (!File.Exists(csvPath)) continue;
                try
                {
                    using (var reader = InspectionCsvReader.OpenShared(csvPath))
                    {
                        string line;
                        while ((line = reader.ReadLine()) != null)
                        {
                            if (!InspectionCsvReader.TryParseRecord(line, out var record)) continue;
                            if (!rangeIds.Contains(record.GrabId) ||
                                !InspectionCsvReader.TryExtractCameraId(record.FileName, out int camId) ||
                                !InspectionCsvReader.TryParseTimestamp(record.FileName, out DateTime timestamp)) continue;

                            if (!recordsByCam.TryGetValue(camId, out var records))
                                recordsByCam[camId] = records = new List<MuraCurveRecord>();
                            string dateDir = CaptureStoragePaths.DateImageDir(rootPath, timestamp);
                            records.Add(new MuraCurveRecord
                            {
                                MeanCPath = CaptureFileNaming.ResolveMeanC(Path.Combine(dateDir, record.FileName)),
                                MaxCPath = CaptureFileNaming.ResolveMaxC(Path.Combine(dateDir, record.FileName)),
                                MaxCMean = record.MaxCMean
                            });
                            totalRows++;
                            if (!float.IsNaN(record.MaxCMean) && !float.IsInfinity(record.MaxCMean))
                                scoredRows++;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine(
                        $"[InspectionMuraProfileRepository.LoadRange] {csvPath}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            foreach (var camera in recordsByCam)
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
                    rankedCams++;
                }
                else
                {
                    maxCandidates = EvenSample(records, limit);
                }

                float[] mean = Aggregate(meanCandidates, true);
                float[] max = Aggregate(maxCandidates, false);
                if (mean != null) meanResult[camera.Key] = mean;
                if (max != null) maxResult[camera.Key] = max;
                meanRows += meanCandidates.Count;
                maxRows += maxCandidates.Count;
            }

            return (meanResult, maxResult, meanRows, maxRows, scoredRows, totalRows,
                rankedCams, recordsByCam.Count);
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

        private static float[] Aggregate(List<MuraCurveRecord> records, bool mean)
        {
            float[] result = null;
            int loaded = 0;
            foreach (var record in records)
            {
                float[] curve = CurveBinFile.Load(mean ? record.MeanCPath : record.MaxCPath);
                if (curve == null || curve.Length == 0) continue;
                if (result == null) result = new float[curve.Length];
                if (result.Length != curve.Length) continue;

                for (int i = 0; i < curve.Length; i++)
                {
                    if (mean) result[i] += curve[i];
                    else if (curve[i] > result[i]) result[i] = curve[i];
                }
                loaded++;
            }

            if (mean && result != null && loaded > 0)
                for (int i = 0; i < result.Length; i++) result[i] /= loaded;
            return result;
        }

    }
}
