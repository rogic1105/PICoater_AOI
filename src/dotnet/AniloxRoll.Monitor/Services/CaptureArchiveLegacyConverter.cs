using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace AniloxRoll.Monitor.Core.Services
{
    internal sealed class CaptureArchiveConversionResult
    {
        public int ArchiveCount { get; set; }
        public int FrameCount { get; set; }
        public long PayloadBytes { get; set; }
        public int SkippedArchiveCount { get; set; }
        public int FailedArchiveCount { get; set; }
    }

    /// <summary>
    /// Reads legacy CSV, frame files, and tick sidecars, then asks CaptureArchiveStore to write
    /// the container. Binary framing and CRC remain owned exclusively by the Store.
    /// </summary>
    internal static class CaptureArchiveLegacyConverter
    {
        private sealed class LegacyFrame
        {
            public string BasePath;
            public int CameraId;
        }

        public static CaptureArchiveConversionResult ConvertRoot(
            string captureRoot,
            bool overwrite,
            Action<string> progress = null)
        {
            var result = new CaptureArchiveConversionResult();
            if (string.IsNullOrWhiteSpace(captureRoot) || !Directory.Exists(captureRoot))
                return result;

            string[] csvPaths = Directory.GetFiles(
                captureRoot, "*.csv", SearchOption.AllDirectories);
            Array.Sort(csvPaths, StringComparer.OrdinalIgnoreCase);
            for (int csvIndex = 0; csvIndex < csvPaths.Length; csvIndex++)
            {
                Dictionary<string, List<LegacyFrame>> byGrab =
                    ReadLegacyFrames(csvPaths[csvIndex]);
                foreach (KeyValuePair<string, List<LegacyFrame>> grab in byGrab)
                    ConvertGrab(captureRoot, grab, overwrite, result, progress);
            }
            return result;
        }

        private static void ConvertGrab(
            string captureRoot,
            KeyValuePair<string, List<LegacyFrame>> grab,
            bool overwrite,
            CaptureArchiveConversionResult result,
            Action<string> progress)
        {
            if (grab.Value.Count == 0) return;
            string firstBase = grab.Value[0].BasePath;
            if (!InspectionCsvReader.TryParseTimestamp(
                Path.GetFileName(firstBase), out DateTime captureDate))
                return;

            string archivePath = CaptureStoragePaths.GrabArchive(
                captureRoot, captureDate, grab.Key);
            if (File.Exists(archivePath) && !overwrite)
            {
                result.SkippedArchiveCount++;
                return;
            }

            try
            {
                Dictionary<string, long> ticks = LoadLegacyTicks(
                    CaptureStoragePaths.DateImageDir(captureRoot, captureDate));
                var frames = new List<CaptureArchiveFrame>(grab.Value.Count);
                for (int i = 0; i < grab.Value.Count; i++)
                {
                    LegacyFrame source = grab.Value[i];
                    string baseName = Path.GetFileName(source.BasePath);
                    ticks.TryGetValue(baseName, out long frameTicks);
                    frames.Add(new CaptureArchiveFrame
                    {
                        BaseName = baseName,
                        CameraId = source.CameraId,
                        FrameTicks = frameTicks,
                        Assets = LoadLegacyAssets(source.BasePath)
                    });
                }

                result.PayloadBytes += CaptureArchiveStore.ReplaceArchive(
                    archivePath, grab.Key, frames);
                result.FrameCount += frames.Count;
                result.ArchiveCount++;
                progress?.Invoke(archivePath);
            }
            catch (Exception ex)
            {
                result.FailedArchiveCount++;
                Trace.WriteLine(
                    $"[CaptureArchive.Convert] {grab.Key}: " +
                    $"{ex.GetType().Name}: {ex.Message}");
            }
        }

        private static Dictionary<string, List<LegacyFrame>> ReadLegacyFrames(
            string csvPath)
        {
            var byGrab = new Dictionary<string, List<LegacyFrame>>(StringComparer.Ordinal);
            try
            {
                using (var reader = InspectionCsvReader.OpenShared(csvPath))
                {
                    string line;
                    var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                    while ((line = reader.ReadLine()) != null)
                    {
                        if (!InspectionCsvReader.TryParseRecord(line, out var record) ||
                            !InspectionCsvReader.TryExtractCameraId(
                                record.FileName, out int cameraId) ||
                            !InspectionCsvReader.TryParseTimestamp(
                                record.FileName, out DateTime timestamp))
                            continue;
                        string identity = record.GrabId + "\n" + record.FileName;
                        if (!seen.Add(identity)) continue;
                        if (!byGrab.TryGetValue(
                            record.GrabId, out List<LegacyFrame> frames))
                            byGrab[record.GrabId] = frames = new List<LegacyFrame>();
                        frames.Add(new LegacyFrame
                        {
                            BasePath = Path.Combine(
                                CaptureStoragePaths.DateImageDir(
                                    CaptureRootFromCsvPath(csvPath), timestamp),
                                record.FileName),
                            CameraId = cameraId
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[CaptureArchive.ReadCsv] {csvPath}: " +
                    $"{ex.GetType().Name}: {ex.Message}");
            }
            return byGrab;
        }

        private static string CaptureRootFromCsvPath(string csvPath)
        {
            DirectoryInfo month = Directory.GetParent(csvPath);
            DirectoryInfo year = month?.Parent;
            return year?.Parent?.FullName ?? string.Empty;
        }

        private static List<CaptureArchiveAsset> LoadLegacyAssets(string basePath)
        {
            var assets = new List<CaptureArchiveAsset>(7);
            AddLegacyAsset(
                assets, CaptureAssetKind.RawJpeg,
                basePath + CaptureFileNaming.RawJpg);
            AddLegacyAsset(
                assets, CaptureAssetKind.ProcessedColumnJpeg,
                ResolveLegacyExisting(
                    basePath, CaptureFileNaming.ProcC,
                    CaptureFileNaming.ProcCPrevious, CaptureFileNaming.ProcLegacy));
            AddLegacyAsset(
                assets, CaptureAssetKind.ProcessedRowJpeg,
                ResolveLegacyExisting(
                    basePath, CaptureFileNaming.ProcR,
                    CaptureFileNaming.ProcRPrevious, CaptureFileNaming.ProcLegacy));
            AddLegacyAsset(
                assets, CaptureAssetKind.MeanColumnCurve,
                ResolveLegacyExisting(
                    basePath, CaptureFileNaming.MeanC,
                    CaptureFileNaming.MeanCPrevious, CaptureFileNaming.MeanCLegacy));
            AddLegacyAsset(
                assets, CaptureAssetKind.MaxColumnCurve,
                ResolveLegacyExisting(
                    basePath, CaptureFileNaming.MaxC,
                    CaptureFileNaming.MaxCPrevious, CaptureFileNaming.MaxCLegacy));
            AddLegacyAsset(
                assets, CaptureAssetKind.MeanRowCurve,
                ResolveLegacyExisting(
                    basePath, CaptureFileNaming.MeanR,
                    CaptureFileNaming.MeanRPrevious, CaptureFileNaming.MeanRLegacy));
            AddLegacyAsset(
                assets, CaptureAssetKind.MaxRowCurve,
                ResolveLegacyExisting(
                    basePath, CaptureFileNaming.MaxR,
                    CaptureFileNaming.MaxRPrevious, CaptureFileNaming.MaxRLegacy));
            return assets;
        }

        private static void AddLegacyAsset(
            List<CaptureArchiveAsset> assets,
            CaptureAssetKind kind,
            string path)
        {
            if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;
            assets.Add(new CaptureArchiveAsset
            {
                Kind = kind,
                Data = File.ReadAllBytes(path)
            });
        }

        private static string ResolveLegacyExisting(
            string basePath,
            string current,
            string previous,
            string legacy)
        {
            string path = basePath + current;
            if (File.Exists(path)) return path;
            path = basePath + previous;
            return File.Exists(path) ? path : basePath + legacy;
        }

        private static Dictionary<string, long> LoadLegacyTicks(string dateDirectory)
        {
            var result = new Dictionary<string, long>(StringComparer.Ordinal);
            string path = Path.Combine(dateDirectory, "_ticks.csv");
            if (!File.Exists(path)) return result;
            try
            {
                foreach (string line in File.ReadLines(path))
                {
                    int comma = line.LastIndexOf(',');
                    if (comma <= 0) continue;
                    if (long.TryParse(line.Substring(comma + 1), out long ticks))
                        result[line.Substring(0, comma)] = ticks;
                }
            }
            catch
            {
                // Missing/corrupt legacy ticks fall back to zero, matching legacy behavior.
            }
            return result;
        }
    }
}
