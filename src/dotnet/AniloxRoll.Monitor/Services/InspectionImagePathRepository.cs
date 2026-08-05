using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>Locates persisted capture images referenced by inspection CSV records.</summary>
    public static class InspectionImagePathRepository
    {
        /// <summary>
        /// Returns the images for one grab, grouped by camera and sorted by capture timestamp.
        /// The current raw JPEG format takes precedence over the legacy BMP format.
        /// </summary>
        public static Dictionary<int, List<string>> LoadForGrabId(
            string captureRootPath, string grabId,
            DateTime hintFrom = default(DateTime), DateTime hintTo = default(DateTime))
        {
            var infos = new[]
            {
                new GrabIdInfo { GrabId = grabId, Earliest = hintFrom, Latest = hintTo }
            };
            Dictionary<string, Dictionary<int, List<string>>> batch =
                LoadForGrabIds(captureRootPath, infos);
            return batch.TryGetValue(grabId, out Dictionary<int, List<string>> result)
                ? result
                : new Dictionary<int, List<string>>();
        }

        /// <summary>
        /// Resolves multiple grabs with one CSV pass. This is used by report verdict indexing so
        /// thousands of historical grabs do not rescan the same daily CSV thousands of times.
        /// </summary>
        public static Dictionary<string, Dictionary<int, List<string>>> LoadForGrabIds(
            string captureRootPath, IList<GrabIdInfo> grabInfos)
        {
            var result = new Dictionary<string, Dictionary<int, List<string>>>(StringComparer.Ordinal);
            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath) ||
                grabInfos == null || grabInfos.Count == 0)
                return result;

            var requested = new HashSet<string>(StringComparer.Ordinal);
            var csvPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            bool requiresFullScan = false;
            foreach (GrabIdInfo info in grabInfos)
            {
                if (info == null || string.IsNullOrWhiteSpace(info.GrabId)) continue;
                requested.Add(info.GrabId);
                if (info.Earliest == default(DateTime) || info.Latest == default(DateTime))
                {
                    requiresFullScan = true;
                    continue;
                }
                for (DateTime date = info.Earliest.Date; date <= info.Latest.Date; date = date.AddDays(1))
                {
                    string path = CaptureStoragePaths.DailyCsv(captureRootPath, date);
                    if (File.Exists(path)) csvPaths.Add(path);
                }
            }
            if (requiresFullScan)
            {
                foreach (string path in Directory.GetFiles(
                    captureRootPath, "*.csv", SearchOption.AllDirectories))
                    csvPaths.Add(path);
            }

            var namesByGrab = new Dictionary<string, Dictionary<int, HashSet<string>>>(StringComparer.Ordinal);
            foreach (string csvPath in csvPaths)
            {
                try
                {
                    using (var reader = InspectionCsvReader.OpenShared(csvPath))
                    {
                        reader.ReadLine();
                        string line;
                        while ((line = reader.ReadLine()) != null)
                        {
                            if (!InspectionCsvReader.TryParseRecord(line, out var record) ||
                                !requested.Contains(record.GrabId) ||
                                !InspectionCsvReader.TryExtractCameraId(record.FileName, out int camId))
                                continue;

                            if (!namesByGrab.TryGetValue(record.GrabId, out var byCamera))
                                namesByGrab[record.GrabId] = byCamera =
                                    new Dictionary<int, HashSet<string>>();
                            if (!byCamera.TryGetValue(camId, out var fileNames))
                                byCamera[camId] = fileNames = new HashSet<string>(StringComparer.Ordinal);
                            fileNames.Add(record.FileName);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine(
                        $"[InspectionImagePathRepository.LoadForGrabIds] {csvPath}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            foreach (var grab in namesByGrab)
            {
                var pathsByCamera = new Dictionary<int, List<string>>();
                foreach (var camera in grab.Value)
                {
                    var sortedNames = new List<string>(camera.Value);
                    sortedNames.Sort(StringComparer.Ordinal);
                    var paths = ResolvePaths(captureRootPath, grab.Key, sortedNames);
                    if (paths.Count > 0) pathsByCamera[camera.Key] = paths;
                }
                if (pathsByCamera.Count > 0) result[grab.Key] = pathsByCamera;
            }
            return result;
        }

        private static List<string> ResolvePaths(
            string captureRootPath, string grabId, IList<string> sortedNames)
        {
            var paths = new List<string>();
            foreach (string fileName in sortedNames)
            {
                if (fileName.Length < 8) continue;
                string directory = CaptureStoragePaths.DateImageDir(captureRootPath, fileName);

                if (InspectionCsvReader.TryParseTimestamp(fileName, out DateTime timestamp))
                {
                    string archivePath = CaptureStoragePaths.GrabArchive(
                        captureRootPath, timestamp, grabId);
                    if (File.Exists(archivePath))
                    {
                        string virtualRaw = CaptureArchiveStore.CreateVirtualRawPath(
                            archivePath, fileName);
                        if (CaptureArchiveStore.Exists(virtualRaw))
                        {
                            paths.Add(virtualRaw);
                            continue;
                        }
                    }
                }

                string rawJpg = Path.Combine(directory, fileName + CaptureFileNaming.RawJpg);
                if (File.Exists(rawJpg))
                {
                    paths.Add(rawJpg);
                    continue;
                }

                string bmp = Path.Combine(directory, fileName + ".bmp");
                if (File.Exists(bmp)) paths.Add(bmp);
            }
            return paths;
        }
    }
}
