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
            var camFileNames = new Dictionary<int, HashSet<string>>();

            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return new Dictionary<int, List<string>>();

            IEnumerable<string> csvPaths;
            if (hintFrom != default(DateTime) && hintTo != default(DateTime))
            {
                var dateCsvs = new List<string>();
                for (DateTime date = hintFrom.Date; date <= hintTo.Date; date = date.AddDays(1))
                {
                    string path = CaptureStoragePaths.DailyCsv(captureRootPath, date);
                    if (File.Exists(path)) dateCsvs.Add(path);
                }
                csvPaths = dateCsvs;
            }
            else
            {
                csvPaths = Directory.GetFiles(captureRootPath, "*.csv", SearchOption.AllDirectories);
            }

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
                            if (!InspectionCsvReader.TryParseRecord(line, out var record)) continue;
                            if (record.GrabId != grabId) continue;
                            if (!InspectionCsvReader.TryExtractCameraId(record.FileName, out int camId)) continue;

                            if (!camFileNames.TryGetValue(camId, out var fileNames))
                                camFileNames[camId] = fileNames = new HashSet<string>();
                            fileNames.Add(record.FileName);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine(
                        $"[InspectionImagePathRepository.LoadForGrabId] {csvPath}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            var result = new Dictionary<int, List<string>>();
            foreach (var camera in camFileNames)
            {
                var sortedNames = new List<string>(camera.Value);
                sortedNames.Sort(StringComparer.Ordinal);

                var paths = new List<string>();
                foreach (string fileName in sortedNames)
                {
                    if (fileName.Length < 8) continue;
                    string directory = CaptureStoragePaths.DateImageDir(captureRootPath, fileName);

                    string rawJpg = Path.Combine(directory, fileName + CaptureFileNaming.RawJpg);
                    if (File.Exists(rawJpg))
                    {
                        paths.Add(rawJpg);
                        continue;
                    }

                    string bmp = Path.Combine(directory, fileName + ".bmp");
                    if (File.Exists(bmp)) paths.Add(bmp);
                }

                if (paths.Count > 0) result[camera.Key] = paths;
            }

            return result;
        }
    }
}
