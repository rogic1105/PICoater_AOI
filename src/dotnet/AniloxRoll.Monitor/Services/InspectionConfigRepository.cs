using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>Queries persisted #CFG snapshots from inspection CSV files.</summary>
    public static class InspectionConfigRepository
    {
        public static CsvConfigSnapshot LoadForGrabId(
            string captureRootPath, string grabId,
            DateTime hintFrom = default(DateTime), DateTime hintTo = default(DateTime))
        {
            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return null;

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
                    CsvConfigSnapshot lastConfig = null;
                    using (var reader = InspectionCsvReader.OpenShared(csvPath))
                    {
                        string line;
                        while ((line = reader.ReadLine()) != null)
                        {
                            if (line.StartsWith("#CFG,"))
                            {
                                if (CsvConfigSnapshot.TryParse(line, out var config))
                                    lastConfig = config;
                                continue;
                            }

                            if (!InspectionCsvReader.TryParseRecord(line, out var record)) continue;
                            if (record.GrabId == grabId && lastConfig != null) return lastConfig;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine(
                        $"[InspectionConfigRepository.LoadForGrabId] {csvPath}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            return null;
        }

        public static CsvConfigSnapshot LoadForDate(string captureRootPath, DateTime date)
        {
            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return null;
            return LoadFromCsv(CaptureStoragePaths.DailyCsv(captureRootPath, date));
        }

        public static CsvConfigSnapshot LoadFromCsv(string csvPath)
        {
            if (string.IsNullOrWhiteSpace(csvPath) || !File.Exists(csvPath))
                return null;

            CsvConfigSnapshot latest = null;
            try
            {
                using (var reader = InspectionCsvReader.OpenShared(csvPath))
                {
                    string line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        if (line.StartsWith("#CFG,") &&
                            CsvConfigSnapshot.TryParse(line, out var config))
                            latest = config;
                    }
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[InspectionConfigRepository.LoadFromCsv] {csvPath}: {ex.GetType().Name}: {ex.Message}");
            }

            return latest;
        }

        public static CsvConfigSnapshot LoadLatest(string captureRootPath)
        {
            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return null;

            CsvConfigSnapshot latest = null;
            try
            {
                var csvFiles = Directory.GetFiles(captureRootPath, "*.csv", SearchOption.AllDirectories);
                Array.Sort(csvFiles, StringComparer.Ordinal);
                foreach (string csvPath in csvFiles)
                {
                    var config = LoadFromCsv(csvPath);
                    if (config != null) latest = config;
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[InspectionConfigRepository.LoadLatest] {ex.GetType().Name}: {ex.Message}");
            }

            return latest;
        }
    }
}
