using System;
using System.Globalization;
using System.IO;

namespace AniloxRoll.Monitor.Core.Services
{
    internal sealed class InspectionCsvRecord
    {
        public string GrabId { get; set; }
        public string FileName { get; set; }
        public int MaxExceed { get; set; }
        public int MeanExceed { get; set; }
        public float MeanPeak { get; set; }
        public float MaxPeak { get; set; }
        public int GrabHeight { get; set; }
        public double LineRateHz { get; set; }
        public double ExposureUs { get; set; }
        public float MaxCMean { get; set; } = float.NaN;
        public float MeanRPeak { get; set; } = float.NaN;
        public float MaxRPeak { get; set; } = float.NaN;
    }

    /// <summary>
    /// Owns the inspection CSV line format and shared-read file access used by report and review queries.
    /// </summary>
    internal static class InspectionCsvReader
    {
        public static StreamReader OpenShared(string path)
        {
            var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            return new StreamReader(stream);
        }

        public static bool TryParseRecord(string line, out InspectionCsvRecord record)
        {
            record = null;
            if (string.IsNullOrWhiteSpace(line) || line[0] == '#') return false;

            string[] columns = line.Split(',');
            if (columns.Length < 4) return false;
            if (!int.TryParse(columns[2].Trim(), out int maxExceed) ||
                !int.TryParse(columns[3].Trim(), out int meanExceed))
                return false;

            record = new InspectionCsvRecord
            {
                GrabId = columns[0].Trim(),
                FileName = columns[1].Trim(),
                MaxExceed = maxExceed,
                MeanExceed = meanExceed
            };

            if (columns.Length >= 9)
            {
                float.TryParse(columns[4].Trim(), NumberStyles.Float,
                    CultureInfo.InvariantCulture, out float meanPeak);
                float.TryParse(columns[5].Trim(), NumberStyles.Float,
                    CultureInfo.InvariantCulture, out float maxPeak);
                int.TryParse(columns[6].Trim(), out int grabHeight);
                double.TryParse(columns[7].Trim(), NumberStyles.Float,
                    CultureInfo.InvariantCulture, out double lineRateHz);
                double.TryParse(columns[8].Trim(), NumberStyles.Float,
                    CultureInfo.InvariantCulture, out double exposureUs);
                record.MeanPeak = meanPeak;
                record.MaxPeak = maxPeak;
                record.GrabHeight = grabHeight;
                record.LineRateHz = lineRateHz;
                record.ExposureUs = exposureUs;
            }

            if (columns.Length >= 10 && float.TryParse(columns[9].Trim(), NumberStyles.Float,
                CultureInfo.InvariantCulture, out float maxCMean))
                record.MaxCMean = maxCMean;
            if (columns.Length >= 12)
            {
                if (float.TryParse(columns[10].Trim(), NumberStyles.Float,
                    CultureInfo.InvariantCulture, out float meanRPeak))
                    record.MeanRPeak = meanRPeak;
                if (float.TryParse(columns[11].Trim(), NumberStyles.Float,
                    CultureInfo.InvariantCulture, out float maxRPeak))
                    record.MaxRPeak = maxRPeak;
            }
            return true;
        }

        public static bool TryUpdateHmFromConfig(string line, ref float captureHmV)
        {
            if (string.IsNullOrEmpty(line) || !line.StartsWith("#CFG,")) return false;
            if (CsvConfigSnapshot.TryParse(line, out var config) && config.HessianMaxFactorV > 0f)
                captureHmV = config.HessianMaxFactorV;
            return true;
        }

        public static bool TryParseTimestamp(string fileName, out DateTime result)
        {
            result = DateTime.MinValue;
            if (string.IsNullOrEmpty(fileName)) return false;
            int underscore = fileName.IndexOf('_');
            if (underscore != 8 || fileName.Length < 19) return false;
            string value = fileName.Substring(0, 8) + fileName.Substring(9, 10);
            return DateTime.TryParseExact(value, "yyyyMMddHHmmss.fff",
                CultureInfo.InvariantCulture, DateTimeStyles.None, out result);
        }

        public static bool TryExtractCameraId(string fileName, out int cameraId)
        {
            cameraId = 0;
            if (string.IsNullOrEmpty(fileName)) return false;
            int dash = fileName.LastIndexOf('-');
            if (dash < 0 || dash >= fileName.Length - 1) return false;
            return int.TryParse(fileName.Substring(dash + 1), out cameraId);
        }
    }
}
