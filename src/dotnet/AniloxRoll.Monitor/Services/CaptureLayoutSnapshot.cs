using System;
using System.Globalization;
using System.Text;
using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// Final machine-layout values for one capture. Pixels and curve samples remain unchanged;
    /// this snapshot controls their physical-coordinate interpretation.
    /// </summary>
    public sealed class CaptureLayoutSnapshot
    {
        public const string CsvPrefix = "#LAYOUT_FINAL,";

        public string GrabId { get; }
        public double[] CamOps { get; }
        public double[] CamPos { get; }
        public double AniloxRollSpeedMPerMin { get; }
        public double TrimHeadMm { get; }
        public double TrimTailMm { get; }
        public DateTime Timestamp { get; }

        public CaptureLayoutSnapshot(
            string grabId,
            double[] camOps,
            double[] camPos,
            double aniloxRollSpeedMPerMin,
            double trimHeadMm,
            double trimTailMm,
            DateTime timestamp)
        {
            GrabId = grabId ?? string.Empty;
            CamOps = CloneSeven(camOps);
            CamPos = CloneSeven(camPos);
            AniloxRollSpeedMPerMin = aniloxRollSpeedMPerMin;
            TrimHeadMm = Math.Max(0, trimHeadMm);
            TrimTailMm = Math.Max(0, trimTailMm);
            Timestamp = timestamp;
        }

        public static CaptureLayoutSnapshot FromSettings(
            string grabId,
            InspectionSettings settings,
            DateTime timestamp)
        {
            if (settings == null) return null;
            return new CaptureLayoutSnapshot(
                grabId,
                settings.GetCameraOpsUmArray(),
                settings.GetCameraStartPositionMmArray(),
                settings.AniloxRollSpeedMPerMin,
                settings.TrimHeadMm,
                settings.TrimTailMm,
                timestamp);
        }

        public string ToCsvLine()
        {
            var sb = new StringBuilder(384);
            sb.Append(CsvPrefix);
            sb.Append(Timestamp.ToString("yyyy-MM-ddTHH:mm:ss.fff", CultureInfo.InvariantCulture));
            sb.Append(",GrabId=").Append(GrabId);
            for (int i = 0; i < 7; i++)
                sb.AppendFormat(CultureInfo.InvariantCulture, ",Cam{0}_Ops={1:F8}", i + 1, CamOps[i]);
            for (int i = 0; i < 7; i++)
                sb.AppendFormat(CultureInfo.InvariantCulture, ",Cam{0}_Pos={1:F4}", i + 1, CamPos[i]);
            sb.AppendFormat(
                CultureInfo.InvariantCulture,
                ",AniloxRollSpeedMPerMin={0:F4},TrimHead={1:F4},TrimTail={2:F4}",
                AniloxRollSpeedMPerMin,
                TrimHeadMm,
                TrimTailMm);
            return sb.ToString();
        }

        public string ToFlowValues()
        {
            return string.Format(
                CultureInfo.InvariantCulture,
                "ops={0} start={1} speed={2:F4} head={3:F2} tail={4:F2}",
                FormatArray(CamOps, "F8"),
                FormatArray(CamPos, "F4"),
                AniloxRollSpeedMPerMin,
                TrimHeadMm,
                TrimTailMm);
        }

        public static bool TryParse(string line, out CaptureLayoutSnapshot result)
        {
            result = null;
            if (string.IsNullOrEmpty(line) ||
                !line.StartsWith(CsvPrefix, StringComparison.Ordinal))
                return false;

            string[] parts = line.Split(',');
            if (parts.Length < 4 ||
                !DateTime.TryParseExact(
                    parts[1].Trim(),
                    "yyyy-MM-ddTHH:mm:ss.fff",
                    CultureInfo.InvariantCulture,
                    DateTimeStyles.None,
                    out DateTime timestamp))
                return false;

            string grabId = string.Empty;
            double[] ops = new double[7];
            double[] pos = new double[7];
            double speed = 0;
            double trimHead = 0;
            double trimTail = 0;

            for (int i = 2; i < parts.Length; i++)
            {
                string part = parts[i].Trim();
                int separator = part.IndexOf('=');
                if (separator <= 0) continue;
                string key = part.Substring(0, separator);
                string value = part.Substring(separator + 1);

                if (key == "GrabId")
                {
                    grabId = value;
                }
                else if (TryGetCameraIndex(key, "_Ops", out int opsIndex))
                {
                    double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out ops[opsIndex]);
                }
                else if (TryGetCameraIndex(key, "_Pos", out int posIndex))
                {
                    double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out pos[posIndex]);
                }
                else if (key == "AniloxRollSpeedMPerMin")
                {
                    double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out speed);
                }
                else if (key == "TrimHead")
                {
                    double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out trimHead);
                }
                else if (key == "TrimTail")
                {
                    double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out trimTail);
                }
            }

            if (string.IsNullOrWhiteSpace(grabId)) return false;
            result = new CaptureLayoutSnapshot(
                grabId, ops, pos, speed, trimHead, trimTail, timestamp);
            return true;
        }

        private static bool TryGetCameraIndex(string key, string suffix, out int index)
        {
            index = -1;
            if (key == null || key.Length != 4 + suffix.Length ||
                !key.StartsWith("Cam", StringComparison.Ordinal) ||
                !key.EndsWith(suffix, StringComparison.Ordinal))
                return false;
            index = key[3] - '1';
            return index >= 0 && index < 7;
        }

        private static double[] CloneSeven(double[] values)
        {
            var result = new double[7];
            if (values != null)
                Array.Copy(values, result, Math.Min(values.Length, result.Length));
            return result;
        }

        private static string FormatArray(double[] values, string format)
        {
            var parts = new string[7];
            for (int i = 0; i < parts.Length; i++)
            {
                double value = values != null && i < values.Length ? values[i] : 0;
                parts[i] = value.ToString(format, CultureInfo.InvariantCulture);
            }
            return string.Join("|", parts);
        }
    }
}
