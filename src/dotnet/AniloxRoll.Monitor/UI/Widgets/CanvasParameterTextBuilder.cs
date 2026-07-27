using System;
using System.Globalization;
using System.Text;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Widgets
{
    internal static class CanvasParameterTextBuilder
    {
        public static string FromCurrentSettings(InspectionSettings settings)
        {
            return Build(CsvConfigSnapshot.FromSettings(settings), "目前 PropertyGrid");
        }

        public static string FromCaptureConfig(CsvConfigSnapshot config)
        {
            string source = config == null
                ? "拍攝 CFG：無資料"
                : $"拍攝 CFG：{config.Timestamp:yyyy-MM-dd HH:mm:ss}";
            return Build(config, source);
        }

        private static string Build(CsvConfigSnapshot config, string source)
        {
            if (config == null) return source;
            var sb = new StringBuilder(640);
            sb.AppendLine(source);
            sb.AppendLine("CAM | OPS(um) | START(mm) | ROW(mm) | HEIGHT | EXP(us) | RATE(Hz)");
            sb.AppendLine("----+---------+-----------+---------+--------+---------+---------");
            int cameraCount = MaxLength(
                config.CamOps, config.CamPos, config.CamRowOffsetMm, config.CamGrabHeight,
                config.CamExposureUs, config.CamLineRateHz);
            for (int i = 0; i < cameraCount; i++)
            {
                sb.Append('C').Append((i + 1).ToString(CultureInfo.InvariantCulture).PadRight(3));
                sb.Append("| ").Append(ValueAt(config.CamOps, i, "0.###").PadLeft(7));
                sb.Append(" | ").Append(ValueAt(config.CamPos, i, "0.###").PadLeft(9));
                sb.Append(" | ").Append(ValueAt(config.CamRowOffsetMm, i, "0.###").PadLeft(7));
                sb.Append(" | ").Append(ValueAt(config.CamGrabHeight, i).PadLeft(6));
                sb.Append(" | ").Append(ValueAt(config.CamExposureUs, i, "0.##").PadLeft(7));
                sb.Append(" | ").AppendLine(ValueAt(config.CamLineRateHz, i, "0.##").PadLeft(7));
            }
            sb.AppendLine($"A軸速度: {config.AniloxRollSpeedMPerMin:0.###} m/min");
            sb.AppendLine($"欄/列正規值: {config.HessianMaxFactorV:0.####} / {config.HessianMaxFactorH:0.####}");
            sb.AppendLine($"欄門檻 Mean/Max: {config.ErrorValueMeanV:0.####} / {config.ErrorValueMaxV:0.####}");
            sb.AppendLine($"列門檻 Mean/Max: {config.ErrorValueMeanH:0.####} / {config.ErrorValueMaxH:0.####}");
            sb.AppendLine($"細線濾除: {config.RidgeSigma:0.####}");
            sb.Append($"CROP 頭/尾: {config.TrimHeadMm:0.###} / {config.TrimTailMm:0.###} mm");
            return sb.ToString();
        }

        private static int MaxLength(params Array[] arrays)
        {
            int length = 0;
            foreach (Array array in arrays)
                length = Math.Max(length, array?.Length ?? 0);
            return length;
        }

        private static string ValueAt(double[] values, int index, string format)
        {
            return values != null && index >= 0 && index < values.Length
                ? values[index].ToString(format, CultureInfo.InvariantCulture)
                : "--";
        }

        private static string ValueAt(int[] values, int index)
        {
            return values != null && index >= 0 && index < values.Length
                ? values[index].ToString(CultureInfo.InvariantCulture)
                : "--";
        }
    }
}
