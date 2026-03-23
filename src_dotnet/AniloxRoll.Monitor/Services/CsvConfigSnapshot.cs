using System;
using System.Globalization;
using System.Text;
using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// CSV #CFG 列的不可變快照，記錄 17 個全域設定值。
    /// </summary>
    public class CsvConfigSnapshot
    {
        public double[] CamOps { get; }   // length 7
        public double[] CamPos { get; }   // length 7
        public float HessianMaxFactor { get; }
        public float ErrorValueMean { get; }
        public float ErrorValueMax { get; }
        public DateTime Timestamp { get; }

        public CsvConfigSnapshot(
            double[] camOps, double[] camPos,
            float hessianMaxFactor, float errorValueMean, float errorValueMax,
            DateTime timestamp)
        {
            CamOps = camOps ?? new double[7];
            CamPos = camPos ?? new double[7];
            HessianMaxFactor = hessianMaxFactor;
            ErrorValueMean = errorValueMean;
            ErrorValueMax = errorValueMax;
            Timestamp = timestamp;
        }

        public static CsvConfigSnapshot FromSettings(InspectionSettings s)
        {
            if (s == null) return null;
            return new CsvConfigSnapshot(
                s.GetCameraOpsUmArray(),
                s.GetCameraStartPositionMmArray(),
                s.HessianMaxFactor,
                s.ErrorValueMean,
                s.ErrorValueMax,
                DateTime.Now);
        }

        /// <summary>不含時間戳的內容鍵，用於偵測設定是否變更。</summary>
        public string ContentKey
        {
            get
            {
                var sb = new StringBuilder(256);
                for (int i = 0; i < 7; i++) sb.Append(CamOps[i].ToString("F2")).Append(',');
                for (int i = 0; i < 7; i++) sb.Append(CamPos[i].ToString("F2")).Append(',');
                sb.Append(HessianMaxFactor.ToString("F4")).Append(',');
                sb.Append(ErrorValueMean.ToString("F4")).Append(',');
                sb.Append(ErrorValueMax.ToString("F4"));
                return sb.ToString();
            }
        }

        /// <summary>序列化為 #CFG CSV 列。</summary>
        public string ToCsvLine()
        {
            var sb = new StringBuilder(512);
            sb.Append("#CFG,");
            sb.Append(Timestamp.ToString("yyyy-MM-ddTHH:mm:ss.fff"));
            for (int i = 0; i < 7; i++)
                sb.Append($",Cam{i + 1}_Ops={CamOps[i]:F2}");
            for (int i = 0; i < 7; i++)
                sb.Append($",Cam{i + 1}_Pos={CamPos[i]:F2}");
            sb.Append($",HessianMaxFactor={HessianMaxFactor:F4}");
            sb.Append($",ErrorValueMean={ErrorValueMean:F4}");
            sb.Append($",ErrorValueMax={ErrorValueMax:F4}");
            return sb.ToString();
        }

        /// <summary>從 #CFG 列解析。</summary>
        public static bool TryParse(string line, out CsvConfigSnapshot result)
        {
            result = null;
            if (string.IsNullOrEmpty(line) || !line.StartsWith("#CFG,")) return false;

            string[] parts = line.Split(',');
            // #CFG, timestamp, 17 key=value pairs
            if (parts.Length < 19) return false;

            if (!DateTime.TryParseExact(parts[1].Trim(), "yyyy-MM-ddTHH:mm:ss.fff",
                CultureInfo.InvariantCulture, DateTimeStyles.None, out DateTime ts))
                return false;

            double[] ops = new double[7];
            double[] pos = new double[7];
            float hessian = 0, errMean = 0, errMax = 0;

            for (int i = 2; i < parts.Length; i++)
            {
                string p = parts[i].Trim();
                int eq = p.IndexOf('=');
                if (eq < 0) continue;
                string key = p.Substring(0, eq);
                string val = p.Substring(eq + 1);

                if (key.StartsWith("Cam") && key.EndsWith("_Ops"))
                {
                    int camIdx = key[3] - '1';
                    if (camIdx >= 0 && camIdx < 7)
                        double.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out ops[camIdx]);
                }
                else if (key.StartsWith("Cam") && key.EndsWith("_Pos"))
                {
                    int camIdx = key[3] - '1';
                    if (camIdx >= 0 && camIdx < 7)
                        double.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out pos[camIdx]);
                }
                else if (key == "HessianMaxFactor")
                    float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out hessian);
                else if (key == "ErrorValueMean")
                    float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out errMean);
                else if (key == "ErrorValueMax")
                    float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out errMax);
            }

            result = new CsvConfigSnapshot(ops, pos, hessian, errMean, errMax, ts);
            return true;
        }
    }
}
