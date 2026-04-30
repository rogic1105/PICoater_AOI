using System;
using System.Globalization;
using System.Text;
using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// CSV #CFG 列的不可變快照，記錄全域設定值。
    /// </summary>
    public class CsvConfigSnapshot
    {
        public double[] CamOps { get; }   // length 7
        public double[] CamPos { get; }   // length 7
        public int[] CamGrabHeight { get; }    // length 7，高度滑桿（line scan 行數）
        public double[] CamExposureUs { get; } // length 7，曝光滑桿（μs）
        public double[] CamLineRateHz { get; } // length 7，線掃滑桿（Hz）
        public float HessianMaxFactor { get; }
        public float ErrorValueMean { get; }
        public float ErrorValueMax { get; }
        public double TrimHeadMm { get; }
        public double TrimTailMm { get; }
        public DateTime Timestamp { get; }

        public CsvConfigSnapshot(
            double[] camOps, double[] camPos, int[] camGrabHeight,
            double[] camExposureUs, double[] camLineRateHz,
            float hessianMaxFactor, float errorValueMean, float errorValueMax,
            double trimHeadMm, double trimTailMm,
            DateTime timestamp)
        {
            CamOps = camOps ?? new double[7];
            CamPos = camPos ?? new double[7];
            CamGrabHeight = camGrabHeight ?? new int[7];
            CamExposureUs = camExposureUs ?? new double[7];
            CamLineRateHz = camLineRateHz ?? new double[7];
            HessianMaxFactor = hessianMaxFactor;
            ErrorValueMean = errorValueMean;
            ErrorValueMax = errorValueMax;
            TrimHeadMm = trimHeadMm;
            TrimTailMm = trimTailMm;
            Timestamp = timestamp;
        }

        public static CsvConfigSnapshot FromSettings(InspectionSettings s)
        {
            if (s == null) return null;
            return new CsvConfigSnapshot(
                s.GetCameraOpsUmArray(),
                s.GetCameraStartPositionMmArray(),
                (int[])s.Acquisition?.CameraGrabHeight?.Clone(),
                (double[])s.Acquisition?.CameraExposureTimeUs?.Clone(),
                (double[])s.Acquisition?.CameraLineRateHz?.Clone(),
                s.HessianMaxFactor,
                s.ErrorValueMean,
                s.ErrorValueMax,
                s.TrimHeadMm,
                s.TrimTailMm,
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
                for (int i = 0; i < 7; i++) sb.Append(CamGrabHeight[i]).Append(',');
                for (int i = 0; i < 7; i++) sb.Append(CamExposureUs[i].ToString("F2")).Append(',');
                for (int i = 0; i < 7; i++) sb.Append(CamLineRateHz[i].ToString("F2")).Append(',');
                sb.Append(HessianMaxFactor.ToString("F4")).Append(',');
                sb.Append(ErrorValueMean.ToString("F4")).Append(',');
                sb.Append(ErrorValueMax.ToString("F4")).Append(',');
                sb.Append(TrimHeadMm.ToString("F2")).Append(',');
                sb.Append(TrimTailMm.ToString("F2"));
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
            for (int i = 0; i < 7; i++)
                sb.Append($",Cam{i + 1}_GrabH={CamGrabHeight[i]}");
            for (int i = 0; i < 7; i++)
                sb.Append($",Cam{i + 1}_Exp={CamExposureUs[i]:F2}");
            for (int i = 0; i < 7; i++)
                sb.Append($",Cam{i + 1}_Lr={CamLineRateHz[i]:F2}");
            sb.Append($",HessianMaxFactor={HessianMaxFactor:F4}");
            sb.Append($",ErrorValueMean={ErrorValueMean:F4}");
            sb.Append($",ErrorValueMax={ErrorValueMax:F4}");
            sb.Append($",TrimHead={TrimHeadMm:F2}");
            sb.Append($",TrimTail={TrimTailMm:F2}");
            return sb.ToString();
        }

        /// <summary>從 #CFG 列解析。</summary>
        public static bool TryParse(string line, out CsvConfigSnapshot result)
        {
            result = null;
            if (string.IsNullOrEmpty(line) || !line.StartsWith("#CFG,")) return false;

            string[] parts = line.Split(',');
            // #CFG, timestamp, key=value pairs（最少 17 個，向下相容舊格式；新格式 31 個）
            if (parts.Length < 19) return false;

            if (!DateTime.TryParseExact(parts[1].Trim(), "yyyy-MM-ddTHH:mm:ss.fff",
                CultureInfo.InvariantCulture, DateTimeStyles.None, out DateTime ts))
                return false;

            double[] ops = new double[7];
            double[] pos = new double[7];
            int[] grabH = new int[7];
            double[] expUs = new double[7];
            double[] lrHz = new double[7];
            float hessian = 0, errMean = 0, errMax = 0;
            double trimHead = 0, trimTail = 0;

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
                else if (key.StartsWith("Cam") && key.EndsWith("_GrabH"))
                {
                    int camIdx = key[3] - '1';
                    if (camIdx >= 0 && camIdx < 7)
                        int.TryParse(val, NumberStyles.Integer, CultureInfo.InvariantCulture, out grabH[camIdx]);
                }
                else if (key.StartsWith("Cam") && key.EndsWith("_Exp"))
                {
                    int camIdx = key[3] - '1';
                    if (camIdx >= 0 && camIdx < 7)
                        double.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out expUs[camIdx]);
                }
                else if (key.StartsWith("Cam") && key.EndsWith("_Lr"))
                {
                    int camIdx = key[3] - '1';
                    if (camIdx >= 0 && camIdx < 7)
                        double.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out lrHz[camIdx]);
                }
                else if (key == "HessianMaxFactor")
                    float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out hessian);
                else if (key == "ErrorValueMean")
                    float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out errMean);
                else if (key == "ErrorValueMax")
                    float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out errMax);
                else if (key == "TrimHead")
                    double.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out trimHead);
                else if (key == "TrimTail")
                    double.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out trimTail);
            }

            result = new CsvConfigSnapshot(ops, pos, grabH, expUs, lrHz, hessian, errMean, errMax, trimHead, trimTail, ts);
            return true;
        }
    }
}
