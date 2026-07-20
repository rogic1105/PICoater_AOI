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
        public double[] CamPos { get; }   // START (mm), length 7
        public int[] CamGrabHeight { get; }    // length 7，高度滑桿（line scan 行數）
        public double[] CamExposureUs { get; } // length 7，曝光滑桿（μs）
        public double[] CamLineRateHz { get; } // length 7，線掃滑桿（Hz）
        /// <summary>欄正規值 — 同時是 capture 時送進 native 的單一 HM，bin 中 baked-in 的縮放係數。</summary>
        public float HessianMaxFactorV { get; }
        /// <summary>列正規值 — view-time only，僅供 H 曲線顯示縮放參考。</summary>
        public float HessianMaxFactorH { get; }
        /// <summary>細線濾除（capture-time ridge_sigma）。</summary>
        public float RidgeSigma { get; }
        public float ErrorValueMeanV { get; }  // 欄平均閾值
        public float ErrorValueMaxV  { get; }  // 欄最大閾值
        public float ErrorValueMeanH { get; }  // 列平均閾值
        public float ErrorValueMaxH  { get; }  // 列最大閾值
        public double TrimHeadMm { get; }
        public double TrimTailMm { get; }
        public DateTime Timestamp { get; }

        public CsvConfigSnapshot(
            double[] camOps, double[] camPos, int[] camGrabHeight,
            double[] camExposureUs, double[] camLineRateHz,
            float hessianMaxFactorV, float hessianMaxFactorH,
            float errorValueMeanV, float errorValueMaxV,
            float errorValueMeanH, float errorValueMaxH,
            double trimHeadMm, double trimTailMm,
            DateTime timestamp)
            : this(
                camOps, camPos, camGrabHeight, camExposureUs, camLineRateHz,
                hessianMaxFactorV, hessianMaxFactorH, 0f,
                errorValueMeanV, errorValueMaxV, errorValueMeanH, errorValueMaxH,
                trimHeadMm, trimTailMm, timestamp)
        {
        }

        public CsvConfigSnapshot(
            double[] camOps, double[] camPos, int[] camGrabHeight,
            double[] camExposureUs, double[] camLineRateHz,
            float hessianMaxFactorV, float hessianMaxFactorH, float ridgeSigma,
            float errorValueMeanV, float errorValueMaxV,
            float errorValueMeanH, float errorValueMaxH,
            double trimHeadMm, double trimTailMm,
            DateTime timestamp)
        {
            CamOps = camOps ?? new double[7];
            CamPos = camPos ?? new double[7];
            CamGrabHeight = camGrabHeight ?? new int[7];
            CamExposureUs = camExposureUs ?? new double[7];
            CamLineRateHz = camLineRateHz ?? new double[7];
            HessianMaxFactorV = hessianMaxFactorV;
            HessianMaxFactorH = hessianMaxFactorH;
            RidgeSigma = ridgeSigma;
            ErrorValueMeanV = errorValueMeanV;
            ErrorValueMaxV  = errorValueMaxV;
            ErrorValueMeanH = errorValueMeanH;
            ErrorValueMaxH  = errorValueMaxH;
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
                s.HessianMaxFactorV, s.HessianMaxFactorH, s.RidgeSigma,
                s.ErrorValueMeanV, s.ErrorValueMaxV,
                s.ErrorValueMeanH, s.ErrorValueMaxH,
                s.TrimHeadMm,
                s.TrimTailMm,
                DateTime.Now);
        }

        /// <summary>不含時間戳的內容鍵，用於偵測設定是否變更。</summary>
        public string ContentKey
        {
            get
            {
                // L5: 用 "R"（round-trip）保證 float 精度不丟失（F4 對 0.0001 級會截斷成相同 key）
                var sb = new StringBuilder(256);
                var inv = CultureInfo.InvariantCulture;
                for (int i = 0; i < 7; i++) sb.Append(CamOps[i].ToString("R", inv)).Append(',');
                for (int i = 0; i < 7; i++) sb.Append(CamPos[i].ToString("R", inv)).Append(',');
                for (int i = 0; i < 7; i++) sb.Append(CamGrabHeight[i]).Append(',');
                for (int i = 0; i < 7; i++) sb.Append(CamExposureUs[i].ToString("R", inv)).Append(',');
                for (int i = 0; i < 7; i++) sb.Append(CamLineRateHz[i].ToString("R", inv)).Append(',');
                sb.Append(HessianMaxFactorV.ToString("R", inv)).Append(',');
                sb.Append(HessianMaxFactorH.ToString("R", inv)).Append(',');
                sb.Append(RidgeSigma.ToString("R", inv)).Append(',');
                sb.Append(ErrorValueMeanV.ToString("R", inv)).Append(',');
                sb.Append(ErrorValueMaxV.ToString("R", inv)).Append(',');
                sb.Append(ErrorValueMeanH.ToString("R", inv)).Append(',');
                sb.Append(ErrorValueMaxH.ToString("R", inv)).Append(',');
                sb.Append(TrimHeadMm.ToString("R", inv)).Append(',');
                sb.Append(TrimTailMm.ToString("R", inv));
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
            sb.Append($",TrimHead={TrimHeadMm:F2}");
            sb.Append($",TrimTail={TrimTailMm:F2}");
            for (int i = 0; i < 7; i++)
                sb.Append($",Cam{i + 1}_GrabH={CamGrabHeight[i]}");
            for (int i = 0; i < 7; i++)
                sb.Append($",Cam{i + 1}_Exp={CamExposureUs[i]:F2}");
            for (int i = 0; i < 7; i++)
                sb.Append($",Cam{i + 1}_Lr={CamLineRateHz[i]:F2}");
            sb.Append($",HessianMaxFactorV={HessianMaxFactorV:F4}");
            sb.Append($",HessianMaxFactorH={HessianMaxFactorH:F4}");
            sb.Append($",RidgeSigma={RidgeSigma:F4}");
            sb.Append($",ErrorValueMeanV={ErrorValueMeanV:F4}");
            sb.Append($",ErrorValueMaxV={ErrorValueMaxV:F4}");
            sb.Append($",ErrorValueMeanH={ErrorValueMeanH:F4}");
            sb.Append($",ErrorValueMaxH={ErrorValueMaxH:F4}");
            return sb.ToString();
        }

        /// <summary>從 #CFG 列解析。只支援新格式（V/H 分裂）。未認得的欄位忽略。</summary>
        public static bool TryParse(string line, out CsvConfigSnapshot result)
        {
            result = null;
            if (string.IsNullOrEmpty(line) || !line.StartsWith("#CFG,")) return false;

            string[] parts = line.Split(',');
            if (parts.Length < 3) return false;

            if (!DateTime.TryParseExact(parts[1].Trim(), "yyyy-MM-ddTHH:mm:ss.fff",
                CultureInfo.InvariantCulture, DateTimeStyles.None, out DateTime ts))
                return false;

            double[] ops = new double[7];
            double[] pos = new double[7];
            int[] grabH = new int[7];
            double[] expUs = new double[7];
            double[] lrHz = new double[7];
            float hessianV = 0, hessianH = 0, ridgeSigma = 0;
            float meanV = 0, maxV = 0, meanH = 0, maxH = 0;
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
                else if (key == "HessianMaxFactorV")
                    float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out hessianV);
                else if (key == "HessianMaxFactorH")
                    float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out hessianH);
                else if (key == "RidgeSigma")
                    float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out ridgeSigma);
                else if (key == "ErrorValueMeanV")
                    float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out meanV);
                else if (key == "ErrorValueMaxV")
                    float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out maxV);
                else if (key == "ErrorValueMeanH")
                    float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out meanH);
                else if (key == "ErrorValueMaxH")
                    float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out maxH);
                else if (key == "TrimHead")
                    double.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out trimHead);
                else if (key == "TrimTail")
                    double.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out trimTail);
            }

            // M4: V/H 互補 — 若只寫了一邊（不太可能但理論上會發生），鏡像到另一邊
            // 避免 hessianH=0 導致下游 rescale ratio 變成 1 退回原 baked-in 值
            if (hessianV > 0f && hessianH <= 0f) hessianH = hessianV;
            else if (hessianH > 0f && hessianV <= 0f) hessianV = hessianH;
            if (meanV > 0f && meanH <= 0f) meanH = meanV;
            else if (meanH > 0f && meanV <= 0f) meanV = meanH;
            if (maxV > 0f && maxH <= 0f) maxH = maxV;
            else if (maxH > 0f && maxV <= 0f) maxV = maxH;

            result = new CsvConfigSnapshot(ops, pos, grabH, expUs, lrHz,
                hessianV, hessianH, ridgeSigma,
                meanV, maxV, meanH, maxH, trimHead, trimTail, ts);
            return true;
        }
    }
}
