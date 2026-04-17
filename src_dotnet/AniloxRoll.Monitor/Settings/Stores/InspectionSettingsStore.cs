using System;
using System.Globalization;
using System.IO;
using System.Text;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// InspectionSettings 儲存層（MachineLayout + Recipe + Storage）。
    /// 讀寫 Config\inspection-settings.json（與 exe 同目錄）。
    /// </summary>
    public static class InspectionSettingsStore
    {
        private static string FullConfigPath =>
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"Config\inspection-settings.json");

        // ── Load ──────────────────────────────────────────────────────────

        public static InspectionSettings Load()
        {
            var settings = SettingsStoreHelper.LoadJsonFile(
                FullConfigPath,
                ParseJson,
                () => { var d = new InspectionSettings(); d.Validate(); return d; });

            if (settings != null) settings.Validate();

            // 檔案不存在時自動建立預設檔
            if (!File.Exists(FullConfigPath))
                Save(settings);

            return settings;
        }

        // ── Save ──────────────────────────────────────────────────────────

        public static void Save(InspectionSettings settings)
        {
            if (settings == null) settings = new InspectionSettings();
            settings.Validate();
            SettingsStoreHelper.SaveJsonFile(FullConfigPath, SerializeJson(settings),
                nameof(InspectionSettingsStore));
        }

        // ── 序列化 ────────────────────────────────────────────────────────

        private static string SerializeJson(InspectionSettings s)
        {
            var L = s.MachineLayout ?? new MachineLayoutConfig();
            var R = s.Recipe        ?? new InspectionRecipe();
            var C = s.Chart         ?? new ChartSettings();
            var V = s.ImageView     ?? new ImageViewSettings();
            var T = s.Storage       ?? new StorageSettings();

            var sb = new StringBuilder();
            sb.AppendLine("{");

            // MachineLayout
            sb.AppendLine("  \"MachineLayout\": {");
            sb.AppendLine($"    \"Cam1_Ops\": {D(L.Cam1_Ops)},");
            sb.AppendLine($"    \"Cam2_Ops\": {D(L.Cam2_Ops)},");
            sb.AppendLine($"    \"Cam3_Ops\": {D(L.Cam3_Ops)},");
            sb.AppendLine($"    \"Cam4_Ops\": {D(L.Cam4_Ops)},");
            sb.AppendLine($"    \"Cam5_Ops\": {D(L.Cam5_Ops)},");
            sb.AppendLine($"    \"Cam6_Ops\": {D(L.Cam6_Ops)},");
            sb.AppendLine($"    \"Cam7_Ops\": {D(L.Cam7_Ops)},");
            sb.AppendLine($"    \"Cam1_Pos\": {D(L.Cam1_Pos)},");
            sb.AppendLine($"    \"Cam2_Pos\": {D(L.Cam2_Pos)},");
            sb.AppendLine($"    \"Cam3_Pos\": {D(L.Cam3_Pos)},");
            sb.AppendLine($"    \"Cam4_Pos\": {D(L.Cam4_Pos)},");
            sb.AppendLine($"    \"Cam5_Pos\": {D(L.Cam5_Pos)},");
            sb.AppendLine($"    \"Cam6_Pos\": {D(L.Cam6_Pos)},");
            sb.AppendLine($"    \"Cam7_Pos\": {D(L.Cam7_Pos)}");
            sb.AppendLine("  },");

            // Recipe
            sb.AppendLine("  \"Recipe\": {");
            sb.AppendLine($"    \"Algorithm\": \"{R.Algorithm}\",");
            sb.AppendLine($"    \"RidgeDir\": \"{R.RidgeDir}\",");
            sb.AppendLine($"    \"HessianMaxFactor\": {F(R.HessianMaxFactor)},");
            sb.AppendLine($"    \"ErrorValueMean\": {F(R.ErrorValueMean)},");
            sb.AppendLine($"    \"ErrorValueMax\": {F(R.ErrorValueMax)},");
            sb.AppendLine($"    \"BackgroundSampleSeconds\": {R.BackgroundSampleSeconds},");
            sb.AppendLine($"    \"AniloxRollSpeedMPerMin\": {D(R.AniloxRollSpeedMPerMin)},");
            sb.AppendLine($"    \"SaveResizeScale\": {R.SaveResizeScale},");
            sb.AppendLine($"    \"SaveJpgQuality\": {R.SaveJpgQuality},");
            sb.AppendLine($"    \"EnableMuraEnhance\": {(R.EnableMuraEnhance ? "true" : "false")},");
            sb.AppendLine($"    \"EnableReviewEnhance\": {(R.EnableReviewEnhance ? "true" : "false")}");
            sb.AppendLine("  },");

            // Chart
            sb.AppendLine("  \"Chart\": {");
            sb.AppendLine($"    \"ScaleMode\": \"{C.ScaleMode}\",");
            sb.AppendLine($"    \"YearlyYMax\": {C.YearlyYMax},");
            sb.AppendLine($"    \"MonthlyYMax\": {C.MonthlyYMax},");
            sb.AppendLine($"    \"DailyYMax\": {C.DailyYMax}");
            sb.AppendLine("  },");

            // ImageView
            sb.AppendLine("  \"ImageView\": {");
            sb.AppendLine($"    \"StitchMode\": \"{V.StitchMode}\"");
            sb.AppendLine("  },");

            // Storage
            sb.AppendLine("  \"Storage\": {");
            sb.AppendLine($"    \"EnableAutoCapture\": {(T.EnableAutoCapture ? "true" : "false")},");
            sb.AppendLine($"    \"CaptureRootPath\": \"{SettingsStoreHelper.EscapeJson(T.CaptureRootPath)}\",");
            sb.AppendLine($"    \"SaveOriginalBmp\": {(T.SaveOriginalBmp ? "true" : "false")},");
            sb.AppendLine($"    \"BackgroundPath\": \"{SettingsStoreHelper.EscapeJson(T.BackgroundPath)}\",");
            sb.AppendLine($"    \"LocalMaxGB\": {T.LocalMaxGB},");
            sb.AppendLine($"    \"FailProtectDays\": {T.FailProtectDays},");
            sb.AppendLine($"    \"RemotePath\": \"{SettingsStoreHelper.EscapeJson(T.RemotePath)}\"");
            sb.AppendLine("  },");

            // PLC
            sb.AppendLine("  \"Plc\": {");
            sb.AppendLine($"    \"Enabled\": {(s.PlcEnabled ? "true" : "false")},");
            sb.AppendLine($"    \"Ip\": \"{SettingsStoreHelper.EscapeJson(s.PlcIp)}\",");
            sb.AppendLine($"    \"Port\": {s.PlcPort}");
            sb.AppendLine("  },");

            // CameraParam
            var P = s.CameraParam ?? new CameraParamSettings();
            sb.AppendLine("  \"CameraParam\": {");
            sb.AppendLine($"    \"DcfPath\": \"{SettingsStoreHelper.EscapeJson(P.DcfPath)}\"");
            sb.AppendLine("  },");

            // Light
            var LT = s.Light ?? new LightSettings();
            sb.AppendLine("  \"Light\": {");
            sb.AppendLine($"    \"Enabled\": {(LT.Enabled ? "true" : "false")},");
            sb.AppendLine($"    \"ComPort\": \"{SettingsStoreHelper.EscapeJson(LT.ComPort)}\",");
            sb.AppendLine($"    \"BrightnessCh1\": {LT.BrightnessCh1},");
            sb.AppendLine($"    \"BrightnessCh2\": {LT.BrightnessCh2}");
            sb.AppendLine("  }");

            sb.Append("}");
            return sb.ToString();
        }

        private static string D(double v) => v.ToString("G", CultureInfo.InvariantCulture);
        private static string F(float  v) => v.ToString("G", CultureInfo.InvariantCulture);

        // ── 解析 ──────────────────────────────────────────────────────────

        private static InspectionSettings ParseJson(string json)
        {
            return new InspectionSettings
            {
                MachineLayout = ParseMachineLayout(json),
                Recipe        = ParseRecipe(json),
                Chart         = ParseChart(json),
                ImageView     = ParseImageView(json),
                Storage       = ParseStorage(json),
                CameraParam   = ParseCameraParam(json),
                Light         = ParseLight(json),
                PlcEnabled    = ParsePlcEnabled(json),
                PlcIp         = ParsePlcIp(json),
                PlcPort       = ParsePlcPort(json),
            };
        }

        private static MachineLayoutConfig ParseMachineLayout(string json)
        {
            string obj = SettingsStoreHelper.ExtractObject(json, "MachineLayout");
            return new MachineLayoutConfig
            {
                Cam1_Ops = SettingsStoreHelper.GetDouble(obj, "Cam1_Ops", 33.0),
                Cam2_Ops = SettingsStoreHelper.GetDouble(obj, "Cam2_Ops", 33.0),
                Cam3_Ops = SettingsStoreHelper.GetDouble(obj, "Cam3_Ops", 33.0),
                Cam4_Ops = SettingsStoreHelper.GetDouble(obj, "Cam4_Ops", 33.0),
                Cam5_Ops = SettingsStoreHelper.GetDouble(obj, "Cam5_Ops", 33.0),
                Cam6_Ops = SettingsStoreHelper.GetDouble(obj, "Cam6_Ops", 33.0),
                Cam7_Ops = SettingsStoreHelper.GetDouble(obj, "Cam7_Ops", 33.0),
                Cam1_Pos = SettingsStoreHelper.GetDouble(obj, "Cam1_Pos",    0.0),
                Cam2_Pos = SettingsStoreHelper.GetDouble(obj, "Cam2_Pos",  400.0),
                Cam3_Pos = SettingsStoreHelper.GetDouble(obj, "Cam3_Pos",  800.0),
                Cam4_Pos = SettingsStoreHelper.GetDouble(obj, "Cam4_Pos", 1200.0),
                Cam5_Pos = SettingsStoreHelper.GetDouble(obj, "Cam5_Pos", 1600.0),
                Cam6_Pos = SettingsStoreHelper.GetDouble(obj, "Cam6_Pos", 2000.0),
                Cam7_Pos = SettingsStoreHelper.GetDouble(obj, "Cam7_Pos", 2400.0),
            };
        }

        private static InspectionRecipe ParseRecipe(string json)
        {
            string obj = SettingsStoreHelper.ExtractObject(json, "Recipe");

            BackgroundAlgorithm algo = BackgroundAlgorithm.SingleFrameBgSub;
            string algoStr = SettingsStoreHelper.GetString(obj, "Algorithm", "SingleFrameBgSub");
            if (!System.Enum.TryParse(algoStr, true, out algo))
                algo = BackgroundAlgorithm.SingleFrameBgSub;

            RidgeDirection ridgeDir = RidgeDirection.Vertical;
            string ridgeDirStr = SettingsStoreHelper.GetString(obj, "RidgeDir", "Vertical");
            if (!System.Enum.TryParse(ridgeDirStr, true, out ridgeDir))
                ridgeDir = RidgeDirection.Vertical;

            return new InspectionRecipe
            {
                Algorithm        = algo,
                RidgeDir         = ridgeDir,
                HessianMaxFactor = SettingsStoreHelper.GetFloat(obj, "HessianMaxFactor", 5.0f),
                ErrorValueMean   = SettingsStoreHelper.GetFloat(obj, "ErrorValueMean",   1.0f),
                ErrorValueMax    = SettingsStoreHelper.GetFloat(obj, "ErrorValueMax",     2.0f),
                // 向後相容：舊 JSON 用 BackgroundSampleRows，新 JSON 用 BackgroundSampleSeconds
                BackgroundSampleSeconds = obj.Contains("BackgroundSampleSeconds")
                    ? SettingsStoreHelper.GetInt(obj, "BackgroundSampleSeconds", 3)
                    : 3,
                AniloxRollSpeedMPerMin = SettingsStoreHelper.GetDouble(obj, "AniloxRollSpeedMPerMin", 10.0),
                SaveResizeScale      = SettingsStoreHelper.GetInt(obj, "SaveResizeScale",   5),
                SaveJpgQuality       = SettingsStoreHelper.GetInt(obj, "SaveJpgQuality",    90),
                EnableMuraEnhance    = SettingsStoreHelper.GetBool(obj, "EnableMuraEnhance",    false),
                EnableReviewEnhance  = SettingsStoreHelper.GetBool(obj, "EnableReviewEnhance",  false),
            };
        }

        private static ChartSettings ParseChart(string json)
        {
            string obj = SettingsStoreHelper.ExtractObject(json, "Chart");

            ChartScaleMode mode = ChartScaleMode.Fixed;
            string modeStr = SettingsStoreHelper.GetString(obj, "ScaleMode", "Fixed");
            if (!System.Enum.TryParse(modeStr, true, out mode))
                mode = ChartScaleMode.Fixed;

            return new ChartSettings
            {
                ScaleMode   = mode,
                YearlyYMax  = SettingsStoreHelper.GetInt(obj, "YearlyYMax",  60000),
                MonthlyYMax = SettingsStoreHelper.GetInt(obj, "MonthlyYMax",  2000),
                DailyYMax   = SettingsStoreHelper.GetInt(obj, "DailyYMax",     300),
            };
        }

        private static ImageViewSettings ParseImageView(string json)
        {
            // 優先從 ImageView 區塊讀取；向後相容：舊 JSON 的 StitchMode 放在 Chart 區塊
            string obj = SettingsStoreHelper.ExtractObject(json, "ImageView");
            string src = !string.IsNullOrEmpty(obj)
                ? obj
                : SettingsStoreHelper.ExtractObject(json, "Chart");

            StitchMode stitchMode = StitchMode.Vertical;
            string stitchStr = SettingsStoreHelper.GetString(src, "StitchMode", "Vertical");
            // 向後相容：舊設定的 "Horizontal" 對映到 Global
            if (string.Equals(stitchStr, "Horizontal", System.StringComparison.OrdinalIgnoreCase))
                stitchMode = StitchMode.Global;
            else if (!System.Enum.TryParse(stitchStr, true, out stitchMode))
                stitchMode = StitchMode.Vertical;

            return new ImageViewSettings
            {
                StitchMode = stitchMode,
            };
        }

        private static StorageSettings ParseStorage(string json)
        {
            string obj = SettingsStoreHelper.ExtractObject(json, "Storage");
            return new StorageSettings
            {
                EnableAutoCapture    = SettingsStoreHelper.GetBool  (obj, "EnableAutoCapture",    false),
                CaptureRootPath      = SettingsStoreHelper.GetString(obj, "CaptureRootPath",      @"D:\AniloxCaptures"),
                // 向後相容：舊 JSON 有 UseCompressedCapture（true=壓縮），反轉為 SaveOriginalBmp
                SaveOriginalBmp      = obj.Contains("SaveOriginalBmp")
                    ? SettingsStoreHelper.GetBool(obj, "SaveOriginalBmp", false)
                    : !SettingsStoreHelper.GetBool(obj, "UseCompressedCapture", true),
                BackgroundPath       = SettingsStoreHelper.GetString(obj, "BackgroundPath", @"D:\AniloxCaptures\bg"),
                LocalMaxGB           = SettingsStoreHelper.GetInt   (obj, "LocalMaxGB",       500),
                FailProtectDays      = SettingsStoreHelper.GetInt   (obj, "FailProtectDays",    30),
                RemotePath           = SettingsStoreHelper.GetString(obj, "RemotePath",        ""),
            };
        }

        private static CameraParamSettings ParseCameraParam(string json)
        {
            string obj = SettingsStoreHelper.ExtractObject(json, "CameraParam");
            return new CameraParamSettings
            {
                DcfPath = SettingsStoreHelper.GetString(obj, "DcfPath", @"D:\AniloxCaptures\dcf\Radient_Config.dcf"),
            };
        }

        private static LightSettings ParseLight(string json)
        {
            string obj = SettingsStoreHelper.ExtractObject(json, "Light");
            return new LightSettings
            {
                Enabled        = SettingsStoreHelper.GetBool  (obj, "Enabled",        false),
                ComPort        = SettingsStoreHelper.GetString(obj, "ComPort",         "COM1"),
                BrightnessCh1  = SettingsStoreHelper.GetInt   (obj, "BrightnessCh1",  128),
                BrightnessCh2  = SettingsStoreHelper.GetInt   (obj, "BrightnessCh2",  128),
            };
        }

        private static bool ParsePlcEnabled(string json)
        {
            string obj = SettingsStoreHelper.ExtractObject(json, "Plc");
            return SettingsStoreHelper.GetBool(obj, "Enabled", true);
        }

        private static string ParsePlcIp(string json)
        {
            string obj = SettingsStoreHelper.ExtractObject(json, "Plc");
            return SettingsStoreHelper.GetString(obj, "Ip", "192.168.255.1");
        }

        private static int ParsePlcPort(string json)
        {
            string obj = SettingsStoreHelper.ExtractObject(json, "Plc");
            return SettingsStoreHelper.GetInt(obj, "Port", 502);
        }
    }
}
