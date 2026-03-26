using System;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// InspectionSettings 儲存層（MachineLayout + Recipe + Storage）。
    /// 讀寫 Config\inspection-settings.json（與 exe 同目錄）。
    /// 不使用 JavaScriptSerializer（user.config 損毀時拋 ConfigurationErrorsException）。
    /// </summary>
    public static class InspectionSettingsStore
    {
        private static string FullConfigPath =>
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"Config\inspection-settings.json");

        // ── Load ──────────────────────────────────────────────────────────

        public static InspectionSettings Load()
        {
            try
            {
                if (!File.Exists(FullConfigPath))
                {
                    var defaults = new InspectionSettings();
                    defaults.Validate();
                    Save(defaults);
                    return defaults;
                }

                string json = File.ReadAllText(FullConfigPath, Encoding.UTF8);
                if (string.IsNullOrWhiteSpace(json)) return new InspectionSettings();

                var settings = ParseJson(json);
                if (settings == null) return new InspectionSettings();
                settings.Validate();
                return settings;
            }
            catch
            {
                return new InspectionSettings();
            }
        }

        // ── Save ──────────────────────────────────────────────────────────

        public static void Save(InspectionSettings settings)
        {
            try
            {
                if (settings == null) settings = new InspectionSettings();
                settings.Validate();

                string dir = Path.GetDirectoryName(FullConfigPath);
                Directory.CreateDirectory(dir);
                byte[] bytes = new UTF8Encoding(false).GetBytes(SerializeJson(settings));
                using (var fs = new FileStream(FullConfigPath, FileMode.Create,
                                               FileAccess.Write, FileShare.ReadWrite))
                {
                    fs.Write(bytes, 0, bytes.Length);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(
                    $"[InspectionSettingsStore.Save] {ex.GetType().Name}: {ex.Message}");
            }
        }

        // ── 序列化 ────────────────────────────────────────────────────────

        private static string SerializeJson(InspectionSettings s)
        {
            var L = s.MachineLayout ?? new MachineLayoutConfig();
            var R = s.Recipe        ?? new InspectionRecipe();
            var C = s.Chart         ?? new ChartSettings();
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
            sb.AppendLine($"    \"BackgroundSampleRows\": {R.BackgroundSampleRows},");
            sb.AppendLine($"    \"AniloxRollSpeedMPerMin\": {D(R.AniloxRollSpeedMPerMin)},");
            sb.AppendLine($"    \"SaveResizeScale\": {R.SaveResizeScale},");
            sb.AppendLine($"    \"SaveJpgQuality\": {R.SaveJpgQuality}");
            sb.AppendLine("  },");

            // Chart
            sb.AppendLine("  \"Chart\": {");
            sb.AppendLine($"    \"ScaleMode\": \"{C.ScaleMode}\",");
            sb.AppendLine($"    \"YearlyYMax\": {C.YearlyYMax},");
            sb.AppendLine($"    \"MonthlyYMax\": {C.MonthlyYMax},");
            sb.AppendLine($"    \"DailyYMax\": {C.DailyYMax}");
            sb.AppendLine("  },");

            // Storage
            sb.AppendLine("  \"Storage\": {");
            sb.AppendLine($"    \"EnableAutoCapture\": {(T.EnableAutoCapture ? "true" : "false")},");
            sb.AppendLine($"    \"CaptureRootPath\": \"{EscapeJson(T.CaptureRootPath)}\",");
            sb.AppendLine($"    \"SaveOriginalBmp\": {(T.SaveOriginalBmp ? "true" : "false")}");
            sb.AppendLine("  }");

            sb.Append("}");
            return sb.ToString();
        }

        private static string D(double v) => v.ToString("G", CultureInfo.InvariantCulture);
        private static string F(float  v) => v.ToString("G", CultureInfo.InvariantCulture);
        private static string EscapeJson(string s) => (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"");

        // ── 解析 ──────────────────────────────────────────────────────────

        private static InspectionSettings ParseJson(string json)
        {
            var settings = new InspectionSettings
            {
                MachineLayout = ParseMachineLayout(json),
                Recipe        = ParseRecipe(json),
                Chart         = ParseChart(json),
                Storage       = ParseStorage(json),
            };
            return settings;
        }

        /// <summary>從整體 JSON 中提取指定頂層 key 的物件內容（支援多行）。</summary>
        private static string ExtractObject(string json, string key)
        {
            var m = Regex.Match(json,
                "\"" + key + "\"\\s*:\\s*\\{([^}]*)\\}",
                RegexOptions.Singleline);
            return m.Success ? m.Groups[1].Value : string.Empty;
        }

        private static double GetDouble(string obj, string key, double fallback)
        {
            var m = Regex.Match(obj, "\"" + key + "\"\\s*:\\s*([\\d.eE+\\-]+)");
            if (!m.Success) return fallback;
            return double.TryParse(m.Groups[1].Value, NumberStyles.Any,
                CultureInfo.InvariantCulture, out double v) ? v : fallback;
        }

        private static int GetInt(string obj, string key, int fallback)
        {
            var m = Regex.Match(obj, "\"" + key + "\"\\s*:\\s*([\\d]+)");
            if (!m.Success) return fallback;
            return int.TryParse(m.Groups[1].Value, out int v) ? v : fallback;
        }

        private static float GetFloat(string obj, string key, float fallback)
        {
            var m = Regex.Match(obj, "\"" + key + "\"\\s*:\\s*([\\d.eE+\\-]+)");
            if (!m.Success) return fallback;
            return float.TryParse(m.Groups[1].Value, NumberStyles.Any,
                CultureInfo.InvariantCulture, out float v) ? v : fallback;
        }

        private static bool GetBool(string obj, string key, bool fallback)
        {
            var m = Regex.Match(obj, "\"" + key + "\"\\s*:\\s*(true|false)",
                RegexOptions.IgnoreCase);
            if (!m.Success) return fallback;
            return string.Equals(m.Groups[1].Value, "true", StringComparison.OrdinalIgnoreCase);
        }

        private static string GetString(string obj, string key, string fallback)
        {
            var m = Regex.Match(obj, "\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
            if (!m.Success) return fallback;
            return m.Groups[1].Value
                .Replace("\\\\", "\x00BS\x00")
                .Replace("\\\"", "\"")
                .Replace("\x00BS\x00", "\\");
        }

        private static MachineLayoutConfig ParseMachineLayout(string json)
        {
            string obj = ExtractObject(json, "MachineLayout");
            return new MachineLayoutConfig
            {
                Cam1_Ops = GetDouble(obj, "Cam1_Ops", 33.0),
                Cam2_Ops = GetDouble(obj, "Cam2_Ops", 33.0),
                Cam3_Ops = GetDouble(obj, "Cam3_Ops", 33.0),
                Cam4_Ops = GetDouble(obj, "Cam4_Ops", 33.0),
                Cam5_Ops = GetDouble(obj, "Cam5_Ops", 33.0),
                Cam6_Ops = GetDouble(obj, "Cam6_Ops", 33.0),
                Cam7_Ops = GetDouble(obj, "Cam7_Ops", 33.0),
                Cam1_Pos = GetDouble(obj, "Cam1_Pos",    0.0),
                Cam2_Pos = GetDouble(obj, "Cam2_Pos",  400.0),
                Cam3_Pos = GetDouble(obj, "Cam3_Pos",  800.0),
                Cam4_Pos = GetDouble(obj, "Cam4_Pos", 1200.0),
                Cam5_Pos = GetDouble(obj, "Cam5_Pos", 1600.0),
                Cam6_Pos = GetDouble(obj, "Cam6_Pos", 2000.0),
                Cam7_Pos = GetDouble(obj, "Cam7_Pos", 2400.0),
            };
        }

        private static InspectionRecipe ParseRecipe(string json)
        {
            string obj = ExtractObject(json, "Recipe");

            BackgroundAlgorithm algo = BackgroundAlgorithm.SingleFrameBgSub;
            string algoStr = GetString(obj, "Algorithm", "SingleFrameBgSub");
            if (!System.Enum.TryParse(algoStr, true, out algo))
                algo = BackgroundAlgorithm.SingleFrameBgSub;

            RidgeDirection ridgeDir = RidgeDirection.Vertical;
            string ridgeDirStr = GetString(obj, "RidgeDir", "Vertical");
            if (!System.Enum.TryParse(ridgeDirStr, true, out ridgeDir))
                ridgeDir = RidgeDirection.Vertical;

            return new InspectionRecipe
            {
                Algorithm        = algo,
                RidgeDir         = ridgeDir,
                HessianMaxFactor = GetFloat(obj, "HessianMaxFactor", 5.0f),
                ErrorValueMean   = GetFloat(obj, "ErrorValueMean",   1.0f),
                ErrorValueMax    = GetFloat(obj, "ErrorValueMax",     2.0f),
                BackgroundSampleRows = GetInt(obj, "BackgroundSampleRows", 10000),
                AniloxRollSpeedMPerMin = GetDouble(obj, "AniloxRollSpeedMPerMin", 10.0),
                SaveResizeScale  = GetInt  (obj, "SaveResizeScale",   5),
                SaveJpgQuality   = GetInt  (obj, "SaveJpgQuality",    90),
            };
        }

        private static ChartSettings ParseChart(string json)
        {
            string obj = ExtractObject(json, "Chart");

            ChartScaleMode mode = ChartScaleMode.Fixed;
            string modeStr = GetString(obj, "ScaleMode", "Fixed");
            if (!System.Enum.TryParse(modeStr, true, out mode))
                mode = ChartScaleMode.Fixed;

            return new ChartSettings
            {
                ScaleMode   = mode,
                YearlyYMax  = GetInt(obj, "YearlyYMax",  60000),
                MonthlyYMax = GetInt(obj, "MonthlyYMax",  2000),
                DailyYMax   = GetInt(obj, "DailyYMax",     300),
            };
        }

        private static StorageSettings ParseStorage(string json)
        {
            string obj = ExtractObject(json, "Storage");
            return new StorageSettings
            {
                EnableAutoCapture    = GetBool  (obj, "EnableAutoCapture",    false),
                CaptureRootPath      = GetString(obj, "CaptureRootPath",      @"D:\AniloxCaptures"),
                // 向後相容：舊 JSON 有 UseCompressedCapture（true=壓縮），反轉為 SaveOriginalBmp
                SaveOriginalBmp      = obj.Contains("SaveOriginalBmp")
                    ? GetBool(obj, "SaveOriginalBmp", false)
                    : !GetBool(obj, "UseCompressedCapture", true),
            };
        }
    }
}
