using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// Settings Store 共用工具：Load/Save 檔案 I/O + regex JSON 欄位提取。
    /// 不使用 JavaScriptSerializer（user.config 損毀時拋 ConfigurationErrorsException）。
    /// </summary>
    internal static class SettingsStoreHelper
    {
        // ── Load/Save ─────────────────────────────────────────────────

        /// <summary>
        /// 從 JSON 檔案載入設定。檔案不存在時呼叫 createDefault 建立預設值。
        /// 解析失敗時回傳 createDefault()。
        /// </summary>
        public static T LoadJsonFile<T>(string configPath,
            Func<string, T> parser, Func<T> createDefault) where T : class
        {
            try
            {
                if (!File.Exists(configPath))
                    return createDefault();

                string json = File.ReadAllText(configPath, Encoding.UTF8);
                if (string.IsNullOrWhiteSpace(json))
                    return createDefault();

                var result = parser(json);
                return result ?? createDefault();
            }
            catch
            {
                return createDefault();
            }
        }

        /// <summary>
        /// 將 JSON 字串寫入檔案（UTF-8 無 BOM）。失敗時 TraceWarning。
        /// </summary>
        public static void SaveJsonFile(string configPath, string json, string callerName)
        {
            try
            {
                string dir = Path.GetDirectoryName(configPath);
                Directory.CreateDirectory(dir);
                byte[] bytes = new UTF8Encoding(false).GetBytes(json);
                using (var fs = new FileStream(configPath, FileMode.Create,
                                               FileAccess.Write, FileShare.ReadWrite))
                {
                    fs.Write(bytes, 0, bytes.Length);
                }
            }
            catch (Exception ex)
            {
                Trace.TraceWarning($"[{callerName}.Save] {ex.GetType().Name}: {ex.Message}");
            }
        }

        // ── Regex JSON 欄位提取 ───────────────────────────────────────

        /// <summary>從整體 JSON 中提取指定頂層 key 的物件內容（支援多行）。</summary>
        public static string ExtractObject(string json, string key)
        {
            var m = Regex.Match(json,
                "\"" + key + "\"\\s*:\\s*\\{([^}]*)\\}",
                RegexOptions.Singleline);
            return m.Success ? m.Groups[1].Value : string.Empty;
        }

        public static double GetDouble(string obj, string key, double fallback)
        {
            var m = Regex.Match(obj, "\"" + key + "\"\\s*:\\s*([\\d.eE+\\-]+)");
            if (!m.Success) return fallback;
            return double.TryParse(m.Groups[1].Value, NumberStyles.Any,
                CultureInfo.InvariantCulture, out double v) ? v : fallback;
        }

        public static int GetInt(string obj, string key, int fallback)
        {
            var m = Regex.Match(obj, "\"" + key + "\"\\s*:\\s*([\\d]+)");
            if (!m.Success) return fallback;
            return int.TryParse(m.Groups[1].Value, out int v) ? v : fallback;
        }

        public static float GetFloat(string obj, string key, float fallback)
        {
            var m = Regex.Match(obj, "\"" + key + "\"\\s*:\\s*([\\d.eE+\\-]+)");
            if (!m.Success) return fallback;
            return float.TryParse(m.Groups[1].Value, NumberStyles.Any,
                CultureInfo.InvariantCulture, out float v) ? v : fallback;
        }

        public static bool GetBool(string obj, string key, bool fallback)
        {
            var m = Regex.Match(obj, "\"" + key + "\"\\s*:\\s*(true|false)",
                RegexOptions.IgnoreCase);
            if (!m.Success) return fallback;
            return string.Equals(m.Groups[1].Value, "true", StringComparison.OrdinalIgnoreCase);
        }

        public static string GetString(string obj, string key, string fallback)
        {
            var m = Regex.Match(obj, "\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
            if (!m.Success) return fallback;
            return m.Groups[1].Value
                .Replace("\\\\", "\x00BS\x00")
                .Replace("\\\"", "\"")
                .Replace("\x00BS\x00", "\\");
        }

        public static string EscapeJson(string s)
            => (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"");
    }
}
