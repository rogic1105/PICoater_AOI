using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace AniloxRoll.Monitor.UI.State
{
    /// <summary>
    /// UI session 狀態（上次選擇的資料夾、時間篩選、是否啟用影像處理）。
    /// 持久化至 Config\session-state.json（與 exe 同目錄），與其他 Store 架構一致。
    /// </summary>
    public static class UserSessionState
    {
        private static readonly string FullConfigPath =
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"Config\session-state.json");

        // 使用 Dictionary 避免 private class 造成 JavaScriptSerializer 反射失敗
        private static Dictionary<string, string> _data = Load();

        private static Dictionary<string, string> Load()
        {
            if (!File.Exists(FullConfigPath))
                return new Dictionary<string, string>(); // 首次執行，正常情況

            try
            {
                string json = File.ReadAllText(FullConfigPath, Encoding.UTF8);
                return ParseJson(json);
            }
            catch (Exception ex)
            {
                // 檔案存在但讀取/解析失敗（磁碟錯誤、權限不足、JSON 損壞）
                System.Diagnostics.Trace.WriteLine(
                    $"[UserSessionState] Load failed ({FullConfigPath}): {ex.GetType().Name}: {ex.Message}. Using empty state.");
                return new Dictionary<string, string>();
            }
        }

        /// <summary>簡單解析 {"key":"value",...} 格式，不依賴 ConfigurationManager。</summary>
        private static Dictionary<string, string> ParseJson(string json)
        {
            var result = new Dictionary<string, string>();
            if (string.IsNullOrWhiteSpace(json)) return result;

            foreach (Match m in Regex.Matches(json, "\"([^\"]+)\"\\s*:\\s*\"([^\"]*)\""))
            {
                string key = m.Groups[1].Value;
                string val = m.Groups[2].Value
                    .Replace("\\\\", "\x00BSLASH\x00")
                    .Replace("\\\"", "\"")
                    .Replace("\x00BSLASH\x00", "\\");
                result[key] = val;
            }
            return result;
        }

        /// <summary>序列化為縮排 JSON，不依賴 ConfigurationManager。</summary>
        private static string SerializeJson(Dictionary<string, string> data)
        {
            var entries = new List<string>();
            foreach (var kv in data)
                entries.Add($"  \"{EscapeJson(kv.Key)}\": \"{EscapeJson(kv.Value)}\"");

            return entries.Count == 0
                ? "{\n}"
                : "{\n" + string.Join(",\n", entries) + "\n}";
        }

        private static string EscapeJson(string s)
            => (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"");

        private static string Get(string key)
        {
            return _data.TryGetValue(key, out string val) ? (val ?? string.Empty) : string.Empty;
        }

        private static void WriteToFile(Dictionary<string, string> data)
        {
            try
            {
                string dir = Path.GetDirectoryName(FullConfigPath);
                Directory.CreateDirectory(dir);
                byte[] bytes = new UTF8Encoding(false).GetBytes(SerializeJson(data));
                using (var fs = new FileStream(FullConfigPath, FileMode.Create,
                                               FileAccess.Write, FileShare.ReadWrite))
                {
                    fs.Write(bytes, 0, bytes.Length);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(
                    $"[UserSessionState] Save failed: {ex.GetType().Name}: {ex.Message}");
            }
        }

        public static void Save() => WriteToFile(_data);

        public static string LastDataPath              => Get("LastDataPath");
        public static string LastYear                  => Get("LastYear");
        public static string LastMonth                 => Get("LastMonth");
        public static string LastDay                   => Get("LastDay");
        public static string LastHour                  => Get("LastHour");
        public static string LastMin                   => Get("LastMin");
        public static string LastSec                   => Get("LastSec");
        public static bool MainWorkspaceFullWidth
        {
            get
            {
                return bool.TryParse(Get("MainWorkspaceFullWidth"), out bool value) && value;
            }
        }

        public static void SetLastDataPath(string path)
            => _data["LastDataPath"] = path ?? string.Empty;

        public static void SaveMainWorkspaceFullWidth(bool fullWidth)
        {
            _data["MainWorkspaceFullWidth"] = fullWidth ? "true" : "false";
            Save();
        }

        public static void SaveDateTimeSelection(string year, string month, string day,
                                                  string hour, string min, string sec)
        {
            _data["LastYear"]  = year  ?? string.Empty;
            _data["LastMonth"] = month ?? string.Empty;
            _data["LastDay"]   = day   ?? string.Empty;
            _data["LastHour"]  = hour  ?? string.Empty;
            _data["LastMin"]   = min   ?? string.Empty;
            _data["LastSec"]   = sec   ?? string.Empty;
        }
    }
}
