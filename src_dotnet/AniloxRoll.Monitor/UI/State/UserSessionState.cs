using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Web.Script.Serialization;
using AniloxRoll.Monitor.Core.Data;

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
            try
            {
                if (!File.Exists(FullConfigPath))
                    return new Dictionary<string, string>();

                string json = File.ReadAllText(FullConfigPath, Encoding.UTF8);
                if (string.IsNullOrWhiteSpace(json))
                    return new Dictionary<string, string>();

                var result = new JavaScriptSerializer()
                    .Deserialize<Dictionary<string, string>>(json);
                return result ?? new Dictionary<string, string>();
            }
            catch
            {
                return new Dictionary<string, string>();
            }
        }

        private static string Get(string key)
        {
            return _data.TryGetValue(key, out string val) ? (val ?? string.Empty) : string.Empty;
        }

        public static void Save()
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(FullConfigPath));
                string json = new JavaScriptSerializer().Serialize(_data);
                File.WriteAllText(FullConfigPath, JsonConfigLoader.FormatJson(json), Encoding.UTF8);
            }
            catch { }
        }

        public static string LastDataPath              => Get("LastDataPath");
        public static string LastYear                  => Get("LastYear");
        public static string LastMonth                 => Get("LastMonth");
        public static string LastDay                   => Get("LastDay");
        public static string LastHour                  => Get("LastHour");
        public static string LastMin                   => Get("LastMin");
        public static string LastSec                   => Get("LastSec");
        public static bool   LastEnableImageProcessing =>
            bool.TryParse(Get("LastEnableImageProcessing"), out bool v) && v;

        public static bool GetLastEnableImageProcessing(bool fallback)
        {
            string raw = Get("LastEnableImageProcessing");
            return string.IsNullOrEmpty(raw) ? fallback : (bool.TryParse(raw, out bool v) && v);
        }

        public static void SetLastDataPath(string path)
            => _data["LastDataPath"] = path ?? string.Empty;

        public static void SetLastEnableImageProcessing(bool enabled)
            => _data["LastEnableImageProcessing"] = enabled.ToString();

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
