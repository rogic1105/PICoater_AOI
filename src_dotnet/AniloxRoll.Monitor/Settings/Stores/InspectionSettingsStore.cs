using System;
using System.IO;
using System.Text;
using System.Web.Script.Serialization;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// InspectionSettings 儲存層。
    /// 直接讀寫 Config\inspection-settings.defaults.json（與 exe 同目錄）。
    /// 該檔案確定存在且可寫，不依賴版本相依的 user.config。
    /// </summary>
    public static class InspectionSettingsStore
    {
        private const string ConfigPath = "Config\\inspection-settings.defaults.json";

        private static string FullConfigPath =>
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ConfigPath);

        public static InspectionSettings Load()
        {
            var settings = InspectionSettingsDefaultsProvider.LoadDefaults();
            settings.Validate();
            return settings;
        }

        public static void Save(InspectionSettings settings)
        {
            try
            {
                if (settings == null) settings = new InspectionSettings();
                settings.Validate();

                var serializer = new JavaScriptSerializer();
                string json = serializer.Serialize(settings);

                // 寫入格式化 JSON，方便人工閱讀
                File.WriteAllText(FullConfigPath, FormatJson(json), Encoding.UTF8);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("[InspectionSettingsStore.Save] " + ex.Message);
            }
        }

        /// <summary>簡單 JSON 格式化（縮排 2 空格）。</summary>
        private static string FormatJson(string json)
        {
            var sb = new StringBuilder();
            int indent = 0;
            bool inString = false;
            foreach (char c in json)
            {
                if (c == '"') inString = !inString;
                if (inString) { sb.Append(c); continue; }
                switch (c)
                {
                    case '{': case '[':
                        sb.Append(c); sb.Append('\n'); sb.Append(new string(' ', ++indent * 2));
                        break;
                    case '}': case ']':
                        sb.Append('\n'); sb.Append(new string(' ', --indent * 2)); sb.Append(c);
                        break;
                    case ',':
                        sb.Append(c); sb.Append('\n'); sb.Append(new string(' ', indent * 2));
                        break;
                    case ':':
                        sb.Append(": ");
                        break;
                    default:
                        sb.Append(c);
                        break;
                }
            }
            return sb.ToString();
        }
    }
}
