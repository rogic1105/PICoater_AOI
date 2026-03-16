using System;
using System.IO;
using System.Text;
using System.Web.Script.Serialization;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// InspectionSettings 儲存層（固定路徑 JSON 檔，不依賴版本相依的 user.config）。
    /// 儲存位置：%LOCALAPPDATA%\AniloxRoll.Monitor\inspection-settings.json
    /// </summary>
    public static class InspectionSettingsStore
    {
        private static readonly string SavePath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "AniloxRoll.Monitor",
            "inspection-settings.json");

        public static InspectionSettings Load()
        {
            InspectionSettings defaults = InspectionSettingsDefaultsProvider.LoadDefaults();
            defaults.Validate();

            try
            {
                if (!File.Exists(SavePath)) return defaults;

                string json = File.ReadAllText(SavePath, Encoding.UTF8);
                if (string.IsNullOrWhiteSpace(json)) return defaults;

                JavaScriptSerializer serializer = new JavaScriptSerializer();
                InspectionSettings settings = serializer.Deserialize<InspectionSettings>(json) ?? defaults;
                settings.Validate();
                return settings;
            }
            catch
            {
                return defaults;
            }
        }

        public static void Save(InspectionSettings settings)
        {
            try
            {
                if (settings == null) settings = new InspectionSettings();
                settings.Validate();

                string dir = Path.GetDirectoryName(SavePath);
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);

                JavaScriptSerializer serializer = new JavaScriptSerializer();
                string json = serializer.Serialize(settings);
                File.WriteAllText(SavePath, json, Encoding.UTF8);
            }
            catch
            {
                // Ignore save error
            }
        }
    }
}
