using System.IO;
using System.Text;
using System.Web.Script.Serialization;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// InspectionSettings 儲存層。
    /// 讀取順序：user JSON → defaults JSON。
    /// 儲存位置：Config\inspection-settings.user.json（與 exe 同目錄下的 Config 資料夾）。
    /// </summary>
    public static class InspectionSettingsStore
    {
        private const string UserConfigPath = "Config\\inspection-settings.user.json";

        private static string FullUserConfigPath =>
            Path.Combine(System.AppDomain.CurrentDomain.BaseDirectory, UserConfigPath);

        public static InspectionSettings Load()
        {
            InspectionSettings defaults = InspectionSettingsDefaultsProvider.LoadDefaults();
            defaults.Validate();

            try
            {
                string path = FullUserConfigPath;
                if (!File.Exists(path)) return defaults;

                string json = File.ReadAllText(path, Encoding.UTF8);
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

                string path = FullUserConfigPath;
                string dir  = Path.GetDirectoryName(path);
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);

                JavaScriptSerializer serializer = new JavaScriptSerializer();
                File.WriteAllText(path, serializer.Serialize(settings), Encoding.UTF8);
            }
            catch
            {
                // Ignore save error
            }
        }
    }
}
