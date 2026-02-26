using System.Web.Script.Serialization;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// InspectionSettings 儲存層（JSON）。
    /// </summary>
    public static class InspectionSettingsStore
    {
        public static InspectionSettings Load()
        {
            InspectionSettings defaults = InspectionSettingsDefaultsProvider.LoadDefaults();
            defaults.Validate();

            try
            {
                string json = UserSettingsService.InspectionConfigJson;
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

                JavaScriptSerializer serializer = new JavaScriptSerializer();
                UserSettingsService.InspectionConfigJson = serializer.Serialize(settings);
                UserSettingsService.Save();
            }
            catch
            {
                // Ignore save error
            }
        }
    }
}
