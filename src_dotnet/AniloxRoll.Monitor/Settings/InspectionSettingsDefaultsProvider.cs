namespace AniloxRoll.Monitor.Core.Data
{
    internal static class InspectionSettingsDefaultsProvider
    {
        private const string DefaultConfigPath = "Config\\inspection-settings.defaults.json";

        public static InspectionSettings LoadDefaults()
        {
            return JsonConfigLoader.LoadOrDefault(DefaultConfigPath, new InspectionSettings());
        }
    }
}
