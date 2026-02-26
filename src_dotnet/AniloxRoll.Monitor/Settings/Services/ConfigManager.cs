namespace AniloxRoll.Monitor.Core.Data
{
    public static class ConfigManager
    {
        public static InspectionSettings LoadInspectionSettings() => InspectionSettingsStore.Load();
        public static void SaveInspectionSettings(InspectionSettings settings) => InspectionSettingsStore.Save(settings);
        public static SystemSettings LoadSystemSettings() => SystemSettings.CreateDefault();
    }
}
