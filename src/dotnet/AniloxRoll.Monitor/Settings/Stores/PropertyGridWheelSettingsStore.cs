using System;
using System.IO;

namespace AniloxRoll.Monitor.Core.Data
{
    internal static class PropertyGridWheelSettingsStore
    {
        private static string FullConfigPath =>
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"Config\ui-interaction-settings.json");

        internal static PropertyGridWheelSettings Load()
        {
            PropertyGridWheelSettings settings = JsonConfigLoader.LoadOrDefault(
                @"Config\ui-interaction-settings.json",
                new PropertyGridWheelSettings());
            settings.Validate();
            Save(settings);
            return settings;
        }

        internal static void Save(PropertyGridWheelSettings settings)
        {
            settings = settings ?? new PropertyGridWheelSettings();
            settings.Validate();
            JsonConfigLoader.SaveJson(FullConfigPath, settings);
        }
    }
}
