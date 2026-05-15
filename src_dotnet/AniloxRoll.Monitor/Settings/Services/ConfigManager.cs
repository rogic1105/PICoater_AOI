namespace AniloxRoll.Monitor.Core.Data
{
    public static class ConfigManager
    {
        /// <summary>
        /// 載入完整設定：inspection-settings.json（PropertyGrid）+ acquisition-settings.json（TrackBar）。
        /// </summary>
        public static InspectionSettings LoadInspectionSettings()
        {
            var settings = InspectionSettingsStore.Load();
            settings.Acquisition = AcquisitionSettingsStore.Load();
            return settings;
        }

        /// <summary>儲存 PropertyGrid 參數（MachineLayout + Recipe + Storage）至 inspection-settings.json。</summary>
        public static void SaveInspectionSettings(InspectionSettings settings)
            => InspectionSettingsStore.Save(settings);

        /// <summary>儲存 TrackBar 參數（Acquisition）至 acquisition-settings.json。</summary>
        public static void SaveAcquisitionSettings(AcquisitionSettings settings)
            => AcquisitionSettingsStore.Save(settings);

    }
}
