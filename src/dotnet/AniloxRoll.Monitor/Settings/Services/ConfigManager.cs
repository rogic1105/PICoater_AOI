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
            // 啟動時 idempotent Save 一次：把新增的 setting 欄位（如 DebugUiActionLog）用 default value
            // 補進現有 json，避免使用者手動編輯。對既有 key 不改值，僅補缺。
            try { InspectionSettingsStore.Save(settings); } catch { }
            try { AcquisitionSettingsStore.Save(settings.Acquisition); } catch { }
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
