using System;
using System.IO;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// InspectionSettings 儲存層（MachineLayout + Recipe + Storage）。
    /// 讀寫 Config\inspection-settings.json（與 exe 同目錄）。
    /// 對應 tabPageInspSettings（PropertyGrid）顯示的參數。
    /// Acquisition 由 AcquisitionSettingsStore 獨立管理。
    /// </summary>
    public static class InspectionSettingsStore
    {
        private static string FullConfigPath =>
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"Config\inspection-settings.json");

        /// <summary>僅序列化 PropertyGrid 顯示的三個群組，不含 Acquisition。</summary>
        private class Dto
        {
            public MachineLayoutConfig MachineLayout { get; set; } = new MachineLayoutConfig();
            public InspectionRecipe    Recipe        { get; set; } = new InspectionRecipe();
            public StorageSettings     Storage       { get; set; } = new StorageSettings();
        }

        public static InspectionSettings Load()
        {
            try
            {
                if (!File.Exists(FullConfigPath))
                {
                    var defaults = new InspectionSettings();
                    defaults.Validate();
                    Save(defaults);
                    return defaults;
                }

                string json = File.ReadAllText(FullConfigPath, System.Text.Encoding.UTF8);
                var dto = string.IsNullOrWhiteSpace(json)
                    ? new Dto()
                    : (new System.Web.Script.Serialization.JavaScriptSerializer().Deserialize<Dto>(json) ?? new Dto());

                var settings = new InspectionSettings
                {
                    MachineLayout = dto.MachineLayout ?? new MachineLayoutConfig(),
                    Recipe        = dto.Recipe        ?? new InspectionRecipe(),
                    Storage       = dto.Storage       ?? new StorageSettings(),
                };
                settings.Validate();
                return settings;
            }
            catch
            {
                return new InspectionSettings();
            }
        }

        public static void Save(InspectionSettings settings)
        {
            try
            {
                if (settings == null) settings = new InspectionSettings();
                settings.Validate();

                var dto = new Dto
                {
                    MachineLayout = settings.MachineLayout,
                    Recipe        = settings.Recipe,
                    Storage       = settings.Storage,
                };
                JsonConfigLoader.SaveJson(FullConfigPath, dto);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("[InspectionSettingsStore.Save] " + ex.Message);
            }
        }
    }
}
