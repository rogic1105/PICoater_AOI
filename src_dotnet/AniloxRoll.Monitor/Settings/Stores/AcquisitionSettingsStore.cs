using System;
using System.IO;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// AcquisitionSettings 儲存層。
    /// 讀寫 Config\acquisition-settings.json（與 exe 同目錄）。
    /// 對應 tabPageCamera（TrackBar）的所有取像參數。
    /// </summary>
    public static class AcquisitionSettingsStore
    {
        private static string FullConfigPath =>
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"Config\acquisition-settings.json");

        public static AcquisitionSettings Load()
        {
            try
            {
                if (!File.Exists(FullConfigPath))
                {
                    var defaults = new AcquisitionSettings();
                    defaults.Validate();
                    Save(defaults);
                    return defaults;
                }

                string json = System.IO.File.ReadAllText(FullConfigPath, System.Text.Encoding.UTF8);
                if (string.IsNullOrWhiteSpace(json)) return new AcquisitionSettings();

                var result = new System.Web.Script.Serialization.JavaScriptSerializer()
                                 .Deserialize<AcquisitionSettings>(json);
                if (result == null) return new AcquisitionSettings();
                result.Validate();
                return result;
            }
            catch
            {
                return new AcquisitionSettings();
            }
        }

        public static void Save(AcquisitionSettings settings)
        {
            try
            {
                if (settings == null) settings = new AcquisitionSettings();
                settings.Validate();
                JsonConfigLoader.SaveJson(FullConfigPath, settings);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("[AcquisitionSettingsStore.Save] " + ex.Message);
            }
        }
    }
}
