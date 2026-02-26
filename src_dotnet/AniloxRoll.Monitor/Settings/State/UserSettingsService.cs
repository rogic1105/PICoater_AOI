using System;
using System.Configuration;
using System.IO;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// Core 設定存取服務（僅保留核心設定資料，不含 UI Session 狀態）。
    /// </summary>
    public static class UserSettingsService
    {
        private static T Execute<T>(Func<Properties.Settings, T> getter, T fallback = default(T))
        {
            try
            {
                return getter(Properties.Settings.Default);
            }
            catch (ConfigurationErrorsException ex)
            {
                RecoverFromCorruptedConfig(ex);
                try { return getter(Properties.Settings.Default); } catch { return fallback; }
            }
        }

        private static void Execute(Action<Properties.Settings> setter)
        {
            try
            {
                setter(Properties.Settings.Default);
            }
            catch (ConfigurationErrorsException ex)
            {
                RecoverFromCorruptedConfig(ex);
                try { setter(Properties.Settings.Default); } catch { }
            }
        }

        private static void RecoverFromCorruptedConfig(ConfigurationErrorsException ex)
        {
            try
            {
                if (!string.IsNullOrWhiteSpace(ex.Filename) && File.Exists(ex.Filename))
                {
                    File.Delete(ex.Filename);
                }
            }
            catch { }

            try { Properties.Settings.Default.Reset(); } catch { }
        }

        public static string InspectionConfigJson
        {
            get => Execute(s => s.InspectionConfigJson, string.Empty);
            set => Execute(s => s.InspectionConfigJson = value);
        }

        public static void Save()
        {
            Execute(s => s.Save());
        }
    }
}
