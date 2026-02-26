using System;
using System.Configuration;
using System.IO;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// 使用者設定存取服務。
    /// 統一封裝 Properties.Settings 的讀寫與損毀設定檔復原流程。
    /// </summary>
    public static class UserSettingsService
    {
        // ===== 內部安全存取 Helper =====
        private static T Execute<T>(Func<Properties.Settings, T> getter, T fallback = default(T))
        {
            try
            {
                return getter(Properties.Settings.Default);
            }
            catch (ConfigurationErrorsException ex)
            {
                RecoverFromCorruptedConfig(ex);
                try
                {
                    return getter(Properties.Settings.Default);
                }
                catch
                {
                    return fallback;
                }
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
                try
                {
                    setter(Properties.Settings.Default);
                }
                catch
                {
                    // ignore
                }
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
            catch
            {
                // ignore
            }

            try
            {
                Properties.Settings.Default.Reset();
            }
            catch
            {
                // ignore
            }
        }

        // ===== 讀取型屬性（UI 狀態、篩選條件、功能開關） =====
        public static string LastDataPath => Execute(s => s.LastDataPath, string.Empty);
        public static string LastYear => Execute(s => s.LastYear, string.Empty);
        public static string LastMonth => Execute(s => s.LastMonth, string.Empty);
        public static string LastDay => Execute(s => s.LastDay, string.Empty);
        public static string LastHour => Execute(s => s.LastHour, string.Empty);
        public static string LastMin => Execute(s => s.LastMin, string.Empty);
        public static string LastSec => Execute(s => s.LastSec, string.Empty);
        public static bool LastEnableImageProcessing => Execute(s => s.LastEnableImageProcessing, true);

        public static string InspectionConfigJson
        {
            get => Execute(s => s.InspectionConfigJson, string.Empty);
            set => Execute(s => s.InspectionConfigJson = value);
        }

        // ===== 對外操作 API =====
        public static void Save()
        {
            Execute(s => s.Save());
        }

        public static void SetLastDataPath(string path)
        {
            Execute(s => s.LastDataPath = path ?? string.Empty);
        }

        public static void SetLastEnableImageProcessing(bool enabled)
        {
            Execute(s => s.LastEnableImageProcessing = enabled);
        }

        public static void SaveDateTimeSelection(string year, string month, string day, string hour, string min, string sec)
        {
            Execute(s =>
            {
                s.LastYear = year ?? string.Empty;
                s.LastMonth = month ?? string.Empty;
                s.LastDay = day ?? string.Empty;
                s.LastHour = hour ?? string.Empty;
                s.LastMin = min ?? string.Empty;
                s.LastSec = sec ?? string.Empty;
            });
        }
    }
}
