using System;
using System.Configuration;
using System.IO;

namespace AniloxRoll.Monitor.UI.State
{
    public static class UserSessionState
    {
        private static T Execute<T>(Func<Properties.Settings, T> getter, T fallback = default(T))
        {
            try { return getter(Properties.Settings.Default); }
            catch (ConfigurationErrorsException ex)
            {
                RecoverFromCorruptedConfig(ex);
                try { return getter(Properties.Settings.Default); } catch { return fallback; }
            }
        }

        private static void Execute(Action<Properties.Settings> setter)
        {
            try { setter(Properties.Settings.Default); }
            catch (ConfigurationErrorsException ex)
            {
                RecoverFromCorruptedConfig(ex);
                try { setter(Properties.Settings.Default); } catch { }
            }
        }

        private static void RecoverFromCorruptedConfig(ConfigurationErrorsException ex)
        {
            try { if (!string.IsNullOrWhiteSpace(ex.Filename) && File.Exists(ex.Filename)) File.Delete(ex.Filename); } catch { }
            try { Properties.Settings.Default.Reset(); } catch { }
        }

        public static string LastDataPath => Execute(s => s.LastDataPath, string.Empty);
        public static string LastYear => Execute(s => s.LastYear, string.Empty);
        public static string LastMonth => Execute(s => s.LastMonth, string.Empty);
        public static string LastDay => Execute(s => s.LastDay, string.Empty);
        public static string LastHour => Execute(s => s.LastHour, string.Empty);
        public static string LastMin => Execute(s => s.LastMin, string.Empty);
        public static string LastSec => Execute(s => s.LastSec, string.Empty);
        public static bool LastEnableImageProcessing => GetLastEnableImageProcessing(false);

        public static bool GetLastEnableImageProcessing(bool fallback)
            => Execute(s => s.LastEnableImageProcessing, fallback);

        public static void Save() => Execute(s => s.Save());
        public static void SetLastDataPath(string path) => Execute(s => s.LastDataPath = path ?? string.Empty);
        public static void SetLastEnableImageProcessing(bool enabled) => Execute(s => s.LastEnableImageProcessing = enabled);

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
