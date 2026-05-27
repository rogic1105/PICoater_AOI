using System.Configuration;

namespace StorageBridge.Control
{
    internal static class AppSettings
    {
        public static string RemotePath => Get("RemotePath", @"\\192.168.10.20\Anilox\Captures");
        public static string LocalPath  => Get("LocalPath",  @"D:\Anilox\Captures");
        public static int ProbeIntervalMs => GetInt("ProbeIntervalMs", 5000);

        private static string Get(string key, string def) =>
            ConfigurationManager.AppSettings[key] ?? def;

        private static int GetInt(string key, int def) =>
            int.TryParse(ConfigurationManager.AppSettings[key], out int v) ? v : def;
    }
}
