using System.Configuration;

namespace LightBridge.Control
{
    internal static class AppSettings
    {
        public static string ComPort => Get("ComPort", "COM17");
        public static int Channel    => GetInt("Channel", 1);
        public static int Brightness => GetInt("Brightness", 200);

        private static string Get(string key, string def) =>
            ConfigurationManager.AppSettings[key] ?? def;

        private static int GetInt(string key, int def) =>
            int.TryParse(ConfigurationManager.AppSettings[key], out int v) ? v : def;
    }
}
