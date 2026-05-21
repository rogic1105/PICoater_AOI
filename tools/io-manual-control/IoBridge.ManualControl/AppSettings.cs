using System.Configuration;

namespace IoBridge.ManualControl
{
    internal static class AppSettings
    {
        public static string PlcIp => Get("PLC_IP", "192.168.1.1");
        public static int PlcPort => GetInt("PLC_Port", 502);
        public static int PollIntervalMs => GetInt("PollIntervalMs", 500);
        public static int ConnectTimeoutMs => GetInt("ConnectTimeoutMs", 5000);
        public static int ReadWriteTimeoutMs => GetInt("ReadWriteTimeoutMs", 2000);

        private static string Get(string key, string defaultValue) =>
            ConfigurationManager.AppSettings[key] ?? defaultValue;

        private static int GetInt(string key, int defaultValue) =>
            int.TryParse(ConfigurationManager.AppSettings[key], out int v) ? v : defaultValue;
    }
}
