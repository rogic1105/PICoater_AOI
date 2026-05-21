using System;
using System.IO;

namespace IoBridge.Core
{
    public static class IoLogger
    {
        private static readonly string _logDir =
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "logs");

        private static readonly object _lock = new object();

        /// <summary>Log 檔名前綴，預設 "PlcBridge"。各 App 可於啟動時設定。</summary>
        public static string FilePrefix { get; set; } = "PlcBridge";

        static IoLogger()
        {
            try { Directory.CreateDirectory(_logDir); } catch { }
        }

        public static void Info(string msg) => Write("INFO ", msg, null);
        public static void Warn(string msg) => Write("WARN ", msg, null);
        public static void Error(string msg, Exception ex = null) => Write("ERROR", msg, ex);

        private static void Write(string level, string msg, Exception ex)
        {
            string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
            string line = ex != null
                ? $"{timestamp} [{level}] {msg} | {ex.GetType().Name}: {ex.Message}"
                : $"{timestamp} [{level}] {msg}";

            string logFile = Path.Combine(_logDir, $"{FilePrefix}_{DateTime.Now:yyyyMMdd}.log");
            lock (_lock)
            {
                try { File.AppendAllText(logFile, line + Environment.NewLine); } catch { }
            }
        }
    }
}
