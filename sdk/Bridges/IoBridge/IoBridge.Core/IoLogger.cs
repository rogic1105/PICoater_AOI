using System;
using System.IO;

namespace IoBridge.Core
{
    public static class IoLogger
    {
        private static readonly object _lock = new object();

        /// <summary>Caller-owned output directory. Bridge samples default beside their executable.</summary>
        public static string LogDirectory { get; set; } =
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "logs");

        /// <summary>Log 檔名前綴，預設 "IoBridge"。各 App 可於啟動時設定。</summary>
        public static string FilePrefix { get; set; } = "IoBridge";

        public static void Info(string msg) => Write("INFO ", msg, null);
        public static void Warn(string msg) => Write("WARN ", msg, null);
        public static void Error(string msg, Exception ex = null) => Write("ERROR", msg, ex);

        private static void Write(string level, string msg, Exception ex)
        {
            string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
            string line = ex != null
                ? $"{timestamp} [{level}] {msg} | {ex.GetType().Name}: {ex.Message}"
                : $"{timestamp} [{level}] {msg}";

            lock (_lock)
            {
                try
                {
                    string logDirectory = LogDirectory;
                    string filePrefix = string.IsNullOrWhiteSpace(FilePrefix) ? "IoBridge" : FilePrefix;
                    Directory.CreateDirectory(logDirectory);
                    string logFile = Path.Combine(logDirectory, $"{filePrefix}-{DateTime.Now:yyyyMMdd}.log");
                    File.AppendAllText(logFile, line + Environment.NewLine);
                }
                catch { }
            }
        }
    }
}
