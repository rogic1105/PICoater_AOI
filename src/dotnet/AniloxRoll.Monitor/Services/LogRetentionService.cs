using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace AniloxRoll.Monitor.Core.Services
{
    internal static class LogFileCatalog
    {
        public static readonly string[] ManagedPatterns =
        {
            "trace-*.log",
            "resource-monitor-*.csv",
            "dropdiag-*.csv",
            "phaselog-*.csv",
            "paramchange-*.csv",
            "ui-actions-*.jsonl",
            "io-*.log",
            "AniloxRoll-crash.log"
        };
    }

    /// <summary>Deletes only cataloged diagnostic files after their configured retention age.</summary>
    internal sealed class LogRetentionService : IDisposable
    {
        private readonly Func<string> _getLogsPath;
        private readonly Func<int> _getRetentionHours;
        private readonly DateTime _processStartUtc;
        private readonly Timer _timer;
        private int _running;

        public LogRetentionService(Func<string> getLogsPath, Func<int> getRetentionHours)
        {
            _getLogsPath = getLogsPath ?? throw new ArgumentNullException(nameof(getLogsPath));
            _getRetentionHours = getRetentionHours ?? throw new ArgumentNullException(nameof(getRetentionHours));
            _processStartUtc = Process.GetCurrentProcess().StartTime.ToUniversalTime();
            _timer = new Timer(_ => RunCleanup(), null, TimeSpan.FromSeconds(5), TimeSpan.FromHours(1));
        }

        public event Action<string> CleanupFailed;
        public event Action CleanupSucceeded;

        internal void RunCleanup()
        {
            if (Interlocked.CompareExchange(ref _running, 1, 0) != 0) return;
            try
            {
                string root = _getLogsPath();
                if (string.IsNullOrWhiteSpace(root) || !Directory.Exists(root)) return;

                int retentionHours = Math.Max(1, _getRetentionHours());
                DateTime cutoffUtc = DateTime.UtcNow.AddHours(-retentionHours);
                int deleted = 0;
                long bytes = 0;
                var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

                foreach (string pattern in LogFileCatalog.ManagedPatterns)
                {
                    foreach (string path in Directory.GetFiles(root, pattern, SearchOption.AllDirectories))
                    {
                        if (!seen.Add(path)) continue;
                        var file = new FileInfo(path);
                        if (file.LastWriteTimeUtc >= cutoffUtc) continue;
                        if (file.CreationTimeUtc >= _processStartUtc.AddMinutes(-1)) continue;

                        long length = file.Length;
                        File.Delete(path);
                        bytes += length;
                        deleted++;
                    }
                }

                if (deleted > 0)
                {
                    Trace.TraceInformation(
                        $"[LogRetention] deleted={deleted} bytes={bytes} retentionHours={retentionHours}");
                }
                CleanupSucceeded?.Invoke();
            }
            catch (Exception ex)
            {
                string error = ex.GetType().Name + ": " + ex.Message;
                Trace.TraceWarning("[LogRetention] cleanup failed: " + error);
                CleanupFailed?.Invoke(error);
            }
            finally
            {
                Interlocked.Exchange(ref _running, 0);
            }
        }

        public void Dispose()
        {
            _timer.Dispose();
        }
    }
}
