using System;
using System.Diagnostics;
using System.IO;
using System.Web.Script.Serialization;
using System.Threading;

namespace AniloxRoll.Monitor.Core.Services
{
    internal sealed class StorageAppHeartbeatRecord
    {
        public int Version { get; set; }
        public int ProcessId { get; set; }
        public DateTime StartedUtc { get; set; }
        public DateTime LastSeenUtc { get; set; }
        public long FreeBytes { get; set; }
        public long TotalBytes { get; set; }
        public DateTime? LastCleanupUtc { get; set; }
        public long LastCleanupFreedBytes { get; set; }
    }

    /// <summary>
    /// Storage-role liveness beacon. The inspection PC reads the shared JSON file so
    /// writable SMB storage and a running storage application remain separate facts.
    /// </summary>
    internal sealed class StorageAppHeartbeatService : IDisposable
    {
        internal const string FileName = "storage-app-heartbeat.json";
        internal static readonly TimeSpan PublishInterval = TimeSpan.FromSeconds(5);
        internal static readonly TimeSpan StaleAfter = TimeSpan.FromSeconds(15);

        private readonly Func<string> _getConfigFolder;
        private readonly Func<string> _getDataRoot;
        private readonly DateTime _startedUtc = DateTime.UtcNow;
        private readonly object _sync = new object();
        private readonly object _publishSync = new object();
        private Timer _timer;
        private DateTime? _lastCleanupUtc;
        private long _lastCleanupFreedBytes;

        public StorageAppHeartbeatService(Func<string> getConfigFolder, Func<string> getDataRoot)
        {
            _getConfigFolder = getConfigFolder ?? throw new ArgumentNullException(nameof(getConfigFolder));
            _getDataRoot = getDataRoot ?? throw new ArgumentNullException(nameof(getDataRoot));
        }

        public void Start()
        {
            lock (_sync)
            {
                if (_timer != null) return;
                PublishNow();
                _timer = new Timer(_ => PublishNow(), null, PublishInterval, PublishInterval);
            }
            FlowTrace.Log("storage heartbeat start interval=5s stale=15s");
        }

        public void RecordCleanup(long freedBytes)
        {
            lock (_sync)
            {
                _lastCleanupUtc = DateTime.UtcNow;
                _lastCleanupFreedBytes = Math.Max(0, freedBytes);
            }
            PublishNow();
        }

        internal void PublishNow()
        {
            lock (_publishSync)
            {
                try
                {
                    string configFolder = _getConfigFolder();
                    if (string.IsNullOrWhiteSpace(configFolder)) return;
                    Directory.CreateDirectory(configFolder);

                    var record = CreateRecord();
                    string path = Path.Combine(configFolder, FileName);
                    string temp = path + ".part-" + Guid.NewGuid().ToString("N");
                    string json = new JavaScriptSerializer().Serialize(record);
                    File.WriteAllText(temp, json);
                    if (File.Exists(path))
                        File.Replace(temp, path, null);
                    else
                        File.Move(temp, path);
                }
                catch (Exception ex)
                {
                    Trace.WriteLine("[StorageHeartbeat] publish failed: " + ex.Message);
                }
            }
        }

        private StorageAppHeartbeatRecord CreateRecord()
        {
            long freeBytes = 0;
            long totalBytes = 0;
            try
            {
                string root = _getDataRoot();
                if (!string.IsNullOrWhiteSpace(root))
                {
                    var drive = new DriveInfo(Path.GetPathRoot(Path.GetFullPath(root)));
                    freeBytes = drive.AvailableFreeSpace;
                    totalBytes = drive.TotalSize;
                }
            }
            catch { }

            lock (_sync)
            {
                return new StorageAppHeartbeatRecord
                {
                    Version = 1,
                    ProcessId = Process.GetCurrentProcess().Id,
                    StartedUtc = _startedUtc,
                    LastSeenUtc = DateTime.UtcNow,
                    FreeBytes = freeBytes,
                    TotalBytes = totalBytes,
                    LastCleanupUtc = _lastCleanupUtc,
                    LastCleanupFreedBytes = _lastCleanupFreedBytes
                };
            }
        }

        public static bool TryRead(string configFolder, DateTime nowUtc,
            out StorageAppHeartbeatRecord record, out string error)
        {
            record = null;
            error = null;
            try
            {
                if (string.IsNullOrWhiteSpace(configFolder))
                {
                    error = "remote config path is empty";
                    return false;
                }
                string path = Path.Combine(configFolder, FileName);
                if (!File.Exists(path))
                {
                    error = "heartbeat file is missing";
                    return false;
                }
                string json = File.ReadAllText(path);
                record = new JavaScriptSerializer().Deserialize<StorageAppHeartbeatRecord>(json);
                if (record == null || record.Version != 1)
                {
                    error = "heartbeat format is invalid";
                    return false;
                }
                TimeSpan age = nowUtc - record.LastSeenUtc.ToUniversalTime();
                if (age < TimeSpan.Zero) age = TimeSpan.Zero;
                if (age > StaleAfter)
                {
                    error = "heartbeat is stale (" + age.TotalSeconds.ToString("F0") + "s)";
                    return false;
                }
                return true;
            }
            catch (Exception ex)
            {
                error = ex.GetType().Name + ": " + ex.Message;
                return false;
            }
        }

        public void Dispose()
        {
            lock (_sync)
            {
                _timer?.Dispose();
                _timer = null;
            }
        }
    }
}
