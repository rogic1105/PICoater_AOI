using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.State;
using StorageBridge.Core;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    internal sealed class StorageHealthSnapshot
    {
        public StorageHealthSnapshot(
            long localFreeBytes,
            long localTotalBytes,
            long remoteFreeBytes,
            long remoteTotalBytes,
            bool? remoteShareConnected,
            bool? remoteAppAlive,
            bool remoteProbeInFlight,
            int reconnectSeconds)
        {
            LocalFreeBytes = localFreeBytes;
            LocalTotalBytes = localTotalBytes;
            RemoteFreeBytes = remoteFreeBytes;
            RemoteTotalBytes = remoteTotalBytes;
            RemoteShareConnected = remoteShareConnected;
            RemoteAppAlive = remoteAppAlive;
            RemoteProbeInFlight = remoteProbeInFlight;
            ReconnectSeconds = reconnectSeconds;
        }

        public long LocalFreeBytes { get; }
        public long LocalTotalBytes { get; }
        public long RemoteFreeBytes { get; }
        public long RemoteTotalBytes { get; }
        public bool? RemoteShareConnected { get; }
        public bool? RemoteAppAlive { get; }
        public bool RemoteProbeInFlight { get; }
        public int ReconnectSeconds { get; }
    }

    /// <summary>
    /// Owns storage capacity and remote availability observation.
    /// Retention and remote-copy policies remain in their dedicated services.
    /// </summary>
    internal sealed class StorageHealthCoordinator : IDisposable
    {
        private const int RemoteProbeIntervalTicks = 4;

        private readonly object _sync = new object();
        private readonly int _telemetryTickMs;
        private readonly bool _storageMachine;
        private readonly Func<string> _localRootProvider;
        private readonly Func<string> _remotePathProvider;
        private readonly Func<string> _remoteConfigPathProvider;
        private readonly RemoteCopyService _remoteCopyService;

        private int _remoteProbeTickCounter;
        private bool _remoteProbeInFlight;
        private bool _disposed;
        private bool _capacityProbeErrorReported;
        private bool? _remoteShareConnected;
        private bool? _remoteAppAlive;
        private DateTime? _lastRemoteHeartbeatUtc;
        private long _localFreeBytes = -1;
        private long _localTotalBytes;
        private long _remoteFreeBytes = -1;
        private long _remoteTotalBytes;

        public StorageHealthCoordinator(
            int telemetryTickMs,
            bool storageMachine,
            Func<string> localRootProvider,
            Func<string> remotePathProvider,
            Func<string> remoteConfigPathProvider,
            RemoteCopyService remoteCopyService)
        {
            if (telemetryTickMs <= 0)
                throw new ArgumentOutOfRangeException(nameof(telemetryTickMs));

            _telemetryTickMs = telemetryTickMs;
            _storageMachine = storageMachine;
            _localRootProvider = localRootProvider ??
                throw new ArgumentNullException(nameof(localRootProvider));
            _remotePathProvider = remotePathProvider ??
                throw new ArgumentNullException(nameof(remotePathProvider));
            _remoteConfigPathProvider = remoteConfigPathProvider ??
                throw new ArgumentNullException(nameof(remoteConfigPathProvider));
            _remoteCopyService = remoteCopyService;
        }

        public event Action StateChanged;

        public StorageHealthSnapshot Snapshot
        {
            get
            {
                lock (_sync)
                {
                    return new StorageHealthSnapshot(
                        _localFreeBytes,
                        _localTotalBytes,
                        _remoteFreeBytes,
                        _remoteTotalBytes,
                        _remoteShareConnected,
                        _remoteAppAlive,
                        _remoteProbeInFlight,
                        CountdownSeconds(_remoteProbeTickCounter));
                }
            }
        }

        public void Tick()
        {
            RefreshLocalCapacity();
            if (_storageMachine) return;

            string remotePath = _remotePathProvider() ?? string.Empty;
            if (string.IsNullOrWhiteSpace(remotePath))
            {
                ClearRemoteState();
                return;
            }

            bool startProbe = false;
            string configPath = null;
            lock (_sync)
            {
                if (_disposed) return;
                if (++_remoteProbeTickCounter >= RemoteProbeIntervalTicks)
                {
                    _remoteProbeTickCounter = 0;
                    if (!_remoteProbeInFlight)
                    {
                        _remoteProbeInFlight = true;
                        configPath = _remoteConfigPathProvider() ?? string.Empty;
                        startProbe = true;
                    }
                }
            }

            if (startProbe)
                Task.Run(() => ProbeRemote(remotePath, configPath));
        }

        public void ForceRemoteProbe()
        {
            lock (_sync)
            {
                if (_disposed) return;
                _remoteProbeTickCounter = RemoteProbeIntervalTicks;
            }
        }

        public StorageHealthSnapshot RefreshLocalCapacity()
        {
            string root = _localRootProvider() ?? string.Empty;
            long freeBytes;
            long totalBytes;
            bool success = TryReadDriveCapacity(root, out freeBytes, out totalBytes);

            lock (_sync)
            {
                if (_disposed) return Snapshot;
                if (success)
                {
                    _localFreeBytes = freeBytes;
                    _localTotalBytes = totalBytes;
                    if (_capacityProbeErrorReported)
                        Trace.WriteLine("[CapacityInfo] local drive probe recovered.");
                    _capacityProbeErrorReported = false;
                }
                else
                {
                    if (!_capacityProbeErrorReported)
                    {
                        Trace.WriteLine(
                            "[CapacityInfo] local drive probe failed: " + root);
                    }
                    _capacityProbeErrorReported = true;
                }

                return CreateSnapshotLocked();
            }
        }

        private void ProbeRemote(string remotePath, string configPath)
        {
            bool shareConnected = false;
            bool appAlive = false;
            string heartbeatError = null;
            StorageAppHeartbeatRecord heartbeatRecord = null;

            try
            {
                bool transportReachable =
                    ProbeStorageTransportReachable(remotePath);
                if (transportReachable)
                {
                    shareConnected =
                        _remoteCopyService?.ProbeRemoteWritable() == true;
                    if (shareConnected)
                    {
                        appAlive = StorageAppHeartbeatService.TryRead(
                            configPath,
                            DateTime.UtcNow,
                            out heartbeatRecord,
                            out heartbeatError);
                    }
                }
                else
                {
                    _remoteCopyService?.ReportRemoteUnavailable(
                        "TCP 445 unavailable.");
                }
            }
            catch (Exception ex)
            {
                _remoteCopyService?.ReportRemoteUnavailable(
                    ex.GetType().Name + ": " + ex.Message);
            }

            bool? previousAppAlive;
            lock (_sync)
            {
                if (_disposed) return;
                previousAppAlive = _remoteAppAlive;
                if (appAlive && heartbeatRecord != null)
                {
                    _lastRemoteHeartbeatUtc =
                        heartbeatRecord.LastSeenUtc.ToUniversalTime();
                }
                else if (shareConnected &&
                         ShouldKeepRemoteAppAlive(
                             previousAppAlive,
                             _lastRemoteHeartbeatUtc,
                             DateTime.UtcNow))
                {
                    // File.Replace over SMB can expose a brief missing-file
                    // window. Keep the last valid heartbeat until it expires.
                    appAlive = true;
                }

                _remoteShareConnected = shareConnected;
                _remoteAppAlive = appAlive;
                if (appAlive && heartbeatRecord != null)
                {
                    _remoteFreeBytes = heartbeatRecord.FreeBytes;
                    _remoteTotalBytes = heartbeatRecord.TotalBytes;
                }
                else if (!appAlive)
                {
                    _remoteFreeBytes = -1;
                    _remoteTotalBytes = 0;
                }
                _remoteProbeInFlight = false;
            }

            if (previousAppAlive != appAlive)
            {
                if (appAlive)
                {
                    double ageSec = Math.Max(
                        0,
                        (DateTime.UtcNow -
                         heartbeatRecord.LastSeenUtc.ToUniversalTime())
                        .TotalSeconds);
                    FlowTrace.Log(
                        $"儲存程式 heartbeat 恢復 pid={heartbeatRecord.ProcessId} " +
                        $"age={ageSec:F1}s");
                }
                else
                {
                    FlowTrace.Log(
                        "⚠ 儲存程式 heartbeat 未回報 reason=" +
                        (heartbeatError ??
                         (shareConnected
                             ? "unknown"
                             : "storage share unavailable")));
                }
            }

            RaiseStateChanged();
        }

        private void ClearRemoteState()
        {
            bool changed;
            lock (_sync)
            {
                if (_disposed) return;
                changed =
                    _remoteShareConnected.HasValue ||
                    _remoteAppAlive.HasValue ||
                    _remoteFreeBytes >= 0 ||
                    _remoteTotalBytes > 0;
                _remoteShareConnected = null;
                _remoteAppAlive = null;
                _lastRemoteHeartbeatUtc = null;
                _remoteFreeBytes = -1;
                _remoteTotalBytes = 0;
                _remoteProbeTickCounter = 0;
            }

            if (changed) RaiseStateChanged();
        }

        private int CountdownSeconds(int elapsedTicks)
        {
            return Math.Max(
                1,
                (int)Math.Ceiling(
                    (RemoteProbeIntervalTicks - elapsedTicks) *
                    _telemetryTickMs /
                    1000.0));
        }

        private StorageHealthSnapshot CreateSnapshotLocked()
        {
            return new StorageHealthSnapshot(
                _localFreeBytes,
                _localTotalBytes,
                _remoteFreeBytes,
                _remoteTotalBytes,
                _remoteShareConnected,
                _remoteAppAlive,
                _remoteProbeInFlight,
                CountdownSeconds(_remoteProbeTickCounter));
        }

        private void RaiseStateChanged()
        {
            StateChanged?.Invoke();
        }

        internal static bool TryReadDriveCapacity(
            string root,
            out long freeBytes,
            out long totalBytes)
        {
            freeBytes = -1;
            totalBytes = 0;
            try
            {
                if (string.IsNullOrWhiteSpace(root)) return false;
                var drive = new DriveInfo(
                    Path.GetPathRoot(Path.GetFullPath(root)));
                if (!drive.IsReady) return false;
                freeBytes = drive.AvailableFreeSpace;
                totalBytes = drive.TotalSize;
                return totalBytes > 0;
            }
            catch
            {
                return false;
            }
        }

        internal static bool ProbeStorageTransportReachable(string uncPath)
        {
            string host = ParseUncHost(uncPath);
            if (string.IsNullOrEmpty(host)) return false;

            Socket socket = null;
            try
            {
                socket = new Socket(
                    AddressFamily.InterNetwork,
                    SocketType.Stream,
                    ProtocolType.Tcp);
                socket.Blocking = false;
                try
                {
                    socket.Connect(host, 445);
                    return socket.Connected;
                }
                catch (SocketException ex)
                {
                    if (!IsConnectInProgress(ex.SocketErrorCode))
                        return false;
                }

                var writable = new List<Socket> { socket };
                var errors = new List<Socket> { socket };
                Socket.Select(null, writable, errors, 1000 * 1000);
                if (errors.Count > 0 || writable.Count == 0)
                    return false;

                int error = (int)socket.GetSocketOption(
                    SocketOptionLevel.Socket,
                    SocketOptionName.Error);
                return error == 0 && socket.Connected;
            }
            finally
            {
                socket?.Dispose();
            }
        }

        private static bool IsConnectInProgress(SocketError error)
        {
            return error == SocketError.WouldBlock ||
                   error == SocketError.InProgress ||
                   error == SocketError.AlreadyInProgress;
        }

        internal static string ParseUncHost(string uncPath)
        {
            if (string.IsNullOrWhiteSpace(uncPath)) return null;
            string path = uncPath.TrimStart('\\', '/');
            int slash = path.IndexOfAny(new[] { '\\', '/' });
            return slash > 0 ? path.Substring(0, slash) : path;
        }

        internal static bool ShouldKeepRemoteAppAlive(
            bool? previousAppAlive,
            DateTime? lastHeartbeatUtc,
            DateTime nowUtc)
        {
            if (previousAppAlive != true || !lastHeartbeatUtc.HasValue)
                return false;

            TimeSpan age =
                nowUtc.ToUniversalTime() -
                lastHeartbeatUtc.Value.ToUniversalTime();
            if (age < TimeSpan.Zero) age = TimeSpan.Zero;
            return age <= StorageAppHeartbeatService.StaleAfter;
        }

        public void Dispose()
        {
            lock (_sync)
            {
                if (_disposed) return;
                _disposed = true;
                _remoteProbeInFlight = false;
            }
        }
    }
}
