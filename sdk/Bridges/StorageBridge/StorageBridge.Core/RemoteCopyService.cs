using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;

namespace StorageBridge.Core
{
    /// <summary>
    /// Durable background copy service. Each pending source path is persisted locally before it
    /// enters the worker queue, so a remote outage or process restart cannot silently drop it.
    /// Remote files are published through a same-directory .part file and atomic replace.
    /// </summary>
    public sealed class RemoteCopyService : IDisposable
    {
        private const string PendingDirectoryName = ".remote-copy-pending";
        private const string PendingExtension = ".pending";
        private const int PendingMarkerMagic = 0x50434F50; // "POCP"
        private const int PendingMarkerVersion = 1;
        private const int IdleWaitMs = 5000;
        private const int MinRetryDelayMs = 2000;
        private const int MaxRetryDelayMs = 30000;

        private readonly Func<string> _getRemotePath;
        private readonly Func<string> _getLocalRoot;
        private readonly ConcurrentQueue<CopyItem> _queue = new ConcurrentQueue<CopyItem>();
        private readonly HashSet<string> _pendingPaths =
            new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        private readonly object _pendingSync = new object();
        private readonly object _stateSync = new object();
        private readonly ManualResetEventSlim _workSignal = new ManualResetEventSlim(false);
        private readonly AutoResetEvent _retrySignal = new AutoResetEvent(false);
        private readonly Thread _workerThread;
        private readonly string _pendingDirectory;

        private volatile bool _disposed;
        private int _remoteWritableState = -1; // -1 unknown, 0 unavailable, 1 accepted
        private int _backlogRecoveryPending;
        private long _totalCopiedFiles;
        private long _totalCopiedBytes;
        private long _totalFailedFiles;
        private long _totalRetryAttempts;
        private string _lastError = string.Empty;

        public RemoteCopyService(Func<string> getRemotePath, Func<string> getLocalRoot)
        {
            _getRemotePath = getRemotePath ?? throw new ArgumentNullException(nameof(getRemotePath));
            _getLocalRoot = getLocalRoot ?? throw new ArgumentNullException(nameof(getLocalRoot));

            _pendingDirectory = ResolvePendingDirectory(_getLocalRoot());
            RestorePendingItems();

            _workerThread = new Thread(WorkerLoop)
            {
                IsBackground = true,
                Name = "RemoteCopyWorker",
                Priority = ThreadPriority.BelowNormal
            };
            _workerThread.Start();
        }

        /// <summary>Files awaiting confirmed remote publication, including the in-flight item.</summary>
        public int QueueCount
        {
            get
            {
                lock (_pendingSync) return _pendingPaths.Count;
            }
        }

        public long TotalCopiedFiles => Interlocked.Read(ref _totalCopiedFiles);
        public long TotalCopiedBytes => Interlocked.Read(ref _totalCopiedBytes);

        /// <summary>Files that have encountered at least one transfer failure.</summary>
        public long TotalFailedFiles => Interlocked.Read(ref _totalFailedFiles);

        public long TotalRetryAttempts => Interlocked.Read(ref _totalRetryAttempts);

        public bool? IsRemoteWritable
        {
            get
            {
                int state = Volatile.Read(ref _remoteWritableState);
                return state < 0 ? (bool?)null : state == 1;
            }
        }

        public string LastError
        {
            get
            {
                lock (_stateSync) return _lastError;
            }
        }

        public void EnqueueFile(string localFilePath)
        {
            if (_disposed || string.IsNullOrWhiteSpace(localFilePath)) return;
            if (string.IsNullOrWhiteSpace(_getRemotePath())) return;

            CopyItem item;
            if (!TryPersistPendingItem(localFilePath, out item)) return;

            _queue.Enqueue(item);
            _workSignal.Set();
        }

        public void EnqueueFiles(string[] localFilePaths)
        {
            if (_disposed || localFilePaths == null) return;
            if (string.IsNullOrWhiteSpace(_getRemotePath())) return;

            bool added = false;
            foreach (string path in localFilePaths)
            {
                if (string.IsNullOrWhiteSpace(path)) continue;

                CopyItem item;
                if (!TryPersistPendingItem(path, out item)) continue;
                _queue.Enqueue(item);
                added = true;
            }

            if (added) _workSignal.Set();
        }

        /// <summary>
        /// Verifies the configured share and directory permissions by creating and deleting a
        /// unique probe file. The caller should first perform a short TCP/445 transport probe to
        /// avoid entering the Windows SMB redirector while the host is offline.
        /// </summary>
        public bool ProbeRemoteWritable()
        {
            string remotePath = _getRemotePath();
            string error;
            bool writable = TryProbeRemoteWritable(remotePath, out error);
            MarkRemoteWritable(writable, error);
            return writable;
        }

        /// <summary>
        /// Verifies a share path by creating, flushing, and deleting a unique probe file.
        /// Call from a background thread because an unavailable UNC path may block in the SMB redirector.
        /// </summary>
        public static bool TryProbeRemoteWritable(string remotePath, out string error)
        {
            error = null;
            if (string.IsNullOrWhiteSpace(remotePath))
            {
                error = "Remote path is empty.";
                return false;
            }

            string probePath = null;
            try
            {
                if (!Directory.Exists(remotePath))
                    throw new DirectoryNotFoundException("Remote share path does not exist.");

                probePath = Path.Combine(
                    remotePath,
                    ".picoater-write-probe-" + Guid.NewGuid().ToString("N") + ".tmp");
                byte[] payload = Encoding.ASCII.GetBytes(DateTime.UtcNow.ToString("O"));
                using (var stream = new FileStream(
                    probePath, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096,
                    FileOptions.WriteThrough))
                {
                    stream.Write(payload, 0, payload.Length);
                    stream.Flush(true);
                }
                File.Delete(probePath);
                probePath = null;
                return true;
            }
            catch (Exception ex)
            {
                error = ex.GetType().Name + ": " + ex.Message;
                return false;
            }
            finally
            {
                TryDelete(probePath);
            }
        }

        /// <summary>Called when the fast transport probe proves that SMB cannot be reached.</summary>
        public void ReportRemoteUnavailable(string reason)
        {
            MarkRemoteWritable(false, string.IsNullOrWhiteSpace(reason)
                ? "SMB transport unavailable."
                : reason);
        }

        /// <summary>Prevents retention from deleting a day folder that still has pending files.</summary>
        public bool HasPendingFilesUnder(string directory)
        {
            string prefix = NormalizeDirectoryPrefix(directory);
            if (prefix == null) return false;

            lock (_pendingSync)
            {
                foreach (string path in _pendingPaths)
                {
                    if (path.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                        return true;
                }
            }
            return false;
        }

        private void WorkerLoop()
        {
            while (!_disposed)
            {
                CopyItem item;
                if (!_queue.TryDequeue(out item))
                {
                    _workSignal.Wait(TimeSpan.FromMilliseconds(IdleWaitMs));
                    _workSignal.Reset();
                    continue;
                }

                if (TryProcessItem(item)) continue;

                if (_disposed) break;
                _queue.Enqueue(item);
                _retrySignal.WaitOne(GetRetryDelayMs(item.Attempt));
            }
        }

        private bool TryProcessItem(CopyItem item)
        {
            string tempPath = null;
            try
            {
                string remotePath = _getRemotePath();
                if (string.IsNullOrWhiteSpace(remotePath))
                    throw new IOException("Remote path is empty.");
                if (!File.Exists(item.LocalPath))
                    throw new FileNotFoundException("Pending source file is missing.", item.LocalPath);

                string destinationPath = Path.Combine(remotePath, item.RelativePath);
                string destinationDirectory = Path.GetDirectoryName(destinationPath);
                if (string.IsNullOrWhiteSpace(destinationDirectory))
                    throw new IOException("Unable to resolve remote destination directory.");

                Directory.CreateDirectory(destinationDirectory);

                long sourceSizeBefore = new FileInfo(item.LocalPath).Length;
                tempPath = destinationPath + ".part-" + Guid.NewGuid().ToString("N");
                File.Copy(item.LocalPath, tempPath, false);

                long sourceSizeAfter = new FileInfo(item.LocalPath).Length;
                long tempSize = new FileInfo(tempPath).Length;
                if (sourceSizeBefore != sourceSizeAfter || tempSize != sourceSizeAfter)
                    throw new IOException("Source changed during remote copy; retrying a stable snapshot.");

                PublishTempFile(tempPath, destinationPath);
                tempPath = null;

                long publishedSize = new FileInfo(destinationPath).Length;
                if (publishedSize != sourceSizeAfter)
                    throw new IOException("Published remote file length does not match the source.");

                if (!TryCompletePendingItem(item))
                    throw new IOException("Remote file was published but pending marker remains.");
                Interlocked.Increment(ref _totalCopiedFiles);
                Interlocked.Add(ref _totalCopiedBytes, sourceSizeAfter);
                MarkRemoteWritable(true, null);

                if (item.Attempt > 0)
                {
                    Trace.TraceInformation(
                        $"[RemoteCopy] recovered file after {item.Attempt} retries: {item.RelativePath}");
                }
                if (QueueCount == 0 && Interlocked.Exchange(ref _backlogRecoveryPending, 0) == 1)
                {
                    Trace.TraceInformation(
                        $"[RemoteCopy] backlog drained: copied={TotalCopiedFiles} bytes={TotalCopiedBytes}");
                }
                return true;
            }
            catch (Exception ex)
            {
                TryDelete(tempPath);
                item.Attempt++;
                Interlocked.Increment(ref _totalRetryAttempts);
                if (item.Attempt == 1) Interlocked.Increment(ref _totalFailedFiles);
                Interlocked.Exchange(ref _backlogRecoveryPending, 1);

                string error = ex.GetType().Name + ": " + ex.Message;
                MarkRemoteWritable(false, error);
                if (item.Attempt == 1 || item.Attempt % 10 == 0)
                {
                    Trace.TraceWarning(
                        $"[RemoteCopy] retry pending attempt={item.Attempt} queue={QueueCount} " +
                        $"file={item.RelativePath} error={error}");
                }
                return false;
            }
        }

        private bool TryPersistPendingItem(string localFilePath, out CopyItem item)
        {
            item = default(CopyItem);
            try
            {
                string localRoot = _getLocalRoot();
                string normalizedPath = NormalizeFullPath(localFilePath);
                string relativePath = GetRelativePath(localRoot, normalizedPath);
                if (relativePath == null)
                    throw new IOException("Source file is outside the configured local capture root.");
                if (string.IsNullOrWhiteSpace(_pendingDirectory))
                    throw new IOException("Pending queue directory is unavailable.");

                lock (_pendingSync)
                {
                    if (_pendingPaths.Contains(normalizedPath)) return false;

                    Directory.CreateDirectory(_pendingDirectory);
                    string markerPath = Path.Combine(
                        _pendingDirectory, ComputeMarkerName(normalizedPath) + PendingExtension);
                    string tempMarkerPath = markerPath + ".tmp-" + Guid.NewGuid().ToString("N");
                    try
                    {
                        WriteMarker(tempMarkerPath, normalizedPath, relativePath);
                        if (!File.Exists(markerPath)) File.Move(tempMarkerPath, markerPath);
                    }
                    finally
                    {
                        TryDelete(tempMarkerPath);
                    }

                    _pendingPaths.Add(normalizedPath);
                    item = new CopyItem
                    {
                        LocalPath = normalizedPath,
                        RelativePath = relativePath,
                        MarkerPath = markerPath,
                        Attempt = 0
                    };
                    return true;
                }
            }
            catch (Exception ex)
            {
                Interlocked.Increment(ref _totalFailedFiles);
                Trace.TraceError(
                    $"[RemoteCopy] unable to persist pending item {localFilePath}: " +
                    $"{ex.GetType().Name}: {ex.Message}");
                return false;
            }
        }

        private void RestorePendingItems()
        {
            if (string.IsNullOrWhiteSpace(_pendingDirectory)) return;

            try
            {
                Directory.CreateDirectory(_pendingDirectory);
                foreach (string temp in Directory.GetFiles(_pendingDirectory, "*.tmp-*"))
                    TryDelete(temp);

                int restored = 0;
                foreach (string markerPath in Directory.GetFiles(
                    _pendingDirectory, "*" + PendingExtension))
                {
                    CopyItem item;
                    if (!TryReadMarker(markerPath, out item)) continue;
                    string relativePath = GetRelativePath(_getLocalRoot(), item.LocalPath);
                    if (relativePath == null)
                    {
                        Trace.TraceError(
                            $"[RemoteCopy] pending source is outside current local root: {item.LocalPath}");
                        continue;
                    }
                    item.RelativePath = relativePath;

                    lock (_pendingSync)
                    {
                        if (!_pendingPaths.Add(item.LocalPath)) continue;
                    }
                    _queue.Enqueue(item);
                    restored++;
                }

                if (restored > 0)
                {
                    Interlocked.Exchange(ref _backlogRecoveryPending, 1);
                    Trace.TraceInformation($"[RemoteCopy] restored pending queue count={restored}");
                    _workSignal.Set();
                }
            }
            catch (Exception ex)
            {
                Trace.TraceError(
                    $"[RemoteCopy] pending queue restore failed: {ex.GetType().Name}: {ex.Message}");
            }
        }

        private static void WriteMarker(
            string markerPath, string localPath, string relativePath)
        {
            using (var stream = new FileStream(
                markerPath, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096,
                FileOptions.WriteThrough))
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
            {
                writer.Write(PendingMarkerMagic);
                writer.Write(PendingMarkerVersion);
                writer.Write(localPath);
                writer.Write(relativePath);
                writer.Flush();
                stream.Flush(true);
            }
        }

        private static bool TryReadMarker(string markerPath, out CopyItem item)
        {
            item = default(CopyItem);
            try
            {
                string localPath;
                string relativePath;
                using (var stream = new FileStream(
                    markerPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                using (var reader = new BinaryReader(stream, Encoding.UTF8, false))
                {
                    if (reader.ReadInt32() != PendingMarkerMagic)
                        throw new InvalidDataException("Pending marker magic is invalid.");
                    if (reader.ReadInt32() != PendingMarkerVersion)
                        throw new InvalidDataException("Pending marker version is unsupported.");
                    localPath = reader.ReadString();
                    relativePath = reader.ReadString();
                }

                item = new CopyItem
                {
                    LocalPath = NormalizeFullPath(localPath),
                    RelativePath = relativePath,
                    MarkerPath = markerPath,
                    Attempt = 0
                };
                return true;
            }
            catch (Exception ex)
            {
                Trace.TraceError(
                    $"[RemoteCopy] invalid pending marker {markerPath}: " +
                    $"{ex.GetType().Name}: {ex.Message}");
                return false;
            }
        }

        private bool TryCompletePendingItem(CopyItem item)
        {
            try
            {
                File.Delete(item.MarkerPath);
                if (File.Exists(item.MarkerPath)) return false;
            }
            catch (Exception ex)
            {
                Trace.TraceWarning(
                    $"[RemoteCopy] copied but marker delete failed {item.MarkerPath}: {ex.Message}");
                return false;
            }

            lock (_pendingSync)
            {
                _pendingPaths.Remove(item.LocalPath);
            }
            return true;
        }

        private void MarkRemoteWritable(bool writable, string error)
        {
            lock (_stateSync)
            {
                _lastError = writable ? string.Empty : (error ?? string.Empty);
            }

            int next = writable ? 1 : 0;
            int previous = Interlocked.Exchange(ref _remoteWritableState, next);
            if (writable && previous != next) _retrySignal.Set();
            if (previous == next) return;

            if (writable)
                Trace.TraceInformation("[RemoteCopy] remote share accepted (write verified)");
            else
                Trace.TraceWarning("[RemoteCopy] remote share unavailable: " + LastError);
        }

        private static void PublishTempFile(string tempPath, string destinationPath)
        {
            if (File.Exists(destinationPath))
                File.Replace(tempPath, destinationPath, null, true);
            else
                File.Move(tempPath, destinationPath);
        }

        private static int GetRetryDelayMs(int attempt)
        {
            if (attempt <= 1) return MinRetryDelayMs;
            int exponent = Math.Min(attempt - 1, 4);
            return Math.Min(MaxRetryDelayMs, MinRetryDelayMs * (1 << exponent));
        }

        private static string ResolvePendingDirectory(string localRoot)
        {
            if (string.IsNullOrWhiteSpace(localRoot)) return null;
            return Path.Combine(NormalizeFullPath(localRoot), PendingDirectoryName);
        }

        private static string GetRelativePath(string basePath, string fullPath)
        {
            string prefix = NormalizeDirectoryPrefix(basePath);
            if (prefix == null || !fullPath.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                return null;
            return fullPath.Substring(prefix.Length);
        }

        private static string NormalizeDirectoryPrefix(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return null;
            string fullPath = NormalizeFullPath(path).TrimEnd(
                Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            return fullPath + Path.DirectorySeparatorChar;
        }

        private static string NormalizeFullPath(string path)
        {
            return Path.GetFullPath(path);
        }

        private static string ComputeMarkerName(string normalizedPath)
        {
            byte[] input = Encoding.UTF8.GetBytes(normalizedPath.ToUpperInvariant());
            using (SHA256 sha = SHA256.Create())
                return BitConverter.ToString(sha.ComputeHash(input)).Replace("-", string.Empty);
        }

        private static void TryDelete(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return;
            try
            {
                if (File.Exists(path)) File.Delete(path);
            }
            catch (Exception ex)
            {
                Trace.TraceWarning(
                    $"[RemoteCopy] temporary file cleanup failed {path}: " +
                    $"{ex.GetType().Name}: {ex.Message}");
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _workSignal.Set();
            _retrySignal.Set();

            if (Thread.CurrentThread != _workerThread)
            {
                if (_workerThread.Join(2000))
                {
                    _workSignal.Dispose();
                    _retrySignal.Dispose();
                }
                else
                {
                    Trace.TraceWarning(
                        "[RemoteCopy] worker did not stop within 2 seconds; pending markers remain recoverable");
                }
            }
        }
    }

    internal struct CopyItem
    {
        public string LocalPath;
        public string RelativePath;
        public string MarkerPath;
        public int Attempt;
    }
}
