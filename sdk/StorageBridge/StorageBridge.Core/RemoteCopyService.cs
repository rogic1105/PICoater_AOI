using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace StorageBridge.Core
{
    /// <summary>
    /// 背景遠端複製服務：將存檔佇列化，背景執行緒逐一 File.Copy 到遠端路徑。
    /// 支援重試（3 次）、斷線時暫存佇列。
    /// 遠端的循環清理由儲存電腦自行負責，檢測電腦只負責複製。
    /// </summary>
    public class RemoteCopyService : IDisposable
    {
        private readonly Func<string> _getRemotePath;
        private readonly Func<string> _getLocalRoot;
        private readonly ConcurrentQueue<CopyItem> _queue = new ConcurrentQueue<CopyItem>();
        private readonly Thread _workerThread;
        private readonly ManualResetEventSlim _signal = new ManualResetEventSlim(false);
        private volatile bool _disposed;

        private const int MaxRetries = 3;
        private const int RetryDelayMs = 2000;

        /// <summary>佇列中待複製的檔案數。</summary>
        public int QueueCount => _queue.Count;
        /// <summary>累計成功複製的檔案數。</summary>
        public long TotalCopiedFiles { get; private set; }
        /// <summary>累計成功複製的位元組數。</summary>
        public long TotalCopiedBytes { get; private set; }
        /// <summary>累計失敗的檔案數。</summary>
        public long TotalFailedFiles { get; private set; }

        public RemoteCopyService(
            Func<string> getRemotePath,
            Func<string> getLocalRoot)
        {
            _getRemotePath = getRemotePath;
            _getLocalRoot  = getLocalRoot;

            _workerThread = new Thread(WorkerLoop)
            {
                IsBackground = true,
                Name = "RemoteCopyWorker",
                Priority = ThreadPriority.BelowNormal
            };
            _workerThread.Start();
        }

        /// <summary>
        /// 將單一檔案排入複製佇列。
        /// </summary>
        public void EnqueueFile(string localFilePath)
        {
            if (string.IsNullOrWhiteSpace(localFilePath)) return;
            string remotePath = _getRemotePath();
            if (string.IsNullOrWhiteSpace(remotePath)) return;

            _queue.Enqueue(new CopyItem { LocalPath = localFilePath });
            _signal.Set();
        }

        /// <summary>
        /// 將多個檔案排入複製佇列。
        /// </summary>
        public void EnqueueFiles(string[] localFilePaths)
        {
            if (localFilePaths == null) return;
            string remotePath = _getRemotePath();
            if (string.IsNullOrWhiteSpace(remotePath)) return;

            foreach (string path in localFilePaths)
            {
                if (!string.IsNullOrWhiteSpace(path))
                    _queue.Enqueue(new CopyItem { LocalPath = path });
            }
            if (localFilePaths.Length > 0)
                _signal.Set();
        }

        // ── 背景執行緒 ──────────────────────────────────────────────────

        private void WorkerLoop()
        {
            while (!_disposed)
            {
                _signal.Wait(TimeSpan.FromSeconds(5));
                _signal.Reset();

                while (!_disposed && _queue.TryDequeue(out CopyItem item))
                {
                    ProcessItem(item);
                }
            }
        }

        private void ProcessItem(CopyItem item)
        {
            string remotePath = _getRemotePath();
            string localRoot  = _getLocalRoot();

            if (string.IsNullOrWhiteSpace(remotePath) || string.IsNullOrWhiteSpace(localRoot))
                return;
            if (!File.Exists(item.LocalPath))
                return;

            // 計算相對路徑：local file 相對於 localRoot
            string relativePath = GetRelativePath(localRoot, item.LocalPath);
            if (relativePath == null) return;

            string destPath = Path.Combine(remotePath, relativePath);
            string destDir  = Path.GetDirectoryName(destPath);

            for (int attempt = 1; attempt <= MaxRetries; attempt++)
            {
                try
                {
                    if (destDir != null)
                        Directory.CreateDirectory(destDir);

                    File.Copy(item.LocalPath, destPath, overwrite: true);

                    long fileSize = new FileInfo(item.LocalPath).Length;
                    TotalCopiedFiles++;
                    TotalCopiedBytes += fileSize;
                    return; // 成功
                }
                catch (Exception ex)
                {
                    if (attempt == MaxRetries)
                    {
                        TotalFailedFiles++;
                        Trace.TraceWarning(
                            $"[RemoteCopy] Failed after {MaxRetries} retries: {item.LocalPath} → {ex.Message}");
                    }
                    else
                    {
                        Thread.Sleep(RetryDelayMs);
                    }
                }
            }
        }

        // ── 工具 ─────────────────────────────────────────────────────────

        private static string GetRelativePath(string basePath, string fullPath)
        {
            if (!basePath.EndsWith(Path.DirectorySeparatorChar.ToString()))
                basePath += Path.DirectorySeparatorChar;
            if (fullPath.StartsWith(basePath, StringComparison.OrdinalIgnoreCase))
                return fullPath.Substring(basePath.Length);
            return null;
        }

        public void Dispose()
        {
            _disposed = true;
            _signal.Set(); // 喚醒 worker 讓它退出
        }
    }

    internal struct CopyItem
    {
        public string LocalPath;
    }
}
