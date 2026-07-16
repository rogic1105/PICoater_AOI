using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;

namespace StorageBridge.Core
{
    /// <summary>
    /// 循環儲存管理：監控磁碟可用空間，低於門檻時從最舊的日資料夾開始刪除圖片。
    /// CSV 永不刪除。
    /// </summary>
    public class StorageRetentionService : IDisposable
    {
        private readonly Func<string> _getRootPath;
        private readonly Func<long>   _getMinFreeBytes;
        private readonly Func<string, bool> _shouldPreserveDayFolder;

        private volatile int _running;
        private bool _invalidThresholdReported;

        /// <summary>最近一次清理的日期資料夾數。</summary>
        public int LastCleanedDayFolders { get; private set; }
        /// <summary>最近一次清理釋放的空間（bytes）。</summary>
        public long LastCleanedBytes { get; private set; }
        /// <summary>最近一次探測的磁碟已用量（bytes，全磁碟）。</summary>
        public long LastScannedTotalBytes { get; private set; }
        /// <summary>最近一次探測的磁碟總容量（bytes）。</summary>
        public long LastDriveTotalBytes { get; private set; }

        /// <summary>清理完成事件（UI 可訂閱顯示狀態）。</summary>
        public event Action<RetentionCleanupResult> OnCleanupCompleted;

        public StorageRetentionService(
            Func<string> getRootPath,
            Func<long>   getMinFreeBytes,
            Func<string, bool> shouldPreserveDayFolder = null)
        {
            _getRootPath = getRootPath;
            _getMinFreeBytes = getMinFreeBytes;
            _shouldPreserveDayFolder = shouldPreserveDayFolder;
        }

        /// <summary>觸發一次清理（事件驅動：grab 結束 / watchdog / 每 10 grab / 啟動時）。</summary>
        public void RunCleanup()
        {
            if (System.Threading.Interlocked.CompareExchange(ref _running, 1, 0) != 0) return;
            try
            {
                string root = _getRootPath();
                if (string.IsNullOrWhiteSpace(root) || !Directory.Exists(root)) return;

                long minFreeBytes = _getMinFreeBytes();
                if (minFreeBytes <= 0) return;

                var (freeBytes, driveTotal) = GetDriveFreeSpace(root);
                LastDriveTotalBytes   = driveTotal;
                LastScannedTotalBytes = driveTotal > 0 ? driveTotal - freeBytes : 0;

                if (driveTotal > 0 && minFreeBytes >= driveTotal)
                {
                    if (!_invalidThresholdReported)
                    {
                        Trace.TraceError(
                            $"[StorageRetention] Cleanup skipped: min free {minFreeBytes} bytes " +
                            $">= volume total {driveTotal} bytes. No files were deleted.");
                        _invalidThresholdReported = true;
                    }
                    return;
                }
                if (_invalidThresholdReported)
                {
                    Trace.TraceInformation(
                        $"[StorageRetention] Cleanup threshold valid again: min free {minFreeBytes} bytes, " +
                        $"volume total {driveTotal} bytes");
                    _invalidThresholdReported = false;
                }

                if (freeBytes >= minFreeBytes) return;

                var dayFolders = CollectDayFolders(root);
                if (dayFolders.Count == 0) return;

                int deletedDayFolders = 0;
                long deletedBytes = 0;

                foreach (var dayFolder in dayFolders)
                {
                    (freeBytes, _) = GetDriveFreeSpace(root);
                    if (freeBytes >= minFreeBytes) break;

                    if (_shouldPreserveDayFolder?.Invoke(dayFolder.Path) == true)
                    {
                        Trace.TraceWarning(
                            $"[StorageRetention] Preserve pending remote-copy folder: {dayFolder.Path}");
                        continue;
                    }

                    long freed = DeleteDayFolderImages(dayFolder.Path);
                    if (freed > 0)
                    {
                        deletedBytes += freed;
                        deletedDayFolders++;
                        TryRemoveEmptyFolder(dayFolder.Path);
                    }
                }

                LastCleanedDayFolders = deletedDayFolders;
                LastCleanedBytes = deletedBytes;
                (freeBytes, _) = GetDriveFreeSpace(root);
                LastScannedTotalBytes = driveTotal > 0 ? driveTotal - freeBytes : 0;

                var result = new RetentionCleanupResult
                {
                    DeletedDayFolders = deletedDayFolders,
                    FreedBytes = deletedBytes,
                    RemainingBytes = LastScannedTotalBytes,
                    MinFreeBytes = minFreeBytes
                };

                Trace.TraceInformation(
                    $"[StorageRetention] Cleanup done: freed {deletedBytes / (1024 * 1024)} MB, " +
                    $"drive free {freeBytes / (1024 * 1024)} MB / min {minFreeBytes / (1024 * 1024)} MB");

                OnCleanupCompleted?.Invoke(result);
            }
            catch (Exception ex)
            {
                Trace.TraceWarning($"[StorageRetention] {ex.GetType().Name}: {ex.Message}");
            }
            finally
            {
                System.Threading.Interlocked.Exchange(ref _running, 0);
            }
        }

        // ── 日資料夾收集 ─────────────────────────────────────────────────

        internal static System.Collections.Generic.List<DayFolder> CollectDayFolders(string root)
        {
            var result = new System.Collections.Generic.List<DayFolder>();

            foreach (string yearDir in SafeGetDirectories(root))
            {
                string yearName = Path.GetFileName(yearDir);
                if (yearName.Length != 4 || !int.TryParse(yearName, out _)) continue;

                foreach (string monthDir in SafeGetDirectories(yearDir))
                {
                    string monthName = Path.GetFileName(monthDir);
                    if (monthName.Length != 6) continue;

                    foreach (string dayDir in SafeGetDirectories(monthDir))
                    {
                        string dayName = Path.GetFileName(dayDir);
                        if (dayName.Length != 8) continue;
                        if (DateTime.TryParseExact(dayName, "yyyyMMdd",
                            CultureInfo.InvariantCulture, DateTimeStyles.None, out DateTime date))
                        {
                            result.Add(new DayFolder { Path = dayDir, Date = date });
                        }
                    }
                }
            }

            result.Sort((a, b) => a.Date.CompareTo(b.Date));
            return result;
        }

        // ── 刪除邏輯 ────────────────────────────────────────────────────

        /// <summary>刪除日資料夾中的圖片/bin 檔案，保留 CSV。回傳釋放的 bytes。</summary>
        private static long DeleteDayFolderImages(string dayDir)
        {
            long freedBytes = 0;
            string[] extensions = { ".jpg", ".bmp", ".bin", ".mcsf" };

            foreach (string file in SafeGetFiles(dayDir, "*.*", SearchOption.AllDirectories))
            {
                string ext = Path.GetExtension(file).ToLowerInvariant();
                bool tickSidecar = string.Equals(
                    Path.GetFileName(file), "_ticks.csv", StringComparison.OrdinalIgnoreCase);
                if (!tickSidecar && Array.IndexOf(extensions, ext) < 0) continue;

                try
                {
                    long size = new FileInfo(file).Length;
                    File.Delete(file);
                    freedBytes += size;
                }
                catch (Exception ex)
                {
                    Trace.TraceWarning($"[StorageRetention] Delete failed {file}: {ex.Message}");
                }
            }

            return freedBytes;
        }

        private static void TryRemoveEmptyFolder(string dayDir)
        {
            try
            {
                string[] children = Directory.GetDirectories(dayDir, "*", SearchOption.AllDirectories);
                Array.Sort(children, (left, right) => right.Length.CompareTo(left.Length));
                foreach (string child in children)
                {
                    if (Directory.GetFileSystemEntries(child).Length == 0)
                        Directory.Delete(child, false);
                }
                if (Directory.GetFileSystemEntries(dayDir).Length > 0) return;
                Directory.Delete(dayDir, false);

                string monthDir = Path.GetDirectoryName(dayDir);
                if (monthDir != null && Directory.GetFileSystemEntries(monthDir).Length == 0)
                {
                    Directory.Delete(monthDir, false);
                    string yearDir = Path.GetDirectoryName(monthDir);
                    if (yearDir != null && Directory.GetFileSystemEntries(yearDir).Length == 0)
                        Directory.Delete(yearDir, false);
                }
            }
            catch (Exception ex)
            {
                Trace.TraceWarning(
                    $"[StorageRetention] Remove empty folder failed {dayDir}: {ex.Message}");
            }
        }

        // ── 工具方法 ─────────────────────────────────────────────────────

        private static (long freeBytes, long totalBytes) GetDriveFreeSpace(string path)
        {
            try
            {
                string root = Path.GetPathRoot(Path.GetFullPath(path));
                var di = new DriveInfo(root);
                return (di.AvailableFreeSpace, di.TotalSize);
            }
            catch (Exception ex)
            {
                Trace.TraceWarning(
                    $"[StorageRetention] Drive probe failed {path}: {ex.Message}");
                return (long.MaxValue, 0);
            }
        }

        private static string[] SafeGetDirectories(string path)
        {
            try { return Directory.GetDirectories(path); }
            catch (Exception ex)
            {
                Trace.TraceWarning(
                    $"[StorageRetention] Directory scan failed {path}: {ex.Message}");
                return Array.Empty<string>();
            }
        }

        private static string[] SafeGetFiles(string path, string pattern)
        {
            return SafeGetFiles(path, pattern, SearchOption.TopDirectoryOnly);
        }

        private static string[] SafeGetFiles(
            string path, string pattern, SearchOption searchOption)
        {
            try { return Directory.GetFiles(path, pattern, searchOption); }
            catch (Exception ex)
            {
                Trace.TraceWarning(
                    $"[StorageRetention] File scan failed {path}: {ex.Message}");
                return Array.Empty<string>();
            }
        }

        public void Dispose() { }
    }

    internal struct DayFolder
    {
        public string Path;
        public DateTime Date;
    }

    public class RetentionCleanupResult
    {
        public int DeletedDayFolders { get; set; }
        public long FreedBytes { get; set; }
        public long RemainingBytes { get; set; }
        public long MinFreeBytes { get; set; }
    }
}
