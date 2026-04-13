using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// 循環儲存管理：監控磁碟用量，超過上限時從最舊的日資料夾開始刪除。
    /// CSV 永不刪除，Fail 圖片在保護期內不刪除。
    /// </summary>
    public class StorageRetentionService : IDisposable
    {
        private readonly Func<string> _getRootPath;
        private readonly Func<long>   _getMaxBytes;
        private readonly Func<int>    _getFailProtectDays;

        private Timer _timer;
        private volatile bool _running;

        /// <summary>最近一次清理刪除的檔案數。</summary>
        public int LastCleanedFileCount { get; private set; }
        /// <summary>最近一次清理釋放的空間（bytes）。</summary>
        public long LastCleanedBytes { get; private set; }
        /// <summary>最近一次掃描的總用量（bytes）。</summary>
        public long LastScannedTotalBytes { get; private set; }

        /// <summary>清理完成事件（UI 可訂閱顯示狀態）。</summary>
        public event Action<RetentionCleanupResult> OnCleanupCompleted;

        public StorageRetentionService(
            Func<string> getRootPath,
            Func<long>   getMaxBytes,
            Func<int>    getFailProtectDays)
        {
            _getRootPath        = getRootPath;
            _getMaxBytes        = getMaxBytes;
            _getFailProtectDays = getFailProtectDays;
        }

        /// <summary>啟動定期檢查（預設每 30 分鐘）。</summary>
        public void Start(int intervalMinutes = 30)
        {
            _timer?.Dispose();
            _timer = new Timer(_ => RunCleanup(), null,
                TimeSpan.FromMinutes(1),         // 啟動後 1 分鐘首次檢查
                TimeSpan.FromMinutes(intervalMinutes));
        }

        /// <summary>手動觸發一次清理。</summary>
        public void RunCleanup()
        {
            if (_running) return;
            _running = true;
            try
            {
                string root = _getRootPath();
                if (string.IsNullOrWhiteSpace(root) || !Directory.Exists(root)) return;

                long maxBytes = _getMaxBytes();
                if (maxBytes <= 0) return;

                // 1. 掃描總用量
                long totalBytes = GetDirectorySize(root);
                LastScannedTotalBytes = totalBytes;

                if (totalBytes <= maxBytes) return;

                // 2. 收集所有日資料夾，按日期排序（最舊優先）
                var dayFolders = CollectDayFolders(root);
                if (dayFolders.Count == 0) return;

                // 3. 讀取 Fail GrabId 集合（保護期內）
                int protectDays = _getFailProtectDays();
                DateTime protectCutoff = DateTime.Today.AddDays(-protectDays);
                var failGrabIds = CollectFailGrabIds(root, protectCutoff);

                // 4. 從最舊開始刪，直到低於上限
                int deletedFiles = 0;
                long deletedBytes = 0;

                foreach (var dayFolder in dayFolders)
                {
                    if (totalBytes <= maxBytes) break;

                    // 跳過今天的資料夾（不刪正在寫入的）
                    if (dayFolder.Date >= DateTime.Today) continue;

                    long freed = DeleteDayFolderImages(dayFolder.Path, failGrabIds);
                    if (freed > 0)
                    {
                        deletedBytes += freed;
                        totalBytes -= freed;
                        deletedFiles++;

                        // 如果資料夾只剩 CSV，也算清理完成
                        TryRemoveEmptyImageFolder(dayFolder.Path);
                    }
                }

                LastCleanedFileCount = deletedFiles;
                LastCleanedBytes = deletedBytes;
                LastScannedTotalBytes = totalBytes;

                var result = new RetentionCleanupResult
                {
                    DeletedFiles = deletedFiles,
                    FreedBytes = deletedBytes,
                    RemainingBytes = totalBytes,
                    MaxBytes = maxBytes
                };

                Trace.TraceInformation(
                    $"[StorageRetention] Cleanup done: freed {deletedBytes / (1024 * 1024)} MB, " +
                    $"remaining {totalBytes / (1024 * 1024)} MB / {maxBytes / (1024 * 1024)} MB");

                OnCleanupCompleted?.Invoke(result);
            }
            catch (Exception ex)
            {
                Trace.TraceWarning($"[StorageRetention] {ex.GetType().Name}: {ex.Message}");
            }
            finally
            {
                _running = false;
            }
        }

        // ── 日資料夾收集 ─────────────────────────────────────────────────

        /// <summary>
        /// 收集所有 YYYYMMDD 格式的日資料夾，按日期升序排列。
        /// 目錄結構：root\YYYY\YYYYMM\YYYYMMDD\
        /// </summary>
        internal static List<DayFolder> CollectDayFolders(string root)
        {
            var result = new List<DayFolder>();

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

        // ── Fail 保護 ────────────────────────────────────────────────────

        /// <summary>
        /// 從 CSV 中讀取保護期內有 Fail 紀錄的 GrabId 集合。
        /// CSV 格式：Id,FileName,MaxExceed,MeanExceed,...
        /// MaxExceed=1 或 MeanExceed=1 → Fail。
        /// </summary>
        internal static HashSet<string> CollectFailGrabIds(string root, DateTime protectCutoff)
        {
            var failIds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (string yearDir in SafeGetDirectories(root))
            {
                string yearName = Path.GetFileName(yearDir);
                if (yearName.Length != 4 || !int.TryParse(yearName, out int year)) continue;
                // 跳過太舊的年份（保護期之前的不需要判斷）
                if (year < protectCutoff.Year - 1) continue;

                foreach (string monthDir in SafeGetDirectories(yearDir))
                {
                    foreach (string csvFile in SafeGetFiles(monthDir, "*.csv"))
                    {
                        string csvName = Path.GetFileNameWithoutExtension(csvFile);
                        if (csvName.Length != 8) continue;
                        if (!DateTime.TryParseExact(csvName, "yyyyMMdd",
                            CultureInfo.InvariantCulture, DateTimeStyles.None, out DateTime csvDate))
                            continue;
                        if (csvDate < protectCutoff) continue;

                        try
                        {
                            foreach (string line in File.ReadLines(csvFile))
                            {
                                if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#") || line.StartsWith("Id,"))
                                    continue;
                                var parts = line.Split(',');
                                if (parts.Length < 4) continue;
                                // MaxExceed=parts[2], MeanExceed=parts[3]
                                if (parts[2] == "1" || parts[3] == "1")
                                    failIds.Add(parts[0]); // GrabId
                            }
                        }
                        catch (Exception ex)
                        {
                            Trace.TraceWarning($"[StorageRetention] CSV read error {csvFile}: {ex.Message}");
                        }
                    }
                }
            }

            return failIds;
        }

        // ── 刪除邏輯 ────────────────────────────────────────────────────

        /// <summary>
        /// 刪除日資料夾中的圖片/bin 檔案，保留 CSV 和 Fail 圖片。
        /// 回傳釋放的 bytes。
        /// </summary>
        private static long DeleteDayFolderImages(string dayDir, HashSet<string> failGrabIds)
        {
            long freedBytes = 0;
            string[] extensions = { ".jpg", ".bmp", ".bin" };

            foreach (string file in SafeGetFiles(dayDir, "*.*"))
            {
                string ext = Path.GetExtension(file).ToLowerInvariant();
                if (!extensions.Contains(ext)) continue; // 保留 CSV 和其他檔案

                // 從檔名提取 GrabId：格式 yyyyMMdd_HHmmss.fff-N_suffix.ext
                // GrabId 在 CSV 中的格式：yyMMdd-HHmmss
                string fileName = Path.GetFileNameWithoutExtension(file);
                string grabId = ExtractGrabIdFromFileName(fileName);

                if (grabId != null && failGrabIds.Contains(grabId))
                    continue; // 保護 Fail 圖片

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

        /// <summary>
        /// 從檔名提取 GrabId。
        /// 檔名格式：yyyyMMdd_HHmmss.fff-N_suffix → GrabId = yyMMdd-HHmmss
        /// </summary>
        internal static string ExtractGrabIdFromFileName(string fileName)
        {
            // 20260413_143012.345-1_raw → 需要前17字元 yyyyMMdd_HHmmss.fff
            if (fileName == null || fileName.Length < 17) return null;

            // 解析 yyyyMMdd_HHmmss 部分
            string datePart = fileName.Substring(0, 8);  // yyyyMMdd
            string timePart = fileName.Length >= 15 ? fileName.Substring(9, 6) : null; // HHmmss
            if (timePart == null) return null;

            // 轉換為 GrabId 格式：yyMMdd-HHmmss
            if (datePart.Length == 8 && timePart.Length == 6)
                return datePart.Substring(2) + "-" + timePart; // yyMMdd-HHmmss

            return null;
        }

        /// <summary>刪除只剩 CSV 的空圖片日資料夾。</summary>
        private static void TryRemoveEmptyImageFolder(string dayDir)
        {
            try
            {
                var remaining = Directory.GetFiles(dayDir);
                if (remaining.Length == 0)
                {
                    Directory.Delete(dayDir, false);

                    // 往上清空的月/年資料夾也嘗試刪除
                    string monthDir = Path.GetDirectoryName(dayDir);
                    if (monthDir != null && Directory.GetFileSystemEntries(monthDir).Length == 0)
                    {
                        Directory.Delete(monthDir, false);
                        string yearDir = Path.GetDirectoryName(monthDir);
                        if (yearDir != null && Directory.GetFileSystemEntries(yearDir).Length == 0)
                            Directory.Delete(yearDir, false);
                    }
                }
            }
            catch { /* 非關鍵，忽略 */ }
        }

        // ── 工具方法 ─────────────────────────────────────────────────────

        private static long GetDirectorySize(string path)
        {
            long size = 0;
            try
            {
                foreach (string file in Directory.EnumerateFiles(path, "*.*", SearchOption.AllDirectories))
                {
                    try { size += new FileInfo(file).Length; }
                    catch { /* 存取被拒 */ }
                }
            }
            catch { /* 目錄不存在或權限不足 */ }
            return size;
        }

        private static string[] SafeGetDirectories(string path)
        {
            try { return Directory.GetDirectories(path); }
            catch { return Array.Empty<string>(); }
        }

        private static string[] SafeGetFiles(string path, string pattern)
        {
            try { return Directory.GetFiles(path, pattern); }
            catch { return Array.Empty<string>(); }
        }

        public void Dispose()
        {
            _timer?.Dispose();
        }
    }

    internal struct DayFolder
    {
        public string Path;
        public DateTime Date;
    }

    public class RetentionCleanupResult
    {
        public int DeletedFiles { get; set; }
        public long FreedBytes { get; set; }
        public long RemainingBytes { get; set; }
        public long MaxBytes { get; set; }
    }
}
