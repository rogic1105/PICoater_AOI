using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Core.Data
{
    public sealed class ImageRepositoryLoadResult
    {
        public int FileCount { get; internal set; }
        public int CsvRecordCount { get; internal set; }
        public int CsvBackedArchiveCount { get; internal set; }
        public int ArchiveFallbackCount { get; internal set; }
        public int LegacyFileCount { get; internal set; }
        public long EnumerationMilliseconds { get; internal set; }
        public long ArchiveIndexMilliseconds { get; internal set; }
        public long MetadataIndexMilliseconds { get; internal set; }
        public long PeriodIndexMilliseconds { get; internal set; }
        public long ElapsedMilliseconds { get; internal set; }
    }

    /// <summary>
    /// [DAO] 影像檔案儲存庫，負責與檔案系統溝通。
    /// 核心功能包含：掃描目錄建立索引、提供時間階層的查詢 (Year->Month->Day...)
    /// 以及將查詢條件轉換為實際檔案路徑。
    /// </summary>
    /// 
    public class ImageRepository
    {
        private volatile ImageMetadata[] _metadataCache = new ImageMetadata[0];
        private volatile DateTime[] _availablePeriods = new DateTime[0];
        public int FileCount => _metadataCache.Length;

        public Task<ImageRepositoryLoadResult> LoadDirectoryAsync(string rootPath)
        {
            return Task.Run(() => LoadDirectory(rootPath));
        }

        public ImageRepositoryLoadResult LoadDirectory(string rootPath)
        {
            var result = new ImageRepositoryLoadResult();
            var watch = Stopwatch.StartNew();
            if (!Directory.Exists(rootPath))
            {
                _metadataCache = new ImageMetadata[0];
                _availablePeriods = new DateTime[0];
                return result;
            }

            // These searches traverse the same large date tree but target disjoint file types.
            // Run them together so a 30,000-grab catalog does not pay three full serial walks.
            Task<string[]> legacyFilesTask = Task.Run(() => Directory.GetFiles(
                rootPath, CaptureFileNaming.RawJpgGlob, SearchOption.AllDirectories));
            Task<string[]> archiveFilesTask = Task.Run(() => Directory.GetFiles(
                rootPath, "*" + CaptureArchiveStore.Extension, SearchOption.AllDirectories));
            Task<string[]> csvFilesTask = Task.Run(() => Directory.GetFiles(
                rootPath, "*.csv", SearchOption.AllDirectories));
            Task.WhenAll(legacyFilesTask, archiveFilesTask, csvFilesTask)
                .GetAwaiter()
                .GetResult();
            long phaseStartedAt = watch.ElapsedMilliseconds;
            result.EnumerationMilliseconds = phaseStartedAt;

            string[] legacyFiles = legacyFilesTask.Result
                .Where(path => !path.EndsWith(
                    CaptureFileNaming.ThumbRawJpg,
                    StringComparison.OrdinalIgnoreCase))
                .ToArray();
            string[] archiveFiles = archiveFilesTask.Result;
            string[] csvFiles = csvFilesTask.Result;
            var archiveFileSet = new HashSet<string>(
                archiveFiles.Select(Path.GetFullPath),
                StringComparer.OrdinalIgnoreCase);
            var csvBackedPaths = new List<string>();
            var csvBackedArchives = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            // Daily CSV records are needed here only to map archive members. Legacy
            // folders already have real raw-image paths, while the report presenter
            // independently reads the CSV once for CFG and verdict indexes. Avoiding
            // this redundant parse removes 210,000 record allocations from the
            // 30,000-grab review catalog load.
            foreach (string csvPath in archiveFiles.Length == 0
                ? Array.Empty<string>()
                : csvFiles)
            {
                string dateName = Path.GetFileNameWithoutExtension(csvPath);
                if (dateName.Length != 8 || !dateName.All(char.IsDigit)) continue;
                string archiveDirectory = Path.Combine(
                    Path.GetDirectoryName(csvPath), dateName);
                try
                {
                    using (StreamReader reader = InspectionCsvReader.OpenShared(csvPath))
                    {
                        string line;
                        while ((line = reader.ReadLine()) != null)
                        {
                            if (!InspectionCsvReader.TryParseRecord(
                                line, out InspectionCsvRecord record))
                                continue;
                            result.CsvRecordCount++;
                            string archivePath = Path.GetFullPath(Path.Combine(
                                archiveDirectory,
                                record.GrabId + CaptureArchiveStore.Extension));
                            if (!archiveFileSet.Contains(archivePath)) continue;
                            string baseName = record.FileName;
                            if (CaptureFileNaming.IsRawJpg(baseName))
                                baseName = CaptureFileNaming.StripRawJpg(baseName);
                            csvBackedPaths.Add(
                                CaptureArchiveStore.CreateVirtualRawPath(
                                    archivePath, baseName));
                            csvBackedArchives.Add(archivePath);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine(
                        $"[ImageRepository] CSV index {csvPath} failed: " +
                        $"{ex.GetType().Name}: {ex.Message}");
                }
            }
            long phaseFinishedAt = watch.ElapsedMilliseconds;
            result.ArchiveIndexMilliseconds = phaseFinishedAt - phaseStartedAt;
            phaseStartedAt = phaseFinishedAt;

            string[] fallbackArchives = archiveFiles
                .Where(path => !csvBackedArchives.Contains(Path.GetFullPath(path)))
                .ToArray();
            var archivePaths = fallbackArchives
                .AsParallel()
                .SelectMany(CaptureArchiveStore.ListAllVirtualRawPaths)
                .ToArray();
            int capacity = legacyFiles.Length + csvBackedPaths.Count +
                archivePaths.Length;
            var metadataByKey = new Dictionary<CaptureKey, ImageMetadata>(capacity);
            foreach (string path in legacyFiles
                .Concat(csvBackedPaths)
                .Concat(archivePaths))
            {
                ImageMetadata item;
                try
                {
                    item = ParsePath(path);
                }
                catch (Exception ex)
                {
                    Trace.WriteLine(
                        $"[ImageRepository] ParsePath({path}) failed: " +
                        $"{ex.GetType().Name}: {ex.Message}");
                    continue;
                }
                if (item == null) continue;

                var key = new CaptureKey(item.Timestamp, item.CameraId);
                if (!metadataByKey.TryGetValue(key, out ImageMetadata existing) ||
                    (!CaptureArchiveStore.IsVirtualPath(existing.FullPath) &&
                     CaptureArchiveStore.IsVirtualPath(item.FullPath)))
                {
                    metadataByKey[key] = item;
                }
            }
            ImageMetadata[] metadata = metadataByKey.Values.ToArray();
            phaseFinishedAt = watch.ElapsedMilliseconds;
            result.MetadataIndexMilliseconds = phaseFinishedAt - phaseStartedAt;
            phaseStartedAt = phaseFinishedAt;

            // Period navigation reads this on every selection change. Build the sorted index once
            // with the file catalog instead of reparsing and sorting every image on the UI thread.
            DateTime[] availablePeriods = metadata
                .Select(x => x.Timestamp)
                .Distinct()
                .OrderBy(x => x)
                .ToArray();
            result.PeriodIndexMilliseconds =
                watch.ElapsedMilliseconds - phaseStartedAt;

            // Publish a complete immutable snapshot in one step. While a background refresh runs,
            // readers continue using the previous catalog rather than observing a half-built list.
            _metadataCache = metadata;
            _availablePeriods = availablePeriods;

            watch.Stop();
            result.FileCount = metadata.Length;
            result.CsvBackedArchiveCount = csvBackedArchives.Count;
            result.ArchiveFallbackCount = fallbackArchives.Length;
            result.LegacyFileCount = legacyFiles.Length;
            result.ElapsedMilliseconds = watch.ElapsedMilliseconds;
            return result;
        }

        private readonly struct CaptureKey : IEquatable<CaptureKey>
        {
            private readonly DateTime _timestamp;
            private readonly int _cameraId;

            public CaptureKey(DateTime timestamp, int cameraId)
            {
                _timestamp = timestamp;
                _cameraId = cameraId;
            }

            public bool Equals(CaptureKey other) =>
                _timestamp == other._timestamp && _cameraId == other._cameraId;

            public override bool Equals(object obj) =>
                obj is CaptureKey other && Equals(other);

            public override int GetHashCode()
            {
                unchecked
                {
                    return (_timestamp.GetHashCode() * 397) ^ _cameraId;
                }
            }
        }

        private static ImageMetadata ParsePath(string path)
        {
            string fileName = CaptureArchiveStore.IsVirtualPath(path)
                ? CaptureArchiveStore.GetVirtualBaseName(path)
                : Path.GetFileName(path);
            if (string.IsNullOrEmpty(fileName) || fileName.Length < 21 ||
                fileName[8] != '_' || fileName[15] != '.' || fileName[19] != '-')
                return null;
            if (!TryReadNumber(fileName, 0, 4, out int year) ||
                !TryReadNumber(fileName, 4, 2, out int month) ||
                !TryReadNumber(fileName, 6, 2, out int day) ||
                !TryReadNumber(fileName, 9, 2, out int hour) ||
                !TryReadNumber(fileName, 11, 2, out int minute) ||
                !TryReadNumber(fileName, 13, 2, out int second) ||
                !TryReadNumber(fileName, 16, 3, out int millisecond) ||
                !TryReadNumber(fileName, 20, 1, out int cameraId))
                return null;

            DateTime timestamp;
            try
            {
                timestamp = new DateTime(
                    year, month, day, hour, minute, second, millisecond);
            }
            catch (ArgumentOutOfRangeException)
            {
                return null;
            }

            return new ImageMetadata
            {
                FullPath = path,
                Year = fileName.Substring(0, 4),
                Month = fileName.Substring(4, 2),
                Day = fileName.Substring(6, 2),
                Hour = fileName.Substring(9, 2),
                Minute = fileName.Substring(11, 2),
                Second = fileName.Substring(13, 2),
                Millisecond = fileName.Substring(16, 3),
                CameraId = cameraId,
                Timestamp = timestamp
            };
        }

        private static bool TryReadNumber(
            string value, int offset, int count, out int number)
        {
            number = 0;
            if (offset < 0 || count <= 0 || offset + count > value.Length)
                return false;
            for (int i = offset; i < offset + count; i++)
            {
                int digit = value[i] - '0';
                if (digit < 0 || digit > 9) return false;
                number = number * 10 + digit;
            }
            return true;
        }

        // ── 簡化 ComboBox 介面（cbReviewDate + cbReviewTime）──────────────────────────

        /// <summary>回傳所有不重複日期（YYYY-MM-DD），已排序。</summary>
        public List<string> GetDates() =>
            ((IEnumerable<ImageMetadata>)_metadataCache)
                .Select(x => $"{x.Year}-{x.Month}-{x.Day}")
                .Distinct()
                .OrderByDescending(x => x)
                .ToList();

        /// <summary>回傳指定日期下所有不重複時間（HH:mm:ss.fff），已排序。</summary>
        public List<string> GetTimesForDate(string date) =>
            ((IEnumerable<ImageMetadata>)_metadataCache)
                .Where(x => $"{x.Year}-{x.Month}-{x.Day}" == date)
                .Select(x => $"{x.Hour}:{x.Minute}:{x.Second}.{x.Millisecond}")
                .Distinct()
                .OrderByDescending(x => x)
                .ToList();

        public IReadOnlyList<DateTime> GetAvailablePeriods() => _availablePeriods;

        /// <summary>
        /// Builds the lightweight Review navigation catalog from the image index.
        /// Full CSV statistics are intentionally not required for first paint.
        /// </summary>
        public List<GrabIdInfo> GetGrabIdInfosDescending()
        {
            var byGrabId = new Dictionary<string, GrabIdInfo>(StringComparer.Ordinal);
            foreach (DateTime timestamp in _availablePeriods)
            {
                string grabId = InspectionLogService.FormatGrabId(timestamp);
                if (!byGrabId.TryGetValue(grabId, out GrabIdInfo info))
                {
                    byGrabId[grabId] = new GrabIdInfo
                    {
                        GrabId = grabId,
                        Earliest = timestamp,
                        Latest = timestamp
                    };
                    continue;
                }

                if (timestamp < info.Earliest) info.Earliest = timestamp;
                if (timestamp > info.Latest) info.Latest = timestamp;
            }

            return byGrabId.Values
                .OrderByDescending(info => info.GrabId, StringComparer.Ordinal)
                .ToList();
        }

        /// <summary>
        /// 從 "ss.fff" 格式解析秒與毫秒，建構 DateTime。
        /// </summary>
        private static DateTime? BuildDateTime(string y, string m, string d, string h, string min, string sFff)
        {
            if (!int.TryParse(y, out int yi) || !int.TryParse(m, out int mi) || !int.TryParse(d, out int di) ||
                !int.TryParse(h, out int hi) || !int.TryParse(min, out int mni))
                return null;

            int si = 0, msi = 0;
            if (sFff != null)
            {
                int dot = sFff.IndexOf('.');
                if (dot >= 0)
                {
                    int.TryParse(sFff.Substring(0, dot), out si);
                    int.TryParse(sFff.Substring(dot + 1), out msi);
                }
                else
                {
                    int.TryParse(sFff, out si);
                }
            }

            try { return new DateTime(yi, mi, di, hi, mni, si, msi); }
            catch { return null; }
        }

        public Dictionary<int, string> GetImages(string y, string m, string d, string h, string min, string sFff)
        {
            string sec = sFff, ms = "";
            if (sFff != null)
            {
                int dot = sFff.IndexOf('.');
                if (dot >= 0) { sec = sFff.Substring(0, dot); ms = sFff.Substring(dot + 1); }
            }
            var result = new Dictionary<int, string>();
            ImageMetadata[] metadata = _metadataCache;
            foreach (var x in metadata)
            {
                if (x.Year != y || x.Month != m || x.Day != d || x.Hour != h || x.Minute != min || x.Second != sec || x.Millisecond != ms)
                    continue;
                result[x.CameraId] = x.FullPath;
            }
            return result;
        }

        /// <summary>以不可變的時點快照查詢影像，避免 async 載入期間再讀 UI ComboBox 的共享選取。</summary>
        public Dictionary<int, string> GetImages(DateTime period)
            => GetImages(
                period.ToString("yyyy", CultureInfo.InvariantCulture),
                period.ToString("MM", CultureInfo.InvariantCulture),
                period.ToString("dd", CultureInfo.InvariantCulture),
                period.ToString("HH", CultureInfo.InvariantCulture),
                period.ToString("mm", CultureInfo.InvariantCulture),
                period.ToString("ss.fff", CultureInfo.InvariantCulture));
    }


}
