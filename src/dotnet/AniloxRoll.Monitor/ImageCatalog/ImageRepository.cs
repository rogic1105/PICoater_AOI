using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// [DAO] 影像檔案儲存庫，負責與檔案系統溝通。
    /// 核心功能包含：掃描目錄建立索引、提供時間階層的查詢 (Year->Month->Day...)
    /// 以及將查詢條件轉換為實際檔案路徑。
    /// </summary>
    /// 
    public class ImageRepository
    {
        private List<ImageMetadata> _metadataCache = new List<ImageMetadata>();
        // Regex: YYYYMMDD_HHMMSS.fff-CamID
        private readonly Regex _fileNameRegex = new Regex(@"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.(\d{3})-(\d)");

        public int FileCount => _metadataCache.Count;

        public void LoadDirectory(string rootPath)
        {
            _metadataCache.Clear();
            if (!Directory.Exists(rootPath)) return;

            var files = Directory.GetFiles(rootPath, CaptureFileNaming.RawJpgGlob, SearchOption.AllDirectories);

            _metadataCache = files.AsParallel()
                .Select(f =>
                {
                    try { return ParsePath(f); }
                    catch (Exception ex)
                    {
                        Trace.WriteLine($"[ImageRepository] ParsePath({f}) failed: {ex.GetType().Name}: {ex.Message}");
                        return null;
                    }
                })
                .Where(x => x != null)
                .ToList();
        }

        private ImageMetadata ParsePath(string path)
        {
            var fileName = Path.GetFileName(path);
            var match = _fileNameRegex.Match(fileName);
            if (!match.Success) return null;

            return new ImageMetadata
            {
                FullPath = path,
                Year = match.Groups[1].Value,
                Month = match.Groups[2].Value,
                Day = match.Groups[3].Value,
                Hour = match.Groups[4].Value,
                Minute = match.Groups[5].Value,
                Second = match.Groups[6].Value,
                Millisecond = match.Groups[7].Value,
                CameraId = int.Parse(match.Groups[8].Value)
            };
        }

        // ── 簡化 ComboBox 介面（cbReviewDate + cbReviewTime）──────────────────────────

        /// <summary>回傳所有不重複日期（YYYY-MM-DD），已排序。</summary>
        public List<string> GetDates() =>
            _metadataCache
                .Select(x => $"{x.Year}-{x.Month}-{x.Day}")
                .Distinct()
                .OrderByDescending(x => x)
                .ToList();

        /// <summary>回傳指定日期下所有不重複時間（HH:mm:ss.fff），已排序。</summary>
        public List<string> GetTimesForDate(string date) =>
            _metadataCache
                .Where(x => $"{x.Year}-{x.Month}-{x.Day}" == date)
                .Select(x => $"{x.Hour}:{x.Minute}:{x.Second}.{x.Millisecond}")
                .Distinct()
                .OrderByDescending(x => x)
                .ToList();

        public List<DateTime> GetAvailablePeriods()
        {
            return _metadataCache
                .Select(x => BuildDateTime(x.Year, x.Month, x.Day, x.Hour, x.Minute, x.Second + "." + x.Millisecond))
                .Where(x => x.HasValue)
                .Select(x => x.Value)
                .Distinct()
                .OrderBy(x => x)
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
            foreach (var x in _metadataCache)
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
