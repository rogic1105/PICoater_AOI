using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

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
        // Regex: YYYYMMDD_HHMMSS-CamID.bmp
        private readonly Regex _fileNameRegex = new Regex(@"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})-(\d)");

        public int FileCount => _metadataCache.Count;

        public void LoadDirectory(string rootPath)
        {
            _metadataCache.Clear();
            if (!Directory.Exists(rootPath)) return;

            var rawFiles = Directory.GetFiles(rootPath, "*.bmp", SearchOption.AllDirectories);

            _metadataCache = rawFiles.AsParallel().Select(ParsePath).Where(x => x != null).ToList();
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
                CameraId = int.Parse(match.Groups[7].Value)
            };
        }

        // 下拉選單資料來源
        public List<string> GetYears() => _metadataCache.Select(x => x.Year).Distinct().OrderBy(x => x).ToList();
        public List<string> GetMonths(string y) => _metadataCache.Where(x => x.Year == y).Select(x => x.Month).Distinct().OrderBy(x => x).ToList();
        public List<string> GetDays(string y, string m) => _metadataCache.Where(x => x.Year == y && x.Month == m).Select(x => x.Day).Distinct().OrderBy(x => x).ToList();
        public List<string> GetHours(string y, string m, string d) => _metadataCache.Where(x => x.Year == y && x.Month == m && x.Day == d).Select(x => x.Hour).Distinct().OrderBy(x => x).ToList();
        public List<string> GetMinutes(string y, string m, string d, string h) => _metadataCache.Where(x => x.Year == y && x.Month == m && x.Day == d && x.Hour == h).Select(x => x.Minute).Distinct().OrderBy(x => x).ToList();
        public List<string> GetSeconds(string y, string m, string d, string h, string min) => _metadataCache.Where(x => x.Year == y && x.Month == m && x.Day == d && x.Hour == h && x.Minute == min).Select(x => x.Second).Distinct().OrderBy(x => x).ToList();


        public List<DateTime> GetAvailablePeriods()
        {
            return _metadataCache
                .Select(x => BuildDateTime(x.Year, x.Month, x.Day, x.Hour, x.Minute, x.Second))
                .Where(x => x.HasValue)
                .Select(x => x.Value)
                .Distinct()
                .OrderBy(x => x)
                .ToList();
        }

        private DateTime? BuildDateTime(string y, string m, string d, string h, string min, string s)
        {
            if (int.TryParse(y, out int yi) && int.TryParse(m, out int mi) && int.TryParse(d, out int di) &&
                int.TryParse(h, out int hi) && int.TryParse(min, out int mni) && int.TryParse(s, out int si))
            {
                try { return new DateTime(yi, mi, di, hi, mni, si); }
                catch { return null; }
            }
            return null;
        }

        // 查詢特定時間點的所有相機圖片
        public Dictionary<int, string> GetImages(string y, string m, string d, string h, string min, string s)
        {
            return _metadataCache
                .Where(x => x.Year == y && x.Month == m && x.Day == d && x.Hour == h && x.Minute == min && x.Second == s)
                .ToDictionary(x => x.CameraId, x => x.FullPath);
        }
    }


}