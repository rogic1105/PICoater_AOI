using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace AniloxRoll.Monitor.Core.Data
{
    public class ImageRepository
    {
        private List<ImageMetadata> _files = new List<ImageMetadata>();
        // Regex: YYYYMMDD_HHMMSS-CamID.bmp
        private readonly Regex _fileNameRegex = new Regex(@"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})-(\d)");

        public int FileCount => _files.Count;

        public void LoadDirectory(string rootPath)
        {
            _files.Clear();
            if (!Directory.Exists(rootPath)) return;

            var rawFiles = Directory.GetFiles(rootPath, "*.bmp", SearchOption.AllDirectories);

            // 使用 PLINQ 加速大量檔案字串解析
            _files = rawFiles.AsParallel().Select(ParsePath).Where(x => x != null).ToList();
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
        public List<string> GetYears() => _files.Select(x => x.Year).Distinct().OrderBy(x => x).ToList();
        public List<string> GetMonths(string y) => _files.Where(x => x.Year == y).Select(x => x.Month).Distinct().OrderBy(x => x).ToList();
        public List<string> GetDays(string y, string m) => _files.Where(x => x.Year == y && x.Month == m).Select(x => x.Day).Distinct().OrderBy(x => x).ToList();
        public List<string> GetHours(string y, string m, string d) => _files.Where(x => x.Year == y && x.Month == m && x.Day == d).Select(x => x.Hour).Distinct().OrderBy(x => x).ToList();
        public List<string> GetMinutes(string y, string m, string d, string h) => _files.Where(x => x.Year == y && x.Month == m && x.Day == d && x.Hour == h).Select(x => x.Minute).Distinct().OrderBy(x => x).ToList();
        public List<string> GetSeconds(string y, string m, string d, string h, string min) => _files.Where(x => x.Year == y && x.Month == m && x.Day == d && x.Hour == h && x.Minute == min).Select(x => x.Second).Distinct().OrderBy(x => x).ToList();

        // 查詢特定時間點的所有相機圖片
        public Dictionary<int, string> GetImages(string y, string m, string d, string h, string min, string s)
        {
            return _files
                .Where(x => x.Year == y && x.Month == m && x.Day == d && x.Hour == h && x.Minute == min && x.Second == s)
                .ToDictionary(x => x.CameraId, x => x.FullPath);
        }
    }
}