// PICoater_AOI\src_dotnet\AniloxRoll.Monitor\ImageDataManager.cs

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace AniloxRoll.Monitor.Core
{
    public class ImageFileInfo
    {
        public string FullPath { get; set; }
        public string Year { get; set; }
        public string Month { get; set; }
        public string Day { get; set; }
        public string Hour { get; set; }
        public string Minute { get; set; }
        public string Second { get; set; }
        public int CamId { get; set; }
    }

    public class ImageDataManager
    {
        private List<ImageFileInfo> _allFiles = new List<ImageFileInfo>();

        // Regex 強制對應檔名格式：YYYYMMDD_HHMMSS-CamID.bmp
        // 這樣就不依賴資料夾名稱，選哪一層資料夾都能跑
        private Regex _regex = new Regex(@"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})-(\d)");

        public int FileCount => _allFiles.Count;

        public void LoadDirectory(string rootPath)
        {
            _allFiles.Clear();
            if (!Directory.Exists(rootPath)) return;

            // 關鍵：使用 AllDirectories，不管選 2026 還是 20260101，都會掃到最底層的圖片
            var files = Directory.GetFiles(rootPath, "*.bmp", SearchOption.AllDirectories);

            foreach (var file in files)
            {
                var filename = Path.GetFileName(file);
                var match = _regex.Match(filename);
                if (match.Success)
                {
                    _allFiles.Add(new ImageFileInfo
                    {
                        FullPath = file,
                        Year = match.Groups[1].Value,
                        Month = match.Groups[2].Value,
                        Day = match.Groups[3].Value,
                        Hour = match.Groups[4].Value,
                        Minute = match.Groups[5].Value,
                        Second = match.Groups[6].Value,
                        CamId = int.Parse(match.Groups[7].Value)
                    });
                }
            }
        }

        // 下拉選單資料來源
        public List<string> GetYears() => _allFiles.Select(x => x.Year).Distinct().OrderBy(x => x).ToList();
        public List<string> GetMonths(string year) => _allFiles.Where(x => x.Year == year).Select(x => x.Month).Distinct().OrderBy(x => x).ToList();
        public List<string> GetDays(string year, string month) => _allFiles.Where(x => x.Year == year && x.Month == month).Select(x => x.Day).Distinct().OrderBy(x => x).ToList();
        public List<string> GetHours(string year, string month, string day) => _allFiles.Where(x => x.Year == year && x.Month == month && x.Day == day).Select(x => x.Hour).Distinct().OrderBy(x => x).ToList();
        public List<string> GetMinutes(string year, string month, string day, string hour) => _allFiles.Where(x => x.Year == year && x.Month == month && x.Day == day && x.Hour == hour).Select(x => x.Minute).Distinct().OrderBy(x => x).ToList();
        public List<string> GetSeconds(string year, string month, string day, string hour, string min) => _allFiles.Where(x => x.Year == year && x.Month == month && x.Day == day && x.Hour == hour && x.Minute == min).Select(x => x.Second).Distinct().OrderBy(x => x).ToList();

        public Dictionary<int, string> GetImages(string year, string month, string day, string hour, string minute, string second)
        {
            var target = _allFiles.Where(x =>
                x.Year == year && x.Month == month && x.Day == day &&
                x.Hour == hour && x.Minute == minute && x.Second == second);
            return target.ToDictionary(x => x.CamId, x => x.FullPath);
        }
    }
}