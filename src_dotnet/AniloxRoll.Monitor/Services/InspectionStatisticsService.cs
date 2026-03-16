using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace AniloxRoll.Monitor.Core.Services
{
    public class CameraStats
    {
        public int   CamId    { get; set; }
        public int   Pass     { get; set; }
        public int   Fail     { get; set; }
        public int   Total    => Pass + Fail;
        public float PassRate => Total == 0 ? 0f : (float)Pass / Total;
    }

    /// <summary>
    /// 從每日 inspection-log CSV 讀取資料，計算各相機的 Pass/Fail 統計。
    /// CSV 格式：Id,FileName,MaxExceed,MeanExceed
    /// Pass 定義：MaxExceed==0 AND MeanExceed==0
    /// </summary>
    public static class InspectionStatisticsService
    {
        /// <summary>
        /// 計算指定時間區間內所有相機的統計數據。
        /// 邊讀邊累加，不一次載入記憶體。
        /// </summary>
        public static Dictionary<int, CameraStats> Compute(
            string   captureRootPath,
            DateTime start,
            DateTime end)
        {
            var stats = new Dictionary<int, CameraStats>();
            for (int i = 1; i <= 7; i++)
                stats[i] = new CameraStats { CamId = i };

            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return stats;

            foreach (string csvPath in EnumerateCsvFiles(captureRootPath, start, end))
            {
                try
                {
                    using (var sr = new StreamReader(csvPath))
                    {
                        string header = sr.ReadLine(); // skip header
                        if (header == null) continue;

                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (!TryParseLine(line, out string fileName,
                                out int maxExceed, out int meanExceed))
                                continue;

                            if (!TryExtractCamId(fileName, out int camId)) continue;
                            if (!stats.TryGetValue(camId, out var s)) continue;

                            if (maxExceed == 0 && meanExceed == 0) s.Pass++;
                            else                                    s.Fail++;
                        }
                    }
                }
                catch { /* 單檔讀取失敗，跳過繼續 */ }
            }

            return stats;
        }

        // ── 私有輔助 ──────────────────────────────────────────────────────

        private static IEnumerable<string> EnumerateCsvFiles(
            string root, DateTime start, DateTime end)
        {
            // 每日 CSV 路徑：{root}\{YYYY}\{YYYYMM}\inspection-log-{YYYYMMDD}.csv
            DateTime cursor = start.Date;
            while (cursor <= end.Date)
            {
                string path = Path.Combine(
                    root,
                    cursor.Year.ToString(CultureInfo.InvariantCulture),
                    cursor.ToString("yyyyMM"),
                    $"inspection-log-{cursor:yyyyMMdd}.csv");

                if (File.Exists(path))
                    yield return path;

                cursor = cursor.AddDays(1);
            }
        }

        private static bool TryParseLine(string line,
            out string fileName, out int maxExceed, out int meanExceed)
        {
            fileName   = null;
            maxExceed  = 0;
            meanExceed = 0;

            if (string.IsNullOrWhiteSpace(line)) return false;
            string[] cols = line.Split(',');
            if (cols.Length < 4) return false;

            fileName = cols[1].Trim();
            return int.TryParse(cols[2].Trim(), out maxExceed) &&
                   int.TryParse(cols[3].Trim(), out meanExceed);
        }

        /// <summary>從 FileName（e.g. "20260316_102301-3"）提取相機 ID。</summary>
        private static bool TryExtractCamId(string fileName, out int camId)
        {
            camId = 0;
            if (string.IsNullOrEmpty(fileName)) return false;
            int dashIdx = fileName.LastIndexOf('-');
            if (dashIdx < 0 || dashIdx >= fileName.Length - 1) return false;
            return int.TryParse(fileName.Substring(dashIdx + 1), out camId);
        }
    }
}
