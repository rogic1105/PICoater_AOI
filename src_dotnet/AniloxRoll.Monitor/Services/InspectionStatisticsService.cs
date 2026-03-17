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
        /// 遞迴掃描 captureRootPath 下所有 CSV，
        /// 只統計 FileName 時間戳落在 [start, end] 範圍內的紀錄。
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

            foreach (string csvPath in Directory.GetFiles(captureRootPath, "*.csv", SearchOption.AllDirectories))
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

                            // 過濾時間範圍（精確到秒）
                            if (!TryParseFileNameDateTime(fileName, out DateTime ts)) continue;
                            if (ts < start || ts > end) continue;

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

        /// <summary>
        /// 遞迴掃描 captureRootPath 下所有 CSV，
        /// 解析每筆 FileName（YYYYMMDD_HHMMSS-camId），
        /// 回傳所有不重複的精確時間（秒）排序集合。
        /// </summary>
        public static SortedSet<DateTime> LoadAvailableTimes(string captureRootPath)
        {
            var times = new SortedSet<DateTime>();
            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return times;

            foreach (string csvPath in Directory.GetFiles(captureRootPath, "*.csv", SearchOption.AllDirectories))
            {
                try
                {
                    using (var sr = new StreamReader(csvPath))
                    {
                        sr.ReadLine(); // skip header
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (!TryParseLine(line, out string fileName, out _, out _)) continue;
                            if (!TryParseFileNameDateTime(fileName, out DateTime dt)) continue;
                            times.Add(dt);
                        }
                    }
                }
                catch { }
            }

            return times;
        }

        /// <summary>取得資料夾內所有 CSV 紀錄的最早與最新時間。</summary>
        public static bool TryGetDateRange(
            string captureRootPath,
            out DateTime earliest,
            out DateTime latest)
        {
            var times = LoadAvailableTimes(captureRootPath);
            earliest = latest = DateTime.MinValue;
            if (times.Count == 0) return false;
            earliest = times.Min;
            latest   = times.Max;
            return true;
        }

        /// <summary>
        /// 從 FileName（e.g. "20260316_102301-3"）解析出完整 DateTime（精確到秒）。
        /// </summary>
        private static bool TryParseFileNameDateTime(string fileName, out DateTime result)
        {
            result = DateTime.MinValue;
            if (string.IsNullOrEmpty(fileName)) return false;
            int underscoreIdx = fileName.IndexOf('_');
            if (underscoreIdx != 8 || fileName.Length < 15) return false;
            string datePart = fileName.Substring(0, 8); // YYYYMMDD
            string timePart = fileName.Substring(9, 6); // HHMMSS
            return DateTime.TryParseExact(datePart + timePart, "yyyyMMddHHmmss",
                CultureInfo.InvariantCulture, DateTimeStyles.None, out result);
        }

        // ── 私有輔助 ──────────────────────────────────────────────────────

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
