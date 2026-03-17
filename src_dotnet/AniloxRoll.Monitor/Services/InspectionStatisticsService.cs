using System;
using System.Collections.Generic;
using System.Diagnostics;
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
    /// 單一序號對 CAM1~7 的 Pass/Fail 結果。
    /// CamResult[i]（i=0~6 對應 CAM1~7）：null=無資料, false=Pass, true=Fail
    /// </summary>
    public class GrabDetail
    {
        public string  GrabId    { get; set; }
        public int     GrabNum   { get; set; }
        public bool?[] CamResult { get; } = new bool?[7];
    }

    /// <summary>每個序號（grabId）的時間範圍資訊。</summary>
    public class GrabIdInfo
    {
        public string   GrabId   { get; set; }
        public int      GrabNum  { get; set; }   // 從 "A00008" 提取的數字
        public DateTime Earliest { get; set; }
        public DateTime Latest   { get; set; }
    }

    /// <summary>
    /// 從每日 CSV（{YYYYMMDD}.csv）讀取資料，計算各相機的 Pass/Fail 統計。
    /// CSV 格式：Id,FileName,MaxExceed,MeanExceed
    /// </summary>
    public static class InspectionStatisticsService
    {
        // ── 時間範圍統計（舊模式：以張數為分母）────────────────────────────

        /// <summary>
        /// 遞迴掃描所有 CSV，只統計 FileName 時間戳落在 [start, end] 的紀錄。
        /// 分母 = 照片張數；每筆獨立判斷 Pass/Fail。
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
                        string header = sr.ReadLine();
                        if (header == null) continue;

                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (!TryParseLine(line, out _, out string fileName,
                                out int maxExceed, out int meanExceed)) continue;

                            if (!TryParseFileNameDateTime(fileName, out DateTime ts)) continue;
                            if (ts < start || ts > end) continue;

                            if (!TryExtractCamId(fileName, out int camId)) continue;
                            if (!stats.TryGetValue(camId, out var s)) continue;

                            if (maxExceed == 0 && meanExceed == 0) s.Pass++;
                            else                                    s.Fail++;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine($"[InspectionStatisticsService.Compute] {csvPath}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            return stats;
        }

        // ── 序號範圍統計（新模式：以唯一序號為分母）────────────────────────

        /// <summary>
        /// 遞迴掃描所有 CSV，統計序號落在 [startGrabNum, endGrabNum] 的紀錄。
        /// 分母 = 唯一序號數；同一序號同一相機只要任一張超標即為 Fail。
        /// </summary>
        public static Dictionary<int, CameraStats> ComputeByGrabIdRange(
            string captureRootPath,
            int    startGrabNum,
            int    endGrabNum)
        {
            var stats = new Dictionary<int, CameraStats>();
            for (int i = 1; i <= 7; i++)
                stats[i] = new CameraStats { CamId = i };

            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return stats;

            // grabNum → camId → hasFail
            var grabCamFail = new Dictionary<int, Dictionary<int, bool>>();

            foreach (string csvPath in Directory.GetFiles(captureRootPath, "*.csv", SearchOption.AllDirectories))
            {
                try
                {
                    using (var sr = new StreamReader(csvPath))
                    {
                        string header = sr.ReadLine();
                        if (header == null) continue;

                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (!TryParseLine(line, out string grabId, out string fileName,
                                out int maxExceed, out int meanExceed)) continue;

                            int grabNum = ParseGrabIdNum(grabId);
                            if (grabNum < startGrabNum || grabNum > endGrabNum) continue;

                            if (!TryExtractCamId(fileName, out int camId)) continue;

                            if (!grabCamFail.TryGetValue(grabNum, out var camMap))
                                grabCamFail[grabNum] = camMap = new Dictionary<int, bool>();

                            bool thisFail = maxExceed > 0 || meanExceed > 0;
                            if (!camMap.TryGetValue(camId, out bool prev))
                                camMap[camId] = thisFail;
                            else if (thisFail)
                                camMap[camId] = true; // 一票否決
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine($"[InspectionStatisticsService.ComputeByGrabIdRange] {csvPath}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            // 彙總：每個 (grabNum, camId) 算一次 Pass 或 Fail
            foreach (var grabKv in grabCamFail)
            {
                foreach (var camKv in grabKv.Value)
                {
                    if (!stats.TryGetValue(camKv.Key, out var s)) continue;
                    if (camKv.Value) s.Fail++;
                    else             s.Pass++;
                }
            }

            return stats;
        }

        // ── 逐序號詳細結果 ───────────────────────────────────────────────

        /// <summary>
        /// 遞迴掃描所有 CSV，回傳 [startGrabNum, endGrabNum] 範圍內每個序號
        /// 對 CAM1~7 的 Pass/Fail 結果，依序號排序。
        /// 同一序號同一相機任一張超標即為 Fail（一票否決）。
        /// </summary>
        public static List<GrabDetail> ComputeDetailedByGrabIdRange(
            string captureRootPath,
            int    startGrabNum,
            int    endGrabNum)
        {
            // grabNum → GrabDetail
            var dict = new SortedDictionary<int, GrabDetail>();

            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return new List<GrabDetail>();

            foreach (string csvPath in Directory.GetFiles(captureRootPath, "*.csv", SearchOption.AllDirectories))
            {
                try
                {
                    using (var sr = new StreamReader(csvPath))
                    {
                        string header = sr.ReadLine();
                        if (header == null) continue;

                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (!TryParseLine(line, out string grabId, out string fileName,
                                out int maxExceed, out int meanExceed)) continue;

                            int grabNum = ParseGrabIdNum(grabId);
                            if (grabNum < startGrabNum || grabNum > endGrabNum) continue;

                            if (!TryExtractCamId(fileName, out int camId)) continue;
                            if (camId < 1 || camId > 7) continue;

                            if (!dict.TryGetValue(grabNum, out var detail))
                            {
                                detail = new GrabDetail { GrabId = grabId, GrabNum = grabNum };
                                dict[grabNum] = detail;
                            }

                            int idx = camId - 1;
                            bool thisFail = maxExceed > 0 || meanExceed > 0;
                            if (detail.CamResult[idx] == null)
                                detail.CamResult[idx] = thisFail;
                            else if (thisFail)
                                detail.CamResult[idx] = true; // 一票否決
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine($"[InspectionStatisticsService.ComputeDetailedByGrabIdRange] {csvPath}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            return new List<GrabDetail>(dict.Values);
        }

        // ── 載入輔助資料 ─────────────────────────────────────────────────

        /// <summary>
        /// 遞迴掃描所有 CSV，回傳每個序號的最早/最晚時間，依序號排序。
        /// </summary>
        public static List<GrabIdInfo> LoadGrabIdInfos(string captureRootPath)
        {
            // grabNum → (grabId string, earliest, latest)
            var dict = new SortedDictionary<int, (string grabId, DateTime earliest, DateTime latest)>();

            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return new List<GrabIdInfo>();

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
                            if (!TryParseLine(line, out string grabId, out string fileName, out _, out _)) continue;
                            if (!TryParseFileNameDateTime(fileName, out DateTime dt)) continue;

                            int grabNum = ParseGrabIdNum(grabId);
                            if (grabNum < 0) continue;

                            if (dict.TryGetValue(grabNum, out var existing))
                            {
                                dict[grabNum] = (existing.grabId,
                                    dt < existing.earliest ? dt : existing.earliest,
                                    dt > existing.latest   ? dt : existing.latest);
                            }
                            else
                            {
                                dict[grabNum] = (grabId, dt, dt);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine($"[InspectionStatisticsService.LoadGrabIdInfos] {csvPath}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            var result = new List<GrabIdInfo>(dict.Count);
            foreach (var kv in dict)
            {
                result.Add(new GrabIdInfo
                {
                    GrabId   = kv.Value.grabId,
                    GrabNum  = kv.Key,
                    Earliest = kv.Value.earliest,
                    Latest   = kv.Value.latest
                });
            }
            return result;
        }

        /// <summary>
        /// 遞迴掃描所有 CSV，回傳所有不重複的精確時間（秒）排序集合。
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
                        sr.ReadLine();
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (!TryParseLine(line, out _, out string fileName, out _, out _)) continue;
                            if (!TryParseFileNameDateTime(fileName, out DateTime dt)) continue;
                            times.Add(dt);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine($"[InspectionStatisticsService.LoadAvailableTimes] {csvPath}: {ex.GetType().Name}: {ex.Message}");
                }
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

        // ── 私有輔助 ──────────────────────────────────────────────────────

        /// <summary>從序號字串（e.g. "A00008"）提取數字部分。</summary>
        internal static int ParseGrabIdNum(string grabId)
        {
            if (string.IsNullOrEmpty(grabId) || grabId.Length < 2) return -1;
            return int.TryParse(grabId.Substring(1), out int n) ? n : -1;
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
            string datePart = fileName.Substring(0, 8);
            string timePart = fileName.Substring(9, 6);
            return DateTime.TryParseExact(datePart + timePart, "yyyyMMddHHmmss",
                CultureInfo.InvariantCulture, DateTimeStyles.None, out result);
        }

        /// <summary>解析一行 CSV：Id,FileName,MaxExceed,MeanExceed</summary>
        private static bool TryParseLine(string line,
            out string grabId, out string fileName,
            out int maxExceed, out int meanExceed)
        {
            grabId     = null;
            fileName   = null;
            maxExceed  = 0;
            meanExceed = 0;

            if (string.IsNullOrWhiteSpace(line)) return false;
            string[] cols = line.Split(',');
            if (cols.Length < 4) return false;

            grabId   = cols[0].Trim();
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
