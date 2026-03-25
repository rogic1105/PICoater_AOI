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

    /// <summary>按時間週期分組的 Pass/Fail 統計（年/月/日）。</summary>
    public class PeriodStats
    {
        public string Label { get; set; }
        public int    Pass  { get; set; }
        public int    Fail  { get; set; }
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

        /// <summary>降序版本（最新在前）。</summary>
        public static List<GrabIdInfo> LoadGrabIdInfosDescending(string captureRootPath)
        {
            var list = LoadGrabIdInfos(captureRootPath);
            list.Reverse();
            return list;
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

        // ── 時間週期分組統計（月份 / 日期 / 小時，固定完整軸）────────────────

        /// <summary>
        /// 按月份（1–12）彙總 Pass/Fail，固定回傳 12 筆，無資料月份為 0。
        /// </summary>
        public static List<PeriodStats> ComputeGroupedByMonthOfYear(
            string captureRootPath, DateTime start, DateTime end)
        {
            var counts = new (int Pass, int Fail)[13]; // index 1-12
            ScanCsvByDateRange(captureRootPath, start, end, (ts, isFail) =>
            {
                if (isFail) counts[ts.Month].Fail++;
                else        counts[ts.Month].Pass++;
            });
            var result = new List<PeriodStats>(12);
            for (int m = 1; m <= 12; m++)
                result.Add(new PeriodStats { Label = m.ToString(), Pass = counts[m].Pass, Fail = counts[m].Fail });
            return result;
        }

        /// <summary>
        /// 按日期（1–31）彙總 Pass/Fail，固定回傳 31 筆，無資料日期為 0。
        /// </summary>
        public static List<PeriodStats> ComputeGroupedByDayOfMonth(
            string captureRootPath, DateTime start, DateTime end)
        {
            var counts = new (int Pass, int Fail)[32]; // index 1-31
            ScanCsvByDateRange(captureRootPath, start, end, (ts, isFail) =>
            {
                if (isFail) counts[ts.Day].Fail++;
                else        counts[ts.Day].Pass++;
            });
            var result = new List<PeriodStats>(31);
            for (int d = 1; d <= 31; d++)
                result.Add(new PeriodStats { Label = d.ToString(), Pass = counts[d].Pass, Fail = counts[d].Fail });
            return result;
        }

        /// <summary>
        /// 按小時（0–23）彙總 Pass/Fail，固定回傳 24 筆，無資料小時為 0。
        /// </summary>
        public static List<PeriodStats> ComputeGroupedByHourOfDay(
            string captureRootPath, DateTime start, DateTime end)
        {
            var counts = new (int Pass, int Fail)[24]; // index 0-23
            ScanCsvByDateRange(captureRootPath, start, end, (ts, isFail) =>
            {
                if (isFail) counts[ts.Hour].Fail++;
                else        counts[ts.Hour].Pass++;
            });
            var result = new List<PeriodStats>(24);
            for (int h = 0; h < 24; h++)
                result.Add(new PeriodStats { Label = h.ToString(), Pass = counts[h].Pass, Fail = counts[h].Fail });
            return result;
        }

        /// <summary>
        /// 掃描所有 CSV，篩選落在 [start, end] 的紀錄，
        /// 以 (GrabId, CamId) 為單位分組：同一序號同一相機任一張超標即 Fail（一票否決），
        /// 每個 (GrabId, CamId) 呼叫一次 onRecord(timestamp, isFail)。
        /// </summary>
        private static void ScanCsvByDateRange(
            string captureRootPath, DateTime start, DateTime end,
            Action<DateTime, bool> onRecord)
        {
            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath)) return;

            // key = (grabId, camId) → (earliest timestamp, hasFail)
            var groups = new Dictionary<(string grabId, int camId), (DateTime ts, bool hasFail)>();

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
                            if (!TryParseFileNameDateTime(fileName, out DateTime ts)) continue;
                            if (ts < start || ts > end) continue;
                            if (!TryExtractCamId(fileName, out int camId)) continue;

                            var key = (grabId, camId);
                            bool thisFail = maxExceed > 0 || meanExceed > 0;

                            if (!groups.TryGetValue(key, out var prev))
                                groups[key] = (ts, thisFail);
                            else
                                groups[key] = (prev.ts < ts ? prev.ts : ts,
                                               prev.hasFail || thisFail);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine($"[InspectionStatisticsService.ScanCsvByDateRange] {csvPath}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            foreach (var kv in groups)
                onRecord(kv.Value.ts, kv.Value.hasFail);
        }

        // ── 私有輔助 ──────────────────────────────────────────────────────

        /// <summary>從序號字串（e.g. "A00008"）提取數字部分。</summary>
        internal static int ParseGrabIdNum(string grabId)
        {
            if (string.IsNullOrEmpty(grabId) || grabId.Length < 2) return -1;
            return int.TryParse(grabId.Substring(1), out int n) ? n : -1;
        }

        /// <summary>
        /// 從 FileName（e.g. "20260316_102301.123-3"）解析出完整 DateTime（精確到毫秒）。
        /// 格式：yyyyMMdd_HHmmss.fff-CamId
        /// </summary>
        private static bool TryParseFileNameDateTime(string fileName, out DateTime result)
        {
            result = DateTime.MinValue;
            if (string.IsNullOrEmpty(fileName)) return false;
            int underscoreIdx = fileName.IndexOf('_');
            if (underscoreIdx != 8 || fileName.Length < 19) return false;  // "yyyyMMdd_HHmmss.fff" = 19 chars
            string datePart = fileName.Substring(0, 8);      // "yyyyMMdd"
            string timePart = fileName.Substring(9, 10);     // "HHmmss.fff"
            return DateTime.TryParseExact(datePart + timePart, "yyyyMMddHHmmss.fff",
                CultureInfo.InvariantCulture, DateTimeStyles.None, out result);
        }

        /// <summary>解析一行 CSV 資料列（跳過 #CFG 等註解列）。
        /// 相容 4 欄舊格式與 9 欄新格式。</summary>
        private static bool TryParseLine(string line,
            out string grabId, out string fileName,
            out int maxExceed, out int meanExceed)
        {
            grabId     = null;
            fileName   = null;
            maxExceed  = 0;
            meanExceed = 0;

            if (string.IsNullOrWhiteSpace(line)) return false;
            if (line[0] == '#') return false; // #CFG 等註解列
            string[] cols = line.Split(',');
            if (cols.Length < 4) return false;

            grabId   = cols[0].Trim();
            fileName = cols[1].Trim();
            return int.TryParse(cols[2].Trim(), out maxExceed) &&
                   int.TryParse(cols[3].Trim(), out meanExceed);
        }

        /// <summary>解析 9 欄新格式，額外取得 MeanPeak/MaxPeak/GrabHeight/LineRateHz/ExposureUs。</summary>
        private static bool TryParseLineEx(string line,
            out string grabId, out string fileName,
            out int maxExceed, out int meanExceed,
            out float meanPeak, out float maxPeak,
            out int grabHeight, out double lineRateHz, out double exposureUs)
        {
            meanPeak = 0; maxPeak = 0; grabHeight = 0; lineRateHz = 0; exposureUs = 0;

            if (!TryParseLine(line, out grabId, out fileName, out maxExceed, out meanExceed))
                return false;

            string[] cols = line.Split(',');
            if (cols.Length >= 9)
            {
                float.TryParse(cols[4].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out meanPeak);
                float.TryParse(cols[5].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out maxPeak);
                int.TryParse(cols[6].Trim(), out grabHeight);
                double.TryParse(cols[7].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out lineRateHz);
                double.TryParse(cols[8].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out exposureUs);
            }
            return true;
        }

        /// <summary>從 FileName（e.g. "20260316_102301.123-3"）提取相機 ID。</summary>
        private static bool TryExtractCamId(string fileName, out int camId)
        {
            camId = 0;
            if (string.IsNullOrEmpty(fileName)) return false;
            int dashIdx = fileName.LastIndexOf('-');
            if (dashIdx < 0 || dashIdx >= fileName.Length - 1) return false;
            return int.TryParse(fileName.Substring(dashIdx + 1), out camId);
        }

        /// <summary>
        /// 掃描 CSV，找出指定序號的各相機所有影像路徑（一台相機可能有多張），
        /// 依檔名（時間戳）排序。優先回傳 _raw.jpg，其次 .bmp。
        /// 回傳 Dictionary&lt;camId, List&lt;sortedFilePaths&gt;&gt;。
        /// hintFrom/hintTo：已知的時間範圍，用於縮小 CSV 搜尋範圍（只掃相關日期）。
        /// </summary>
        public static Dictionary<int, List<string>> LoadImagePathsForGrabId(
            string captureRootPath, string grabId,
            DateTime hintFrom = default(DateTime), DateTime hintTo = default(DateTime))
        {
            // camId → unique set of fileNames (無副檔名)
            var camFileNames = new Dictionary<int, HashSet<string>>();

            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return new Dictionary<int, List<string>>();

            // 若有時間提示，只掃對應日期的 CSV（通常只有 1 個）；否則掃全部
            IEnumerable<string> csvPaths;
            if (hintFrom != default(DateTime) && hintTo != default(DateTime))
            {
                var dateCsvs = new List<string>();
                for (DateTime d = hintFrom.Date; d <= hintTo.Date; d = d.AddDays(1))
                {
                    string p = Path.Combine(captureRootPath,
                        d.ToString("yyyy"), d.ToString("yyyyMM"), d.ToString("yyyyMMdd") + ".csv");
                    if (File.Exists(p)) dateCsvs.Add(p);
                }
                csvPaths = dateCsvs;
            }
            else
            {
                csvPaths = Directory.GetFiles(captureRootPath, "*.csv", SearchOption.AllDirectories);
            }

            foreach (string csvPath in csvPaths)
            {
                try
                {
                    using (var sr = new StreamReader(csvPath))
                    {
                        sr.ReadLine(); // skip header
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (!TryParseLine(line, out string id, out string fileName, out _, out _)) continue;
                            if (id != grabId) continue;
                            if (!TryExtractCamId(fileName, out int camId)) continue;
                            if (!camFileNames.ContainsKey(camId))
                                camFileNames[camId] = new HashSet<string>();
                            camFileNames[camId].Add(fileName);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine($"[InspectionStatisticsService.LoadImagePathsForGrabId] {csvPath}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            // 將 fileName → 完整檔案路徑（目錄結構：root\yyyy\yyyyMM\yyyyMMdd\），依時間排序
            var result = new Dictionary<int, List<string>>();
            foreach (var kv in camFileNames)
            {
                var sortedNames = new List<string>(kv.Value);
                sortedNames.Sort(); // "YYYYMMDD_HHMMSS.fff-n" 字典序 = 時間序

                var paths = new List<string>();
                foreach (string fn in sortedNames)
                {
                    if (fn.Length < 8) continue;
                    string dateStr = fn.Substring(0, 8);
                    string dir = Path.Combine(captureRootPath,
                        dateStr.Substring(0, 4),
                        dateStr.Substring(0, 6),
                        dateStr.Substring(0, 8));

                    string rawJpg = Path.Combine(dir, fn + "_raw.jpg");
                    if (File.Exists(rawJpg)) { paths.Add(rawJpg); continue; }

                    string bmp = Path.Combine(dir, fn + ".bmp");
                    if (File.Exists(bmp)) paths.Add(bmp);
                }
                if (paths.Count > 0) result[kv.Key] = paths;
            }

            return result;
        }

        /// <summary>
        /// 從 CSV 中找出指定 grabId 對應的 #CFG（該 grabId 上方最近的 #CFG 列）。
        /// 回傳 null 表示無 #CFG（舊格式 CSV）。
        /// </summary>
        public static CsvConfigSnapshot LoadConfigForGrabId(
            string captureRootPath, string grabId,
            DateTime hintFrom = default(DateTime), DateTime hintTo = default(DateTime))
        {
            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return null;

            IEnumerable<string> csvPaths;
            if (hintFrom != default(DateTime) && hintTo != default(DateTime))
            {
                var dateCsvs = new List<string>();
                for (DateTime d = hintFrom.Date; d <= hintTo.Date; d = d.AddDays(1))
                {
                    string p = Path.Combine(captureRootPath,
                        d.ToString("yyyy"), d.ToString("yyyyMM"), d.ToString("yyyyMMdd") + ".csv");
                    if (File.Exists(p)) dateCsvs.Add(p);
                }
                csvPaths = dateCsvs;
            }
            else
            {
                csvPaths = Directory.GetFiles(captureRootPath, "*.csv", SearchOption.AllDirectories);
            }

            foreach (string csvPath in csvPaths)
            {
                try
                {
                    CsvConfigSnapshot lastCfg = null;
                    using (var sr = new StreamReader(csvPath))
                    {
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (line.StartsWith("#CFG,"))
                            {
                                if (CsvConfigSnapshot.TryParse(line, out var cfg))
                                    lastCfg = cfg;
                                continue;
                            }
                            if (!TryParseLine(line, out string id, out _, out _, out _)) continue;
                            if (id == grabId && lastCfg != null) return lastCfg;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine(
                        $"[InspectionStatisticsService.LoadConfigForGrabId] {csvPath}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            return null;
        }
    }
}
