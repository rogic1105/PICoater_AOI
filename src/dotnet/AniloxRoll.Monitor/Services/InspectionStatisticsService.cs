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
        public bool?[] CamResult { get; } = new bool?[7];
    }

    /// <summary>每個序號（grabId）的時間範圍資訊。</summary>
    public class GrabIdInfo
    {
        public string   GrabId   { get; set; }
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
    /// View-time Pass/Fail 重算 context：以「當前 Settings」的閾值 + 正規值，
    /// 對 CSV 內 raw peak（capture-time 已 baked by HM_V_capture）重算判定，
    /// 而非沿用 CSV 內 MaxExceed/MeanExceed（capture-time 寫死）。
    /// 公式：display_peak = raw_peak × (HM_V_capture / HM_V_current)，
    ///       isFail = display_peak > current_threshold。
    /// 傳 null → 沿用 CSV 內 MaxExceed/MeanExceed（legacy capture-time 判定）。
    /// </summary>
    public class ThresholdContext
    {
        public float CurrentHmV     { get; }
        public float CurrentErrMean { get; }
        public float CurrentErrMax  { get; }

        public ThresholdContext(float currentHmV, float currentErrMean, float currentErrMax)
        {
            CurrentHmV     = currentHmV;
            CurrentErrMean = currentErrMean;
            CurrentErrMax  = currentErrMax;
        }

        public bool IsFail(float meanPeak, float maxPeak, float captureHmV)
        {
            float ratio = (captureHmV > 0f && CurrentHmV > 0f) ? captureHmV / CurrentHmV : 1f;
            float displayMean = meanPeak * ratio;
            float displayMax  = maxPeak  * ratio;
            return displayMean > CurrentErrMean || displayMax > CurrentErrMax;
        }
    }

    /// <summary>
    /// 從每日 CSV（{YYYYMMDD}.csv）讀取資料，計算各相機的 Pass/Fail 統計。
    /// CSV 格式：Id,FileName,MaxExceed,MeanExceed
    /// </summary>
    public static class InspectionStatisticsService
    {
        /// <summary>
        /// 共用的 CSV 讀取 helper — 用 FileShare.ReadWrite 開檔，避免與 InspectionLogService
        /// 寫入競爭（特別是 Storage PC 讀 / Inspection PC 寫的跨 process 情境）。
        /// </summary>
        private static StreamReader OpenCsvShared(string path)
        {
            var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            return new StreamReader(fs);
        }

        // ── 時間範圍統計（舊模式：以張數為分母）────────────────────────────

        /// <summary>
        /// 遞迴掃描所有 CSV，只統計 FileName 時間戳落在 [start, end] 的紀錄。
        /// 分母 = 照片張數；每筆獨立判斷 Pass/Fail。
        /// </summary>
        public static Dictionary<int, CameraStats> Compute(
            string   captureRootPath,
            DateTime start,
            DateTime end,
            ThresholdContext ctx = null)
        {
            var stats = new Dictionary<int, CameraStats>();
            for (int i = 1; i <= 7; i++)
                stats[i] = new CameraStats { CamId = i };

            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return stats;

            // M5: 跨 CSV 檔保留 captureHmV（按日期排序）
            float captureHmV = ctx?.CurrentHmV ?? 0f;
            var csvFiles = Directory.GetFiles(captureRootPath, "*.csv", SearchOption.AllDirectories);
            Array.Sort(csvFiles, StringComparer.Ordinal);
            foreach (string csvPath in csvFiles)
            {
                try
                {
                    using (var sr = OpenCsvShared(csvPath))
                    {
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (TryUpdateHmFromCfg(line, ref captureHmV)) continue;
                            if (!TryParseLineEx(line, out _, out string fileName,
                                out int maxExceed, out int meanExceed,
                                out float meanPeak, out float maxPeak,
                                out _, out _, out _)) continue;

                            if (!TryParseFileNameDateTime(fileName, out DateTime ts)) continue;
                            if (ts < start || ts > end) continue;

                            if (!TryExtractCamId(fileName, out int camId)) continue;
                            if (!stats.TryGetValue(camId, out var s)) continue;

                            bool isFail = ctx != null
                                ? ctx.IsFail(meanPeak, maxPeak, captureHmV)
                                : (maxExceed > 0 || meanExceed > 0);
                            if (isFail) s.Fail++;
                            else        s.Pass++;
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

        /// <summary>
        /// 偵測 #CFG 列並更新 captureHmV。回傳 true 表示這行是 #CFG（呼叫端應 continue 不當資料列處理）。
        /// </summary>
        private static bool TryUpdateHmFromCfg(string line, ref float captureHmV)
        {
            if (string.IsNullOrEmpty(line) || !line.StartsWith("#CFG,")) return false;
            if (CsvConfigSnapshot.TryParse(line, out var cfg) && cfg.HessianMaxFactorV > 0f)
                captureHmV = cfg.HessianMaxFactorV;
            return true;
        }

        // ── 序號範圍統計（新模式：以唯一序號為分母）────────────────────────

        /// <summary>
        /// 遞迴掃描所有 CSV，統計序號落在 [startGrabId, endGrabId]（字串比較）的紀錄。
        /// 分母 = 唯一序號數；同一序號同一相機只要任一張超標即為 Fail。
        /// </summary>
        public static Dictionary<int, CameraStats> ComputeByGrabIdRange(
            string captureRootPath,
            string startGrabId,
            string endGrabId,
            ThresholdContext ctx = null)
        {
            var stats = new Dictionary<int, CameraStats>();
            for (int i = 1; i <= 7; i++)
                stats[i] = new CameraStats { CamId = i };

            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return stats;

            string lo = StringComparer.Ordinal.Compare(startGrabId, endGrabId) <= 0 ? startGrabId : endGrabId;
            string hi = lo == startGrabId ? endGrabId : startGrabId;

            // grabId → camId → hasFail
            var grabCamFail = new Dictionary<string, Dictionary<int, bool>>(StringComparer.Ordinal);

            // M5: 跨 CSV 檔保留 captureHmV（按日期排序）
            float captureHmV = ctx?.CurrentHmV ?? 0f;
            var csvFiles = Directory.GetFiles(captureRootPath, "*.csv", SearchOption.AllDirectories);
            Array.Sort(csvFiles, StringComparer.Ordinal);
            foreach (string csvPath in csvFiles)
            {
                try
                {
                    using (var sr = OpenCsvShared(csvPath))
                    {
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (TryUpdateHmFromCfg(line, ref captureHmV)) continue;
                            if (!TryParseLineEx(line, out string grabId, out string fileName,
                                out int maxExceed, out int meanExceed,
                                out float meanPeak, out float maxPeak,
                                out _, out _, out _)) continue;

                            if (StringComparer.Ordinal.Compare(grabId, lo) < 0 ||
                                StringComparer.Ordinal.Compare(grabId, hi) > 0) continue;

                            if (!TryExtractCamId(fileName, out int camId)) continue;

                            if (!grabCamFail.TryGetValue(grabId, out var camMap))
                                grabCamFail[grabId] = camMap = new Dictionary<int, bool>();

                            bool thisFail = ctx != null
                                ? ctx.IsFail(meanPeak, maxPeak, captureHmV)
                                : (maxExceed > 0 || meanExceed > 0);
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

            // 彙總：每個 (grabId, camId) 算一次 Pass 或 Fail
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

        // ── Mura 空間分布曲線（.bin 抽樣平均）──────────────────────────────

        /// <summary>
        /// 從指定 grabIdInfos 讀取每台相機的 _mean_v.bin / _max_v.bin，按位置平均後返回。
        /// grabIds 由呼叫方依三種模式完成抽樣（≤50 筆）。
        /// 值域：0-255（raw，與 ColumnCurveChartHelper 一致）。
        /// </summary>
        public static (Dictionary<int, float[]> Mean, Dictionary<int, float[]> Max)
            LoadAvgMuraProfile(string rootPath, IList<GrabIdInfo> grabIds)
        {
            var accMean  = new Dictionary<int, float[]>();
            var accMax   = new Dictionary<int, float[]>();
            var counts   = new Dictionary<int, int>();

            foreach (var info in grabIds)
            {
                string dateDir = CaptureStoragePaths.DateImageDir(rootPath, info.Earliest);
                if (!Directory.Exists(dateDir)) continue;

                string prefix = info.Earliest.ToString("yyyyMMdd_HHmmss");

                for (int camId = 1; camId <= 7; camId++)
                {
                    string[] mFiles = Directory.GetFiles(dateDir, $"{prefix}*-{camId}{CaptureFileNaming.MeanV}");
                    string[] xFiles = Directory.GetFiles(dateDir, $"{prefix}*-{camId}{CaptureFileNaming.MaxV}");
                    if (mFiles.Length == 0) continue;

                    float[] mean = TryLoadBin(mFiles[0]);
                    if (mean == null || mean.Length == 0) continue;
                    float[] max  = xFiles.Length > 0 ? TryLoadBin(xFiles[0]) : null;

                    if (!accMean.TryGetValue(camId, out float[] am))
                    {
                        accMean[camId] = new float[mean.Length];
                        accMax[camId]  = new float[mean.Length];
                        counts[camId]  = 0;
                    }
                    else if (am.Length != mean.Length)
                        continue;

                    float[] sumM = accMean[camId];
                    float[] sumX = accMax[camId];
                    for (int j = 0; j < mean.Length; j++)
                    {
                        sumM[j] += mean[j];
                        if (max != null && j < max.Length && max[j] > sumX[j]) sumX[j] = max[j];
                    }
                    counts[camId]++;
                }
            }

            var resultMean = new Dictionary<int, float[]>();
            var resultMax  = new Dictionary<int, float[]>();
            foreach (var kvp in accMean)
            {
                int n = counts[kvp.Key];
                if (n == 0) continue;
                float[] avgM = new float[kvp.Value.Length];
                for (int j = 0; j < avgM.Length; j++)
                    avgM[j] = kvp.Value[j] / n;
                resultMean[kvp.Key] = avgM;
                resultMax[kvp.Key]  = accMax[kvp.Key];   // max 取各幀最大值（非平均）
            }
            return (resultMean, resultMax);
        }

        private static float[] TryLoadBin(string path)
        {
            // MCBF: magic(4) + version(4) + scale_factor(4f) + [v2: light(4)+exposure(4f)] + length(4) + float[]
            // scale_factor 僅作 metadata，不參與讀值（與 InspectionEngine.LoadCurveBin 相同）
            try
            {
                using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
                using (var br = new BinaryReader(fs))
                {
                    if (fs.Length < 16) return null;
                    var magic = br.ReadBytes(4);
                    if (magic[0] != 'M' || magic[1] != 'C' || magic[2] != 'B' || magic[3] != 'F') return null;
                    int version = br.ReadInt32();
                    br.ReadSingle();  // scale_factor (ignored)
                    if (version >= 2) { br.ReadInt32(); br.ReadSingle(); }  // lightLevel + exposureUs
                    int length = br.ReadInt32();
                    if (length <= 0 || length > 200000) return null;
                    var arr = new float[length];
                    for (int i = 0; i < length; i++)
                        arr[i] = br.ReadSingle();
                    return arr;
                }
            }
            catch { return null; }
        }

        // ── 逐序號詳細結果 ───────────────────────────────────────────────

        /// <summary>
        /// 遞迴掃描所有 CSV，回傳 [startGrabId, endGrabId]（字串比較）範圍內每個序號
        /// 對 CAM1~7 的 Pass/Fail 結果，依序號排序。
        /// 同一序號同一相機任一張超標即為 Fail（一票否決）。
        /// </summary>
        public static List<GrabDetail> ComputeDetailedByGrabIdRange(
            string captureRootPath,
            string startGrabId,
            string endGrabId,
            ThresholdContext ctx = null)
        {
            // grabId → GrabDetail（字串排序 = 時間排序）
            var dict = new SortedDictionary<string, GrabDetail>(StringComparer.Ordinal);

            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return new List<GrabDetail>();

            string lo = StringComparer.Ordinal.Compare(startGrabId, endGrabId) <= 0 ? startGrabId : endGrabId;
            string hi = lo == startGrabId ? endGrabId : startGrabId;

            // M5: 跨 CSV 檔保留 captureHmV（按日期排序）
            float captureHmV = ctx?.CurrentHmV ?? 0f;
            var csvFiles = Directory.GetFiles(captureRootPath, "*.csv", SearchOption.AllDirectories);
            Array.Sort(csvFiles, StringComparer.Ordinal);
            foreach (string csvPath in csvFiles)
            {
                try
                {
                    using (var sr = OpenCsvShared(csvPath))
                    {
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (TryUpdateHmFromCfg(line, ref captureHmV)) continue;
                            if (!TryParseLineEx(line, out string grabId, out string fileName,
                                out int maxExceed, out int meanExceed,
                                out float meanPeak, out float maxPeak,
                                out _, out _, out _)) continue;

                            if (StringComparer.Ordinal.Compare(grabId, lo) < 0 ||
                                StringComparer.Ordinal.Compare(grabId, hi) > 0) continue;

                            if (!TryExtractCamId(fileName, out int camId)) continue;
                            if (camId < 1 || camId > 7) continue;

                            if (!dict.TryGetValue(grabId, out var detail))
                            {
                                detail = new GrabDetail { GrabId = grabId };
                                dict[grabId] = detail;
                            }

                            int idx = camId - 1;
                            bool thisFail = ctx != null
                                ? ctx.IsFail(meanPeak, maxPeak, captureHmV)
                                : (maxExceed > 0 || meanExceed > 0);
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
        /// 遞迴掃描所有 CSV，回傳每個序號的最早/最晚時間，依 GrabId 字串排序（= 時間排序）。
        /// </summary>
        public static List<GrabIdInfo> LoadGrabIdInfos(string captureRootPath)
        {
            // grabId → (earliest, latest)
            var dict = new SortedDictionary<string, (DateTime earliest, DateTime latest)>(StringComparer.Ordinal);

            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return new List<GrabIdInfo>();

            foreach (string csvPath in Directory.GetFiles(captureRootPath, "*.csv", SearchOption.AllDirectories))
            {
                try
                {
                    using (var sr = OpenCsvShared(csvPath))
                    {
                        sr.ReadLine(); // skip header
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (!TryParseLine(line, out string grabId, out string fileName, out _, out _)) continue;
                            if (string.IsNullOrEmpty(grabId)) continue;
                            if (!TryParseFileNameDateTime(fileName, out DateTime dt)) continue;

                            if (dict.TryGetValue(grabId, out var existing))
                            {
                                dict[grabId] = (
                                    dt < existing.earliest ? dt : existing.earliest,
                                    dt > existing.latest   ? dt : existing.latest);
                            }
                            else
                            {
                                dict[grabId] = (dt, dt);
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
                    GrabId   = kv.Key,
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
                    using (var sr = OpenCsvShared(csvPath))
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


        // ── 時間週期分組統計（月份 / 日期 / 小時，固定完整軸）────────────────

        /// <summary>
        /// 按月份（1–12）彙總 Pass/Fail，固定回傳 12 筆，無資料月份為 0。
        /// </summary>
        public static List<PeriodStats> ComputeGroupedByMonthOfYear(
            string captureRootPath, DateTime start, DateTime end,
            ThresholdContext ctx = null)
        {
            var counts = new (int Pass, int Fail)[13]; // index 1-12
            ScanCsvByDateRange(captureRootPath, start, end, (ts, isFail) =>
            {
                if (isFail) counts[ts.Month].Fail++;
                else        counts[ts.Month].Pass++;
            }, ctx);
            var result = new List<PeriodStats>(12);
            for (int m = 1; m <= 12; m++)
                result.Add(new PeriodStats { Label = m.ToString(), Pass = counts[m].Pass, Fail = counts[m].Fail });
            return result;
        }

        /// <summary>
        /// 按日期（1–31）彙總 Pass/Fail，固定回傳 31 筆，無資料日期為 0。
        /// </summary>
        public static List<PeriodStats> ComputeGroupedByDayOfMonth(
            string captureRootPath, DateTime start, DateTime end,
            ThresholdContext ctx = null)
        {
            var counts = new (int Pass, int Fail)[32]; // index 1-31
            ScanCsvByDateRange(captureRootPath, start, end, (ts, isFail) =>
            {
                if (isFail) counts[ts.Day].Fail++;
                else        counts[ts.Day].Pass++;
            }, ctx);
            var result = new List<PeriodStats>(31);
            for (int d = 1; d <= 31; d++)
                result.Add(new PeriodStats { Label = d.ToString(), Pass = counts[d].Pass, Fail = counts[d].Fail });
            return result;
        }

        /// <summary>
        /// 按小時（0–23）彙總 Pass/Fail，固定回傳 24 筆，無資料小時為 0。
        /// </summary>
        public static List<PeriodStats> ComputeGroupedByHourOfDay(
            string captureRootPath, DateTime start, DateTime end,
            ThresholdContext ctx = null)
        {
            var counts = new (int Pass, int Fail)[24]; // index 0-23
            ScanCsvByDateRange(captureRootPath, start, end, (ts, isFail) =>
            {
                if (isFail) counts[ts.Hour].Fail++;
                else        counts[ts.Hour].Pass++;
            }, ctx);
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
            Action<DateTime, bool> onRecord,
            ThresholdContext ctx = null)
        {
            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath)) return;

            // key = (grabId, camId) → (earliest timestamp, hasFail)
            var groups = new Dictionary<(string grabId, int camId), (DateTime ts, bool hasFail)>();

            // M5: 跨 CSV 檔保留 captureHmV — 若 day1.csv 結尾沒 #CFG、day2.csv 開頭也沒 #CFG，
            // 沿用 day1 最後一筆。CSV 路徑按字串排序 = 按日期排序（yyyy/yyyyMM/yyyyMMdd.csv）。
            float captureHmV = ctx?.CurrentHmV ?? 0f;
            var csvFiles = Directory.GetFiles(captureRootPath, "*.csv", SearchOption.AllDirectories);
            Array.Sort(csvFiles, StringComparer.Ordinal);
            foreach (string csvPath in csvFiles)
            {
                try
                {
                    using (var sr = OpenCsvShared(csvPath))
                    {
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (TryUpdateHmFromCfg(line, ref captureHmV)) continue;
                            if (!TryParseLineEx(line, out string grabId, out string fileName,
                                out int maxExceed, out int meanExceed,
                                out float meanPeak, out float maxPeak,
                                out _, out _, out _)) continue;
                            if (!TryParseFileNameDateTime(fileName, out DateTime ts)) continue;
                            if (ts < start || ts > end) continue;
                            if (!TryExtractCamId(fileName, out int camId)) continue;

                            var key = (grabId, camId);
                            bool thisFail = ctx != null
                                ? ctx.IsFail(meanPeak, maxPeak, captureHmV)
                                : (maxExceed > 0 || meanExceed > 0);

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
                    string p = CaptureStoragePaths.DailyCsv(captureRootPath, d);
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
                    using (var sr = OpenCsvShared(csvPath))
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
                    string dir = CaptureStoragePaths.DateImageDir(captureRootPath, fn);

                    string rawJpg = Path.Combine(dir, fn + CaptureFileNaming.RawJpg);
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
                    string p = CaptureStoragePaths.DailyCsv(captureRootPath, d);
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
                    using (var sr = OpenCsvShared(csvPath))
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

        /// <summary>
        /// 載入指定日期的 #CFG（從 {root}/{yyyy}/{yyyyMM}/{yyyyMMdd}.csv 讀取最後一個 #CFG 列）。
        /// 回傳 null 表示找不到 #CFG。
        /// </summary>
        public static CsvConfigSnapshot LoadConfigForDate(string captureRootPath, DateTime date)
        {
            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return null;

            // CSV 路徑：{root}/{yyyy}/{yyyyMM}/{yyyyMMdd}.csv
            string csvPath = CaptureStoragePaths.DailyCsv(captureRootPath, date);

            return LoadConfigFromCsv(csvPath);
        }

        /// <summary>
        /// 從單一 CSV 檔讀取最後一個 #CFG 列。
        /// 回傳 null 表示找不到 #CFG 或檔案不存在。
        /// </summary>
        public static CsvConfigSnapshot LoadConfigFromCsv(string csvPath)
        {
            if (string.IsNullOrWhiteSpace(csvPath) || !File.Exists(csvPath))
                return null;

            CsvConfigSnapshot latest = null;
            try
            {
                using (var sr = OpenCsvShared(csvPath))
                {
                    string line;
                    while ((line = sr.ReadLine()) != null)
                    {
                        if (line.StartsWith("#CFG,") &&
                            CsvConfigSnapshot.TryParse(line, out var cfg))
                            latest = cfg;
                    }
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[InspectionStatisticsService.LoadConfigFromCsv] {csvPath}: {ex.GetType().Name}: {ex.Message}");
            }

            return latest;
        }

        /// <summary>
        /// 載入指定目錄下最新的 #CFG（掃描所有 CSV，取最後一個 #CFG 列）。
        /// 回傳 null 表示找不到 #CFG。
        /// </summary>
        public static CsvConfigSnapshot LoadLatestConfig(string captureRootPath)
        {
            if (string.IsNullOrWhiteSpace(captureRootPath) || !Directory.Exists(captureRootPath))
                return null;

            CsvConfigSnapshot latest = null;
            try
            {
                var csvFiles = Directory.GetFiles(captureRootPath, "*.csv", SearchOption.AllDirectories);
                Array.Sort(csvFiles, StringComparer.Ordinal);  // B-L1：跟其他 4 處統一

                foreach (string csvPath in csvFiles)
                {
                    var cfg = LoadConfigFromCsv(csvPath);
                    if (cfg != null) latest = cfg;
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[InspectionStatisticsService.LoadLatestConfig] {ex.GetType().Name}: {ex.Message}");
            }

            return latest;
        }
    }
}
