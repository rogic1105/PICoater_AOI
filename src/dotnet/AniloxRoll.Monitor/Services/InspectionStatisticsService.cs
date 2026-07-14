using System;
using System.Collections.Generic;
using System.Diagnostics;
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
        /// <summary>null=舊資料未保存列峰值，false=Pass，true=任一列曲線超標。</summary>
        public bool? RowResult { get; set; }
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
    /// One-pass, bounded in-memory view of report CSV data. UI navigators, detail
    /// lists, and period charts share this snapshot instead of rescanning disk.
    /// </summary>
    public sealed class InspectionStatisticsSnapshot
    {
        internal InspectionStatisticsSnapshot(
            SortedSet<DateTime> availableTimes,
            List<GrabIdInfo> grabIdsDescending,
            Dictionary<string, GrabDetail> detailsByGrabId,
            int csvFileCount,
            int recordCount)
        {
            AvailableTimes = availableTimes;
            GrabIdsDescending = grabIdsDescending;
            DetailsByGrabId = detailsByGrabId;
            CsvFileCount = csvFileCount;
            RecordCount = recordCount;
        }

        public SortedSet<DateTime> AvailableTimes { get; }
        public List<GrabIdInfo> GrabIdsDescending { get; }
        public Dictionary<string, GrabDetail> DetailsByGrabId { get; }
        public int CsvFileCount { get; }
        public int RecordCount { get; }
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
        public float CurrentHmH { get; }
        public float CurrentRowErrMean { get; }
        public float CurrentRowErrMax { get; }

        public ThresholdContext(float currentHmV, float currentErrMean, float currentErrMax)
            : this(currentHmV, currentErrMean, currentErrMax,
                currentHmV, currentErrMean, currentErrMax)
        {
        }

        public ThresholdContext(
            float currentHmV, float currentErrMean, float currentErrMax,
            float currentHmH, float currentRowErrMean, float currentRowErrMax)
        {
            CurrentHmV     = currentHmV;
            CurrentErrMean = currentErrMean;
            CurrentErrMax  = currentErrMax;
            CurrentHmH = currentHmH;
            CurrentRowErrMean = currentRowErrMean;
            CurrentRowErrMax = currentRowErrMax;
        }

        public bool IsFail(float meanPeak, float maxPeak, float captureHmV)
        {
            float ratio = (captureHmV > 0f && CurrentHmV > 0f) ? captureHmV / CurrentHmV : 1f;
            float displayMean = meanPeak * ratio;
            float displayMax  = maxPeak  * ratio;
            return displayMean > CurrentErrMean || displayMax > CurrentErrMax;
        }

        public bool? IsRowFail(float meanPeak, float maxPeak, float captureHmV)
        {
            if (float.IsNaN(meanPeak) || float.IsNaN(maxPeak)) return null;
            float ratio = (captureHmV > 0f && CurrentHmH > 0f)
                ? captureHmV / CurrentHmH : 1f;
            return meanPeak * ratio > CurrentRowErrMean ||
                   maxPeak * ratio > CurrentRowErrMax;
        }
    }

    /// <summary>
    /// 從每日 CSV（{YYYYMMDD}.csv）讀取資料，計算各相機的 Pass/Fail 統計。
    /// CSV 格式：Id,FileName,MaxExceed,MeanExceed
    /// </summary>
    public static class InspectionStatisticsService
    {
        /// <summary>
        /// Parses every report CSV once and materializes all shared report indexes.
        /// The snapshot retains one GrabIdInfo and one seven-camera result per grab,
        /// not every CSV record.
        /// </summary>
        public static InspectionStatisticsSnapshot LoadSnapshot(
            string captureRootPath,
            ThresholdContext ctx = null)
        {
            var availableTimes = new SortedSet<DateTime>();
            var infosByGrabId =
                new SortedDictionary<string, GrabIdInfo>(StringComparer.Ordinal);
            var detailsByGrabId =
                new Dictionary<string, GrabDetail>(StringComparer.Ordinal);
            int recordCount = 0;

            if (string.IsNullOrWhiteSpace(captureRootPath) ||
                !Directory.Exists(captureRootPath))
            {
                return new InspectionStatisticsSnapshot(
                    availableTimes, new List<GrabIdInfo>(), detailsByGrabId, 0, 0);
            }

            string[] csvFiles = GetInspectionCsvFiles(captureRootPath);
            Array.Sort(csvFiles, StringComparer.Ordinal);
            float captureHmV = ctx?.CurrentHmV ?? 0f;
            foreach (string csvPath in csvFiles)
            {
                try
                {
                    using (var reader = InspectionCsvReader.OpenShared(csvPath))
                    {
                        string line;
                        while ((line = reader.ReadLine()) != null)
                        {
                            if (InspectionCsvReader.TryUpdateHmFromConfig(
                                line, ref captureHmV))
                                continue;
                            if (!InspectionCsvReader.TryParseRecord(line, out var record))
                                continue;
                            if (!InspectionCsvReader.TryParseTimestamp(
                                record.FileName, out DateTime timestamp))
                                continue;

                            recordCount++;
                            availableTimes.Add(timestamp);
                            if (infosByGrabId.TryGetValue(
                                record.GrabId, out GrabIdInfo info))
                            {
                                if (timestamp < info.Earliest) info.Earliest = timestamp;
                                if (timestamp > info.Latest) info.Latest = timestamp;
                            }
                            else
                            {
                                info = new GrabIdInfo
                                {
                                    GrabId = record.GrabId,
                                    Earliest = timestamp,
                                    Latest = timestamp
                                };
                                infosByGrabId[record.GrabId] = info;
                            }

                            if (!InspectionCsvReader.TryExtractCameraId(
                                record.FileName, out int cameraId) ||
                                cameraId < 1 || cameraId > 7)
                                continue;
                            if (!detailsByGrabId.TryGetValue(
                                record.GrabId, out GrabDetail detail))
                            {
                                detail = new GrabDetail { GrabId = record.GrabId };
                                detailsByGrabId[record.GrabId] = detail;
                            }

                            bool failed = ctx != null
                                ? ctx.IsFail(record.MeanPeak, record.MaxPeak, captureHmV)
                                : (record.MaxExceed > 0 || record.MeanExceed > 0);
                            int cameraIndex = cameraId - 1;
                            if (!detail.CamResult[cameraIndex].HasValue || failed)
                                detail.CamResult[cameraIndex] = failed;
                            MergeRowResult(detail, ctx?.IsRowFail(
                                record.MeanRPeak, record.MaxRPeak, captureHmV));
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine(
                        $"[InspectionStatisticsService.LoadSnapshot] {csvPath}: " +
                        $"{ex.GetType().Name}: {ex.Message}");
                }
            }

            var grabIds = new List<GrabIdInfo>(infosByGrabId.Values);
            grabIds.Reverse();
            return new InspectionStatisticsSnapshot(
                availableTimes, grabIds, detailsByGrabId,
                csvFiles.Length, recordCount);
        }

        private static string[] GetInspectionCsvFiles(string captureRootPath)
        {
            string[] candidates = Directory.GetFiles(
                captureRootPath, "*.csv", SearchOption.AllDirectories);
            var result = new List<string>(candidates.Length);
            foreach (string path in candidates)
            {
                string name = Path.GetFileNameWithoutExtension(path);
                if (name.Length != 8) continue;

                bool allDigits = true;
                for (int i = 0; i < name.Length; i++)
                {
                    if (name[i] < '0' || name[i] > '9')
                    {
                        allDigits = false;
                        break;
                    }
                }
                if (allDigits) result.Add(path);
            }
            result.Sort(StringComparer.Ordinal);
            return result.ToArray();
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
                    using (var sr = InspectionCsvReader.OpenShared(csvPath))
                    {
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (InspectionCsvReader.TryUpdateHmFromConfig(line, ref captureHmV)) continue;
                            if (!InspectionCsvReader.TryParseRecord(line, out var record)) continue;

                            if (!InspectionCsvReader.TryParseTimestamp(record.FileName, out DateTime ts)) continue;
                            if (ts < start || ts > end) continue;

                            if (!InspectionCsvReader.TryExtractCameraId(record.FileName, out int camId)) continue;
                            if (!stats.TryGetValue(camId, out var s)) continue;

                            bool isFail = ctx != null
                                ? ctx.IsFail(record.MeanPeak, record.MaxPeak, captureHmV)
                                : (record.MaxExceed > 0 || record.MeanExceed > 0);
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
                    using (var sr = InspectionCsvReader.OpenShared(csvPath))
                    {
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (InspectionCsvReader.TryUpdateHmFromConfig(line, ref captureHmV)) continue;
                            if (!InspectionCsvReader.TryParseRecord(line, out var record)) continue;

                            if (StringComparer.Ordinal.Compare(record.GrabId, lo) < 0 ||
                                StringComparer.Ordinal.Compare(record.GrabId, hi) > 0) continue;

                            if (!InspectionCsvReader.TryExtractCameraId(record.FileName, out int camId)) continue;

                            if (!grabCamFail.TryGetValue(record.GrabId, out var camMap))
                                grabCamFail[record.GrabId] = camMap = new Dictionary<int, bool>();

                            bool thisFail = ctx != null
                                ? ctx.IsFail(record.MeanPeak, record.MaxPeak, captureHmV)
                                : (record.MaxExceed > 0 || record.MeanExceed > 0);
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
                    using (var sr = InspectionCsvReader.OpenShared(csvPath))
                    {
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (InspectionCsvReader.TryUpdateHmFromConfig(line, ref captureHmV)) continue;
                            if (!InspectionCsvReader.TryParseRecord(line, out var record)) continue;

                            if (StringComparer.Ordinal.Compare(record.GrabId, lo) < 0 ||
                                StringComparer.Ordinal.Compare(record.GrabId, hi) > 0) continue;

                            if (!InspectionCsvReader.TryExtractCameraId(record.FileName, out int camId)) continue;
                            if (camId < 1 || camId > 7) continue;

                            if (!dict.TryGetValue(record.GrabId, out var detail))
                            {
                                detail = new GrabDetail { GrabId = record.GrabId };
                                dict[record.GrabId] = detail;
                            }

                            int idx = camId - 1;
                            bool thisFail = ctx != null
                                ? ctx.IsFail(record.MeanPeak, record.MaxPeak, captureHmV)
                                : (record.MaxExceed > 0 || record.MeanExceed > 0);
                            if (detail.CamResult[idx] == null)
                                detail.CamResult[idx] = thisFail;
                            else if (thisFail)
                                detail.CamResult[idx] = true; // 一票否決
                            MergeRowResult(detail, ctx?.IsRowFail(
                                record.MeanRPeak, record.MaxRPeak, captureHmV));
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine($"[InspectionStatisticsService.ComputeDetailedByGrabIdRange] {csvPath}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            // 明細列表顯示序：新→舊（dict 為字串升冪＝舊→新，反轉成降冪）。
            // 處理迴圈仍走升冪（captureHmV 跨 CSV carry 依賴日期順序），只反轉輸出。
            var ordered = new List<GrabDetail>(dict.Values);
            ordered.Reverse();
            return ordered;
        }

        private static void MergeRowResult(GrabDetail detail, bool? failed)
        {
            if (detail == null || !failed.HasValue) return;
            if (!detail.RowResult.HasValue || failed.Value)
                detail.RowResult = failed.Value;
        }

        public static Dictionary<int, CameraStats> ComputeStatsFromDetails(
            IList<GrabDetail> details)
        {
            var stats = new Dictionary<int, CameraStats>();
            for (int camId = 1; camId <= 7; camId++)
                stats[camId] = new CameraStats { CamId = camId };
            if (details == null) return stats;

            foreach (GrabDetail detail in details)
            {
                if (detail == null) continue;
                for (int i = 0; i < detail.CamResult.Length && i < 7; i++)
                {
                    bool? failed = detail.CamResult[i];
                    if (!failed.HasValue) continue;
                    if (failed.Value) stats[i + 1].Fail++;
                    else stats[i + 1].Pass++;
                }
            }
            return stats;
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
                    using (var sr = InspectionCsvReader.OpenShared(csvPath))
                    {
                        sr.ReadLine(); // skip header
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (!InspectionCsvReader.TryParseRecord(line, out var record)) continue;
                            if (string.IsNullOrEmpty(record.GrabId)) continue;
                            if (!InspectionCsvReader.TryParseTimestamp(record.FileName, out DateTime dt)) continue;

                            if (dict.TryGetValue(record.GrabId, out var existing))
                            {
                                dict[record.GrabId] = (
                                    dt < existing.earliest ? dt : existing.earliest,
                                    dt > existing.latest   ? dt : existing.latest);
                            }
                            else
                            {
                                dict[record.GrabId] = (dt, dt);
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
                    using (var sr = InspectionCsvReader.OpenShared(csvPath))
                    {
                        sr.ReadLine();
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (!InspectionCsvReader.TryParseRecord(line, out var record)) continue;
                            if (!InspectionCsvReader.TryParseTimestamp(record.FileName, out DateTime dt)) continue;
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
            IList<GrabIdInfo> grabIds,
            IDictionary<string, GrabDetail> details,
            DateTime start,
            DateTime end)
        {
            return ComputeGroupedFromIndex(
                grabIds, details, start, end, 12,
                timestamp => timestamp.Month - 1,
                index => (index + 1).ToString());
        }

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
            IList<GrabIdInfo> grabIds,
            IDictionary<string, GrabDetail> details,
            DateTime start,
            DateTime end)
        {
            return ComputeGroupedFromIndex(
                grabIds, details, start, end, 31,
                timestamp => timestamp.Day - 1,
                index => (index + 1).ToString());
        }

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
            IList<GrabIdInfo> grabIds,
            IDictionary<string, GrabDetail> details,
            DateTime start,
            DateTime end)
        {
            return ComputeGroupedFromIndex(
                grabIds, details, start, end, 24,
                timestamp => timestamp.Hour,
                index => index.ToString());
        }

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

        private static List<PeriodStats> ComputeGroupedFromIndex(
            IList<GrabIdInfo> grabIds,
            IDictionary<string, GrabDetail> details,
            DateTime start,
            DateTime end,
            int bucketCount,
            Func<DateTime, int> getBucket,
            Func<int, string> getLabel)
        {
            var pass = new int[bucketCount];
            var fail = new int[bucketCount];
            if (grabIds != null && details != null)
            {
                foreach (GrabIdInfo info in grabIds)
                {
                    DateTime timestamp = info.Earliest;
                    if (timestamp < start || timestamp > end) continue;
                    if (!details.TryGetValue(info.GrabId, out GrabDetail detail))
                        continue;

                    int bucket = getBucket(timestamp);
                    if (bucket < 0 || bucket >= bucketCount) continue;
                    for (int cameraIndex = 0;
                        cameraIndex < detail.CamResult.Length;
                        cameraIndex++)
                    {
                        bool? failed = detail.CamResult[cameraIndex];
                        if (!failed.HasValue) continue;
                        if (failed.Value) fail[bucket]++;
                        else pass[bucket]++;
                    }
                }
            }

            var result = new List<PeriodStats>(bucketCount);
            for (int index = 0; index < bucketCount; index++)
            {
                result.Add(new PeriodStats
                {
                    Label = getLabel(index),
                    Pass = pass[index],
                    Fail = fail[index]
                });
            }
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
                    using (var sr = InspectionCsvReader.OpenShared(csvPath))
                    {
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (InspectionCsvReader.TryUpdateHmFromConfig(line, ref captureHmV)) continue;
                            if (!InspectionCsvReader.TryParseRecord(line, out var record)) continue;
                            if (!InspectionCsvReader.TryParseTimestamp(record.FileName, out DateTime ts)) continue;
                            if (ts < start || ts > end) continue;
                            if (!InspectionCsvReader.TryExtractCameraId(record.FileName, out int camId)) continue;

                            var key = (record.GrabId, camId);
                            bool thisFail = ctx != null
                                ? ctx.IsFail(record.MeanPeak, record.MaxPeak, captureHmV)
                                : (record.MaxExceed > 0 || record.MeanExceed > 0);

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

    }
}
