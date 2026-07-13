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

        // ── Mura 空間分布曲線（.bin 抽樣平均）──────────────────────────────

        /// <summary>
        /// 從指定 grabIdInfos 讀取每台相機的 MeanC / MaxC bin，按位置平均後返回。
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
                    string[] mFiles = FindCurveFiles(dateDir, prefix, camId,
                        CaptureFileNaming.MeanC, CaptureFileNaming.MeanCPrevious, CaptureFileNaming.MeanCLegacy);
                    string[] xFiles = FindCurveFiles(dateDir, prefix, camId,
                        CaptureFileNaming.MaxC, CaptureFileNaming.MaxCPrevious, CaptureFileNaming.MaxCLegacy);
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

        private sealed class MuraCurveRecord
        {
            public string MeanCPath;
            public string MaxCPath;
            public float MaxCMean;
        }

        public static (
            Dictionary<int, float[]> Mean,
            Dictionary<int, float[]> Max,
            int MeanRows,
            int MaxRows,
            int ScoredRows,
            int TotalRows,
            int RankedCams,
            int TotalCams)
            LoadRangeMuraProfile(string rootPath, IList<GrabIdInfo> rangeInfos, int limit)
        {
            var meanResult = new Dictionary<int, float[]>();
            var maxResult = new Dictionary<int, float[]>();
            int meanRows = 0, maxRows = 0, scoredRows = 0, totalRows = 0;
            int rankedCams = 0;
            if (string.IsNullOrWhiteSpace(rootPath) || rangeInfos == null ||
                rangeInfos.Count == 0 || limit <= 0)
                return (meanResult, maxResult, 0, 0, 0, 0, 0, 0);

            var rangeIds = new HashSet<string>(StringComparer.Ordinal);
            var dates = new HashSet<DateTime>();
            foreach (var info in rangeInfos)
            {
                if (info == null || string.IsNullOrEmpty(info.GrabId)) continue;
                rangeIds.Add(info.GrabId);
                dates.Add(info.Earliest.Date);
            }

            var recordsByCam = new Dictionary<int, List<MuraCurveRecord>>();
            foreach (DateTime date in dates)
            {
                string csvPath = CaptureStoragePaths.DailyCsv(rootPath, date);
                if (!File.Exists(csvPath)) continue;
                try
                {
                    using (var sr = InspectionCsvReader.OpenShared(csvPath))
                    {
                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            if (!InspectionCsvReader.TryParseRecord(line, out var record)) continue;
                            if (!rangeIds.Contains(record.GrabId) ||
                                !InspectionCsvReader.TryExtractCameraId(record.FileName, out int camId) ||
                                !InspectionCsvReader.TryParseTimestamp(record.FileName, out DateTime timestamp)) continue;

                            if (!recordsByCam.TryGetValue(camId, out var records))
                                recordsByCam[camId] = records = new List<MuraCurveRecord>();
                            string dateDir = CaptureStoragePaths.DateImageDir(rootPath, timestamp);
                            records.Add(new MuraCurveRecord
                            {
                                MeanCPath = CaptureFileNaming.ResolveMeanC(Path.Combine(dateDir, record.FileName)),
                                MaxCPath = CaptureFileNaming.ResolveMaxC(Path.Combine(dateDir, record.FileName)),
                                MaxCMean = record.MaxCMean
                            });
                            totalRows++;
                            if (!float.IsNaN(record.MaxCMean) && !float.IsInfinity(record.MaxCMean)) scoredRows++;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine($"[InspectionStatisticsService.LoadRangeMuraProfile] {csvPath}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            foreach (var camRecords in recordsByCam)
            {
                List<MuraCurveRecord> records = camRecords.Value;
                List<MuraCurveRecord> meanCandidates = EvenSampleCurveRecords(records, limit);
                var scored = records.FindAll(r =>
                    !float.IsNaN(r.MaxCMean) && !float.IsInfinity(r.MaxCMean));
                List<MuraCurveRecord> maxCandidates;
                if (scored.Count == records.Count && scored.Count > 0)
                {
                    scored.Sort((a, b) => b.MaxCMean.CompareTo(a.MaxCMean));
                    maxCandidates = scored.GetRange(0, Math.Min(limit, scored.Count));
                    rankedCams++;
                }
                else
                {
                    maxCandidates = EvenSampleCurveRecords(records, limit);
                }

                float[] mean = AggregateCurveRecords(meanCandidates, true);
                float[] max = AggregateCurveRecords(maxCandidates, false);
                if (mean != null) meanResult[camRecords.Key] = mean;
                if (max != null) maxResult[camRecords.Key] = max;
                meanRows += meanCandidates.Count;
                maxRows += maxCandidates.Count;
            }

            return (meanResult, maxResult, meanRows, maxRows, scoredRows, totalRows,
                rankedCams, recordsByCam.Count);
        }

        private static List<MuraCurveRecord> EvenSampleCurveRecords(
            List<MuraCurveRecord> records, int limit)
        {
            if (records.Count <= limit) return new List<MuraCurveRecord>(records);
            if (limit == 1) return new List<MuraCurveRecord> { records[0] };
            var sampled = new List<MuraCurveRecord>(limit);
            for (int i = 0; i < limit; i++)
            {
                int index = (int)((long)i * (records.Count - 1) / (limit - 1));
                sampled.Add(records[index]);
            }
            return sampled;
        }

        private static string[] FindCurveFiles(
            string directory, string prefix, int camId,
            string current, string previous, string legacy)
        {
            string[] files = Directory.GetFiles(directory, $"{prefix}*-{camId}{current}");
            if (files.Length > 0) return files;
            files = Directory.GetFiles(directory, $"{prefix}*-{camId}{previous}");
            return files.Length > 0
                ? files
                : Directory.GetFiles(directory, $"{prefix}*-{camId}{legacy}");
        }

        private static float[] AggregateCurveRecords(List<MuraCurveRecord> records, bool mean)
        {
            float[] result = null;
            int loaded = 0;
            foreach (var record in records)
            {
                float[] curve = TryLoadBin(mean ? record.MeanCPath : record.MaxCPath);
                if (curve == null || curve.Length == 0) continue;
                if (result == null) result = new float[curve.Length];
                if (result.Length != curve.Length) continue;

                for (int i = 0; i < curve.Length; i++)
                {
                    if (mean) result[i] += curve[i];
                    else if (curve[i] > result[i]) result[i] = curve[i];
                }
                loaded++;
            }

            if (mean && result != null && loaded > 0)
                for (int i = 0; i < result.Length; i++) result[i] /= loaded;
            return result;
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
