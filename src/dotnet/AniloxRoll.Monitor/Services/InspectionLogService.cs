using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// 逐圖檢測結果記錄服務。
    /// 每次抓圖事件分配一個以首筆擷取時間為基礎的序號（yyMMdd-HHmmss），
    /// 並以 CSV 格式寫入 {CaptureRootPath}\{YYYY}\{YYYYMM}\{YYYYMMDD}.csv。
    /// </summary>
    public class InspectionLogService
    {
        private const string Header =
            "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs,MaxCMean,MeanRPeak,MaxRPeak";

        private readonly Func<string> _getCaptureRoot;
        private readonly object _csvLock = new object();

        // #CFG 變更偵測
        private string _lastWrittenConfigKey;
        private string _lastCsvPath;
        private string _lastFlowRecordGrabId;
        private string _schemaCheckedCsvPath;

        /// <summary>最近一次成功寫入的 CSV 完整路徑（供呼叫端排入遠端複製佇列）。</summary>
        public string LastCsvPath => _lastCsvPath;
        public event Action<string> WriteFailed;
        public event Action WriteSucceeded;

        /// <param name="getCaptureRoot">取得 CaptureRootPath 的委派（支援動態更新）</param>
        public InspectionLogService(Func<string> getCaptureRoot)
        {
            _getCaptureRoot = getCaptureRoot ?? (() => string.Empty);
        }

        /// <summary>
        /// 產生抓圖序號（yyMMdd-HHmmss），以當前時間為基礎。
        /// 每次按下「開始抓取」呼叫一次。
        /// </summary>
        public string NextGrabId()
        {
            return FormatGrabId(DateTime.Now);
        }

        /// <summary>將 DateTime 格式化為 GrabId（yyMMdd-HHmmss）。</summary>
        internal static string FormatGrabId(DateTime dt)
        {
            return dt.ToString("yyMMdd-HHmmss", CultureInfo.InvariantCulture);
        }

        /// <summary>
        /// 寫入一筆單相機檢測結果到當日 CSV（12 欄 + #CFG）。
        /// </summary>
        public void AppendRecord(
            string grabId,
            string fileName,
            float  meanPeak,
            float  maxPeak,
            float  maxCMean,
            float  meanRPeak,
            float  maxRPeak,
            float  errMean,
            float  errMax,
            int    grabHeight,
            double lineRateHz,
            double exposureUs,
            CsvConfigSnapshot config)
        {
            AppendRecord(grabId, fileName, meanPeak, maxPeak, maxCMean,
                meanRPeak, maxRPeak, errMean, errMax,
                grabHeight, lineRateHz, exposureUs, config, DateTime.Now);
        }

        public void AppendRecord(
            string grabId, string fileName, float meanPeak, float maxPeak, float maxCMean,
            float errMean, float errMax, int grabHeight, double lineRateHz,
            double exposureUs, CsvConfigSnapshot config)
        {
            AppendRecord(grabId, fileName, meanPeak, maxPeak, maxCMean,
                float.NaN, float.NaN, errMean, errMax,
                grabHeight, lineRateHz, exposureUs, config, DateTime.Now);
        }

        public void AppendRecord(
            string grabId, string fileName, float meanPeak, float maxPeak,
            float errMean, float errMax, int grabHeight, double lineRateHz,
            double exposureUs, CsvConfigSnapshot config)
        {
            AppendRecord(grabId, fileName, meanPeak, maxPeak, float.NaN,
                float.NaN, float.NaN, errMean, errMax,
                grabHeight, lineRateHz, exposureUs, config, DateTime.Now);
        }

        internal void AppendRecord(
            string grabId, string fileName, float meanPeak, float maxPeak,
            float errMean, float errMax, int grabHeight, double lineRateHz,
            double exposureUs, CsvConfigSnapshot config, DateTime timestamp)
        {
            AppendRecord(grabId, fileName, meanPeak, maxPeak, float.NaN,
                float.NaN, float.NaN, errMean, errMax,
                grabHeight, lineRateHz, exposureUs, config, timestamp);
        }

        internal void AppendRecord(
            string grabId, string fileName, float meanPeak, float maxPeak,
            float maxCMean, float errMean, float errMax, int grabHeight,
            double lineRateHz, double exposureUs, CsvConfigSnapshot config,
            DateTime timestamp)
        {
            AppendRecord(grabId, fileName, meanPeak, maxPeak, maxCMean,
                float.NaN, float.NaN, errMean, errMax,
                grabHeight, lineRateHz, exposureUs, config, timestamp);
        }

        internal void AppendRecord(
            string   grabId,
            string   fileName,
            float    meanPeak,
            float    maxPeak,
            float    maxCMean,
            float    meanRPeak,
            float    maxRPeak,
            float    errMean,
            float    errMax,
            int      grabHeight,
            double   lineRateHz,
            double   exposureUs,
            CsvConfigSnapshot config,
            DateTime timestamp)
        {
            try
            {
                string root = _getCaptureRoot();
                if (string.IsNullOrWhiteSpace(root)) return;

                string csvPath = CaptureStoragePaths.DailyCsv(root, timestamp);
                Directory.CreateDirectory(Path.GetDirectoryName(csvPath));

                int maxExceed  = maxPeak  > errMax  ? 1 : 0;
                int meanExceed = meanPeak > errMean ? 1 : 0;
                bool flowCsvOpen = false;
                bool flowCfgWrite = false;
                bool flowFirstRecordForGrab = false;

                lock (_csvLock)
                {
                    bool isNewFile = !File.Exists(csvPath);
                    bool isNewDay  = !string.Equals(_lastCsvPath, csvPath, StringComparison.OrdinalIgnoreCase);
                    if (!isNewFile && !string.Equals(_schemaCheckedCsvPath, csvPath, StringComparison.OrdinalIgnoreCase))
                    {
                        UpgradeHeaderIfNeeded(csvPath);
                        _schemaCheckedCsvPath = csvPath;
                    }

                    // B-H1：FileShare.ReadWrite 對齊 InspectionCsvReader.OpenShared，避免跨 process race
                    // 造成 reader 偶發 IOException。
                    using (var fs = new FileStream(csvPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite))
                    using (var sw = new StreamWriter(fs, new UTF8Encoding(false)))
                    {
                        // 新檔案或新的一天 → 寫 #CFG + header
                        if (isNewFile || isNewDay)
                        {
                            if (config != null)
                            {
                                sw.WriteLine(config.ToCsvLine());
                                _lastWrittenConfigKey = config.ContentKey;
                                flowCfgWrite = true;
                            }
                            if (isNewFile)
                                sw.WriteLine(Header);
                            _lastCsvPath = csvPath;
                            flowCsvOpen = true;
                        }
                        else if (config != null && config.ContentKey != _lastWrittenConfigKey)
                        {
                            // 設定變更 → 插入新的 #CFG 列
                            sw.WriteLine(config.ToCsvLine());
                            _lastWrittenConfigKey = config.ContentKey;
                            flowCfgWrite = true;
                        }

                        sw.WriteLine(string.Format(CultureInfo.InvariantCulture,
                            "{0},{1},{2},{3},{4:F4},{5:F4},{6},{7:F1},{8:F1},{9:F6},{10:F4},{11:F4}",
                            grabId, fileName, maxExceed, meanExceed,
                            meanPeak, maxPeak, grabHeight, lineRateHz, exposureUs, maxCMean,
                            meanRPeak, maxRPeak));

                        if (!string.Equals(_lastFlowRecordGrabId, grabId, StringComparison.Ordinal))
                        {
                            _lastFlowRecordGrabId = grabId;
                            flowFirstRecordForGrab = true;
                        }
                    }
                }

                if (flowCsvOpen)
                    FlowTrace.Log($"capture csv open path={csvPath} cfg={(flowCfgWrite ? "yes" : "no")}");
                if (flowCfgWrite && config != null)
                    FlowConfigWrite(csvPath, config);
                if (flowFirstRecordForGrab)
                    FlowTrace.Log($"capture csv firstRecord grab={grabId} path={csvPath} file={fileName} " +
                        $"verdict=max{maxExceed}/mean{meanExceed} peak={meanPeak:F4}/{maxPeak:F4} " +
                        $"rowPeak={meanRPeak:F4}/{maxRPeak:F4} maxCMean={maxCMean:F6} " +
                        $"thrV={errMean:F4}/{errMax:F4}");
                WriteSucceeded?.Invoke();
            }
            catch (Exception ex)
            {
                string error = ex.GetType().Name + ": " + ex.Message;
                Trace.WriteLine("[InspectionLogService] " + error);
                WriteFailed?.Invoke(error);
            }
        }

        private static void UpgradeHeaderIfNeeded(string csvPath)
        {
            string[] lines = File.ReadAllLines(csvPath, Encoding.UTF8);
            int headerIndex = Array.FindIndex(lines, line =>
                line.StartsWith("Id,FileName,", StringComparison.Ordinal));
            if (headerIndex < 0 || string.Equals(lines[headerIndex], Header, StringComparison.Ordinal))
                return;

            lines[headerIndex] = Header;
            string tempPath = csvPath + ".schema.tmp";
            try
            {
                File.WriteAllLines(tempPath, lines, new UTF8Encoding(false));
                File.Replace(tempPath, csvPath, null);
            }
            finally
            {
                if (File.Exists(tempPath)) File.Delete(tempPath);
            }
        }

        /// <summary>
        /// 抓圖進行中設定變更時呼叫，立刻在當日 CSV 插入一行 #CFG（不伴隨資料列）。
        /// </summary>
        public void ForceWriteConfig(CsvConfigSnapshot config)
        {
            if (config == null) return;
            try
            {
                string root = _getCaptureRoot();
                if (string.IsNullOrWhiteSpace(root)) return;

                DateTime now = DateTime.Now;
                string csvPath = CaptureStoragePaths.DailyCsv(root, now);
                Directory.CreateDirectory(Path.GetDirectoryName(csvPath));

                lock (_csvLock)
                {
                    if (config.ContentKey == _lastWrittenConfigKey &&
                        string.Equals(_lastCsvPath, csvPath, StringComparison.OrdinalIgnoreCase))
                        return; // 沒有變更，不寫

                    bool isNewFile = !File.Exists(csvPath) ||
                        new FileInfo(csvPath).Length == 0;
                    // B-H1：FileShare.ReadWrite 對齊 reader 端
                    using (var fs = new FileStream(csvPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite))
                    using (var sw = new StreamWriter(fs, new UTF8Encoding(false)))
                    {

                        if (isNewFile)
                            sw.WriteLine(Header);

                        sw.WriteLine(config.ToCsvLine());
                    }

                    _lastWrittenConfigKey = config.ContentKey;
                    _lastCsvPath = csvPath;
                }

                FlowConfigWrite(csvPath, config);
                WriteSucceeded?.Invoke();
            }
            catch (Exception ex)
            {
                string error = ex.GetType().Name + ": " + ex.Message;
                Trace.WriteLine("[InspectionLogService.ForceWriteConfig] " + error);
                WriteFailed?.Invoke(error);
            }
        }

        /// <summary>
        /// Appends the final machine layout for one completed grab. The marker is grab-scoped,
        /// so review/report readers can apply the last layout value to the complete grab without
        /// rewriting already persisted inspection records.
        /// </summary>
        public void WriteFinalLayout(CaptureLayoutSnapshot layout, DateTime captureDate)
        {
            if (layout == null || string.IsNullOrWhiteSpace(layout.GrabId)) return;
            try
            {
                string root = _getCaptureRoot();
                if (string.IsNullOrWhiteSpace(root)) return;

                string csvPath = CaptureStoragePaths.DailyCsv(root, captureDate);
                Directory.CreateDirectory(Path.GetDirectoryName(csvPath));
                lock (_csvLock)
                {
                    using (var fs = new FileStream(
                        csvPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite))
                    using (var sw = new StreamWriter(fs, new UTF8Encoding(false)))
                        sw.WriteLine(layout.ToCsvLine());
                }

                FlowTrace.Log(
                    $"capture layout final grab={layout.GrabId} " +
                    $"{layout.ToFlowValues()} path={csvPath}");
                WriteSucceeded?.Invoke();
            }
            catch (Exception ex)
            {
                string error = ex.GetType().Name + ": " + ex.Message;
                Trace.WriteLine("[InspectionLogService.WriteFinalLayout] " + error);
                WriteFailed?.Invoke(error);
            }
        }

        private static void FlowConfigWrite(string csvPath, CsvConfigSnapshot config)
        {
            double lineRate = config.CamLineRateHz != null && config.CamLineRateHz.Length > 0
                ? config.CamLineRateHz[0]
                : 0;
            FlowTrace.Log($"capture csv cfg path={csvPath} speed={config.AniloxRollSpeedMPerMin:F4} lr={lineRate:F2} " +
                $"HM={config.HessianMaxFactorV:F4}/{config.HessianMaxFactorH:F4} ridge={config.RidgeSigma:F4} " +
                $"thrV={config.ErrorValueMeanV:F4}/{config.ErrorValueMaxV:F4} thrH={config.ErrorValueMeanH:F4}/{config.ErrorValueMaxH:F4}");
        }

    }
}
