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
            "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs";

        private readonly Func<string> _getCaptureRoot;
        private readonly object _csvLock = new object();

        // #CFG 變更偵測
        private string _lastWrittenConfigKey;
        private string _lastCsvPath;
        private string _lastFlowRecordGrabId;

        /// <summary>最近一次成功寫入的 CSV 完整路徑（供呼叫端排入遠端複製佇列）。</summary>
        public string LastCsvPath => _lastCsvPath;

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
        /// 寫入一筆單相機檢測結果到當日 CSV（新格式 9 欄 + #CFG）。
        /// </summary>
        public void AppendRecord(
            string grabId,
            string fileName,
            float  meanPeak,
            float  maxPeak,
            float  errMean,
            float  errMax,
            int    grabHeight,
            double lineRateHz,
            double exposureUs,
            CsvConfigSnapshot config)
        {
            AppendRecord(grabId, fileName, meanPeak, maxPeak, errMean, errMax,
                grabHeight, lineRateHz, exposureUs, config, DateTime.Now);
        }

        internal void AppendRecord(
            string   grabId,
            string   fileName,
            float    meanPeak,
            float    maxPeak,
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

                    // B-H1：FileShare.ReadWrite 對齊 reader 端的 OpenCsvShared，避免跨 process race
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
                            "{0},{1},{2},{3},{4:F4},{5:F4},{6},{7:F1},{8:F1}",
                            grabId, fileName, maxExceed, meanExceed,
                            meanPeak, maxPeak, grabHeight, lineRateHz, exposureUs));

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
                    FlowTrace.Log($"capture csv cfg path={csvPath} HM={config.HessianMaxFactorV:F4}/{config.HessianMaxFactorH:F4} " +
                        $"thrV={config.ErrorValueMeanV:F4}/{config.ErrorValueMaxV:F4} thrH={config.ErrorValueMeanH:F4}/{config.ErrorValueMaxH:F4}");
                if (flowFirstRecordForGrab)
                    FlowTrace.Log($"capture csv firstRecord grab={grabId} path={csvPath} file={fileName} " +
                        $"verdict=max{maxExceed}/mean{meanExceed} peak={meanPeak:F4}/{maxPeak:F4} thrV={errMean:F4}/{errMax:F4}");
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[InspectionLogService] {ex.GetType().Name}: {ex.Message}");
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

                FlowTrace.Log($"capture csv cfg path={csvPath} HM={config.HessianMaxFactorV:F4}/{config.HessianMaxFactorH:F4} " +
                    $"thrV={config.ErrorValueMeanV:F4}/{config.ErrorValueMaxV:F4} thrH={config.ErrorValueMeanH:F4}/{config.ErrorValueMaxH:F4}");
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[InspectionLogService.ForceWriteConfig] {ex.GetType().Name}: {ex.Message}");
            }
        }

    }
}
