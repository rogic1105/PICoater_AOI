using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// 逐圖檢測結果記錄服務。
    /// 每次抓圖事件分配一個唯一編號（A00001 起），
    /// 並以 CSV 格式寫入 {CaptureRootPath}\{YYYY}\{YYYYMM}\{YYYYMMDD}.csv。
    /// </summary>
    public class InspectionLogService
    {
        private const string Header =
            "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs";

        private int _lastIdNum;
        private readonly Func<string> _getCaptureRoot;
        private readonly object _csvLock = new object();

        // #CFG 變更偵測
        private string _lastWrittenConfigKey;
        private string _lastCsvPath;

        /// <summary>
        /// <param name="getCaptureRoot">取得 CaptureRootPath 的委派（支援動態更新）</param>
        /// <param name="startIdNum">從 session-state 讀回的上次編號，下次 NextGrabId() 從 +1 開始</param>
        /// </summary>
        public InspectionLogService(Func<string> getCaptureRoot, int startIdNum = 0)
        {
            _getCaptureRoot = getCaptureRoot ?? (() => string.Empty);
            _lastIdNum = startIdNum;
        }

        public int LastIdNum => _lastIdNum;

        /// <summary>
        /// 產生下一個抓圖編號（e.g. A00001），同時持久化計數器。
        /// 每次按下「開始抓取」呼叫一次。
        /// </summary>
        public string NextGrabId()
        {
            _lastIdNum++;
            UI.State.UserSessionState.SetLastGrabIdNum(_lastIdNum);
            UI.State.UserSessionState.Save();
            return FormatId(_lastIdNum);
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

                string dir = Path.Combine(
                    root,
                    timestamp.Year.ToString(CultureInfo.InvariantCulture),
                    timestamp.ToString("yyyyMM"));
                Directory.CreateDirectory(dir);

                string csvPath = Path.Combine(dir, $"{timestamp:yyyyMMdd}.csv");

                int maxExceed  = maxPeak  > errMax  ? 1 : 0;
                int meanExceed = meanPeak > errMean ? 1 : 0;

                lock (_csvLock)
                {
                    bool isNewFile = !File.Exists(csvPath);
                    bool isNewDay  = !string.Equals(_lastCsvPath, csvPath, StringComparison.OrdinalIgnoreCase);

                    using (var sw = new StreamWriter(csvPath, append: true, new UTF8Encoding(false)))
                    {
                        // 新檔案或新的一天 → 寫 #CFG + header
                        if (isNewFile || isNewDay)
                        {
                            if (config != null)
                            {
                                sw.WriteLine(config.ToCsvLine());
                                _lastWrittenConfigKey = config.ContentKey;
                            }
                            if (isNewFile)
                                sw.WriteLine(Header);
                            _lastCsvPath = csvPath;
                        }
                        else if (config != null && config.ContentKey != _lastWrittenConfigKey)
                        {
                            // 設定變更 → 插入新的 #CFG 列
                            sw.WriteLine(config.ToCsvLine());
                            _lastWrittenConfigKey = config.ContentKey;
                        }

                        sw.WriteLine(string.Format(CultureInfo.InvariantCulture,
                            "{0},{1},{2},{3},{4:F4},{5:F4},{6},{7:F1},{8:F1}",
                            grabId, fileName, maxExceed, meanExceed,
                            meanPeak, maxPeak, grabHeight, lineRateHz, exposureUs));
                    }
                }
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
                string dir = Path.Combine(root,
                    now.Year.ToString(CultureInfo.InvariantCulture),
                    now.ToString("yyyyMM"));
                Directory.CreateDirectory(dir);

                string csvPath = Path.Combine(dir, $"{now:yyyyMMdd}.csv");

                lock (_csvLock)
                {
                    if (config.ContentKey == _lastWrittenConfigKey &&
                        string.Equals(_lastCsvPath, csvPath, StringComparison.OrdinalIgnoreCase))
                        return; // 沒有變更，不寫

                    using (var sw = new StreamWriter(csvPath, append: true, new UTF8Encoding(false)))
                    {
                        bool isNewFile = !File.Exists(csvPath) ||
                            new FileInfo(csvPath).Length == 0;

                        if (isNewFile)
                            sw.WriteLine(Header);

                        sw.WriteLine(config.ToCsvLine());
                    }

                    _lastWrittenConfigKey = config.ContentKey;
                    _lastCsvPath = csvPath;
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[InspectionLogService.ForceWriteConfig] {ex.GetType().Name}: {ex.Message}");
            }
        }

        private static string FormatId(int n) => $"A{n:D5}";
    }
}
