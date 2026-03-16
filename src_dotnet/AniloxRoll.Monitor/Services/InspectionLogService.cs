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
    /// 並以 CSV 格式寫入 {CaptureRootPath}\{YYYY}\{YYYYMM}\inspection-log-{YYYYMMDD}.csv。
    /// </summary>
    public class InspectionLogService
    {
        private int _lastIdNum;
        private readonly Func<string> _getCaptureRoot;

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
        /// 寫入一筆單相機檢測結果到當日 CSV。
        /// maxPeak / meanPeak 為 0–1 normalized（MuraCurve.Max() / 255f）。
        /// </summary>
        public void AppendRecord(
            string grabId,
            string fileName,
            float  meanPeak,
            float  maxPeak,
            float  errMean,
            float  errMax)
        {
            AppendRecord(grabId, fileName, meanPeak, maxPeak, errMean, errMax, DateTime.Now);
        }

        internal void AppendRecord(
            string   grabId,
            string   fileName,
            float    meanPeak,
            float    maxPeak,
            float    errMean,
            float    errMax,
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

                string path = Path.Combine(dir,
                    $"inspection-log-{timestamp:yyyyMMdd}.csv");

                bool writeHeader = !File.Exists(path);
                int  maxExceed   = maxPeak  > errMax  ? 1 : 0;
                int  meanExceed  = meanPeak > errMean ? 1 : 0;

                using (var sw = new StreamWriter(path, append: true, new UTF8Encoding(false)))
                {
                    if (writeHeader)
                        sw.WriteLine("Id,FileName,MaxExceed,MeanExceed");

                    sw.WriteLine($"{grabId},{fileName},{maxExceed},{meanExceed}");
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[InspectionLogService] {ex.GetType().Name}: {ex.Message}");
            }
        }

        private static string FormatId(int n) => $"A{n:D5}";
    }
}
