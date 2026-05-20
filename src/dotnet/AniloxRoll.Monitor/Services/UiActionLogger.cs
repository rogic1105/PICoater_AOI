using System;
using System.Collections.Concurrent;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Settings.Services;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// FSM Action Logger：記錄 UI 互動 → setting 變更 → state transition。
    /// Runtime flag 控制（PG 設定），Release build 內保留邏輯，預設 Off 零 overhead。
    /// 對應 docs/dev/fsm/{state-catalog,transition-table}.csv 跟 viewer.html。
    ///
    /// 用法：
    ///   click handler 開頭：UiActionLogger.SetSource("muraChartVerticalLive.Click");
    ///   SettingsHub.Changed 訂閱：UiActionLogger.OnSettingChanged(c);
    /// </summary>
    public static class UiActionLogger
    {
        // ── 開關 + 路徑 ────────────────────────────────────────────────
        public static volatile bool Enabled;
        public static string LogFolder = @"D:\Anilox\Logs";

        // ── ThreadStatic / AsyncLocal source（跨 await 保留）─────────────
        private static readonly AsyncLocal<string> _currentSource = new AsyncLocal<string>();

        // ── 背景寫檔 queue（避免 IO block UI thread）────────────────────
        private static readonly BlockingCollection<string> _queue
            = new BlockingCollection<string>(new ConcurrentQueue<string>());
        private static Thread _writerThread;
        private static InspectionSettings _settings;

        // ── State 計算 ─────────────────────────────────────────────────
        private static int _lastStateId;

        public static void Init(InspectionSettings settings)
        {
            _settings = settings;
            _lastStateId = ComputeStateId();
            _writerThread = new Thread(WriterLoop) { IsBackground = true, Name = "UiActionLogger.Writer" };
            _writerThread.Start();
        }

        /// <summary>
        /// click handler 開頭呼，記錄這次 UI input 的來源。
        /// 後續同 thread / 同 async flow 觸發的 SettingsHub.Changed 會記到這個 source 上。
        /// </summary>
        public static void SetSource(string source)
        {
            if (!Enabled) return;
            _currentSource.Value = source;
        }

        /// <summary>SettingsHub.Changed 訂閱，每筆 setting 變更寫一筆 log entry。</summary>
        public static void OnSettingChanged(SettingChange c)
        {
            if (!Enabled || _settings == null) return;
            int newState = ComputeStateId();
            string source = _currentSource.Value ?? "(unknown)";
            string entry = BuildJsonl(source, c.Name, c.OldValue, c.NewValue,
                                      _lastStateId, newState, c.Source.ToString());
            _queue.TryAdd(entry);
            _lastStateId = newState;
        }

        /// <summary>純 view-only 互動（雙擊 FitToScreen / 拖曳）— 不改 setting 但要記。</summary>
        public static void RecordViewOnly(string source)
        {
            if (!Enabled || _settings == null) return;
            int s = ComputeStateId();
            string entry = BuildJsonl(source, "(none)", null, null, s, s, "ViewOnly");
            _queue.TryAdd(entry);
        }

        // ── State id 計算（對齊 docs/dev/fsm/state-catalog.csv）────────────
        // 規則：3 個 setting → 8 state，編碼 = bit0(hb=Global) + bit1(hc) + bit2(hd) + 1
        // 1=V/F/F, 2=V/T/F, 3=V/F/T, 4=V/T/T, 5=G/F/F, 6=G/T/F, 7=G/F/T, 8=G/T/T
        public static int ComputeStateId()
        {
            if (_settings == null) return 0;
            int b0 = _settings.StitchMode == StitchMode.Global ? 4 : 0;
            int b1 = _settings.EnableMuraEnhance ? 1 : 0;
            int b2 = _settings.EnableReviewEnhance ? 2 : 0;
            return b0 + b1 + b2 + 1;
        }

        // ── Internal ───────────────────────────────────────────────────
        private static string BuildJsonl(string source, string setting, object oldVal,
                                         object newVal, int beforeState, int afterState, string changeSrc)
        {
            string oldStr = oldVal == null ? "null" : $"\"{oldVal}\"";
            string newStr = newVal == null ? "null" : $"\"{newVal}\"";
            return "{\"ts\":\"" + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff") +
                   "\",\"source\":\"" + source +
                   "\",\"setting\":\"" + setting +
                   "\",\"from\":" + oldStr +
                   ",\"to\":" + newStr +
                   ",\"stateBefore\":" + beforeState +
                   ",\"stateAfter\":" + afterState +
                   ",\"changeSrc\":\"" + changeSrc + "\"}";
        }

        private static void WriterLoop()
        {
            foreach (var entry in _queue.GetConsumingEnumerable())
            {
                try
                {
                    Directory.CreateDirectory(LogFolder);
                    string path = Path.Combine(LogFolder,
                        $"ui-actions-{DateTime.Now:yyyyMMdd}.jsonl");
                    File.AppendAllText(path, entry + "\n");
                }
                catch { /* swallow — 不影響 UI */ }
            }
        }
    }
}
