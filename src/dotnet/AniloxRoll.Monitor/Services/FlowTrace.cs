using System;
using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// [Flow] 顯示資料流跡唯一出口：咽喉點各一行（按鈕/設定→接線→首幀→顯示），落 Logs\trace-*.log。
    /// 每行帶「時間戳 + 執行緒 ID」：多執行緒交錯時可按執行緒拆回各自序列（執行緒內順序絕對正確）、
    /// 用時間戳看跨執行緒先後。Trace 有全域鎖 → 落檔順序＝實際呼叫順序，不會半行交錯。
    /// 驗證（DVT）寫「偏序」不寫「全序」：因果關係（StartGrab 先於 firstFrame）+ 完整性（每台首幀必出現），
    /// 不規定跨執行緒的交錯順序（本來就不定）。
    /// </summary>
    public static class FlowTrace
    {
        private static int _mode = (int)LogRecordingMode.Operational;

        public static LogRecordingMode Mode => (LogRecordingMode)_mode;
        public static bool DvtEnabled => _mode >= (int)LogRecordingMode.FlowVerification;
        public static bool DiagnosticEnabled => _mode >= (int)LogRecordingMode.FullDiagnostic;

        public static void Configure(LogRecordingMode mode)
        {
            _mode = (int)mode;
            Log($"log mode={mode}");
        }

        public static void Log(string msg) => Write(msg);

        public static void Dvt(string msg)
        {
            if (DvtEnabled) Write(msg);
        }

        public static void Diagnostic(string msg)
        {
            if (DiagnosticEnabled) Write(msg);
        }

        /// <summary>SDK 顯示元件共用出口；依訊息種類分流，不讓 SDK 依賴 app 的設定 enum。</summary>
        public static void Display(string owner, string message)
        {
            if (message == null) return;
            string full = owner + " " + message;
            if (message.StartsWith("stats ", StringComparison.Ordinal))
                Diagnostic(full);
            else if (message.StartsWith("state ", StringComparison.Ordinal) ||
                     message.StartsWith("viewEdges ", StringComparison.Ordinal))
                Dvt(full);
            else
                Log(full);
        }

        private static void Write(string msg) =>
            System.Diagnostics.Trace.WriteLine(
                $"[Flow] {DateTime.Now:HH:mm:ss.fff} T{System.Threading.Thread.CurrentThread.ManagedThreadId,2} {msg}");
    }

    /// <summary>UI 卡頓偵測器（常駐儀器）：
    /// ① 33ms UI timer 量 tick 間隔 → `[UiStall]`＋GC 增量（分辨 GC vs 同步卡住）。
    /// ② 背景執行緒每 100ms BeginInvoke ping 量往返 → `[UiPing]`。
    /// 組合判讀：WM_TIMER 是最低優先權訊息、BeginInvoke（posted message）優先權較高——
    /// UiStall 大 + UiPing 小＝佇列被 paint/input 飽和（timer 被餓，無單一兇手）；
    /// UiStall 大 + UiPing 也大＝UI 執行緒真的被某同步呼叫卡住。</summary>
    public sealed class UiStallDetector : IDisposable
    {
        private readonly System.Windows.Forms.Timer _timer;
        private readonly System.Windows.Forms.Control _pingTarget;
        private readonly System.Threading.Thread _pingThread;
        private readonly System.Threading.ManualResetEventSlim _ponged =
            new System.Threading.ManualResetEventSlim(false);
        private volatile bool _disposed;
        private volatile bool _measurementActive;
        private int _pingGeneration;
        private int _lastTick;
        private int _g0, _g1, _g2;
        private const int ThresholdMs = 100;   // 33ms timer 遲到 3 倍以上才算卡（避免正常排程抖動洗版）

        public UiStallDetector(System.Windows.Forms.Control pingTarget)
        {
            _pingTarget = pingTarget;
            _lastTick = Environment.TickCount;
            _g0 = GC.CollectionCount(0); _g1 = GC.CollectionCount(1); _g2 = GC.CollectionCount(2);
            _timer = new System.Windows.Forms.Timer { Interval = 33 };
            _timer.Tick += (s, e) =>
            {
                int now = Environment.TickCount;
                int gap = now - _lastTick;
                _lastTick = now;
                int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2);
                if (_measurementActive && gap >= ThresholdMs)
                    FlowTrace.Log($"[UiStall] {gap}ms（GC0+{g0 - _g0} GC1+{g1 - _g1} GC2+{g2 - _g2}）");
                _g0 = g0; _g1 = g1; _g2 = g2;
            };
            _timer.Start();

            var uiThread = System.Threading.Thread.CurrentThread;   // ctor 在 UI 執行緒跑 → 取得 UI 執行緒參考（堆疊取樣用）
            _pingThread = new System.Threading.Thread(() =>
            {
                while (!_disposed)
                {
                    System.Threading.Thread.Sleep(100);
                    if (_disposed) break;
                    if (!_measurementActive) continue;
                     var t = _pingTarget;
                     if (t == null || t.IsDisposed || !t.IsHandleCreated) continue;
                     int sent = Environment.TickCount;
                     int generation = System.Threading.Interlocked.Increment(
                         ref _pingGeneration);
                     _ponged.Reset();
                     try
                     {
                         t.BeginInvoke(new Action(() =>
                         {
                             if (_disposed ||
                                 System.Threading.Volatile.Read(ref _pingGeneration) != generation)
                                 return;

                             try { _ponged.Set(); }
                             catch (ObjectDisposedException) { return; }
                             int rtt = Environment.TickCount - sent;
                             if (rtt >= ThresholdMs)
                                 FlowTrace.Log($"[UiPing] {rtt}ms");
                        }));
                    }
                    catch (InvalidOperationException) { continue; }

                     // 200ms 沒回應＝UI 正卡住 → 當場取 UI 執行緒堆疊（卡在哪一行直接點名）。
                     // Suspend+StackTrace 是 deprecated 診斷手段（.NET Framework 可用）：只在已卡住時取樣、立刻 Resume。
                     if (!_ponged.Wait(200) && !_disposed)
                    {
                        try
                        {
#pragma warning disable 618
                            uiThread.Suspend();
                            string frames;
                            try
                            {
                                var st = new System.Diagnostics.StackTrace(uiThread, false);
                                var sb = new System.Text.StringBuilder();
                                int taken = 0;
                                for (int i = 0; i < st.FrameCount && taken < 10; i++)
                                {
                                    var mth = st.GetFrame(i).GetMethod();
                                    if (mth == null) continue;
                                    sb.Append(mth.DeclaringType?.Name).Append('.').Append(mth.Name).Append(" ← ");
                                    taken++;
                                }
                                frames = sb.ToString();
                            }
                            finally { uiThread.Resume(); }
#pragma warning restore 618
                            FlowTrace.Log($"[UiStack] {frames}");
                        }
                        catch (Exception ex) { FlowTrace.Log($"[UiStack] 取樣失敗 {ex.GetType().Name}"); }
                    }
                }
            })
            { IsBackground = true, Name = "UiPing" };
            _pingThread.Start();
        }

        /// <summary>
        /// Form 首次顯示與不可見 tab 預熱完成後才開始量測。建構期尚不能互動，
        /// 若從 ctor 起算，第一個 WM_TIMER 會把整段啟動時間誤報成 UI stall。
        /// </summary>
        public void BeginInteractiveMeasurement()
        {
            _lastTick = Environment.TickCount;
            _g0 = GC.CollectionCount(0);
            _g1 = GC.CollectionCount(1);
            _g2 = GC.CollectionCount(2);
            _measurementActive = true;
        }

        public void Dispose()
        {
            _disposed = true;
            System.Threading.Interlocked.Increment(ref _pingGeneration);
            _timer.Stop();
            _timer.Dispose();
            if (System.Threading.Thread.CurrentThread != _pingThread)
                _pingThread.Join(500);
            _ponged.Dispose();
        }
    }

    /// <summary>WM_PAINT 探針（卡頓歸因儀器）：subclass 目標控制項的 WndProc，量「真正畫」的時間。
    /// 補 [UiSlow]（只量資料更新）的盲區——MSChart 等控制項 UpdateData 快、但之後的 WM_PAINT 才是重活
    /// （densely-pointed chart 畫一次可達數百 ms 且滑鼠掠過就重畫）。&gt;50ms 記 `[UiPaint] 名稱 Nms`。</summary>
    public sealed class PaintProbe : System.Windows.Forms.NativeWindow
    {
        private const int WM_PAINT = 0x000F;
        private readonly string _name;

        public PaintProbe(System.Windows.Forms.Control target, string name)
        {
            _name = name;
            if (target.IsHandleCreated) AssignHandle(target.Handle);
            else target.HandleCreated += (s, e) => AssignHandle(target.Handle);
            target.HandleDestroyed += (s, e) => ReleaseHandle();
        }

        protected override void WndProc(ref System.Windows.Forms.Message m)
        {
            if (m.Msg == WM_PAINT)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                base.WndProc(ref m);
                if (sw.ElapsedMilliseconds > 50)
                    FlowTrace.Log($"[UiPaint] {_name} {sw.ElapsedMilliseconds}ms");
                return;
            }
            base.WndProc(ref m);
        }
    }
}
