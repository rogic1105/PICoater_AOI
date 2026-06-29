using System;
using System.Diagnostics;

namespace TanukiCv.Core
{
    /// <summary>
    /// 量一段程式碼耗時 + 記錄「回報視窗內最大值（worst-case）」的通用計時器（純邏輯、無 UI/MIL 依賴）。
    /// 「計時很常用」的單一來源：原本同一個 pattern 分散在 ImageCanvas 繪製計時(_paintSw) 與
    /// 範例縮圖計時(_resizeSw)，2026-06 收斂於此。
    ///
    /// 用法：
    ///   t.Start(); …量的程式碼…; t.Stop();      // Stop 自動更新 LastMs / MaxMs
    ///   double worst = t.ResetMax();             // 每回報視窗讀出最大並歸零（取「這段時間內最差」）
    ///   using (t.Measure()) { …量的程式碼… }     // 範圍計時（等同 Start/Stop）
    ///
    /// 非執行緒安全（一個量測點一個實例；跨執行緒請各自持有）。
    /// </summary>
    public sealed class PerfTimer
    {
        private readonly Stopwatch _sw = new Stopwatch();

        /// <summary>上次 <see cref="Stop"/> 量到的耗時（ms，含小數）。</summary>
        public double LastMs { get; private set; }

        /// <summary>自上次 <see cref="ResetMax"/> 以來的最大耗時（worst-case，判斷卡頓用）。</summary>
        public double MaxMs { get; private set; }

        /// <summary>開始計時（重置碼錶）。</summary>
        public void Start() => _sw.Restart();

        /// <summary>停止計時：更新 <see cref="LastMs"/>，並在更大時更新 <see cref="MaxMs"/>。</summary>
        public void Stop()
        {
            _sw.Stop();
            LastMs = _sw.Elapsed.TotalMilliseconds;
            if (LastMs > MaxMs) MaxMs = LastMs;
        }

        /// <summary>讀出視窗內最大耗時並歸零（下個回報視窗重新累計 worst-case）。</summary>
        public double ResetMax() { double m = MaxMs; MaxMs = 0; return m; }

        /// <summary>範圍計時：<c>using (t.Measure()) { … }</c>，Dispose 時自動 <see cref="Stop"/>。</summary>
        public IDisposable Measure() { Start(); return new Scope(this); }
        private sealed class Scope : IDisposable { private readonly PerfTimer _t; public Scope(PerfTimer t) { _t = t; } public void Dispose() => _t.Stop(); }
    }
}
