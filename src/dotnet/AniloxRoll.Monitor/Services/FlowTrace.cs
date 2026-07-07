using System;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// [Flow] 顯示資料流跡唯一出口：咽喉點各一行（按鈕/設定→接線→首幀→顯示），落 Logs\trace-*.log。
    /// 每行帶「時間戳 + 執行緒 ID」：多執行緒交錯時可按執行緒拆回各自序列（執行緒內順序絕對正確）、
    /// 用時間戳看跨執行緒先後。Trace 有全域鎖 → 落檔順序＝實際呼叫順序，不會半行交錯。
    /// 驗證（EVT）寫「偏序」不寫「全序」：因果關係（StartGrab 先於 firstFrame）+ 完整性（每台首幀必出現），
    /// 不規定跨執行緒的交錯順序（本來就不定）。
    /// </summary>
    public static class FlowTrace
    {
        public static void Log(string msg) =>
            System.Diagnostics.Trace.WriteLine(
                $"[Flow] {DateTime.Now:HH:mm:ss.fff} T{System.Threading.Thread.CurrentThread.ManagedThreadId,2} {msg}");
    }
}
