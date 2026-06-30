using System;
using System.Collections.Generic;

namespace AniloxRoll.Monitor.UI.Managers
{
    /// <summary>
    /// 相機 stall（取像卡死）偵測 —— **純邏輯、零依賴**（無 MIL / WinForms）：
    /// 判據＝<c>M_PROCESS_FRAME_COUNT</c> 有沒有前進，**非 FPS 門檻**（低線掃 + 高高度時合法 FPS 極低，
    /// 固定門檻會把「慢但正常」誤判）。真 stall＝幀數凍住不動；慢速 grab＝幀數仍慢慢加（不誤判）。
    ///
    /// 偵測窗自動依「預期幀週期＝高度/線掃」拉長：高速 ~2s 偵到、低速等久一點（仍偵得到，只是慢）。
    /// 自己擁有 per-camera 累計狀態（<see cref="Update"/> 餵幀數序列 → 判 stall），故可**單獨單元測**，
    /// 不依賴 MIL/UI（從 LiveCameraManager.CameraStatusTimer_Tick 提取出的職責；2026-06-26 重構示範第一刀）。
    /// </summary>
    internal sealed class CameraStallDetector
    {
        private const int BaseTicks = 4;          // 基準窗（4×500ms＝2s，避開重啟暫態）
        private const double PeriodFactor = 1.5;  // 額外等「預期幀週期 × 此倍數」才判（容忍合法慢速抖動）
        private const int TickMs = 500;           // 呼叫端 timer 間隔（與 CameraStatusTimer Interval 一致）

        private readonly Dictionary<int, int> _stallTicks = new Dictionary<int, int>();
        private readonly Dictionary<int, long> _lastFrameCount = new Dictionary<int, long>();

        /// <summary>餵本 tick 的累計幀數 + 預期 FPS（=線掃/高度，0=未知）→ 回 true 表示「已判定 stall」
        /// （幀數凍住超過偵測窗）。幀數有任何變化（前進，或重啟歸零）＝grab 活著 → 重置累計、回 false。</summary>
        public bool Update(int camId, long frameCount, double expFps)
        {
            long lastFc = _lastFrameCount.TryGetValue(camId, out var lv) ? lv : -1;
            _lastFrameCount[camId] = frameCount;

            bool advanced = (lastFc < 0) || (frameCount != lastFc);
            if (advanced) { _stallTicks[camId] = 0; return false; }

            int t = (_stallTicks.TryGetValue(camId, out var v) ? v : 0) + 1;
            _stallTicks[camId] = t;

            int needed = BaseTicks;
            if (expFps > 0)
            {
                double framePeriodMs = 1000.0 / expFps;
                needed = BaseTicks + (int)Math.Ceiling(framePeriodMs * PeriodFactor / TickMs);
            }
            return t >= needed;   // 幀數凍住超過窗＝真卡死
        }

        /// <summary>相機斷線 / 未在 grab 等非取像狀態 → 重置該台累計（避免暫態誤判）。</summary>
        public void Reset(int camId) => _stallTicks[camId] = 0;
    }
}
