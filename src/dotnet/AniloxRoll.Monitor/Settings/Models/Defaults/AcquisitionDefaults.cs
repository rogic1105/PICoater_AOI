using System.Linq;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// AcquisitionSettings 預設值集中定義（同 InspectionDefaults 風格）。
    /// 原本 3001/149/3001 在 AcquisitionSettings.cs 寫 5 處（屬性初始 + Validate 兩處），
    /// 收斂到唯一來源避免改一處漏改其他。
    /// </summary>
    internal static class AcquisitionDefaults
    {
        public const int    CamCount       = 7;
        public const int    GrabHeight     = 3001;
        public const double ExposureTimeUs = 50.0;     // PICoater 機台實測
        public const double LineRateHz     = 3001.0;

        /// <summary>★ grab buffer 高度上限（px）。**0 = 自動依板載算**（MilCamera 開機算 autoMax，跨 grabber 通用）；
        /// >0 = 手動覆寫（特殊情況才設）。防「grab buffer 撞 grabber 板載記憶體 → stall」。
        ///
        /// 自動算（通用物理公式，換 grabber 自動適應、不需改 code）：
        ///   autoMax = 板載總量 × Safety% ÷ 同板台數 ÷ (影像寬 × ring buffer 數 ÷ 1MB)
        /// 現役 Radient eV-CL 實測（2026-06-22）：每板 1024MB（4 bank×256MB，同板共用）；
        ///   物理上界 perRow = 寬16384 × 2 ring ÷ 1MB ≈ 0.03125（實測有效約 0.0233，物理較保守）；
        ///   板0 4 台、Safety 80% → 自動算 ≈ 6500（保守，實測 9000×4 仍剩 17% 故安全有餘）。
        ///   單台勿接近 bank size（~15000→234MB/buffer 撞 bank 非線性）。要榨更高可手動設此值（如 9000）。</summary>
        public const int    MaxGrabBufferHeightPx = 0;        // 0 = 自動

        /// <summary>板載安全使用率（%）。autoMax 用 `板載 × 此% ÷ 台數 ÷ 每台每行`。預設 80（留 20% 給 MIL 內部/突發）。</summary>
        public const int    GrabBufferSafetyPercent = 80;

        /// <summary>每張板（grabber System）板載總量（MB）。autoMax 算高度上限用。**不查 MIL（MsysInquire 會害第一台
        /// 相機 stall）→ 用此設定常數**。現役 Radient eV-CL 每板 1024MB（實測）。換 grabber 量一次改此值（開機 log
        /// 的板載可用 ≈ 此值），或改 acquisition-settings.json。</summary>
        public const int    BoardTotalMemMB = 1024;

        public static int[]    NewGrabHeightArray()   => Enumerable.Repeat(GrabHeight,     CamCount).ToArray();
        public static double[] NewExposureTimeArray() => Enumerable.Repeat(ExposureTimeUs, CamCount).ToArray();
        public static double[] NewLineRateArray()     => Enumerable.Repeat(LineRateHz,     CamCount).ToArray();
    }
}
