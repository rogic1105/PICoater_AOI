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

        /// <summary>★ grab 高度硬上限（px）。各台高度一律 clamp 到 min(設定高度, 此值)。**目標＝避免 stall**
        /// （尤其 grab 中調相機參數/拉高度時）。
        ///
        /// ⚠ **此值只能 hardcode、實測決定**（2026-06-24 定案）：
        ///   • 現役 Radient eV-CL（QB，板載 1GB / **4 acquisition path**）+ Linea LA-CM-16K05A（影像寬 16384）：
        ///     **grab 進行中**把高度從小往大拉，拉到 **~12062 就 stall**（每次 12062~12065 略有浮動）。
        ///     （開機「初配」可到 ~14303，但連續取像時 on-board 還要兼當 PCIe latency 緩衝 → 動態拉的上限較低＝12062，
        ///      官方 BoardSpecificNotes/radient_ev-cl/Minimum_latency… + Grabbing_large_images.htm。以動態上限為準才不 stall。）
        ///   • **板載是 4 個 path 各自獨立配到 ~12062（即 12062×4）**，故 **per-camera 固定上限、不分同板台數**
        ///     （6/23 一度按台數往下減＝過度保守、偏離實測，已移除）。
        ///   • 要更高只能改架構（Host 大 buffer + child-buffer 分段擷取，官方 Grabbing_large_images.htm），非本專案需求。
        ///
        /// 取 **12000**（= 實測動態 stall 邊界 ~12062 保守取整、留 ~60 餘裕）。**換相機（寬）或 grabber（path/板載）必須
        /// 重新實測**（測法：grab 中高度從小往大拉，找剛好開始 stall 的值，再保守取整）。詳見 sdk/MIL/docs/grab-height-param-stall.md
        /// + AGENTS.md / $modify-acquisition skill。</summary>
        public const int    MaxGrabHeightPx = 12000;

        public static int[]    NewGrabHeightArray()   => Enumerable.Repeat(GrabHeight,     CamCount).ToArray();
        public static double[] NewExposureTimeArray() => Enumerable.Repeat(ExposureTimeUs, CamCount).ToArray();
        public static double[] NewLineRateArray()     => Enumerable.Repeat(LineRateHz,     CamCount).ToArray();
    }
}
