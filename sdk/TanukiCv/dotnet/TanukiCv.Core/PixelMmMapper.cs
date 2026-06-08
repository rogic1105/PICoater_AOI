namespace TanukiCv.Core
{
    /// <summary>
    /// 像素 ↔ 實際 mm 座標換算 + 實體放大倍率的單一公式來源（純靜態、無 MIL/WinForms 依賴）。
    /// 即時 / 回顧 / 範例共用同一組公式，差別只在 mmPerPx / startMm / screenMmPerPx 的來源不同。
    /// 原本住在 AniloxRoll.Monitor.Core.Services，2026-06 收進 TanukiCv.Core 當跨專案唯一來源。
    /// </summary>
    public static class PixelMmMapper
    {
        /// <summary>像素 → 實際 mm：startMm + pixel × mmPerPx</summary>
        public static double PixelToMm(double pixel, double startMm, double mmPerPx) => startMm + pixel * mmPerPx;

        /// <summary>實際 mm → 像素：(mm - startMm) / mmPerPx</summary>
        public static double MmToPixel(double mm, double startMm, double mmPerPx) => (mm - startMm) / mmPerPx;

        /// <summary>顯示放大倍率（實體倍率）：zoom × screenMmPerPx / mmPerPx</summary>
        public static double PhysicalMagnification(double zoom, double screenMmPerPx, double mmPerPx) => zoom * screenMmPerPx / mmPerPx;

        /// <summary>1:1（實際大小）所需 zoom：mmPerPx / screenMmPerPx</summary>
        public static double OneToOneZoom(double mmPerPx, double screenMmPerPx) => mmPerPx / screenMmPerPx;
    }
}
