namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// 像素 ↔ 實際 mm 座標換算的單一公式來源（純靜態、無 MIL/WinForms 依賴）。
    /// 即時（LiveCameraManager）與回顧（CanvasInteractionHelper）共用同一組公式，
    /// 差別只在 mmPerPx / startMm 的來源不同（合圖參考值 / 各相機 OPS、回顧另含 _imageScaleFactor）。
    /// </summary>
    public static class PixelMmMapper
    {
        /// <summary>像素 → 實際 mm：startMm + pixel × mmPerPx</summary>
        public static double PixelToMm(double pixel, double startMm, double mmPerPx) => startMm + pixel * mmPerPx;

        /// <summary>實際 mm → 像素：(mm - startMm) / mmPerPx</summary>
        public static double MmToPixel(double mm, double startMm, double mmPerPx) => (mm - startMm) / mmPerPx;

        /// <summary>顯示放大倍率：zoom × screenMmPerPx / mmPerPx</summary>
        public static double PhysicalMagnification(double zoom, double screenMmPerPx, double mmPerPx) => zoom * screenMmPerPx / mmPerPx;

        /// <summary>1:1（實際大小）所需 zoom：mmPerPx / screenMmPerPx</summary>
        public static double OneToOneZoom(double mmPerPx, double screenMmPerPx) => mmPerPx / screenMmPerPx;
    }
}
