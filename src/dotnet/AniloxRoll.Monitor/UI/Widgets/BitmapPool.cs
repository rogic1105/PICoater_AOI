using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// 大尺寸 Bitmap 物件池，減少 LOH 分配與 Gen2 GC 壓力。
    /// 保留最近歸還的 1 張 Bitmap，若下次 Rent 尺寸相同則重用（清零後返回）。
    /// 僅用於 GrabImageStitcher / ReviewStitchCoordinator 的合成圖（數十~數百 MB），
    /// 不用於小型中間 Bitmap（如 JPEG 重編碼、縮圖）。
    /// Thread-safe：lock 保護單一 slot。
    /// </summary>
    public static class BitmapPool
    {
        private static readonly object _lock = new object();
        private static Bitmap _cached;

        /// <summary>
        /// 取得指定尺寸的 Bitmap（Format32bppArgb）。
        /// 若池中有同尺寸的 Bitmap 則重用（已清零），否則 new。
        /// </summary>
        public static Bitmap Rent(int width, int height)
        {
            lock (_lock)
            {
                if (_cached != null &&
                    _cached.Width == width &&
                    _cached.Height == height)
                {
                    var bmp = _cached;
                    _cached = null;
                    ClearBitmap(bmp);
                    return bmp;
                }
            }
            return new Bitmap(width, height, PixelFormat.Format32bppArgb);
        }

        /// <summary>
        /// 歸還 Bitmap 到池中。若池已有不同尺寸的 Bitmap，則舊的會被 Dispose。
        /// 傳入 null 不會出錯。
        /// </summary>
        public static void Return(Bitmap bmp)
        {
            if (bmp == null) return;
            lock (_lock)
            {
                var old = _cached;
                _cached = bmp;
                if (old != null && old != bmp)
                    old.Dispose();
            }
        }

        private static void ClearBitmap(Bitmap bmp)
        {
            using (var g = Graphics.FromImage(bmp))
                g.Clear(Color.Transparent);
        }
    }
}
