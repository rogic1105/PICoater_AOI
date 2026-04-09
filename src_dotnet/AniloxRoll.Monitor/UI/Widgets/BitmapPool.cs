using System;
using System.Collections.Concurrent;
using System.Drawing;
using System.Drawing.Imaging;

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// 輕量 Bitmap 池，減少 Review 大型影像重複配置造成的 LOH/GC 壓力。
    /// 以 (Width, Height) 為 key，同尺寸保留 1 張；總量超過 <see cref="MaxTotalBytes"/> 時清除全部。
    /// </summary>
    internal static class BitmapPool
    {
        private static readonly ConcurrentDictionary<(int w, int h), Bitmap> _pool
            = new ConcurrentDictionary<(int w, int h), Bitmap>();

        /// <summary>池中所有 Bitmap 的 ARGB 像素總位元組上限（預設 512 MB）。</summary>
        private const long MaxTotalBytes = 512L * 1024 * 1024;
        private static long _totalBytes;

        /// <summary>從池中取出或建立指定尺寸的 Bitmap（已清為透明）。</summary>
        public static Bitmap Rent(int width, int height, PixelFormat format = PixelFormat.Format32bppArgb)
        {
            var key = (width, height);
            if (_pool.TryRemove(key, out var bmp))
            {
                long bytes = (long)width * height * 4;
                System.Threading.Interlocked.Add(ref _totalBytes, -bytes);
                using (var g = Graphics.FromImage(bmp))
                    g.Clear(Color.Transparent);
                return bmp;
            }
            return new Bitmap(width, height, format);
        }

        /// <summary>歸還 Bitmap 至池中（同尺寸已有或超過總量上限時直接 Dispose）。</summary>
        public static void Return(Bitmap bmp)
        {
            if (bmp == null) return;
            var key = (bmp.Width, bmp.Height);
            long bytes = (long)bmp.Width * bmp.Height * 4;

            // 超過記憶體上限：清空池再嘗試放入
            if (System.Threading.Interlocked.Read(ref _totalBytes) + bytes > MaxTotalBytes)
                Flush();

            if (_pool.TryAdd(key, bmp))
            {
                System.Threading.Interlocked.Add(ref _totalBytes, bytes);
            }
            else
            {
                // 同尺寸已有一張，直接釋放
                bmp.Dispose();
            }
        }

        private static void Flush()
        {
            foreach (var key in _pool.Keys)
            {
                if (_pool.TryRemove(key, out var old))
                    old.Dispose();
            }
            System.Threading.Interlocked.Exchange(ref _totalBytes, 0);
        }
    }
}
