using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace TanukiCv.Controls
{
    /// <summary>
    /// 8-bit 灰階 byte[] → Format8bppIndexed Bitmap 的唯一來源（灰階調色盤一次算好，每幀重算是純浪費）。
    /// 原分散三份（LiveDisplayView / ThumbStrip / 範例 BuildGrayBitmap）→ 收斂於此。
    /// <paramref name="flip"/> 上下翻轉：線掃相機由下往上拍 / GPU resize 輸出 bottom-up 時用（只是來源列反向、零額外成本）。
    /// </summary>
    public static class GrayBitmap
    {
        private static readonly Color[] _grayEntries = BuildGrayEntries();
        private static Color[] BuildGrayEntries() { var e = new Color[256]; for (int i = 0; i < 256; i++) e[i] = Color.FromArgb(i, i, i); return e; }

        /// <summary>灰階 bytes → 8bppIndexed Bitmap；data 長度需 ≥ w*h。</summary>
        public static Bitmap From(byte[] data, int w, int h, bool flip = false)
        {
            var bmp = new Bitmap(w, h, PixelFormat.Format8bppIndexed);
            ColorPalette pal = bmp.Palette;                  // ColorPalette 不可跨 Bitmap 共用，但 entry 內容用快取免重算
            for (int i = 0; i < 256; i++) pal.Entries[i] = _grayEntries[i];
            bmp.Palette = pal;

            BitmapData bd = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
            try
            {
                for (int y = 0; y < h; y++)
                {
                    int srcRow = flip ? (h - 1 - y) : y;
                    Marshal.Copy(data, srcRow * w, IntPtr.Add(bd.Scan0, y * bd.Stride), w);
                }
            }
            finally { bmp.UnlockBits(bd); }
            return bmp;
        }
    }
}
