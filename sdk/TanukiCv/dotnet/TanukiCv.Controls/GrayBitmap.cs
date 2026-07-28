using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace TanukiCv.Controls
{
    /// <summary>8-bit intensity 的顯示調色盤；只改顯示顏色，不改原始 byte 值。</summary>
    public enum IntensityColorMap
    {
        Grayscale = 0,
        HeatmapCold = 1,
        HeatmapWarm = 2,
        HeatmapBlueYellowRed = 3,
        HeatmapGreen = 4
    }

    /// <summary>
    /// 8-bit 灰階 byte[] → Format8bppIndexed Bitmap 的唯一來源（灰階調色盤一次算好，每幀重算是純浪費）。
    /// 原分散三份（ImageDisplayView / ThumbStrip / 範例 BuildGrayBitmap）→ 收斂於此。
    /// <paramref name="flip"/> 上下翻轉：線掃相機由下往上拍 / GPU resize 輸出 bottom-up 時用（只是來源列反向、零額外成本）。
    /// </summary>
    public static class GrayBitmap
    {
        private static readonly Color[] _grayEntries = BuildGrayEntries();
        private static readonly Color[] _coldEntries = BuildColdEntries();
        private static readonly Color[] _warmEntries = BuildWarmEntries();
        private static readonly Color[] _blueYellowRedEntries = BuildBlueYellowRedEntries();
        private static readonly Color[] _greenEntries = BuildGreenEntries();
        private static readonly Func<Color, int> _redIntensity = color => color.R;
        private static readonly Func<Color, int> _blueIntensity = color => color.B;
        private static readonly Func<Color, int> _greenIntensity = color => color.G;
        private static readonly Func<Color, int> _blueYellowRedIntensity = DecodeBlueYellowRedIntensity;
        private static Color[] BuildGrayEntries() { var e = new Color[256]; for (int i = 0; i < 256; i++) e[i] = Color.FromArgb(i, i, i); return e; }

        private static Color[] BuildColdEntries()
        {
            var entries = new Color[256];
            for (int intensity = 0; intensity < entries.Length; intensity++)
            {
                // 固定 0..255 尺度：黑→深藍→藍→青藍→白。B 恒等於原 intensity，
                // 讓畫布的熱力圖亮度取樣在偽彩色下仍回報真實數值。
                int green = intensity <= 96 ? 0 : (intensity - 96) * 255 / 159;
                int red = intensity <= 224 ? 0 : (intensity - 224) * 255 / 31;
                entries[intensity] = Color.FromArgb(red, green, intensity);
            }
            return entries;
        }

        private static Color[] BuildWarmEntries()
        {
            var entries = new Color[256];
            for (int intensity = 0; intensity < entries.Length; intensity++)
            {
                // 黑→深紅→橘→黃→白；R 恒等於原 intensity。
                int green = intensity <= 96 ? 0 : (intensity - 96) * 255 / 159;
                int blue = intensity <= 224 ? 0 : (intensity - 224) * 255 / 31;
                entries[intensity] = Color.FromArgb(intensity, green, blue);
            }
            return entries;
        }

        private static Color[] BuildBlueYellowRedEntries()
        {
            var entries = new Color[256];
            entries[0] = Color.Black;
            for (int intensity = 1; intensity < entries.Length; intensity++)
            {
                // 明確三段：0=黑、85=藍、170=黃、255=紅。每段 85 階，反算亮度無歧義。
                if (intensity <= 85)
                {
                    entries[intensity] = Color.FromArgb(0, 0, intensity * 3);
                }
                else if (intensity <= 170)
                {
                    int step = intensity - 85;
                    entries[intensity] = Color.FromArgb(step * 3, step * 3, 255 - step * 3);
                }
                else
                {
                    int step = intensity - 170;
                    entries[intensity] = Color.FromArgb(255, 255 - step * 3, 0);
                }
            }
            return entries;
        }

        private static Color[] BuildGreenEntries()
        {
            var entries = new Color[256];
            for (int intensity = 0; intensity < entries.Length; intensity++)
            {
                // 固定 0..255 尺度：黑→綠→白；G 恒等於原 intensity。
                int redBlue = intensity <= 224 ? 0 : (intensity - 224) * 255 / 31;
                entries[intensity] = Color.FromArgb(redBlue, intensity, redBlue);
            }
            return entries;
        }

        /// <summary>取得與調色盤配對的原始亮度解碼器，供畫布游標/狀態列共用。</summary>
        public static Func<Color, int> GetBrightnessSelector(IntensityColorMap colorMap)
        {
            if (colorMap == IntensityColorMap.HeatmapCold) return _blueIntensity;
            if (colorMap == IntensityColorMap.HeatmapBlueYellowRed) return _blueYellowRedIntensity;
            if (colorMap == IntensityColorMap.HeatmapGreen) return _greenIntensity;
            return _redIntensity;
        }

        private static int DecodeBlueYellowRedIntensity(Color color)
        {
            if (color.R == 0 && color.G == 0)
                return (color.B + 1) / 3;
            if (color.B > 0 || color.R < 255)
                return 85 + (((color.R + color.G) / 2) + 1) / 3;
            return 170 + ((255 - color.G) + 1) / 3;
        }

        /// <summary>灰階 bytes → 8bppIndexed Bitmap；data 長度需 ≥ w*h。</summary>
        public static Bitmap From(byte[] data, int w, int h, bool flip = false,
            IntensityColorMap colorMap = IntensityColorMap.Grayscale)
        {
            var bmp = new Bitmap(w, h, PixelFormat.Format8bppIndexed);
            ColorPalette pal = bmp.Palette;                  // ColorPalette 不可跨 Bitmap 共用，但 entry 內容用快取免重算
            Color[] entries;
            switch (colorMap)
            {
                case IntensityColorMap.HeatmapCold: entries = _coldEntries; break;
                case IntensityColorMap.HeatmapWarm: entries = _warmEntries; break;
                case IntensityColorMap.HeatmapBlueYellowRed: entries = _blueYellowRedEntries; break;
                case IntensityColorMap.HeatmapGreen: entries = _greenEntries; break;
                default: entries = _grayEntries; break;
            }
            for (int i = 0; i < 256; i++) pal.Entries[i] = entries[i];
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
