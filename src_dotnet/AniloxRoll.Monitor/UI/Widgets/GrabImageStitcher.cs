using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// 將同一台相機的多張影像垂直拼接（依時間由上到下）。
    /// JPEG（_raw.jpg）直接載入；BMP 先以 bmpResizeScale 縮小再拼接。
    /// </summary>
    public static class GrabImageStitcher
    {
        /// <summary>
        /// 拼接同一相機的多張影像（依 sortedPaths 順序，第一張在最上方）。
        /// 若只有一張則直接回傳該張的 Bitmap。
        /// 全部失敗則回傳 null。
        /// </summary>
        public static Bitmap StitchCamera(
            IList<string> sortedPaths,
            int bmpResizeScale = InspectionEngineConfig.DefaultSaveResizeScale)
        {
            var images = new List<Bitmap>();
            int refW = 0, refH = 0;

            foreach (string path in sortedPaths)
            {
                if (!File.Exists(path)) continue;
                try
                {
                    var bmp = LoadCameraImage(path, bmpResizeScale);
                    if (bmp == null) continue;
                    images.Add(bmp);
                    if (refW == 0) { refW = bmp.Width; refH = bmp.Height; }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[GrabImageStitcher] {System.IO.Path.GetFileName(path)}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            if (images.Count == 0) return null;
            if (images.Count == 1) return images[0]; // 單張直接回傳，不需拼接

            var result = new Bitmap(refW, refH * images.Count, PixelFormat.Format32bppArgb);
            try
            {
                using (var g = Graphics.FromImage(result))
                {
                    g.InterpolationMode = InterpolationMode.NearestNeighbor;
                    g.PixelOffsetMode   = PixelOffsetMode.Half;
                    for (int i = 0; i < images.Count; i++)
                        g.DrawImage(images[i],
                            new Rectangle(0, i * refH, refW, refH),
                            new Rectangle(0, 0, images[i].Width, images[i].Height),
                            GraphicsUnit.Pixel);
                }
            }
            finally
            {
                foreach (var img in images) img.Dispose();
            }

            return result;
        }

        private static Bitmap LoadCameraImage(string path, int bmpResizeScale)
        {
            if (path.EndsWith("_raw.jpg", StringComparison.OrdinalIgnoreCase))
            {
                byte[] bytes = File.ReadAllBytes(path);
                using (var ms = new MemoryStream(bytes))
                    return new Bitmap(ms);
            }

            // BMP：縮小 bmpResizeScale 倍
            using (var orig = new Bitmap(path))
            {
                int w = Math.Max(1, orig.Width  / bmpResizeScale);
                int h = Math.Max(1, orig.Height / bmpResizeScale);
                var resized = new Bitmap(w, h, PixelFormat.Format32bppArgb);
                using (var g = Graphics.FromImage(resized))
                {
                    g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    g.PixelOffsetMode   = PixelOffsetMode.Half;
                    g.DrawImage(orig,
                        new Rectangle(0, 0, w, h),
                        new Rectangle(0, 0, orig.Width, orig.Height),
                        GraphicsUnit.Pixel);
                }
                return resized;
            }
        }
    }
}
