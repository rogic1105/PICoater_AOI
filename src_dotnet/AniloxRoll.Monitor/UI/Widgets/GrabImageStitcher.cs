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
    /// JPEG（_raw.jpg）直接載入；BMP 優先使用 bmpLoader（CoreCV_FastReadBMP + GPU resize），
    /// 縮小 bmpResizeScale 倍後 JPEG 95% 重編碼，對齊 _raw.jpg 的視覺品質。
    /// </summary>
    public static class GrabImageStitcher
    {
        // JPEG codec（靜態快取，避免每次呼叫 GetImageEncoders）
        private static readonly ImageCodecInfo _jpegCodec = FindJpegCodec();

        private static ImageCodecInfo FindJpegCodec()
        {
            foreach (var c in ImageCodecInfo.GetImageEncoders())
                if (c.FormatID == ImageFormat.Jpeg.Guid) return c;
            return null;
        }

        /// <summary>
        /// 拼接同一相機的多張影像（依 sortedPaths 順序，第一張在最上方）。
        /// bmpLoader：BMP 檔案的快速載入器（CoreCV_FastReadBMP + GPU resize）；
        ///            為 null 時退回 GDI+ 路徑（慢）。
        /// 若只有一張則直接回傳該張的 Bitmap。全部失敗則回傳 null。
        /// </summary>
        public static Bitmap StitchCamera(
            IList<string> sortedPaths,
            int bmpResizeScale = InspectionEngineConfig.DefaultSaveResizeScale,
            Func<string, Bitmap> bmpLoader = null)
        {
            var images = new List<Bitmap>();
            int refW = 0, refH = 0;

            foreach (string path in sortedPaths)
            {
                if (!File.Exists(path)) continue;
                try
                {
                    var bmp = LoadCameraImage(path, bmpResizeScale, bmpLoader);
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

        private static Bitmap LoadCameraImage(string path, int bmpResizeScale, Func<string, Bitmap> bmpLoader)
        {
            if (path.EndsWith("_raw.jpg", StringComparison.OrdinalIgnoreCase))
            {
                byte[] bytes = File.ReadAllBytes(path);
                using (var ms = new MemoryStream(bytes))
                    return new Bitmap(ms);
            }

            // BMP：優先用 bmpLoader（CoreCV_FastReadBMP + GPU resize），fallback GDI+
            Bitmap resized = bmpLoader != null
                ? bmpLoader(path)
                : LoadGdiBmpResized(path, bmpResizeScale);

            if (resized == null) return null;

            // 縮小後重新以 JPEG 95% 編碼再解碼，對齊 _raw.jpg 的視覺品質
            return ReencodeAsJpeg(resized, 95);
        }

        /// <summary>GDI+ fallback：當 bmpLoader 為 null 時使用。</summary>
        private static Bitmap LoadGdiBmpResized(string path, int scale)
        {
            using (var orig = new Bitmap(path))
            {
                int w = Math.Max(1, orig.Width  / scale);
                int h = Math.Max(1, orig.Height / scale);
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

        /// <summary>
        /// 將 Bitmap 以 JPEG quality 重新編碼再解碼。
        /// GDI+ JPEG encoder 不支援 8bpp indexed，先轉 24bpp RGB。
        /// </summary>
        private static Bitmap ReencodeAsJpeg(Bitmap src, int quality)
        {
            if (_jpegCodec == null) return src; // codec 找不到時直接回傳

            Bitmap toEncode = src;
            if (src.PixelFormat == PixelFormat.Format8bppIndexed)
            {
                // 8bpp → 24bpp（GDI+ JPEG 不接受 indexed format）
                toEncode = new Bitmap(src.Width, src.Height, PixelFormat.Format24bppRgb);
                using (var g = Graphics.FromImage(toEncode))
                    g.DrawImage(src, 0, 0);
                src.Dispose();
            }

            var encoderParams = new EncoderParameters(1);
            encoderParams.Param[0] = new EncoderParameter(Encoder.Quality, (long)quality);
            using (var ms = new MemoryStream())
            {
                toEncode.Save(ms, _jpegCodec, encoderParams);
                toEncode.Dispose();
                ms.Position = 0;
                return new Bitmap(ms);
            }
        }
    }
}
