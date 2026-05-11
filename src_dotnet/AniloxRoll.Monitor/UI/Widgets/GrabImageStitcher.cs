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
        /// useProcessed：true 時 _raw.jpg 改讀 _proc_v/h.jpg（若存在）；BMP 路徑不受影響。
        /// ridgeDirection：處理圖方向 "v"（預設）或 "h"。
        /// 若只有一張則直接回傳該張的 Bitmap。全部失敗則回傳 null。
        /// </summary>
        public static Bitmap StitchCamera(
            IList<string> sortedPaths,
            int bmpResizeScale = InspectionEngineConfig.DefaultSaveResizeScale,
            Func<string, Bitmap> bmpLoader = null,
            bool useProcessed = false,
            string ridgeDirection = "v")
        {
            var images = new List<Bitmap>();
            int refW = 0, refH = 0;

            foreach (string path in sortedPaths)
            {
                if (!File.Exists(path)) continue;
                try
                {
                    var bmp = LoadCameraImage(path, bmpResizeScale, bmpLoader, useProcessed, ridgeDirection);
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
            if (images.Count == 1)
            {
                // 取像時序需上下翻轉顯示
                images[0].RotateFlip(RotateFlipType.RotateNoneFlipY);
                return images[0];
            }

            var result = BitmapPool.Rent(refW, refH * images.Count);
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

            // 合成完一次翻轉（取像時序需上下翻轉顯示，與 Period 模式一致）
            result.RotateFlip(RotateFlipType.RotateNoneFlipY);
            return result;
        }

        /// <summary>
        /// 將多台相機的影像水平合併為一張全域圖。
        /// 重疊規則：相鄰相機重疊區域各取一半（中點分界），無重疊則留黑。
        /// opsUm[i]：第 i 台相機的解析度（um/pixel）。
        /// posMm[i]：第 i 台相機的起始位置（mm）。
        /// scaleFactor：影像已縮小的倍率（用於將 pixel 換算回實際 mm）。
        /// </summary>
        public static Bitmap MergeHorizontal(
            Bitmap[] cameraImages, double[] opsUm, double[] posMm, int scaleFactor)
        {
            if (cameraImages == null) return null;

            // Pass 1：ALL slots（含空缺）計算全域 mm 範圍；空缺槽以 MaxWidth 為標準寬度
            double globalMinMm = double.MaxValue;
            double globalMaxMm = double.MinValue;
            double refOpsPxMm = 0;

            for (int i = 0; i < cameraImages.Length; i++)
            {
                double ops = (i < opsUm.Length) ? opsUm[i] : opsUm[0];
                double pos = (i < posMm.Length) ? posMm[i] : 0;
                double storedWidthPx = (cameraImages[i] != null)
                    ? cameraImages[i].Width
                    : InspectionEngineConfig.MaxWidth / (double)scaleFactor;
                double widthMm = storedWidthPx * ops * scaleFactor / 1000.0;
                double endMm = pos + widthMm;
                if (pos < globalMinMm) globalMinMm = pos;
                if (endMm > globalMaxMm) globalMaxMm = endMm;
                if (refOpsPxMm == 0) refOpsPxMm = ops * scaleFactor / 1000.0;
            }

            // Pass 2：只收集有影像的相機（供後續繪製）
            var validCams = new List<(Bitmap bmp, double posMm)>();
            int maxH = 0;

            for (int i = 0; i < cameraImages.Length; i++)
            {
                if (cameraImages[i] == null) continue;
                double pos = (i < posMm.Length) ? posMm[i] : 0;
                if (cameraImages[i].Height > maxH) maxH = cameraImages[i].Height;
                validCams.Add((cameraImages[i], pos));
            }

            if (refOpsPxMm == 0 || validCams.Count == 0) return null;

            int totalW = (int)Math.Ceiling((globalMaxMm - globalMinMm) / refOpsPxMm);
            if (totalW <= 0 || maxH <= 0) return null;

            var entries = new List<(Bitmap bmp, int xOffset)>(validCams.Count);
            for (int i = 0; i < validCams.Count; i++)
            {
                int xOff = (int)Math.Round((validCams[i].posMm - globalMinMm) / refOpsPxMm);
                entries.Add((validCams[i].bmp, xOff));
            }

            // 按 xOffset 排序，計算每台相機的有效繪製範圍
            entries.Sort((a, b) => a.xOffset.CompareTo(b.xOffset));
            int n = entries.Count;
            var drawLeft  = new int[n];
            var drawRight = new int[n];
            for (int i = 0; i < n; i++)
            {
                drawLeft[i]  = entries[i].xOffset;
                drawRight[i] = entries[i].xOffset + entries[i].bmp.Width;
            }

            // 重疊區域各取一半（中點分界）
            for (int i = 0; i < n - 1; i++)
            {
                int rightEdge = entries[i].xOffset + entries[i].bmp.Width;
                int leftEdge  = entries[i + 1].xOffset;
                int overlap   = rightEdge - leftEdge;
                if (overlap > 0)
                {
                    int mid = leftEdge + overlap / 2;
                    drawRight[i]     = Math.Min(drawRight[i], mid);
                    drawLeft[i + 1]  = Math.Max(drawLeft[i + 1], mid);
                }
                // overlap <= 0：間隙留黑（ARGB 預設透明/黑）
            }

            var result = BitmapPool.Rent(totalW, maxH);
            using (var g = Graphics.FromImage(result))
            {
                g.InterpolationMode = InterpolationMode.NearestNeighbor;
                g.PixelOffsetMode = PixelOffsetMode.Half;
                for (int i = 0; i < n; i++)
                {
                    var (bmp, xOff) = entries[i];
                    int srcLeft  = drawLeft[i] - xOff;
                    int srcWidth = drawRight[i] - drawLeft[i];
                    if (srcWidth <= 0) continue;
                    g.DrawImage(bmp,
                        new Rectangle(drawLeft[i], 0, srcWidth, bmp.Height),
                        new Rectangle(srcLeft, 0, srcWidth, bmp.Height),
                        GraphicsUnit.Pixel);
                }
            }

            return result;
        }

        internal static Bitmap LoadCameraImage(string path, int bmpResizeScale,
            Func<string, Bitmap> bmpLoader, bool useProcessed, string ridgeDirection = "v")
        {
            if (!path.EndsWith("_raw.jpg", StringComparison.OrdinalIgnoreCase))
                return null;

            string loadPath = path;
            if (useProcessed)
            {
                string baseName = path.Substring(0, path.Length - "_raw.jpg".Length);
                string procSuffix = (ridgeDirection == "h") ? "_proc_h.jpg" : "_proc_v.jpg";
                string procPath = baseName + procSuffix;
                if (!File.Exists(procPath)) procPath = baseName + "_proc.jpg";
                if (File.Exists(procPath)) loadPath = procPath;
            }
            byte[] bytes = File.ReadAllBytes(loadPath);
            using (var ms = new MemoryStream(bytes))
                return new Bitmap(ms);
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
