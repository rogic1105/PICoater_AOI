using TanukiCv.Controls;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using AniloxRoll.Monitor.Core.Services;
using TanukiCv.Core; // MergeLayout：合圖佈局 + 重疊中點分界單一來源（純算法，app → sdk 合法）

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// 將同一台相機的多張影像垂直拼接（依時間由上到下）。
    /// JPEG（_raw.jpg）直接載入；BMP 優先使用 bmpLoader（TanukiCv_FastReadBMP + GPU resize），
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
        /// bmpLoader：BMP 檔案的快速載入器（TanukiCv_FastReadBMP + GPU resize）；
        ///            為 null 時退回 GDI+ 路徑（慢）。
        /// useProcessed：true 時 _raw.jpg 改讀 _proc_c/r.jpg（若存在）；BMP 路徑不受影響。
        /// ridgeDirection：處理圖方向 "v"（預設）或 "h"。
        /// 若只有一張則直接回傳該張的 Bitmap。全部失敗則回傳 null。
        /// </summary>
        public static Bitmap StitchCamera(
            IList<string> sortedPaths,
            int bmpResizeScale = InspectionEngineConfig.DefaultSaveResizeScale,
            Func<string, Bitmap> bmpLoader = null,
            bool useProcessed = false,
            string ridgeDirection = "v",
            bool useThumbnail = false)
        {
            // slot-based：保留位置（null/缺檔 = 黑布幀），讓掉偵那格補黑、各台高度一致對齊（不縮短不錯位）。
            // sortedPaths 可含 null（呼叫端對齊參考時間軸後，缺的位置塞 null）；no-null 時行為與舊版一致。
            var slots = new Bitmap[sortedPaths.Count];
            int refW = 0, refH = 0, realCount = 0;

            for (int i = 0; i < sortedPaths.Count; i++)
            {
                string path = sortedPaths[i];
                if (string.IsNullOrEmpty(path) || !CaptureArchiveStore.Exists(path)) continue;
                try
                {
                    var bmp = LoadCameraImage(
                        path, bmpResizeScale, bmpLoader,
                        useProcessed, ridgeDirection, useThumbnail);
                    if (bmp == null) continue;
                    slots[i] = bmp; realCount++;
                    if (refW == 0) { refW = bmp.Width; refH = bmp.Height; }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[GrabImageStitcher] {System.IO.Path.GetFileName(path)}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            if (realCount == 0) return null;
            if (slots.Length == 1 && slots[0] != null)
            {
                return slots[0];
            }

            var result = BitmapPool.Rent(refW, refH * slots.Length);
            try
            {
                using (var g = Graphics.FromImage(result))
                {
                    g.Clear(Color.Black);   // null slot = 黑布占位（掉偵那格）
                    g.InterpolationMode = InterpolationMode.NearestNeighbor;
                    g.PixelOffsetMode   = PixelOffsetMode.Half;
                    for (int i = 0; i < slots.Length; i++)
                        if (slots[i] != null)
                            g.DrawImage(slots[i],
                                new Rectangle(0, i * refH, refW, refH),
                                new Rectangle(0, 0, slots[i].Width, slots[i].Height),
                                GraphicsUnit.Pixel);
                }
            }
            finally
            {
                foreach (var img in slots) img?.Dispose();
            }

            return result;
        }

        /// <summary>
        /// Reads only enough JPEG metadata to predict the stitched bitmap size. This follows the
        /// same slot and source-selection rules as <see cref="StitchCamera"/>, without decoding all
        /// frames or allocating the stitched bitmap.
        /// </summary>
        internal static bool TryGetStitchedSize(
            IList<string> sortedPaths, bool useProcessed, string ridgeDirection,
            out int width, out int height)
        {
            width = 0;
            height = 0;
            if (sortedPaths == null || sortedPaths.Count == 0) return false;

            for (int i = 0; i < sortedPaths.Count; i++)
            {
                string loadPath = ResolveLoadPath(sortedPaths[i], useProcessed, ridgeDirection);
                if (string.IsNullOrEmpty(loadPath) || !CaptureArchiveStore.Exists(loadPath)) continue;
                try
                {
                    if (CaptureArchiveStore.IsVirtualPath(loadPath) &&
                        CaptureArchiveStore.TryReadJpegSize(
                            loadPath, out int archiveWidth, out int archiveHeight))
                    {
                        width = archiveWidth;
                        height = checked(archiveHeight * sortedPaths.Count);
                        return width > 0 && height > 0;
                    }

                    byte[] bytes = CaptureArchiveStore.ReadAllBytes(loadPath);
                    if (bytes == null) continue;
                    using (var stream = new MemoryStream(bytes, false))
                    using (var image = Image.FromStream(stream, false, false))
                    {
                        width = image.Width;
                        height = checked(image.Height * sortedPaths.Count);
                        return width > 0 && height > 0;
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine(
                        $"[GrabImageStitcher.Size] {Path.GetFileName(loadPath)}: {ex.GetType().Name}: {ex.Message}");
                }
            }
            return false;
        }

        // 註：舊「水平合圖 MergeHorizontal」已移除（死碼）—— 顯示用合圖統一走 sdk ImageDisplayView.BuildMerge
        // （MergeLayout + MergeAll 黑占位，與 live/瀑布/曲線同一單一來源）。本檔只留 StitchCamera（垂直拼）+ LoadCameraImage。

        internal static Bitmap LoadCameraImage(string path, int bmpResizeScale,
            Func<string, Bitmap> bmpLoader, bool useProcessed, string ridgeDirection = "v",
            bool useThumbnail = false)
        {
            if (!CaptureFileNaming.IsRawJpg(path))
                return null;

            string loadPath = ResolveLoadPath(path, useProcessed, ridgeDirection);
            if (useThumbnail)
            {
                string thumbnailPath = CaptureFileNaming.ResolveThumbnailJpg(
                    path, useProcessed, ridgeDirection);
                if (string.IsNullOrEmpty(thumbnailPath)) return null;
                loadPath = thumbnailPath;
            }
            byte[] bytes = CaptureArchiveStore.ReadAllBytes(loadPath);
            if (bytes == null) return null;
            using (var ms = new MemoryStream(bytes, false))
            using (var decoded = new Bitmap(ms))
                return new Bitmap(decoded);
        }

        private static string ResolveLoadPath(
            string path, bool useProcessed, string ridgeDirection)
        {
            return CaptureFileNaming.ResolveDisplayJpg(
                path, useProcessed, ridgeDirection);
        }

    }
}
