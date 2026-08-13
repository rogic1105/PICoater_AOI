using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;

namespace AniloxRoll.Monitor.Core.Services
{
    internal sealed class CaptureArchiveThumbnailResult
    {
        public int ArchiveCount { get; set; }
        public int FrameCount { get; set; }
        public int ThumbnailCount { get; set; }
        public long ThumbnailBytes { get; set; }
        public int SkippedThumbnailCount { get; set; }
        public int FailedFrameCount { get; set; }
    }

    /// <summary>
    /// One-time maintenance for legacy per-frame thumbnails. Runtime archive IO stays in
    /// CaptureArchiveStore; new captures use preview atlases instead of this fallback format.
    /// </summary>
    internal static class CaptureArchiveThumbnailMaintenance
    {
        private sealed class ThumbnailFrame
        {
            public string BaseName;
            public int CameraId;
            public long FrameTicks;
            public List<CaptureArchiveAsset> Assets;
        }

        public static CaptureArchiveThumbnailResult AddThumbnails(
            string captureRoot,
            int targetWidth,
            Action<string> progress = null)
        {
            var result = new CaptureArchiveThumbnailResult();
            if (string.IsNullOrWhiteSpace(captureRoot) ||
                !Directory.Exists(captureRoot) ||
                targetWidth <= 0)
                return result;

            string[] archives = Directory.GetFiles(
                captureRoot, "*" + CaptureArchiveStore.Extension, SearchOption.AllDirectories);
            Array.Sort(archives, StringComparer.OrdinalIgnoreCase);
            foreach (string archivePath in archives)
            {
                bool archiveChanged = false;
                string grabId = Path.GetFileNameWithoutExtension(archivePath);
                List<string> rawPaths = CaptureArchiveStore.ListAllVirtualRawPaths(archivePath);
                var pending = new List<ThumbnailFrame>(rawPaths.Count);
                for (int i = 0; i < rawPaths.Count; i++)
                {
                    string rawPath = rawPaths[i];
                    string baseName = CaptureArchiveStore.GetVirtualBaseName(rawPath);
                    if (string.IsNullOrEmpty(baseName) ||
                        !InspectionCsvReader.TryExtractCameraId(baseName, out int cameraId))
                    {
                        result.FailedFrameCount++;
                        continue;
                    }

                    var assets = new List<CaptureArchiveAsset>(3);
                    AddThumbnailAsset(
                        assets, rawPath,
                        CaptureFileNaming.StripRawJpg(rawPath) + CaptureFileNaming.ThumbRawJpg,
                        CaptureAssetKind.ThumbnailRawJpeg,
                        targetWidth, result);
                    string basePath = CaptureFileNaming.StripRawJpg(rawPath);
                    AddThumbnailAsset(
                        assets, basePath + CaptureFileNaming.ProcC,
                        basePath + CaptureFileNaming.ThumbProcC,
                        CaptureAssetKind.ThumbnailColumnJpeg,
                        targetWidth, result);
                    AddThumbnailAsset(
                        assets, basePath + CaptureFileNaming.ProcR,
                        basePath + CaptureFileNaming.ThumbProcR,
                        CaptureAssetKind.ThumbnailRowJpeg,
                        targetWidth, result);

                    if (assets.Count == 0) continue;
                    CaptureArchiveStore.TryGetFrameTicks(rawPath, out long frameTicks);
                    pending.Add(new ThumbnailFrame
                    {
                        BaseName = baseName,
                        CameraId = cameraId,
                        FrameTicks = frameTicks,
                        Assets = assets
                    });
                }

                for (int i = 0; i < pending.Count; i++)
                {
                    ThumbnailFrame frame = pending[i];
                    try
                    {
                        result.ThumbnailBytes += CaptureArchiveStore.AppendFrame(
                            archivePath, grabId, frame.BaseName, frame.CameraId,
                            frame.FrameTicks, frame.Assets);
                        result.ThumbnailCount += frame.Assets.Count;
                        result.FrameCount++;
                        archiveChanged = true;
                    }
                    catch (Exception ex)
                    {
                        result.FailedFrameCount++;
                        Trace.WriteLine(
                            $"[CaptureArchive.Thumbnail] {frame.BaseName}: " +
                            $"{ex.GetType().Name}: {ex.Message}");
                    }
                }
                if (archiveChanged)
                {
                    result.ArchiveCount++;
                    progress?.Invoke(archivePath);
                }
            }
            return result;
        }

        private static void AddThumbnailAsset(
            List<CaptureArchiveAsset> assets,
            string sourcePath,
            string thumbnailPath,
            CaptureAssetKind thumbnailKind,
            int targetWidth,
            CaptureArchiveThumbnailResult result)
        {
            if (!CaptureArchiveStore.Exists(sourcePath)) return;
            if (CaptureArchiveStore.Exists(thumbnailPath))
            {
                result.SkippedThumbnailCount++;
                return;
            }

            byte[] source = CaptureArchiveStore.ReadAllBytes(sourcePath);
            byte[] thumbnail = CreateJpegThumbnail(source, targetWidth);
            if (thumbnail == null || thumbnail.Length == 0)
            {
                result.FailedFrameCount++;
                return;
            }
            assets.Add(new CaptureArchiveAsset
            {
                Kind = thumbnailKind,
                Data = thumbnail
            });
        }

        private static byte[] CreateJpegThumbnail(byte[] jpeg, int targetWidth)
        {
            if (jpeg == null || jpeg.Length == 0) return null;
            try
            {
                using (var input = new MemoryStream(jpeg, false))
                using (var source = Image.FromStream(input, false, false))
                {
                    int width = Math.Min(targetWidth, source.Width);
                    int height = Math.Max(
                        1, (int)Math.Round(source.Height * width / (double)source.Width));
                    using (var resized = new Bitmap(width, height, PixelFormat.Format24bppRgb))
                    {
                        using (Graphics graphics = Graphics.FromImage(resized))
                        {
                            graphics.Clear(Color.Black);
                            graphics.InterpolationMode = InterpolationMode.HighQualityBilinear;
                            graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
                            graphics.DrawImage(source, 0, 0, width, height);
                        }
                        using (var output = new MemoryStream())
                        {
                            ImageCodecInfo codec = GetJpegCodec();
                            if (codec == null)
                                resized.Save(output, ImageFormat.Jpeg);
                            else
                            {
                                using (var parameters = new EncoderParameters(1))
                                {
                                    parameters.Param[0] = new EncoderParameter(
                                        System.Drawing.Imaging.Encoder.Quality, 75L);
                                    resized.Save(output, codec, parameters);
                                }
                            }
                            return output.ToArray();
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[CaptureArchive.Thumbnail] encode failed: " +
                    $"{ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        private static ImageCodecInfo GetJpegCodec()
        {
            ImageCodecInfo[] codecs = ImageCodecInfo.GetImageEncoders();
            for (int i = 0; i < codecs.Length; i++)
                if (codecs[i].FormatID == ImageFormat.Jpeg.Guid)
                    return codecs[i];
            return null;
        }
    }
}
