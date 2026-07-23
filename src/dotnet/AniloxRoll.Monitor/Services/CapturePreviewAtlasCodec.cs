using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Text;

namespace AniloxRoll.Monitor.Core.Services
{
    internal sealed class CapturePreviewAtlasData : IDisposable
    {
        public Bitmap[] CameraImages { get; set; }
        public int[] SourceWidths { get; set; }
        public int[] SourceHeights { get; set; }
        public int AtlasWidth { get; set; }
        public int AtlasHeight { get; set; }
        public double PixelScaleRatio { get; set; }

        public void Dispose()
        {
            if (CameraImages == null) return;
            for (int i = 0; i < CameraImages.Length; i++)
                CameraImages[i]?.Dispose();
            CameraImages = null;
        }
    }

    /// <summary>
    /// Builds one rebuildable preview atlas per grab and display variant. Each camera is first
    /// stacked on the shared frame timeline, then all camera strips are packed horizontally and
    /// scaled once. Review decodes one JPEG and splits it in memory; full JPEG records remain truth.
    /// </summary>
    internal static class CapturePreviewAtlasCodec
    {
        internal const string AtlasBaseName = "__preview_atlas__";
        private const int PayloadVersion = 1;
        private const int MaxCameraSlots = 32;
        private const long JpegQuality = 75L;
        private static readonly byte[] PayloadMagic =
            { (byte)'P', (byte)'A', (byte)'T', (byte)'L' };

        private sealed class AtlasSlot
        {
            public int CameraId;
            public int X;
            public int Y;
            public int Width;
            public int Height;
            public int SourceWidth;
            public int SourceHeight;
            public int FrameWidth;
            public int FrameHeight;
            public IList<string> AlignedPaths;
        }

        internal static CaptureArchivePreviewAtlasResult AddToRoot(
            string captureRoot,
            int maxWidth,
            int maxHeight,
            bool replaceExisting,
            Action<string> progress)
        {
            var result = new CaptureArchivePreviewAtlasResult();
            if (string.IsNullOrWhiteSpace(captureRoot) ||
                !Directory.Exists(captureRoot) ||
                maxWidth <= 0 ||
                maxHeight <= 0)
                return result;

            string[] archives = Directory.GetFiles(
                captureRoot, "*" + CaptureArchiveStore.Extension,
                SearchOption.AllDirectories);
            Array.Sort(archives, StringComparer.OrdinalIgnoreCase);
            for (int archiveIndex = 0; archiveIndex < archives.Length; archiveIndex++)
            {
                string archivePath = archives[archiveIndex];
                string grabId = Path.GetFileNameWithoutExtension(archivePath);
                try
                {
                    Dictionary<int, List<string>> grouped =
                        GroupRawPaths(archivePath);
                    if (grouped.Count == 0)
                    {
                        result.FailedArchiveCount++;
                        continue;
                    }

                    FrameAlignmentResult alignment =
                        FrameTickIndex.ResolveAlignment(grouped);
                    var assets = new List<CaptureArchiveAsset>(3);
                    AddVariant(
                        archivePath, alignment, false, "c",
                        CaptureAssetKind.PreviewAtlasRaw,
                        maxWidth, maxHeight, replaceExisting, assets, result);
                    AddVariant(
                        archivePath, alignment, true, "c",
                        CaptureAssetKind.PreviewAtlasColumn,
                        maxWidth, maxHeight, replaceExisting, assets, result);
                    AddVariant(
                        archivePath, alignment, true, "r",
                        CaptureAssetKind.PreviewAtlasRow,
                        maxWidth, maxHeight, replaceExisting, assets, result);

                    if (assets.Count == 0) continue;
                    result.AtlasBytes += CaptureArchiveStore.AppendFrame(
                        archivePath, grabId, AtlasBaseName, 0, 0, assets);
                    result.AtlasCount += assets.Count;
                    result.ArchiveCount++;
                    progress?.Invoke(archivePath);
                }
                catch (Exception ex)
                {
                    result.FailedArchiveCount++;
                    Trace.WriteLine(
                        $"[CapturePreviewAtlas] {grabId}: " +
                        $"{ex.GetType().Name}: {ex.Message}");
                }
            }
            return result;
        }

        internal static bool TryLoad(
            IDictionary<int, List<string>> groupedPaths,
            int cameraCount,
            bool useProcessed,
            string ridgeDirection,
            out CapturePreviewAtlasData data)
        {
            data = null;
            if (groupedPaths == null || cameraCount <= 0) return false;

            string archivePath = FindSingleArchive(groupedPaths);
            if (string.IsNullOrEmpty(archivePath)) return false;
            CaptureAssetKind kind = SelectKind(useProcessed, ridgeDirection);
            byte[] payload = CaptureArchiveStore.ReadAsset(
                archivePath, AtlasBaseName, kind);
            return TryDecode(payload, cameraCount, out data);
        }

        private static void AddVariant(
            string archivePath,
            FrameAlignmentResult alignment,
            bool useProcessed,
            string ridgeDirection,
            CaptureAssetKind kind,
            int maxWidth,
            int maxHeight,
            bool replaceExisting,
            List<CaptureArchiveAsset> assets,
            CaptureArchivePreviewAtlasResult result)
        {
            if (CaptureArchiveStore.ContainsAsset(
                archivePath, AtlasBaseName, kind) &&
                !replaceExisting)
            {
                result.SkippedAtlasCount++;
                return;
            }

            byte[] payload = BuildPayload(
                alignment, useProcessed, ridgeDirection,
                maxWidth, maxHeight);
            if (payload == null || payload.Length == 0)
                throw new InvalidDataException(
                    "Unable to build preview atlas " + kind + ".");
            assets.Add(new CaptureArchiveAsset { Kind = kind, Data = payload });
        }

        private static byte[] BuildPayload(
            FrameAlignmentResult alignment,
            bool useProcessed,
            string ridgeDirection,
            int maxWidth,
            int maxHeight)
        {
            if (alignment?.ByCamera == null || alignment.ByCamera.Count == 0)
                return null;

            List<AtlasSlot> slots = CreateSourceSlots(
                alignment.ByCamera, useProcessed, ridgeDirection);
            if (slots.Count == 0) return null;

            long totalSourceWidth = 0;
            int maxSourceHeight = 0;
            for (int i = 0; i < slots.Count; i++)
            {
                totalSourceWidth += slots[i].SourceWidth;
                maxSourceHeight = Math.Max(
                    maxSourceHeight, slots[i].SourceHeight);
            }
            if (totalSourceWidth <= 0 || maxSourceHeight <= 0)
                return null;

            double scale = Math.Min(
                1.0,
                Math.Min(
                    maxWidth / (double)totalSourceWidth,
                    maxHeight / (double)maxSourceHeight));
            if (scale <= 0 || double.IsNaN(scale) || double.IsInfinity(scale))
                return null;

            int atlasWidth = 0;
            int atlasHeight = 0;
            for (int i = 0; i < slots.Count; i++)
            {
                AtlasSlot slot = slots[i];
                slot.X = atlasWidth;
                slot.Y = 0;
                slot.Width = Math.Max(
                    1, (int)Math.Floor(slot.SourceWidth * scale));
                slot.Height = Math.Max(
                    1, (int)Math.Floor(slot.SourceHeight * scale));
                atlasWidth += slot.Width;
                atlasHeight = Math.Max(atlasHeight, slot.Height);
            }

            byte[] jpeg;
            using (var atlas = new Bitmap(
                atlasWidth, atlasHeight, PixelFormat.Format24bppRgb))
            {
                using (Graphics graphics = Graphics.FromImage(atlas))
                {
                    graphics.Clear(Color.Black);
                    graphics.CompositingMode = CompositingMode.SourceCopy;
                    graphics.CompositingQuality =
                        CompositingQuality.HighSpeed;
                    graphics.InterpolationMode =
                        InterpolationMode.HighQualityBilinear;
                    graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
                    for (int i = 0; i < slots.Count; i++)
                        DrawSlot(
                            graphics, slots[i], useProcessed,
                            ridgeDirection, scale);
                }
                jpeg = EncodeJpeg(atlas);
            }
            if (jpeg == null || jpeg.Length == 0) return null;

            using (var stream = new MemoryStream())
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
            {
                writer.Write(PayloadMagic);
                writer.Write(PayloadVersion);
                writer.Write(atlasWidth);
                writer.Write(atlasHeight);
                writer.Write(1.0 / scale);
                writer.Write(slots.Count);
                for (int i = 0; i < slots.Count; i++)
                {
                    AtlasSlot slot = slots[i];
                    writer.Write(slot.CameraId);
                    writer.Write(slot.X);
                    writer.Write(slot.Y);
                    writer.Write(slot.Width);
                    writer.Write(slot.Height);
                    writer.Write(slot.SourceWidth);
                    writer.Write(slot.SourceHeight);
                }
                writer.Write(jpeg.Length);
                writer.Write(jpeg);
                writer.Flush();
                return stream.ToArray();
            }
        }

        private static List<AtlasSlot> CreateSourceSlots(
            IDictionary<int, List<string>> alignedByCamera,
            bool useProcessed,
            string ridgeDirection)
        {
            var cameraIds = new List<int>(alignedByCamera.Keys);
            cameraIds.Sort();
            var slots = new List<AtlasSlot>(cameraIds.Count);
            for (int i = 0; i < cameraIds.Count; i++)
            {
                int cameraId = cameraIds[i];
                IList<string> paths = alignedByCamera[cameraId];
                if (paths == null || paths.Count == 0) continue;
                if (!TryGetFrameSize(
                    paths, useProcessed, ridgeDirection,
                    out int frameWidth, out int frameHeight))
                    continue;
                slots.Add(new AtlasSlot
                {
                    CameraId = cameraId,
                    SourceWidth = frameWidth,
                    SourceHeight = checked(frameHeight * paths.Count),
                    FrameWidth = frameWidth,
                    FrameHeight = frameHeight,
                    AlignedPaths = paths
                });
            }
            return slots;
        }

        private static bool TryGetFrameSize(
            IList<string> paths,
            bool useProcessed,
            string ridgeDirection,
            out int width,
            out int height)
        {
            width = 0;
            height = 0;
            for (int i = 0; i < paths.Count; i++)
            {
                string source = CaptureFileNaming.ResolveDisplayJpg(
                    paths[i], useProcessed, ridgeDirection);
                if (string.IsNullOrEmpty(source) ||
                    !CaptureArchiveStore.Exists(source))
                    continue;
                if (CaptureArchiveStore.TryReadJpegSize(
                    source, out width, out height))
                    return width > 0 && height > 0;
            }
            return false;
        }

        private static void DrawSlot(
            Graphics graphics,
            AtlasSlot slot,
            bool useProcessed,
            string ridgeDirection,
            double scale)
        {
            for (int frameIndex = 0;
                frameIndex < slot.AlignedPaths.Count;
                frameIndex++)
            {
                string sourcePath = CaptureFileNaming.ResolveDisplayJpg(
                    slot.AlignedPaths[frameIndex],
                    useProcessed, ridgeDirection);
                if (string.IsNullOrEmpty(sourcePath) ||
                    !CaptureArchiveStore.Exists(sourcePath))
                    continue;

                byte[] sourceBytes =
                    CaptureArchiveStore.ReadAllBytes(sourcePath);
                if (sourceBytes == null || sourceBytes.Length == 0)
                    continue;
                int top = slot.Y + (int)Math.Round(
                    frameIndex * slot.FrameHeight * scale);
                int bottom = slot.Y + (int)Math.Round(
                    (frameIndex + 1) * slot.FrameHeight * scale);
                top = Math.Min(slot.Y + slot.Height - 1, top);
                bottom = Math.Min(slot.Y + slot.Height, Math.Max(top + 1, bottom));
                using (var stream = new MemoryStream(sourceBytes, false))
                using (var source = Image.FromStream(stream, false, false))
                {
                    graphics.DrawImage(
                        source,
                        new Rectangle(
                            slot.X, top, slot.Width, bottom - top),
                        new Rectangle(
                            0, 0, source.Width, source.Height),
                        GraphicsUnit.Pixel);
                }
            }
        }

        private static bool TryDecode(
            byte[] payload,
            int cameraCount,
            out CapturePreviewAtlasData data)
        {
            data = null;
            if (payload == null || payload.Length < 32) return false;
            var cameraImages = new Bitmap[cameraCount];
            try
            {
                using (var stream = new MemoryStream(payload, false))
                using (var reader = new BinaryReader(stream, Encoding.UTF8, true))
                {
                    if (!Matches(reader.ReadBytes(PayloadMagic.Length), PayloadMagic) ||
                        reader.ReadInt32() != PayloadVersion)
                        return false;
                    int atlasWidth = reader.ReadInt32();
                    int atlasHeight = reader.ReadInt32();
                    double ratio = reader.ReadDouble();
                    int slotCount = reader.ReadInt32();
                    if (atlasWidth <= 0 || atlasHeight <= 0 ||
                        atlasWidth > 16384 || atlasHeight > 16384 ||
                        ratio < 1.0 || double.IsNaN(ratio) ||
                        double.IsInfinity(ratio) ||
                        slotCount <= 0 || slotCount > MaxCameraSlots)
                        return false;

                    var slots = new List<AtlasSlot>(slotCount);
                    for (int i = 0; i < slotCount; i++)
                    {
                        var slot = new AtlasSlot
                        {
                            CameraId = reader.ReadInt32(),
                            X = reader.ReadInt32(),
                            Y = reader.ReadInt32(),
                            Width = reader.ReadInt32(),
                            Height = reader.ReadInt32(),
                            SourceWidth = reader.ReadInt32(),
                            SourceHeight = reader.ReadInt32()
                        };
                        if (slot.CameraId <= 0 ||
                            slot.CameraId > cameraCount ||
                            slot.X < 0 || slot.Y < 0 ||
                            slot.Width <= 0 || slot.Height <= 0 ||
                            slot.X + slot.Width > atlasWidth ||
                            slot.Y + slot.Height > atlasHeight ||
                            slot.SourceWidth <= 0 || slot.SourceHeight <= 0)
                            return false;
                        slots.Add(slot);
                    }

                    int jpegLength = reader.ReadInt32();
                    if (jpegLength <= 0 ||
                        jpegLength != stream.Length - stream.Position)
                        return false;
                    byte[] jpeg = reader.ReadBytes(jpegLength);
                    using (var jpegStream = new MemoryStream(jpeg, false))
                    using (var atlas = new Bitmap(jpegStream))
                    {
                        if (atlas.Width != atlasWidth ||
                            atlas.Height != atlasHeight)
                            return false;
                        var sourceWidths = new int[cameraCount];
                        var sourceHeights = new int[cameraCount];
                        for (int i = 0; i < slots.Count; i++)
                        {
                            AtlasSlot slot = slots[i];
                            int index = slot.CameraId - 1;
                            var camera = new Bitmap(
                                slot.Width, slot.Height,
                                PixelFormat.Format24bppRgb);
                            using (Graphics graphics = Graphics.FromImage(camera))
                            {
                                graphics.DrawImage(
                                    atlas,
                                    new Rectangle(
                                        0, 0, slot.Width, slot.Height),
                                    new Rectangle(
                                        slot.X, slot.Y,
                                        slot.Width, slot.Height),
                                    GraphicsUnit.Pixel);
                            }
                            cameraImages[index] = camera;
                            sourceWidths[index] = slot.SourceWidth;
                            sourceHeights[index] = slot.SourceHeight;
                        }
                        data = new CapturePreviewAtlasData
                        {
                            CameraImages = cameraImages,
                            SourceWidths = sourceWidths,
                            SourceHeights = sourceHeights,
                            AtlasWidth = atlasWidth,
                            AtlasHeight = atlasHeight,
                            PixelScaleRatio = ratio
                        };
                        cameraImages = null;
                        return true;
                    }
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[CapturePreviewAtlas] decode failed: " +
                    $"{ex.GetType().Name}: {ex.Message}");
                return false;
            }
            finally
            {
                if (cameraImages != null)
                    for (int i = 0; i < cameraImages.Length; i++)
                        cameraImages[i]?.Dispose();
            }
        }

        private static Dictionary<int, List<string>> GroupRawPaths(
            string archivePath)
        {
            var grouped = new Dictionary<int, List<string>>();
            List<string> rawPaths =
                CaptureArchiveStore.ListAllVirtualRawPaths(archivePath);
            for (int i = 0; i < rawPaths.Count; i++)
            {
                string baseName =
                    CaptureArchiveStore.GetVirtualBaseName(rawPaths[i]);
                if (!InspectionCsvReader.TryExtractCameraId(
                    baseName, out int cameraId))
                    continue;
                if (!grouped.TryGetValue(
                    cameraId, out List<string> paths))
                    grouped[cameraId] = paths = new List<string>();
                paths.Add(rawPaths[i]);
            }
            foreach (List<string> paths in grouped.Values)
                paths.Sort(StringComparer.Ordinal);
            return grouped;
        }

        private static string FindSingleArchive(
            IDictionary<int, List<string>> groupedPaths)
        {
            string archivePath = null;
            foreach (KeyValuePair<int, List<string>> camera in groupedPaths)
            {
                if (camera.Value == null) continue;
                for (int i = 0; i < camera.Value.Count; i++)
                {
                    string rawPath = camera.Value[i];
                    if (!CaptureArchiveStore.TryGetArchivePath(
                        rawPath, out string current))
                        return null;
                    if (archivePath == null)
                        archivePath = current;
                    else if (!string.Equals(
                        archivePath, current,
                        StringComparison.OrdinalIgnoreCase))
                        return null;
                }
            }
            return archivePath;
        }

        private static CaptureAssetKind SelectKind(
            bool useProcessed, string ridgeDirection)
        {
            if (!useProcessed) return CaptureAssetKind.PreviewAtlasRaw;
            return ridgeDirection == "r" || ridgeDirection == "h"
                ? CaptureAssetKind.PreviewAtlasRow
                : CaptureAssetKind.PreviewAtlasColumn;
        }

        private static byte[] EncodeJpeg(Bitmap bitmap)
        {
            using (var output = new MemoryStream())
            {
                ImageCodecInfo codec = FindJpegCodec();
                if (codec == null)
                    bitmap.Save(output, ImageFormat.Jpeg);
                else
                {
                    using (var parameters = new EncoderParameters(1))
                    {
                        parameters.Param[0] = new EncoderParameter(
                            System.Drawing.Imaging.Encoder.Quality,
                            JpegQuality);
                        bitmap.Save(output, codec, parameters);
                    }
                }
                return output.ToArray();
            }
        }

        private static ImageCodecInfo FindJpegCodec()
        {
            ImageCodecInfo[] codecs = ImageCodecInfo.GetImageEncoders();
            for (int i = 0; i < codecs.Length; i++)
                if (codecs[i].FormatID == ImageFormat.Jpeg.Guid)
                    return codecs[i];
            return null;
        }

        private static bool Matches(byte[] actual, byte[] expected)
        {
            if (actual == null || actual.Length != expected.Length)
                return false;
            for (int i = 0; i < actual.Length; i++)
                if (actual[i] != expected[i])
                    return false;
            return true;
        }
    }
}
