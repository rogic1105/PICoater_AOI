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
    internal enum CaptureAssetKind : byte
    {
        RawJpeg = 1,
        ProcessedColumnJpeg = 2,
        ProcessedRowJpeg = 3,
        MeanColumnCurve = 4,
        MaxColumnCurve = 5,
        MeanRowCurve = 6,
        MaxRowCurve = 7,
        ThumbnailRawJpeg = 8,
        ThumbnailColumnJpeg = 9,
        ThumbnailRowJpeg = 10,
        PreviewAtlasRaw = 11,
        PreviewAtlasColumn = 12,
        PreviewAtlasRow = 13
    }

    internal sealed class CaptureArchiveAsset
    {
        public CaptureAssetKind Kind { get; set; }
        public byte[] Data { get; set; }
    }

    internal sealed class CaptureArchiveConversionResult
    {
        public int ArchiveCount { get; set; }
        public int FrameCount { get; set; }
        public long PayloadBytes { get; set; }
        public int SkippedArchiveCount { get; set; }
        public int FailedArchiveCount { get; set; }
    }

    internal sealed class CaptureArchiveValidationResult
    {
        public int ArchiveCount { get; set; }
        public int RawFrameCount { get; set; }
        public int RecordCount { get; set; }
        public long PayloadBytes { get; set; }
        public int PreviewAtlasCount { get; set; }
        public int InvalidArchiveCount { get; set; }
        public int InvalidRecordCount { get; set; }
        public int PartialFileCount { get; set; }
    }

    internal sealed class CaptureArchiveThumbnailResult
    {
        public int ArchiveCount { get; set; }
        public int FrameCount { get; set; }
        public int ThumbnailCount { get; set; }
        public long ThumbnailBytes { get; set; }
        public int SkippedThumbnailCount { get; set; }
        public int FailedFrameCount { get; set; }
    }

    internal sealed class CaptureArchivePreviewAtlasResult
    {
        public int ArchiveCount { get; set; }
        public int AtlasCount { get; set; }
        public long AtlasBytes { get; set; }
        public int SkippedAtlasCount { get; set; }
        public int FailedArchiveCount { get; set; }
    }

    /// <summary>
    /// One-grab appendable capture container. JPEG and MCBF payloads remain independent records,
    /// so readers keep parallel decode and random curve access without thousands of file opens.
    /// A truncated final record is ignored; every earlier record remains readable.
    /// </summary>
    internal static class CaptureArchiveStore
    {
        public const string Extension = ".acap";
        private const string VirtualPrefix = "acap://";
        private const int FormatVersion = 1;
        private const int RecordVersion = 1;
        private const int MaxNameBytes = 1024;
        private const int MaxPayloadBytes = 512 * 1024 * 1024;

        private static readonly byte[] FileMagic =
            { (byte)'P', (byte)'I', (byte)'C', (byte)'A', (byte)'C', (byte)'A', (byte)'P', 0 };
        private static readonly byte[] RecordMagic =
            { (byte)'A', (byte)'R', (byte)'E', (byte)'C' };
        private static readonly object LockMapSync = new object();
        private static readonly Dictionary<string, object> PathLocks =
            new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
        private static readonly object IndexCacheSync = new object();
        private static readonly Dictionary<string, CachedIndex> IndexCache =
            new Dictionary<string, CachedIndex>(StringComparer.OrdinalIgnoreCase);

        private sealed class ArchiveEntry
        {
            public long PayloadOffset;
            public int PayloadLength;
            public uint Crc32;
            public long FrameTicks;
            public int CameraId;
        }

        private sealed class CachedIndex
        {
            public long FileLength;
            public long LastWriteTicks;
            public string GrabId;
            public Dictionary<string, ArchiveEntry> Entries;
        }

        private sealed class LegacyFrame
        {
            public string BaseName;
            public int CameraId;
            public long FrameTicks;
        }

        private sealed class ThumbnailFrame
        {
            public string BaseName;
            public int CameraId;
            public long FrameTicks;
            public List<CaptureArchiveAsset> Assets;
        }

        public static string CreateVirtualRawPath(string archivePath, string baseName)
        {
            return CreateVirtualBasePath(archivePath, baseName) + CaptureFileNaming.RawJpg;
        }

        public static string CreateVirtualBasePath(string archivePath, string baseName)
        {
            if (string.IsNullOrWhiteSpace(archivePath)) throw new ArgumentNullException(nameof(archivePath));
            if (string.IsNullOrWhiteSpace(baseName)) throw new ArgumentNullException(nameof(baseName));
            return VirtualPrefix + Uri.EscapeDataString(Path.GetFullPath(archivePath)) + "/" +
                Uri.EscapeDataString(baseName);
        }

        public static bool IsVirtualPath(string path)
        {
            return !string.IsNullOrEmpty(path) &&
                path.StartsWith(VirtualPrefix, StringComparison.OrdinalIgnoreCase);
        }

        public static string GetVirtualBaseName(string path)
        {
            if (!TryParseVirtualPath(path, out _, out string baseName, out _)) return null;
            return baseName;
        }

        internal static bool TryGetArchivePath(string virtualPath, out string archivePath)
        {
            archivePath = null;
            if (!TryParseVirtualPath(
                virtualPath, out string parsed, out _, out _))
                return false;
            archivePath = parsed;
            return true;
        }

        internal static bool ContainsAsset(
            string archivePath, string baseName, CaptureAssetKind kind)
        {
            CachedIndex index = GetIndex(archivePath);
            return index != null &&
                index.Entries.ContainsKey(EntryKey(baseName, kind));
        }

        public static bool Exists(string path)
        {
            if (!IsVirtualPath(path)) return File.Exists(path);
            if (!TryParseVirtualPath(path, out string archivePath, out string baseName, out CaptureAssetKind kind))
                return false;
            CachedIndex index = GetIndex(archivePath);
            return index != null && index.Entries.ContainsKey(EntryKey(baseName, kind));
        }

        public static byte[] ReadAllBytes(string path)
        {
            if (!IsVirtualPath(path))
                return string.IsNullOrEmpty(path) || !File.Exists(path) ? null : File.ReadAllBytes(path);
            if (!TryParseVirtualPath(path, out string archivePath, out string baseName, out CaptureAssetKind kind))
                return null;
            return ReadAsset(archivePath, baseName, kind);
        }

        public static bool TryReadJpegSize(string virtualPath, out int width, out int height)
        {
            width = 0;
            height = 0;
            if (!TryParseVirtualPath(
                virtualPath, out string archivePath, out string baseName, out CaptureAssetKind kind) ||
                (kind != CaptureAssetKind.RawJpeg &&
                 kind != CaptureAssetKind.ProcessedColumnJpeg &&
                 kind != CaptureAssetKind.ProcessedRowJpeg))
                return false;

            CachedIndex index = GetIndex(archivePath);
            if (index == null ||
                !index.Entries.TryGetValue(EntryKey(baseName, kind), out ArchiveEntry entry))
                return false;

            try
            {
                using (var stream = new FileStream(
                    archivePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite,
                    4096, FileOptions.RandomAccess))
                {
                    stream.Position = entry.PayloadOffset;
                    return TryParseJpegSize(
                        stream, entry.PayloadOffset + entry.PayloadLength,
                        out width, out height);
                }
            }
            catch
            {
                width = 0;
                height = 0;
                return false;
            }
        }

        public static bool TryGetFrameTicks(string virtualImagePath, out long frameTicks)
        {
            frameTicks = 0;
            if (!TryParseVirtualPath(
                virtualImagePath, out string archivePath, out string baseName, out CaptureAssetKind kind))
                return false;
            CachedIndex index = GetIndex(archivePath);
            if (index == null || !index.Entries.TryGetValue(EntryKey(baseName, kind), out ArchiveEntry entry))
                return false;
            frameTicks = entry.FrameTicks;
            return frameTicks > 0;
        }

        public static List<string> ListVirtualRawPaths(string archivePath, int cameraId)
        {
            var result = new List<string>();
            CachedIndex index = GetIndex(archivePath);
            if (index == null) return result;
            foreach (KeyValuePair<string, ArchiveEntry> item in index.Entries)
            {
                if (item.Value.CameraId != cameraId || !KeyHasKind(item.Key, CaptureAssetKind.RawJpeg))
                    continue;
                string baseName = KeyBaseName(item.Key);
                result.Add(CreateVirtualRawPath(archivePath, baseName));
            }
            result.Sort(StringComparer.Ordinal);
            return result;
        }

        private static bool TryParseJpegSize(
            Stream stream, long payloadEnd, out int width, out int height)
        {
            width = 0;
            height = 0;
            if (stream.Position + 2 > payloadEnd ||
                stream.ReadByte() != 0xff || stream.ReadByte() != 0xd8)
                return false;

            while (stream.Position + 4 <= payloadEnd)
            {
                int prefix;
                do { prefix = stream.ReadByte(); }
                while (prefix != 0xff && prefix >= 0 && stream.Position < payloadEnd);
                if (prefix != 0xff) return false;

                int marker;
                do { marker = stream.ReadByte(); }
                while (marker == 0xff && stream.Position < payloadEnd);
                if (marker < 0 || marker == 0xd9 || marker == 0xda) return false;
                if (marker == 0x00 || marker == 0x01 ||
                    marker >= 0xd0 && marker <= 0xd8)
                    continue;

                int lengthHigh = stream.ReadByte();
                int lengthLow = stream.ReadByte();
                if (lengthHigh < 0 || lengthLow < 0) return false;
                int segmentLength = (lengthHigh << 8) | lengthLow;
                if (segmentLength < 2 ||
                    stream.Position + segmentLength - 2 > payloadEnd)
                    return false;

                if (IsStartOfFrame(marker))
                {
                    if (segmentLength < 7 || stream.ReadByte() < 0) return false;
                    int heightHigh = stream.ReadByte();
                    int heightLow = stream.ReadByte();
                    int widthHigh = stream.ReadByte();
                    int widthLow = stream.ReadByte();
                    if (heightHigh < 0 || heightLow < 0 ||
                        widthHigh < 0 || widthLow < 0)
                        return false;
                    height = (heightHigh << 8) | heightLow;
                    width = (widthHigh << 8) | widthLow;
                    return width > 0 && height > 0;
                }

                stream.Position += segmentLength - 2;
            }
            return false;
        }

        private static bool IsStartOfFrame(int marker)
        {
            return marker >= 0xc0 && marker <= 0xcf &&
                marker != 0xc4 && marker != 0xc8 && marker != 0xcc;
        }

        public static List<string> ListAllVirtualRawPaths(string archivePath)
        {
            var result = new List<string>();
            CachedIndex index = GetIndex(archivePath);
            if (index == null) return result;
            foreach (KeyValuePair<string, ArchiveEntry> item in index.Entries)
            {
                if (!KeyHasKind(item.Key, CaptureAssetKind.RawJpeg)) continue;
                result.Add(CreateVirtualRawPath(archivePath, KeyBaseName(item.Key)));
            }
            result.Sort(StringComparer.Ordinal);
            return result;
        }

        public static long AppendFrame(
            string archivePath,
            string grabId,
            string baseName,
            int cameraId,
            long frameTicks,
            IList<CaptureArchiveAsset> assets)
        {
            if (string.IsNullOrWhiteSpace(archivePath)) throw new ArgumentNullException(nameof(archivePath));
            if (string.IsNullOrWhiteSpace(grabId)) throw new ArgumentNullException(nameof(grabId));
            if (string.IsNullOrWhiteSpace(baseName)) throw new ArgumentNullException(nameof(baseName));
            if (assets == null || assets.Count == 0) return 0;

            string fullPath = Path.GetFullPath(archivePath);
            Directory.CreateDirectory(Path.GetDirectoryName(fullPath));
            object pathLock = GetPathLock(fullPath);
            long written = 0;
            lock (pathLock)
            {
                using (var stream = new FileStream(
                    fullPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read,
                    128 * 1024, FileOptions.SequentialScan))
                using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
                {
                    if (stream.Length == 0)
                        WriteFileHeader(writer, grabId);
                    else
                        ValidateFileHeader(stream, grabId);
                    stream.Position = stream.Length;

                    for (int i = 0; i < assets.Count; i++)
                    {
                        CaptureArchiveAsset asset = assets[i];
                        if (asset == null || asset.Data == null || asset.Data.Length == 0) continue;
                        WriteRecord(writer, baseName, cameraId, frameTicks, asset.Kind, asset.Data);
                        written += asset.Data.Length;
                    }
                    writer.Flush();
                    stream.Flush(true);
                }
                InvalidateIndex(fullPath);
            }
            return written;
        }

        public static CaptureArchiveConversionResult ConvertLegacyRoot(
            string captureRoot,
            bool overwrite,
            Action<string> progress = null)
        {
            var result = new CaptureArchiveConversionResult();
            if (string.IsNullOrWhiteSpace(captureRoot) || !Directory.Exists(captureRoot))
                return result;

            string[] csvPaths = Directory.GetFiles(captureRoot, "*.csv", SearchOption.AllDirectories);
            Array.Sort(csvPaths, StringComparer.OrdinalIgnoreCase);
            foreach (string csvPath in csvPaths)
            {
                var byGrab = ReadLegacyFrames(csvPath);
                foreach (KeyValuePair<string, List<LegacyFrame>> grab in byGrab)
                {
                    if (grab.Value.Count == 0) continue;
                    string firstBase = grab.Value[0].BaseName;
                    if (!InspectionCsvReader.TryParseTimestamp(
                        Path.GetFileName(firstBase), out DateTime captureDate))
                        continue;
                    string archivePath = CaptureStoragePaths.GrabArchive(
                        captureRoot, captureDate, grab.Key);
                    if (File.Exists(archivePath) && !overwrite)
                    {
                        result.SkippedArchiveCount++;
                        continue;
                    }

                    string tempPath = archivePath + ".part-" + Guid.NewGuid().ToString("N");
                    try
                    {
                        Dictionary<string, long> ticks = LoadLegacyTicks(
                            CaptureStoragePaths.DateImageDir(captureRoot, captureDate));
                        Directory.CreateDirectory(Path.GetDirectoryName(archivePath));
                        using (var stream = new FileStream(
                            tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.None,
                            1024 * 1024, FileOptions.SequentialScan))
                        using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
                        {
                            WriteFileHeader(writer, grab.Key);
                            foreach (LegacyFrame frame in grab.Value)
                            {
                                if (ticks.TryGetValue(Path.GetFileName(frame.BaseName), out long frameTicks))
                                    frame.FrameTicks = frameTicks;
                                List<CaptureArchiveAsset> assets = LoadLegacyAssets(frame.BaseName);
                                for (int i = 0; i < assets.Count; i++)
                                {
                                    CaptureArchiveAsset asset = assets[i];
                                    WriteRecord(
                                        writer, Path.GetFileName(frame.BaseName), frame.CameraId,
                                        frame.FrameTicks, asset.Kind, asset.Data);
                                    result.PayloadBytes += asset.Data.Length;
                                }
                                result.FrameCount++;
                            }
                            writer.Flush();
                            stream.Flush(true);
                        }
                        if (File.Exists(archivePath)) File.Delete(archivePath);
                        File.Move(tempPath, archivePath);
                        InvalidateIndex(archivePath);
                        result.ArchiveCount++;
                        progress?.Invoke(archivePath);
                    }
                    catch (Exception ex)
                    {
                        result.FailedArchiveCount++;
                        Trace.WriteLine(
                            $"[CaptureArchive.Convert] {grab.Key}: {ex.GetType().Name}: {ex.Message}");
                        TryDelete(tempPath);
                    }
                }
            }
            return result;
        }

        public static CaptureArchiveValidationResult ValidateRoot(string captureRoot)
        {
            var result = new CaptureArchiveValidationResult();
            if (string.IsNullOrWhiteSpace(captureRoot) || !Directory.Exists(captureRoot))
                return result;

            result.PartialFileCount = Directory.GetFiles(
                captureRoot, "*.part-*", SearchOption.AllDirectories).Length;
            string[] archives = Directory.GetFiles(
                captureRoot, "*" + Extension, SearchOption.AllDirectories);
            Array.Sort(archives, StringComparer.OrdinalIgnoreCase);
            result.ArchiveCount = archives.Length;

            foreach (string archivePath in archives)
            {
                CachedIndex index = GetIndex(archivePath);
                if (index == null || index.Entries.Count == 0 ||
                    !string.Equals(
                        Path.GetFileNameWithoutExtension(archivePath),
                        index.GrabId,
                        StringComparison.Ordinal))
                {
                    result.InvalidArchiveCount++;
                    continue;
                }

                foreach (KeyValuePair<string, ArchiveEntry> item in index.Entries)
                {
                    result.RecordCount++;
                    result.PayloadBytes += item.Value.PayloadLength;
                    if (KeyHasKind(item.Key, CaptureAssetKind.RawJpeg))
                        result.RawFrameCount++;
                    if (!TryGetKeyKind(item.Key, out CaptureAssetKind kind))
                    {
                        result.InvalidRecordCount++;
                        continue;
                    }
                    if (kind == CaptureAssetKind.PreviewAtlasRaw ||
                        kind == CaptureAssetKind.PreviewAtlasColumn ||
                        kind == CaptureAssetKind.PreviewAtlasRow)
                        result.PreviewAtlasCount++;
                    if (ReadAsset(
                        archivePath, KeyBaseName(item.Key), kind) == null)
                        result.InvalidRecordCount++;
                }
            }
            return result;
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
                captureRoot, "*" + Extension, SearchOption.AllDirectories);
            Array.Sort(archives, StringComparer.OrdinalIgnoreCase);
            foreach (string archivePath in archives)
            {
                bool archiveChanged = false;
                string grabId = Path.GetFileNameWithoutExtension(archivePath);
                List<string> rawPaths = ListAllVirtualRawPaths(archivePath);
                var pending = new List<ThumbnailFrame>(rawPaths.Count);
                for (int i = 0; i < rawPaths.Count; i++)
                {
                    string rawPath = rawPaths[i];
                    string baseName = GetVirtualBaseName(rawPath);
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
                    TryGetFrameTicks(rawPath, out long frameTicks);
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
                        result.ThumbnailBytes += AppendFrame(
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

        public static CaptureArchivePreviewAtlasResult AddPreviewAtlases(
            string captureRoot,
            int maxWidth,
            int maxHeight,
            bool replaceExisting = false,
            Action<string> progress = null)
        {
            return CapturePreviewAtlasCodec.AddToRoot(
                captureRoot, maxWidth, maxHeight,
                replaceExisting, progress);
        }

        public static CaptureArchivePreviewAtlasResult AddPreviewAtlasesToArchive(
            string archivePath,
            int maxWidth,
            int maxHeight,
            bool replaceExisting = false,
            Action<string> progress = null)
        {
            return CapturePreviewAtlasCodec.AddToArchive(
                archivePath,
                maxWidth,
                maxHeight,
                replaceExisting,
                progress);
        }

        private static void AddThumbnailAsset(
            List<CaptureArchiveAsset> assets,
            string sourcePath,
            string thumbnailPath,
            CaptureAssetKind thumbnailKind,
            int targetWidth,
            CaptureArchiveThumbnailResult result)
        {
            if (!Exists(sourcePath)) return;
            if (Exists(thumbnailPath))
            {
                result.SkippedThumbnailCount++;
                return;
            }

            byte[] source = ReadAllBytes(sourcePath);
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

        private static Dictionary<string, List<LegacyFrame>> ReadLegacyFrames(string csvPath)
        {
            var byGrab = new Dictionary<string, List<LegacyFrame>>(StringComparer.Ordinal);
            try
            {
                using (var reader = InspectionCsvReader.OpenShared(csvPath))
                {
                    string line;
                    var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                    while ((line = reader.ReadLine()) != null)
                    {
                        if (!InspectionCsvReader.TryParseRecord(line, out var record) ||
                            !InspectionCsvReader.TryExtractCameraId(record.FileName, out int cameraId) ||
                            !InspectionCsvReader.TryParseTimestamp(record.FileName, out DateTime timestamp))
                            continue;
                        string identity = record.GrabId + "\n" + record.FileName;
                        if (!seen.Add(identity)) continue;
                        if (!byGrab.TryGetValue(record.GrabId, out List<LegacyFrame> frames))
                            byGrab[record.GrabId] = frames = new List<LegacyFrame>();
                        frames.Add(new LegacyFrame
                        {
                            BaseName = Path.Combine(
                                CaptureStoragePaths.DateImageDir(
                                    CaptureRootFromCsvPath(csvPath), timestamp), record.FileName),
                            CameraId = cameraId
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[CaptureArchive.ReadCsv] {csvPath}: {ex.GetType().Name}: {ex.Message}");
            }
            return byGrab;
        }

        private static string CaptureRootFromCsvPath(string csvPath)
        {
            DirectoryInfo month = Directory.GetParent(csvPath);
            DirectoryInfo year = month?.Parent;
            return year?.Parent?.FullName ?? string.Empty;
        }

        private static List<CaptureArchiveAsset> LoadLegacyAssets(string basePath)
        {
            var assets = new List<CaptureArchiveAsset>(7);
            AddLegacyAsset(assets, CaptureAssetKind.RawJpeg, basePath + CaptureFileNaming.RawJpg);
            AddLegacyAsset(assets, CaptureAssetKind.ProcessedColumnJpeg,
                ResolveLegacyExisting(basePath, CaptureFileNaming.ProcC,
                    CaptureFileNaming.ProcCPrevious, CaptureFileNaming.ProcLegacy));
            AddLegacyAsset(assets, CaptureAssetKind.ProcessedRowJpeg,
                ResolveLegacyExisting(basePath, CaptureFileNaming.ProcR,
                    CaptureFileNaming.ProcRPrevious, CaptureFileNaming.ProcLegacy));
            AddLegacyAsset(assets, CaptureAssetKind.MeanColumnCurve,
                ResolveLegacyExisting(basePath, CaptureFileNaming.MeanC,
                    CaptureFileNaming.MeanCPrevious, CaptureFileNaming.MeanCLegacy));
            AddLegacyAsset(assets, CaptureAssetKind.MaxColumnCurve,
                ResolveLegacyExisting(basePath, CaptureFileNaming.MaxC,
                    CaptureFileNaming.MaxCPrevious, CaptureFileNaming.MaxCLegacy));
            AddLegacyAsset(assets, CaptureAssetKind.MeanRowCurve,
                ResolveLegacyExisting(basePath, CaptureFileNaming.MeanR,
                    CaptureFileNaming.MeanRPrevious, CaptureFileNaming.MeanRLegacy));
            AddLegacyAsset(assets, CaptureAssetKind.MaxRowCurve,
                ResolveLegacyExisting(basePath, CaptureFileNaming.MaxR,
                    CaptureFileNaming.MaxRPrevious, CaptureFileNaming.MaxRLegacy));
            return assets;
        }

        private static void AddLegacyAsset(
            List<CaptureArchiveAsset> assets, CaptureAssetKind kind, string path)
        {
            if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;
            assets.Add(new CaptureArchiveAsset { Kind = kind, Data = File.ReadAllBytes(path) });
        }

        private static string ResolveLegacyExisting(
            string basePath, string current, string previous, string legacy)
        {
            string path = basePath + current;
            if (File.Exists(path)) return path;
            path = basePath + previous;
            return File.Exists(path) ? path : basePath + legacy;
        }

        private static Dictionary<string, long> LoadLegacyTicks(string dateDirectory)
        {
            var result = new Dictionary<string, long>(StringComparer.Ordinal);
            string path = Path.Combine(dateDirectory, "_ticks.csv");
            if (!File.Exists(path)) return result;
            try
            {
                foreach (string line in File.ReadLines(path))
                {
                    int comma = line.LastIndexOf(',');
                    if (comma <= 0) continue;
                    if (long.TryParse(line.Substring(comma + 1), out long ticks))
                        result[line.Substring(0, comma)] = ticks;
                }
            }
            catch { }
            return result;
        }

        internal static byte[] ReadAsset(
            string archivePath, string baseName, CaptureAssetKind kind)
        {
            CachedIndex index = GetIndex(archivePath);
            if (index == null || !index.Entries.TryGetValue(
                EntryKey(baseName, kind), out ArchiveEntry entry))
                return null;
            try
            {
                byte[] payload = new byte[entry.PayloadLength];
                using (var stream = new FileStream(
                    archivePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite,
                    128 * 1024, FileOptions.RandomAccess))
                {
                    stream.Position = entry.PayloadOffset;
                    if (!ReadExactly(stream, payload, 0, payload.Length)) return null;
                }
                return Crc32.Compute(payload) == entry.Crc32 ? payload : null;
            }
            catch
            {
                return null;
            }
        }

        private static CachedIndex GetIndex(string archivePath)
        {
            if (string.IsNullOrWhiteSpace(archivePath)) return null;
            FileInfo file;
            try { file = new FileInfo(archivePath); }
            catch { return null; }
            if (!file.Exists) return null;

            lock (IndexCacheSync)
            {
                if (IndexCache.TryGetValue(file.FullName, out CachedIndex cached) &&
                    cached.FileLength == file.Length &&
                    cached.LastWriteTicks == file.LastWriteTimeUtc.Ticks)
                    return cached;
            }

            CachedIndex built = BuildIndex(file.FullName);
            if (built == null) return null;
            lock (IndexCacheSync) IndexCache[file.FullName] = built;
            return built;
        }

        private static CachedIndex BuildIndex(string archivePath)
        {
            try
            {
                var entries = new Dictionary<string, ArchiveEntry>(StringComparer.Ordinal);
                string grabId;
                using (var stream = new FileStream(
                    archivePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite,
                    128 * 1024, FileOptions.SequentialScan))
                using (var reader = new BinaryReader(stream, Encoding.UTF8, true))
                {
                    grabId = ReadFileHeader(reader);
                    while (stream.Position + 36 <= stream.Length)
                    {
                        byte[] magic = reader.ReadBytes(4);
                        if (!Matches(magic, RecordMagic)) break;
                        if (reader.ReadInt32() != RecordVersion) break;
                        var kind = (CaptureAssetKind)reader.ReadByte();
                        reader.ReadBytes(3);
                        int cameraId = reader.ReadInt32();
                        long ticks = reader.ReadInt64();
                        int nameLength = reader.ReadInt32();
                        int payloadLength = reader.ReadInt32();
                        uint crc = reader.ReadUInt32();
                        if (nameLength <= 0 || nameLength > MaxNameBytes ||
                            payloadLength <= 0 || payloadLength > MaxPayloadBytes ||
                            stream.Position + nameLength + payloadLength > stream.Length)
                            break;
                        string baseName = Encoding.UTF8.GetString(reader.ReadBytes(nameLength));
                        long payloadOffset = stream.Position;
                        entries[EntryKey(baseName, kind)] = new ArchiveEntry
                        {
                            PayloadOffset = payloadOffset,
                            PayloadLength = payloadLength,
                            Crc32 = crc,
                            FrameTicks = ticks,
                            CameraId = cameraId
                        };
                        stream.Position = payloadOffset + payloadLength;
                    }
                }
                var file = new FileInfo(archivePath);
                return new CachedIndex
                {
                    FileLength = file.Length,
                    LastWriteTicks = file.LastWriteTimeUtc.Ticks,
                    GrabId = grabId,
                    Entries = entries
                };
            }
            catch
            {
                return null;
            }
        }

        private static void WriteFileHeader(BinaryWriter writer, string grabId)
        {
            byte[] grabBytes = Encoding.UTF8.GetBytes(grabId);
            writer.Write(FileMagic);
            writer.Write(FormatVersion);
            writer.Write(DateTime.UtcNow.Ticks);
            writer.Write(grabBytes.Length);
            writer.Write(grabBytes);
        }

        private static string ReadFileHeader(BinaryReader reader)
        {
            if (!Matches(reader.ReadBytes(FileMagic.Length), FileMagic))
                throw new InvalidDataException("Invalid ACAP header.");
            if (reader.ReadInt32() != FormatVersion)
                throw new InvalidDataException("Unsupported ACAP version.");
            reader.ReadInt64();
            int length = reader.ReadInt32();
            if (length <= 0 || length > 128) throw new InvalidDataException("Invalid grab id.");
            return Encoding.UTF8.GetString(reader.ReadBytes(length));
        }

        private static void ValidateFileHeader(Stream stream, string expectedGrabId)
        {
            long original = stream.Position;
            stream.Position = 0;
            using (var reader = new BinaryReader(stream, Encoding.UTF8, true))
            {
                string actual = ReadFileHeader(reader);
                if (!string.Equals(actual, expectedGrabId, StringComparison.Ordinal))
                    throw new InvalidDataException("ACAP grab id mismatch.");
            }
            stream.Position = original;
        }

        private static void WriteRecord(
            BinaryWriter writer,
            string baseName,
            int cameraId,
            long frameTicks,
            CaptureAssetKind kind,
            byte[] payload)
        {
            byte[] nameBytes = Encoding.UTF8.GetBytes(baseName);
            if (nameBytes.Length == 0 || nameBytes.Length > MaxNameBytes)
                throw new InvalidDataException("Invalid ACAP record name.");
            if (payload.Length > MaxPayloadBytes)
                throw new InvalidDataException("ACAP record payload is too large.");
            writer.Write(RecordMagic);
            writer.Write(RecordVersion);
            writer.Write((byte)kind);
            writer.Write(new byte[3]);
            writer.Write(cameraId);
            writer.Write(frameTicks);
            writer.Write(nameBytes.Length);
            writer.Write(payload.Length);
            writer.Write(Crc32.Compute(payload));
            writer.Write(nameBytes);
            writer.Write(payload);
        }

        private static bool TryParseVirtualPath(
            string path,
            out string archivePath,
            out string baseName,
            out CaptureAssetKind kind)
        {
            archivePath = null;
            baseName = null;
            kind = 0;
            if (!IsVirtualPath(path)) return false;
            int slash = path.IndexOf('/', VirtualPrefix.Length);
            if (slash <= VirtualPrefix.Length || slash >= path.Length - 1) return false;
            try
            {
                archivePath = Uri.UnescapeDataString(
                    path.Substring(VirtualPrefix.Length, slash - VirtualPrefix.Length));
                string assetName = Uri.UnescapeDataString(path.Substring(slash + 1));
                if (!TryStripAssetSuffix(assetName, out baseName, out kind)) return false;
                return !string.IsNullOrEmpty(archivePath) && !string.IsNullOrEmpty(baseName);
            }
            catch
            {
                return false;
            }
        }

        private static bool TryStripAssetSuffix(
            string assetName, out string baseName, out CaptureAssetKind kind)
        {
            baseName = null;
            kind = 0;
            string[] suffixes =
            {
                CaptureFileNaming.ThumbRawJpg,
                CaptureFileNaming.ThumbProcC,
                CaptureFileNaming.ThumbProcR,
                CaptureFileNaming.RawJpg,
                CaptureFileNaming.ProcC,
                CaptureFileNaming.ProcR,
                CaptureFileNaming.MeanC,
                CaptureFileNaming.MaxC,
                CaptureFileNaming.MeanR,
                CaptureFileNaming.MaxR
            };
            CaptureAssetKind[] kinds =
            {
                CaptureAssetKind.ThumbnailRawJpeg,
                CaptureAssetKind.ThumbnailColumnJpeg,
                CaptureAssetKind.ThumbnailRowJpeg,
                CaptureAssetKind.RawJpeg,
                CaptureAssetKind.ProcessedColumnJpeg,
                CaptureAssetKind.ProcessedRowJpeg,
                CaptureAssetKind.MeanColumnCurve,
                CaptureAssetKind.MaxColumnCurve,
                CaptureAssetKind.MeanRowCurve,
                CaptureAssetKind.MaxRowCurve
            };
            for (int i = 0; i < suffixes.Length; i++)
            {
                if (!assetName.EndsWith(suffixes[i], StringComparison.OrdinalIgnoreCase)) continue;
                baseName = assetName.Substring(0, assetName.Length - suffixes[i].Length);
                kind = kinds[i];
                return true;
            }
            return false;
        }

        private static string EntryKey(string baseName, CaptureAssetKind kind)
        {
            return baseName + "\n" + ((int)kind).ToString();
        }

        private static bool KeyHasKind(string key, CaptureAssetKind kind)
        {
            return key.EndsWith("\n" + ((int)kind).ToString(), StringComparison.Ordinal);
        }

        private static string KeyBaseName(string key)
        {
            int separator = key.LastIndexOf('\n');
            return separator > 0 ? key.Substring(0, separator) : key;
        }

        private static bool TryGetKeyKind(string key, out CaptureAssetKind kind)
        {
            kind = 0;
            int separator = key.LastIndexOf('\n');
            if (separator <= 0 || separator >= key.Length - 1) return false;
            if (!int.TryParse(key.Substring(separator + 1), out int value)) return false;
            kind = (CaptureAssetKind)value;
            return Enum.IsDefined(typeof(CaptureAssetKind), kind);
        }

        private static object GetPathLock(string path)
        {
            lock (LockMapSync)
            {
                if (!PathLocks.TryGetValue(path, out object value))
                    PathLocks[path] = value = new object();
                return value;
            }
        }

        private static void InvalidateIndex(string path)
        {
            try { path = Path.GetFullPath(path); } catch { }
            lock (IndexCacheSync) IndexCache.Remove(path);
        }

        private static bool Matches(byte[] actual, byte[] expected)
        {
            if (actual == null || actual.Length != expected.Length) return false;
            for (int i = 0; i < actual.Length; i++)
                if (actual[i] != expected[i]) return false;
            return true;
        }

        private static bool ReadExactly(Stream stream, byte[] buffer, int offset, int count)
        {
            while (count > 0)
            {
                int read = stream.Read(buffer, offset, count);
                if (read <= 0) return false;
                offset += read;
                count -= read;
            }
            return true;
        }

        private static void TryDelete(string path)
        {
            try { if (!string.IsNullOrEmpty(path) && File.Exists(path)) File.Delete(path); }
            catch { }
        }

        private static class Crc32
        {
            private static readonly uint[] Table = BuildTable();

            public static uint Compute(byte[] bytes)
            {
                uint crc = 0xffffffffu;
                for (int i = 0; i < bytes.Length; i++)
                    crc = Table[(crc ^ bytes[i]) & 0xff] ^ (crc >> 8);
                return ~crc;
            }

            private static uint[] BuildTable()
            {
                var table = new uint[256];
                for (uint i = 0; i < table.Length; i++)
                {
                    uint value = i;
                    for (int bit = 0; bit < 8; bit++)
                        value = (value & 1) != 0 ? 0xedb88320u ^ (value >> 1) : value >> 1;
                    table[i] = value;
                }
                return table;
            }
        }
    }

    /// <summary>PowerShell-facing entry point for the one-time legacy capture migration.</summary>
    public static class CaptureArchiveMigration
    {
        public static string ConvertLegacyRoot(string captureRoot, bool overwrite)
        {
            CaptureArchiveConversionResult result = CaptureArchiveStore.ConvertLegacyRoot(
                captureRoot, overwrite);
            return string.Format(
                "archives={0};frames={1};payloadBytes={2};skipped={3};failed={4}",
                result.ArchiveCount,
                result.FrameCount,
                result.PayloadBytes,
                result.SkippedArchiveCount,
                result.FailedArchiveCount);
        }

        public static string ValidateRoot(string captureRoot)
        {
            CaptureArchiveValidationResult result = CaptureArchiveStore.ValidateRoot(captureRoot);
            return string.Format(
                "archives={0};rawFrames={1};records={2};payloadBytes={3};" +
                "previewAtlases={4};invalidArchives={5};invalidRecords={6};partialFiles={7}",
                result.ArchiveCount,
                result.RawFrameCount,
                result.RecordCount,
                result.PayloadBytes,
                result.PreviewAtlasCount,
                result.InvalidArchiveCount,
                result.InvalidRecordCount,
                result.PartialFileCount);
        }

        public static string AddThumbnails(string captureRoot, int targetWidth)
        {
            CaptureArchiveThumbnailResult result =
                CaptureArchiveStore.AddThumbnails(captureRoot, targetWidth);
            return string.Format(
                "archives={0};frames={1};thumbnails={2};thumbnailBytes={3};" +
                "skipped={4};failed={5}",
                result.ArchiveCount,
                result.FrameCount,
                result.ThumbnailCount,
                result.ThumbnailBytes,
                result.SkippedThumbnailCount,
                result.FailedFrameCount);
        }

        public static string AddPreviewAtlases(
            string captureRoot, int maxWidth, int maxHeight,
            bool replaceExisting)
        {
            CaptureArchivePreviewAtlasResult result =
                CaptureArchiveStore.AddPreviewAtlases(
                    captureRoot, maxWidth, maxHeight,
                    replaceExisting);
            return string.Format(
                "archives={0};atlases={1};atlasBytes={2};skipped={3};failed={4}",
                result.ArchiveCount,
                result.AtlasCount,
                result.AtlasBytes,
                result.SkippedAtlasCount,
                result.FailedArchiveCount);
        }

        public static string AddPreviewAtlases(
            string captureRoot, int maxWidth, int maxHeight)
        {
            return AddPreviewAtlases(
                captureRoot, maxWidth, maxHeight, false);
        }
    }
}
