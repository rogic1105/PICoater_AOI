using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Widgets;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Integration.Tests
{
    [TestFixture]
    public class CaptureArchiveStoreTests
    {
        private string _root;

        [SetUp]
        public void SetUp()
        {
            _root = Path.Combine(Path.GetTempPath(), "picoater-acap-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_root);
        }

        [TearDown]
        public void TearDown()
        {
            try { if (Directory.Exists(_root)) Directory.Delete(_root, true); }
            catch { }
        }

        [Test]
        public void AppendFrame_ReadsIndependentAssetsAndTicks()
        {
            string archive = Path.Combine(_root, "260722-120000.acap");
            byte[] raw = { 1, 2, 3, 4 };
            byte[] curve = { 9, 8, 7 };

            CaptureArchiveStore.AppendFrame(
                archive, "260722-120000", "20260722_120000.000-1", 1, 123456,
                new List<CaptureArchiveAsset>
                {
                    new CaptureArchiveAsset { Kind = CaptureAssetKind.RawJpeg, Data = raw },
                    new CaptureArchiveAsset { Kind = CaptureAssetKind.MeanColumnCurve, Data = curve }
                });

            string rawPath = CaptureArchiveStore.CreateVirtualRawPath(
                archive, "20260722_120000.000-1");
            string curvePath = CaptureFileNaming.StripRawJpg(rawPath) + CaptureFileNaming.MeanC;

            Assert.That(CaptureArchiveStore.Exists(rawPath), Is.True);
            Assert.That(CaptureArchiveStore.ReadAllBytes(rawPath), Is.EqualTo(raw));
            Assert.That(CaptureArchiveStore.ReadAllBytes(curvePath), Is.EqualTo(curve));
            Assert.That(CaptureArchiveStore.TryGetFrameTicks(rawPath, out long ticks), Is.True);
            Assert.That(ticks, Is.EqualTo(123456));
        }

        [Test]
        public void CameraFrameSaver_WritesOneArchiveWithAllAssetsAndNoLooseFiles()
        {
            const string grabId = "260724-120000";
            const string baseName = "20260724_120000.000-1";
            string[] savedFiles = null;
            string resultGrabId = null;
            var pixels = new byte[12];
            for (int i = 0; i < pixels.Length; i++)
                pixels[i] = (byte)(i * 10);
            byte[] hessianC = { 0x00, 0x00, 0x00, 0x38, 0x00, 0x3c, 0x00, 0x40 };
            byte[] hessianR = { 0x00, 0x40, 0x00, 0x3c, 0x00, 0x38, 0x00, 0x00 };

            var saver = new CameraFrameSaver();
            saver.SaveCapture(new CaptureContext
            {
                RawBytes = pixels,
                ProcCBytes = pixels,
                ProcRBytes = pixels,
                HessianCStandard = hessianC,
                HessianRStandard = hessianR,
                MeanC = new[] { 1f, 2f, 3f },
                MaxC = new[] { 4f, 5f, 6f },
                MeanR = new[] { 7f, 8f, 9f },
                MaxR = new[] { 10f, 11f, 12f },
                ResizeWidth = 4,
                ResizeHeight = 3,
                StandardWidth = 2,
                StandardHeight = 2,
                JpgQuality = 90,
                ScaleForHeader = 5,
                SaveDir = _root,
                GrabId = grabId,
                BaseName = baseName,
                CameraId = 1,
                OrigWidth = 40,
                OrigHeight = 30,
                FrameStartTicks = 1234,
                OnFilesSaved = files => savedFiles = files,
                OnResult = (id, cameraId, name, meanPeak, maxPeak, maxCMean, meanRPeak, maxRPeak) =>
                    resultGrabId = id
            });

            string archive = Path.Combine(_root, grabId + CaptureArchiveStore.Extension);
            string rawPath = CaptureArchiveStore.CreateVirtualRawPath(archive, baseName);
            string basePath = CaptureFileNaming.StripRawJpg(rawPath);
            string[] expectedAssets =
            {
                rawPath,
                basePath + CaptureFileNaming.ProcC,
                basePath + CaptureFileNaming.ProcR,
                basePath + CaptureFileNaming.HessianC,
                basePath + CaptureFileNaming.HessianR,
                basePath + CaptureFileNaming.MeanC,
                basePath + CaptureFileNaming.MaxC,
                basePath + CaptureFileNaming.MeanR,
                basePath + CaptureFileNaming.MaxR
            };

            Assert.That(File.Exists(archive), Is.True);
            Assert.That(savedFiles, Is.EqualTo(new[] { archive }));
            Assert.That(resultGrabId, Is.EqualTo(grabId));
            foreach (string assetPath in expectedAssets)
                Assert.That(CaptureArchiveStore.Exists(assetPath), Is.True, assetPath);
            CollectionAssert.AreEqual(
                hessianC,
                HessianStandardMapCodec.Decode(CaptureArchiveStore.ReadAllBytes(
                    basePath + CaptureFileNaming.HessianC)).HalfBytes);
            CollectionAssert.AreEqual(
                hessianR,
                HessianStandardMapCodec.Decode(CaptureArchiveStore.ReadAllBytes(
                    basePath + CaptureFileNaming.HessianR)).HalfBytes);
            using (Bitmap remapped = GrabImageStitcher.LoadCameraImage(
                rawPath, 5, null, true, "c", false, 0.5f))
            {
                Assert.That(remapped.Size, Is.EqualTo(new Size(2, 2)));
                Assert.That(remapped.GetPixel(1, 0).R, Is.EqualTo(64).Within(1));
                Assert.That(remapped.GetPixel(0, 1).R, Is.EqualTo(128).Within(1));
                Assert.That(remapped.GetPixel(1, 1).R, Is.EqualTo(255));
            }
            Assert.That(
                Directory.GetFiles(_root, "*", SearchOption.AllDirectories),
                Is.EqualTo(new[] { archive }));
            Assert.That(File.Exists(Path.Combine(_root, CameraFrameSaver.TickSidecarName)), Is.False);
        }

        [Test]
        public void Reader_IgnoresTruncatedFinalRecordAndKeepsEarlierFrame()
        {
            string archive = Path.Combine(_root, "260722-120001.acap");
            CaptureArchiveStore.AppendFrame(
                archive, "260722-120001", "20260722_120001.000-1", 1, 1,
                new[]
                {
                    new CaptureArchiveAsset
                    {
                        Kind = CaptureAssetKind.RawJpeg,
                        Data = new byte[] { 10, 20, 30 }
                    }
                });
            using (var stream = new FileStream(archive, FileMode.Append, FileAccess.Write))
                stream.Write(new byte[] { (byte)'A', (byte)'R', (byte)'E' }, 0, 3);

            string rawPath = CaptureArchiveStore.CreateVirtualRawPath(
                archive, "20260722_120001.000-1");
            Assert.That(CaptureArchiveStore.ReadAllBytes(rawPath), Is.EqualTo(new byte[] { 10, 20, 30 }));
        }

        [Test]
        public void ListVirtualRawPaths_FiltersCameraAndSortsFrames()
        {
            string archive = Path.Combine(_root, "260722-120002.acap");
            AppendRaw(archive, "260722-120002", "20260722_120002.100-2", 2, 2);
            AppendRaw(archive, "260722-120002", "20260722_120002.000-1", 1, 1);
            AppendRaw(archive, "260722-120002", "20260722_120002.200-1", 1, 3);

            List<string> camera1 = CaptureArchiveStore.ListVirtualRawPaths(archive, 1);
            Assert.That(camera1.Count, Is.EqualTo(2));
            Assert.That(CaptureArchiveStore.GetVirtualBaseName(camera1[0]),
                Is.EqualTo("20260722_120002.000-1"));
            Assert.That(CaptureArchiveStore.GetVirtualBaseName(camera1[1]),
                Is.EqualTo("20260722_120002.200-1"));
        }

        [Test]
        public void ConvertLegacyRoot_CreatesOneArchivePerGrab()
        {
            DateTime date = new DateTime(2026, 7, 22, 12, 3, 4, 5);
            string grabId = "260722-120304";
            string baseName = "20260722_120304.005-1";
            string dateDir = CaptureStoragePaths.DateImageDir(_root, date);
            string csvPath = CaptureStoragePaths.DailyCsv(_root, date);
            Directory.CreateDirectory(dateDir);
            Directory.CreateDirectory(Path.GetDirectoryName(csvPath));
            File.WriteAllText(csvPath,
                "Id,FileName,MaxExceed,MeanExceed\r\n" +
                grabId + "," + baseName + ",0,0\r\n");
            File.WriteAllBytes(
                Path.Combine(dateDir, baseName + CaptureFileNaming.RawJpg),
                new byte[] { 4, 5, 6 });
            Assert.That(InspectionCsvReader.TryParseRecord(
                grabId + "," + baseName + ",0,0", out InspectionCsvRecord parsed), Is.True);
            Assert.That(parsed.GrabId, Is.EqualTo(grabId));
            Assert.That(InspectionCsvReader.TryParseTimestamp(baseName, out _), Is.True);

            CaptureArchiveConversionResult converted = CaptureArchiveStore.ConvertLegacyRoot(
                _root, false);

            Assert.That(converted.ArchiveCount, Is.EqualTo(1),
                $"frames={converted.FrameCount} failed={converted.FailedArchiveCount} skipped={converted.SkippedArchiveCount}");
            string archive = CaptureStoragePaths.GrabArchive(_root, date, grabId);
            string rawPath = CaptureArchiveStore.CreateVirtualRawPath(archive, baseName);
            Assert.That(CaptureArchiveStore.ReadAllBytes(rawPath), Is.EqualTo(new byte[] { 4, 5, 6 }));
        }

        [Test]
        public void RepositoryAndCurveReader_PreferArchiveRecords()
        {
            DateTime date = new DateTime(2026, 7, 22, 12, 4, 5, 6);
            string grabId = "260722-120405";
            string baseName = "20260722_120405.006-1";
            string csvPath = CaptureStoragePaths.DailyCsv(_root, date);
            Directory.CreateDirectory(Path.GetDirectoryName(csvPath));
            File.WriteAllText(csvPath,
                "Id,FileName,MaxExceed,MeanExceed\r\n" +
                grabId + "," + baseName + ",0,0\r\n");
            byte[] curveBytes = EncodeCurveBin(
                new float[] { 1.25f, 2.5f }, 5);
            string archive = CaptureStoragePaths.GrabArchive(_root, date, grabId);
            CaptureArchiveStore.AppendFrame(
                archive, grabId, baseName, 1, 88,
                new[]
                {
                    new CaptureArchiveAsset
                    {
                        Kind = CaptureAssetKind.RawJpeg,
                        Data = new byte[] { 1, 2, 3 }
                    },
                    new CaptureArchiveAsset
                    {
                        Kind = CaptureAssetKind.MeanColumnCurve,
                        Data = curveBytes
                    }
                });

            Dictionary<int, List<string>> grouped = InspectionImagePathRepository.LoadForGrabId(
                _root, grabId, date.Date, date.Date);
            Assert.That(grouped[1].Count, Is.EqualTo(1));
            Assert.That(CaptureArchiveStore.IsVirtualPath(grouped[1][0]), Is.True);
            string curvePath = CaptureFileNaming.BaseFromImagePath(grouped[1][0]) +
                CaptureFileNaming.MeanC;
            Assert.That(CurveBinFile.Load(curvePath), Is.EqualTo(new float[] { 1.25f, 2.5f }));
        }

        [Test]
        public void TryReadJpegSize_ReadsHeaderWithoutDecodingPayload()
        {
            const string grabId = "260722-120406";
            const string baseName = "20260722_120406.000-1";
            string archive = Path.Combine(_root, grabId + ".acap");
            CaptureArchiveStore.AppendFrame(
                archive, grabId, baseName, 1, 99,
                new[]
                {
                    new CaptureArchiveAsset
                    {
                        Kind = CaptureAssetKind.RawJpeg,
                        Data = EncodeJpeg(13, 17)
                    }
                });
            string virtualPath = CaptureArchiveStore.CreateVirtualRawPath(
                archive, baseName);

            bool ok = CaptureArchiveStore.TryReadJpegSize(
                virtualPath, out int width, out int height);

            Assert.That(ok, Is.True);
            Assert.That(width, Is.EqualTo(13));
            Assert.That(height, Is.EqualTo(17));
        }

        [Test]
        public void AddThumbnails_EmbedsThreePreviewVariantsAndIsIdempotent()
        {
            const string grabId = "260722-120407";
            const string baseName = "20260722_120407.000-1";
            string archive = Path.Combine(_root, grabId + ".acap");
            CaptureArchiveStore.AppendFrame(
                archive, grabId, baseName, 1, 100,
                new[]
                {
                    new CaptureArchiveAsset
                    {
                        Kind = CaptureAssetKind.RawJpeg,
                        Data = EncodeJpeg(320, 120)
                    },
                    new CaptureArchiveAsset
                    {
                        Kind = CaptureAssetKind.ProcessedColumnJpeg,
                        Data = EncodeJpeg(320, 120)
                    },
                    new CaptureArchiveAsset
                    {
                        Kind = CaptureAssetKind.ProcessedRowJpeg,
                        Data = EncodeJpeg(320, 120)
                    }
                });

            CaptureArchiveThumbnailResult first =
                CaptureArchiveStore.AddThumbnails(_root, 64);
            long lengthAfterFirst = new FileInfo(archive).Length;
            CaptureArchiveThumbnailResult second =
                CaptureArchiveStore.AddThumbnails(_root, 64);

            string rawPath = CaptureArchiveStore.CreateVirtualRawPath(
                archive, baseName);
            string rawThumb = CaptureFileNaming.ResolveThumbnailJpg(
                rawPath, false, "c");
            string columnThumb = CaptureFileNaming.ResolveThumbnailJpg(
                rawPath, true, "c");
            string rowThumb = CaptureFileNaming.ResolveThumbnailJpg(
                rawPath, true, "r");
            Assert.That(first.ThumbnailCount, Is.EqualTo(3));
            Assert.That(CaptureArchiveStore.Exists(rawThumb), Is.True);
            Assert.That(CaptureArchiveStore.Exists(columnThumb), Is.True);
            Assert.That(CaptureArchiveStore.Exists(rowThumb), Is.True);
            using (var image = Image.FromStream(
                new MemoryStream(CaptureArchiveStore.ReadAllBytes(rawThumb))))
            {
                Assert.That(image.Width, Is.EqualTo(64));
                Assert.That(image.Height, Is.EqualTo(24));
            }
            Assert.That(second.ThumbnailCount, Is.Zero);
            Assert.That(second.SkippedThumbnailCount, Is.EqualTo(3));
            Assert.That(new FileInfo(archive).Length, Is.EqualTo(lengthAfterFirst));
        }

        [Test]
        public void AddPreviewAtlases_EmbedsBoundedVariantsAndCanReplaceExisting()
        {
            const string grabId = "260722-120408";
            string archive = Path.Combine(_root, grabId + ".acap");
            for (int frame = 0; frame < 2; frame++)
            {
                for (int cameraId = 1; cameraId <= 2; cameraId++)
                {
                    string baseName = string.Format(
                        "20260722_120408.{0:000}-{1}",
                        frame * 100, cameraId);
                    CaptureArchiveStore.AppendFrame(
                        archive, grabId, baseName, cameraId,
                        1000 + frame * 100 + cameraId,
                        new[]
                        {
                            new CaptureArchiveAsset
                            {
                                Kind = CaptureAssetKind.RawJpeg,
                                Data = EncodeJpeg(80, 60)
                            },
                            new CaptureArchiveAsset
                            {
                                Kind = CaptureAssetKind.ProcessedColumnJpeg,
                                Data = EncodeJpeg(80, 60)
                            },
                            new CaptureArchiveAsset
                            {
                                Kind = CaptureAssetKind.ProcessedRowJpeg,
                                Data = EncodeJpeg(80, 60)
                            }
                        });
                }
            }

            CaptureArchivePreviewAtlasResult first =
                CaptureArchiveStore.AddPreviewAtlasesToArchive(archive, 100, 60);
            long lengthAfterFirst = new FileInfo(archive).Length;
            CaptureArchivePreviewAtlasResult second =
                CaptureArchiveStore.AddPreviewAtlasesToArchive(archive, 100, 60);
            CaptureArchivePreviewAtlasResult replaced =
                CaptureArchiveStore.AddPreviewAtlasesToArchive(
                    archive, 50, 30, replaceExisting: true);
            var grouped = new Dictionary<int, List<string>>
            {
                { 1, CaptureArchiveStore.ListVirtualRawPaths(archive, 1) },
                { 2, CaptureArchiveStore.ListVirtualRawPaths(archive, 2) }
            };

            bool loaded = CapturePreviewAtlasCodec.TryLoad(
                grouped, 2, true, "r",
                out CapturePreviewAtlasData atlas);

            Assert.That(first.ArchiveCount, Is.EqualTo(1));
            Assert.That(first.AtlasCount, Is.EqualTo(3));
            Assert.That(first.AtlasBytes, Is.GreaterThan(0));
            Assert.That(second.AtlasCount, Is.Zero);
            Assert.That(second.SkippedAtlasCount, Is.EqualTo(3));
            Assert.That(replaced.AtlasCount, Is.EqualTo(3));
            Assert.That(replaced.SkippedAtlasCount, Is.Zero);
            Assert.That(new FileInfo(archive).Length, Is.GreaterThan(lengthAfterFirst));
            Assert.That(loaded, Is.True);
            using (atlas)
            {
                Assert.That(atlas.AtlasWidth, Is.LessThanOrEqualTo(50));
                Assert.That(atlas.AtlasHeight, Is.LessThanOrEqualTo(30));
                Assert.That(atlas.PixelScaleRatio, Is.EqualTo(4.0).Within(0.01));
                Assert.That(atlas.CameraImages[0], Is.Not.Null);
                Assert.That(atlas.CameraImages[1], Is.Not.Null);
                Assert.That(atlas.CameraImages[0].Width, Is.EqualTo(20));
                Assert.That(atlas.CameraImages[0].Height, Is.EqualTo(30));
                Assert.That(atlas.SourceWidths[0], Is.EqualTo(80));
                Assert.That(atlas.SourceHeights[0], Is.EqualTo(120));
            }
        }

        [Test]
        public void ValidateRoot_ReportsCorruptPayload()
        {
            string grabId = "260722-120004";
            string archive = Path.Combine(_root, grabId + ".acap");
            AppendRaw(archive, grabId, "20260722_120004.000-1", 1, 42);

            CaptureArchiveValidationResult healthy = CaptureArchiveStore.ValidateRoot(_root);
            Assert.That(healthy.InvalidArchiveCount, Is.Zero);
            Assert.That(healthy.InvalidRecordCount, Is.Zero);
            Assert.That(healthy.RawFrameCount, Is.EqualTo(1));

            using (var stream = new FileStream(archive, FileMode.Open, FileAccess.ReadWrite))
            {
                stream.Position = stream.Length - 1;
                int value = stream.ReadByte();
                stream.Position = stream.Length - 1;
                stream.WriteByte((byte)(value ^ 0xff));
            }

            CaptureArchiveValidationResult corrupt = CaptureArchiveStore.ValidateRoot(_root);
            Assert.That(corrupt.InvalidArchiveCount, Is.Zero);
            Assert.That(corrupt.InvalidRecordCount, Is.EqualTo(1));
        }

        private static void AppendRaw(
            string archive, string grabId, string baseName, int cameraId, byte value)
        {
            CaptureArchiveStore.AppendFrame(
                archive, grabId, baseName, cameraId, value,
                new[]
                {
                    new CaptureArchiveAsset
                    {
                        Kind = CaptureAssetKind.RawJpeg,
                        Data = new[] { value }
                    }
                });
        }

        private static byte[] EncodeCurveBin(float[] values, int scale)
        {
            using (var stream = new MemoryStream())
            using (var writer = new BinaryWriter(stream))
            {
                writer.Write(new byte[] { (byte)'M', (byte)'C', (byte)'B', (byte)'F' });
                writer.Write(1);
                writer.Write((float)scale);
                writer.Write(values.Length);
                for (int i = 0; i < values.Length; i++)
                    writer.Write(values[i]);
                writer.Flush();
                return stream.ToArray();
            }
        }

        private static byte[] EncodeJpeg(int width, int height)
        {
            using (var bitmap = new Bitmap(width, height))
            using (var stream = new MemoryStream())
            {
                bitmap.Save(stream, ImageFormat.Jpeg);
                return stream.ToArray();
            }
        }
    }
}
