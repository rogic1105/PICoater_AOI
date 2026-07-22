using System;
using System.Collections.Generic;
using System.IO;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Services;
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
            byte[] curveBytes = CameraFrameSaver.EncodeCurveBin(
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
    }
}
