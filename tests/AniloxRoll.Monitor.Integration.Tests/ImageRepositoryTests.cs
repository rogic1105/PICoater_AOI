using System;
using System.IO;
using NUnit.Framework;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Integration.Tests
{
    [TestFixture]
    public class ImageRepositoryTests
    {
        private string _tempRoot;

        [SetUp]
        public void SetUp()
        {
            _tempRoot = Path.Combine(Path.GetTempPath(),
                "ImageRepositoryTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tempRoot);
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_tempRoot, true); } catch { }
        }

        [Test]
        public void LoadDirectory_BuildsSortedDistinctPeriodIndexAndReplacesItOnReload()
        {
            WriteRaw("20260714_120001.250-2_raw.jpg");
            WriteRaw("20260714_120000.125-1_raw.jpg");
            WriteRaw("20260714_120000.125-2_raw.jpg");

            var repository = new ImageRepository();
            repository.LoadDirectory(_tempRoot);

            Assert.That(repository.GetAvailablePeriods(), Is.EqualTo(new[]
            {
                new DateTime(2026, 7, 14, 12, 0, 0, 125),
                new DateTime(2026, 7, 14, 12, 0, 1, 250),
            }));

            string secondRoot = Path.Combine(_tempRoot, "reload");
            Directory.CreateDirectory(secondRoot);
            File.WriteAllText(Path.Combine(secondRoot, "20260715_010203.004-1_raw.jpg"), "raw");
            repository.LoadDirectory(secondRoot);

            Assert.That(repository.GetAvailablePeriods(), Is.EqualTo(new[]
            {
                new DateTime(2026, 7, 15, 1, 2, 3, 4),
            }));
        }

        [Test]
        public void LoadDirectory_IndexesArchiveFramesAndPrefersThemOverLegacyFiles()
        {
            const string baseName = "20260722_120405.006-1";
            string legacy = Path.Combine(
                _tempRoot, baseName + CaptureFileNaming.RawJpg);
            File.WriteAllText(legacy, "legacy");
            string archive = Path.Combine(_tempRoot, "260722-120405.acap");
            CaptureArchiveStore.AppendFrame(
                archive, "260722-120405", baseName, 1, 88,
                new[]
                {
                    new CaptureArchiveAsset
                    {
                        Kind = CaptureAssetKind.RawJpeg,
                        Data = new byte[] { 1, 2, 3 }
                    }
                });

            var repository = new ImageRepository();
            repository.LoadDirectory(_tempRoot);

            Assert.That(repository.FileCount, Is.EqualTo(1));
            var images = repository.GetImages(
                new DateTime(2026, 7, 22, 12, 4, 5, 6));
            Assert.That(images.ContainsKey(1), Is.True);
            Assert.That(CaptureArchiveStore.IsVirtualPath(images[1]), Is.True);
        }

        [Test]
        public void LoadDirectory_UsesDailyCsvAsArchiveCatalogWithoutArchiveFallback()
        {
            const string grabId = "260722-120405";
            const string baseName = "20260722_120405.006-1";
            string month = Path.Combine(_tempRoot, "2026", "202607");
            string day = Path.Combine(month, "20260722");
            Directory.CreateDirectory(day);
            string archive = Path.Combine(
                day, grabId + CaptureArchiveStore.Extension);
            CaptureArchiveStore.AppendFrame(
                archive, grabId, baseName, 1, 88,
                new[]
                {
                    new CaptureArchiveAsset
                    {
                        Kind = CaptureAssetKind.RawJpeg,
                        Data = new byte[] { 1, 2, 3 }
                    }
                });
            File.WriteAllLines(Path.Combine(month, "20260722.csv"), new[]
            {
                "Id,FileName,MaxExceed,MeanExceed",
                grabId + "," + baseName + ",0,0"
            });

            var repository = new ImageRepository();
            ImageRepositoryLoadResult result = repository.LoadDirectory(_tempRoot);

            Assert.That(result.CsvRecordCount, Is.EqualTo(1));
            Assert.That(result.CsvBackedArchiveCount, Is.EqualTo(1));
            Assert.That(result.ArchiveFallbackCount, Is.EqualTo(0));
            Assert.That(repository.FileCount, Is.EqualTo(1));
            Assert.That(CaptureArchiveStore.IsVirtualPath(
                repository.GetImages(new DateTime(2026, 7, 22, 12, 4, 5, 6))[1]),
                Is.True);
        }

        [Test]
        public void GetGrabIdInfosDescending_GroupsCameraTimestampsByCaptureSecond()
        {
            WriteRaw("20260714_120001.250-1_raw.jpg");
            WriteRaw("20260714_120001.750-2_raw.jpg");
            WriteRaw("20260714_120000.125-1_raw.jpg");

            var repository = new ImageRepository();
            repository.LoadDirectory(_tempRoot);

            var infos = repository.GetGrabIdInfosDescending();

            Assert.That(infos.Count, Is.EqualTo(2));
            Assert.That(infos[0].GrabId, Is.EqualTo("260714-120001"));
            Assert.That(infos[0].Earliest.Millisecond, Is.EqualTo(250));
            Assert.That(infos[0].Latest.Millisecond, Is.EqualTo(750));
            Assert.That(infos[1].GrabId, Is.EqualTo("260714-120000"));
        }

        [Test]
        public void LoadDirectory_LegacyCatalogSkipsReportCsvAndThumbnailFiles()
        {
            const string baseName = "20260722_120405.006-1";
            File.WriteAllText(Path.Combine(
                _tempRoot, baseName + CaptureFileNaming.RawJpg), "raw");
            File.WriteAllText(Path.Combine(
                _tempRoot, baseName + CaptureFileNaming.ThumbRawJpg), "thumbnail");
            File.WriteAllLines(Path.Combine(_tempRoot, "20260722.csv"), new[]
            {
                "Id,FileName,MaxExceed,MeanExceed",
                "260722-120405," + baseName + ",0,0"
            });

            var repository = new ImageRepository();
            ImageRepositoryLoadResult result = repository.LoadDirectory(_tempRoot);

            Assert.That(result.CsvRecordCount, Is.EqualTo(0));
            Assert.That(result.LegacyFileCount, Is.EqualTo(1));
            Assert.That(repository.FileCount, Is.EqualTo(1));
            Assert.That(repository.GetImages(
                new DateTime(2026, 7, 22, 12, 4, 5, 6))[1],
                Is.EqualTo(Path.Combine(
                    _tempRoot, baseName + CaptureFileNaming.RawJpg)));
        }

        private void WriteRaw(string fileName)
        {
            File.WriteAllText(Path.Combine(_tempRoot, fileName), "raw");
        }
    }
}
