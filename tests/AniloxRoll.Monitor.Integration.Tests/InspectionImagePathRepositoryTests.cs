using System;
using System.IO;
using NUnit.Framework;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class InspectionImagePathRepositoryTests
    {
        private string _tempRoot;

        [SetUp]
        public void SetUp()
        {
            _tempRoot = Path.Combine(Path.GetTempPath(),
                "InspectionImagePathRepositoryTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tempRoot);
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_tempRoot, true); } catch { }
        }

        [Test]
        public void LoadForGrabId_MixedFormats_ReturnsSortedUniqueExistingPaths()
        {
            DateTime date = new DateTime(2026, 7, 13);
            string grabId = "260713-100000";
            string earlier = "20260713_100000.000-1";
            string later = "20260713_100001.000-1";
            string missing = "20260713_100000.000-2";
            WriteCsv(date, grabId, later, earlier, earlier, missing);

            string imageDir = CaptureStoragePaths.DateImageDir(_tempRoot, date);
            Directory.CreateDirectory(imageDir);
            string earlierRaw = Path.Combine(imageDir, earlier + CaptureFileNaming.RawJpg);
            File.WriteAllText(earlierRaw, "raw");
            File.WriteAllText(Path.Combine(imageDir, earlier + ".bmp"), "legacy");
            string laterBmp = Path.Combine(imageDir, later + ".bmp");
            File.WriteAllText(laterBmp, "legacy");

            var result = InspectionImagePathRepository.LoadForGrabId(
                _tempRoot, grabId, date, date);

            Assert.That(result.Keys, Is.EquivalentTo(new[] { 1 }));
            Assert.That(result[1], Is.EqualTo(new[] { earlierRaw, laterBmp }));
        }

        [Test]
        public void LoadForGrabId_WithDateHint_ScansOnlyHintedDates()
        {
            string grabId = "260713-100000";
            DateTime firstDate = new DateTime(2026, 7, 12);
            DateTime secondDate = firstDate.AddDays(1);
            string firstName = "20260712_100000.000-1";
            string secondName = "20260713_100000.000-1";
            WriteCsv(firstDate, grabId, firstName);
            WriteCsv(secondDate, grabId, secondName);
            string firstPath = WriteRaw(firstDate, firstName);
            string secondPath = WriteRaw(secondDate, secondName);

            var hinted = InspectionImagePathRepository.LoadForGrabId(
                _tempRoot, grabId, secondDate, secondDate);
            var allDates = InspectionImagePathRepository.LoadForGrabId(_tempRoot, grabId);

            Assert.That(hinted[1], Is.EqualTo(new[] { secondPath }));
            Assert.That(allDates[1], Is.EqualTo(new[] { firstPath, secondPath }));
        }

        private void WriteCsv(DateTime date, string grabId, params string[] fileNames)
        {
            string csvPath = CaptureStoragePaths.DailyCsv(_tempRoot, date);
            Directory.CreateDirectory(Path.GetDirectoryName(csvPath));
            using (var writer = new StreamWriter(csvPath))
            {
                writer.WriteLine("Id,FileName,MaxExceed,MeanExceed");
                foreach (string fileName in fileNames)
                    writer.WriteLine($"{grabId},{fileName},0,0");
            }
        }

        private string WriteRaw(DateTime date, string fileName)
        {
            string imageDir = CaptureStoragePaths.DateImageDir(_tempRoot, date);
            Directory.CreateDirectory(imageDir);
            string path = Path.Combine(imageDir, fileName + CaptureFileNaming.RawJpg);
            File.WriteAllText(path, "raw");
            return path;
        }
    }
}
