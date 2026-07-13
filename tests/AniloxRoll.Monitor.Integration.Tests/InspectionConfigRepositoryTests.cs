using System;
using System.IO;
using NUnit.Framework;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class InspectionConfigRepositoryTests
    {
        private string _tempRoot;

        [SetUp]
        public void SetUp()
        {
            _tempRoot = Path.Combine(Path.GetTempPath(),
                "InspectionConfigRepositoryTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tempRoot);
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_tempRoot, true); } catch { }
        }

        [Test]
        public void LoadFromCsv_MultipleConfigs_ReturnsLastConfig()
        {
            string csvPath = Path.Combine(_tempRoot, "inspection.csv");
            File.WriteAllLines(csvPath, new[]
            {
                "Id,FileName,MaxExceed,MeanExceed",
                CreateConfig(1.5f, new DateTime(2026, 7, 13, 10, 0, 0)).ToCsvLine(),
                "260713-100000,20260713_100000.000-1,0,0",
                CreateConfig(2.5f, new DateTime(2026, 7, 13, 11, 0, 0)).ToCsvLine()
            });

            var result = InspectionConfigRepository.LoadFromCsv(csvPath);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.HessianMaxFactorV, Is.EqualTo(2.5f));
        }

        [Test]
        public void LoadForGrabId_MultipleConfigs_ReturnsNearestPrecedingConfig()
        {
            DateTime date = new DateTime(2026, 7, 13);
            string csvPath = CaptureStoragePaths.DailyCsv(_tempRoot, date);
            Directory.CreateDirectory(Path.GetDirectoryName(csvPath));
            File.WriteAllLines(csvPath, new[]
            {
                "Id,FileName,MaxExceed,MeanExceed",
                CreateConfig(1.5f, date.AddHours(10)).ToCsvLine(),
                "260713-100000,20260713_100000.000-1,0,0",
                CreateConfig(2.5f, date.AddHours(11)).ToCsvLine(),
                "260713-110000,20260713_110000.000-1,0,0"
            });

            var first = InspectionConfigRepository.LoadForGrabId(
                _tempRoot, "260713-100000", date, date);
            var second = InspectionConfigRepository.LoadForGrabId(
                _tempRoot, "260713-110000", date, date);

            Assert.That(first.HessianMaxFactorV, Is.EqualTo(1.5f));
            Assert.That(second.HessianMaxFactorV, Is.EqualTo(2.5f));
        }

        [Test]
        public void LoadLatest_MultipleDays_ReturnsLatestFileConfig()
        {
            DateTime firstDate = new DateTime(2026, 7, 12);
            DateTime secondDate = firstDate.AddDays(1);
            WriteDailyConfig(firstDate, 1.5f);
            WriteDailyConfig(secondDate, 2.5f);

            var result = InspectionConfigRepository.LoadLatest(_tempRoot);

            Assert.That(result, Is.Not.Null);
            Assert.That(result.HessianMaxFactorV, Is.EqualTo(2.5f));
        }

        private void WriteDailyConfig(DateTime date, float hessian)
        {
            string path = CaptureStoragePaths.DailyCsv(_tempRoot, date);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllLines(path, new[]
            {
                "Id,FileName,MaxExceed,MeanExceed",
                CreateConfig(hessian, date).ToCsvLine()
            });
        }

        private static CsvConfigSnapshot CreateConfig(float hessian, DateTime timestamp)
        {
            return new CsvConfigSnapshot(
                new double[7], new double[7], new int[7], new double[7], new double[7],
                hessian, 1f, 0f, 0f, 0f, 0f, 0d, 0d, timestamp);
        }
    }
}
