using System;
using System.IO;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Integration.Tests
{
    [TestFixture]
    public class ReviewCurveDataLoaderTests
    {
        private string _tempRoot;

        [SetUp]
        public void SetUp()
        {
            _tempRoot = Path.Combine(Path.GetTempPath(),
                "ReviewCurveDataLoaderTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tempRoot);
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_tempRoot, true); } catch { }
        }

        [Test]
        public void Load_TwoCaptures_MergesColumnAndRowCurvesWithConfig()
        {
            DateTime date = new DateTime(2026, 7, 21);
            string grabId = "260721-080000";
            string first = "20260721_080000.000-1";
            string second = "20260721_080000.100-1";
            string imageDir = CaptureStoragePaths.DateImageDir(_tempRoot, date);
            Directory.CreateDirectory(imageDir);
            WriteCapture(imageDir, first,
                new[] { 1f, 3f }, new[] { 5f, 2f },
                new[] { 10f, 20f }, new[] { 30f, 40f });
            WriteCapture(imageDir, second,
                new[] { 3f, 5f }, new[] { 4f, 7f },
                new[] { 50f, 60f }, new[] { 70f, 80f });
            WriteCsv(date, grabId, first, second);

            var loader = new ReviewCurveDataLoader();
            ReviewCurveData result = loader.Load(_tempRoot, grabId, date, date, 2);

            Assert.That(result.ImageCount, Is.EqualTo(2));
            Assert.That(result.MatchedCameraCount, Is.EqualTo(1));
            Assert.That(result.AlignmentMode, Is.EqualTo("filename"));
            Assert.That(result.Config, Is.Not.Null);
            Assert.That(result.Config.HessianMaxFactorV, Is.EqualTo(0.75f));
            Assert.That(result.ColumnMean[0], Is.EqualTo(new[] { 2f, 4f }));
            Assert.That(result.ColumnMax[0], Is.EqualTo(new[] { 5f, 7f }));
            Assert.That(result.RowMean[0], Is.EqualTo(new[] { 10f, 20f, 50f, 60f }));
            Assert.That(result.RowMax[0], Is.EqualTo(new[] { 30f, 40f, 70f, 80f }));
            Assert.That(result.ColumnMean[1], Is.Null);
        }

        private void WriteCsv(DateTime date, string grabId, params string[] fileNames)
        {
            string csvPath = CaptureStoragePaths.DailyCsv(_tempRoot, date);
            Directory.CreateDirectory(Path.GetDirectoryName(csvPath));
            var config = new CsvConfigSnapshot(
                new double[7], new double[7], new int[7], new double[7], new double[7],
                0.75f, 1f, 0f, 0f, 0f, 0f, 0d, 0d, date.AddHours(8));
            using (var writer = new StreamWriter(csvPath))
            {
                writer.WriteLine("Id,FileName,MaxExceed,MeanExceed");
                writer.WriteLine(config.ToCsvLine());
                foreach (string fileName in fileNames)
                    writer.WriteLine($"{grabId},{fileName},0,0");
            }
        }

        private static void WriteCapture(
            string directory,
            string baseName,
            float[] columnMean,
            float[] columnMax,
            float[] rowMean,
            float[] rowMax)
        {
            File.WriteAllText(Path.Combine(directory, baseName + CaptureFileNaming.RawJpg), "raw");
            WriteCurveBin(Path.Combine(directory, baseName + CaptureFileNaming.MeanC), columnMean);
            WriteCurveBin(Path.Combine(directory, baseName + CaptureFileNaming.MaxC), columnMax);
            WriteCurveBin(Path.Combine(directory, baseName + CaptureFileNaming.MeanR), rowMean);
            WriteCurveBin(Path.Combine(directory, baseName + CaptureFileNaming.MaxR), rowMax);
        }

        private static void WriteCurveBin(string path, float[] values)
        {
            using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write))
            using (var writer = new BinaryWriter(stream))
            {
                writer.Write(new[] { (byte)'M', (byte)'C', (byte)'B', (byte)'F' });
                writer.Write(1);
                writer.Write(1.0f);
                writer.Write(values.Length);
                foreach (float value in values)
                    writer.Write(value);
            }
        }
    }
}
