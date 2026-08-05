using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Integration.Tests
{
    [TestFixture]
    public class ColumnCurvePeakIndexTests
    {
        private string _tempRoot;

        [SetUp]
        public void SetUp()
        {
            _tempRoot = Path.Combine(Path.GetTempPath(),
                "ColumnCurvePeakIndexTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tempRoot);
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_tempRoot, true); } catch { }
        }

        [Test]
        public void Build_MissingSummary_UsesMergedColumnBins()
        {
            DateTime date = new DateTime(2026, 8, 4, 13, 54, 0);
            const string grabId = "260804-135400";
            string first = "20260804_135400.000-1";
            string second = "20260804_135401.000-1";
            string imageDir = CaptureStoragePaths.DateImageDir(_tempRoot, date);
            Directory.CreateDirectory(imageDir);
            WriteCapture(imageDir, first, new[] { 25.5f, 51f }, new[] { 76.5f, 102f });
            WriteCapture(imageDir, second, new[] { 76.5f, 102f }, new[] { 127.5f, 153f });
            WriteCsv(date, grabId, first, second);

            var infos = new List<GrabIdInfo>
            {
                new GrabIdInfo { GrabId = grabId, Earliest = date, Latest = date }
            };
            var captureHm = new Dictionary<string, float> { [grabId] = 0.5f };

            ColumnCurvePeakIndexResult result = ColumnCurvePeakIndex.Build(
                _tempRoot, infos, captureHm,
                new Dictionary<string, CsvConfigSnapshot>(),
                2, CancellationToken.None);

            Assert.That(result.SummaryGrabCount, Is.EqualTo(0));
            Assert.That(result.BinFallbackGrabCount, Is.EqualTo(1));
            Assert.That(result.MissingGrabCount, Is.EqualTo(0));
            Assert.That(result.CameraCount, Is.EqualTo(1));
            Assert.That(result.ByGrabId[grabId][0].MeanPeak, Is.EqualTo(0.3f).Within(0.0001f));
            Assert.That(result.ByGrabId[grabId][0].MaxPeak, Is.EqualTo(0.6f).Within(0.0001f));
            Assert.That(result.ByGrabId[grabId][0].CaptureHmV, Is.EqualTo(0.5f));
            Assert.That(result.ByGrabId[grabId][1], Is.Null);
        }

        [Test]
        public void LoadForGrabIds_ReturnsMultipleGrabsFromOneDailyCsv()
        {
            DateTime date = new DateTime(2026, 8, 4, 13, 54, 0);
            string first = "20260804_135400.000-1";
            string second = "20260804_135401.000-2";
            string imageDir = CaptureStoragePaths.DateImageDir(_tempRoot, date);
            Directory.CreateDirectory(imageDir);
            File.WriteAllText(Path.Combine(imageDir, first + CaptureFileNaming.RawJpg), "raw");
            File.WriteAllText(Path.Combine(imageDir, second + CaptureFileNaming.RawJpg), "raw");
            string csvPath = CaptureStoragePaths.DailyCsv(_tempRoot, date);
            Directory.CreateDirectory(Path.GetDirectoryName(csvPath));
            File.WriteAllLines(csvPath, new[]
            {
                "Id,FileName,MaxExceed,MeanExceed",
                "260804-135400," + first + ",0,0",
                "260804-135401," + second + ",0,0"
            });

            var infos = new List<GrabIdInfo>
            {
                new GrabIdInfo { GrabId = "260804-135400", Earliest = date, Latest = date },
                new GrabIdInfo { GrabId = "260804-135401", Earliest = date, Latest = date }
            };
            Dictionary<string, Dictionary<int, List<string>>> result =
                InspectionImagePathRepository.LoadForGrabIds(_tempRoot, infos);

            Assert.That(result["260804-135400"][1], Has.Count.EqualTo(1));
            Assert.That(result["260804-135401"][2], Has.Count.EqualTo(1));
        }

        private void WriteCsv(DateTime date, string grabId, params string[] fileNames)
        {
            string csvPath = CaptureStoragePaths.DailyCsv(_tempRoot, date);
            Directory.CreateDirectory(Path.GetDirectoryName(csvPath));
            using (var writer = new StreamWriter(csvPath))
            {
                writer.WriteLine("Id,FileName,MaxExceed,MeanExceed");
                foreach (string fileName in fileNames)
                    writer.WriteLine(grabId + "," + fileName + ",0,0");
            }
        }

        private static void WriteCapture(
            string directory, string baseName, float[] mean, float[] max)
        {
            File.WriteAllText(Path.Combine(directory, baseName + CaptureFileNaming.RawJpg), "raw");
            WriteCurveBin(Path.Combine(directory, baseName + CaptureFileNaming.MeanC), mean);
            WriteCurveBin(Path.Combine(directory, baseName + CaptureFileNaming.MaxC), max);
        }

        private static void WriteCurveBin(string path, float[] values)
        {
            using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write))
            using (var writer = new BinaryWriter(stream))
            {
                writer.Write(new[] { (byte)'M', (byte)'C', (byte)'B', (byte)'F' });
                writer.Write(1);
                writer.Write(1f);
                writer.Write(values.Length);
                foreach (float value in values) writer.Write(value);
            }
        }
    }
}
