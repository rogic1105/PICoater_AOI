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
        public void BuildSummaries_ReturnsAvailableRecordsBeforeBinFallback()
        {
            DateTime date = new DateTime(2026, 8, 11, 11, 0, 0);
            var ready = new GrabIdInfo
            {
                GrabId = "260811-110000",
                Earliest = date,
                Latest = date
            };
            var pending = new GrabIdInfo
            {
                GrabId = "260811-110100",
                Earliest = date.AddMinutes(1),
                Latest = date.AddMinutes(1)
            };
            var summary = new SingleGrabCurveSummary(
                new[] { new[] { 25.5f, 51f }, (float[])null },
                new[] { new[] { 76.5f, 102f }, (float[])null },
                new[] { 25.5f, 51f },
                new[] { 76.5f, 102f },
                1);
            Assert.That(SingleGrabCurveSummaryStore.TrySave(
                _tempRoot, ready, 2, summary), Is.True);

            ColumnCurvePeakIndexResult result = ColumnCurvePeakIndex.BuildSummaries(
                _tempRoot,
                new List<GrabIdInfo> { ready, pending },
                new Dictionary<string, float>(),
                new Dictionary<string, CsvConfigSnapshot>(),
                2,
                CancellationToken.None);

            Assert.That(result.SummaryGrabCount, Is.EqualTo(1));
            Assert.That(result.BinFallbackGrabCount, Is.EqualTo(0));
            Assert.That(result.ByGrabId.ContainsKey(ready.GrabId), Is.True);
            Assert.That(result.ByGrabId.ContainsKey(pending.GrabId), Is.False);
            Assert.That(result.PendingBinGrabInfos, Has.Count.EqualTo(1));
            Assert.That(result.PendingBinGrabInfos[0].GrabId, Is.EqualTo(pending.GrabId));
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

            ColumnCurvePeakIndexResult summaryPhase = ColumnCurvePeakIndex.BuildSummaries(
                _tempRoot, infos, captureHm,
                new Dictionary<string, CsvConfigSnapshot>(),
                2, CancellationToken.None);
            Assert.That(summaryPhase.SummaryGrabCount, Is.EqualTo(0));
            Assert.That(summaryPhase.PendingBinGrabInfos, Has.Count.EqualTo(1));

            var progressBatches = new List<ColumnCurvePeakIndexResult>();
            ColumnCurvePeakIndexResult binPhase = ColumnCurvePeakIndex.BuildBinFallback(
                _tempRoot, summaryPhase.PendingBinGrabInfos, captureHm,
                new Dictionary<string, CsvConfigSnapshot>(),
                2, CancellationToken.None,
                progressBatches.Add,
                progressBatchSize: 1);
            Assert.That(binPhase.BinFallbackGrabCount, Is.EqualTo(1));
            Assert.That(binPhase.ByGrabId.ContainsKey(grabId), Is.True);
            Assert.That(progressBatches, Has.Count.EqualTo(1));
            Assert.That(progressBatches[0].ByGrabId.ContainsKey(grabId), Is.True);
            Assert.That(progressBatches[0].RowByGrabId.ContainsKey(grabId), Is.True);

            // Exercise the complete cold path independently from the projection index
            // materialized by the explicit fallback phase above.
            File.Delete(CaptureStoragePaths.DailyCurvePeakIndex(_tempRoot, date));

            ColumnCurvePeakIndexResult result = ColumnCurvePeakIndex.Build(
                _tempRoot, infos, captureHm,
                new Dictionary<string, CsvConfigSnapshot>(),
                2, CancellationToken.None);

            Assert.That(result.SummaryGrabCount, Is.EqualTo(0));
            Assert.That(result.BinFallbackGrabCount, Is.EqualTo(1));
            Assert.That(result.MissingGrabCount, Is.EqualTo(0));
            Assert.That(result.CameraCount, Is.EqualTo(1));
            Assert.That(result.ByGrabId[grabId][0].RawMeanPeak, Is.EqualTo(0.3f).Within(0.0001f));
            Assert.That(result.ByGrabId[grabId][0].RawMaxPeak, Is.EqualTo(0.6f).Within(0.0001f));
            Assert.That(result.ByGrabId[grabId][0].CaptureHmV, Is.EqualTo(0.5f));
            Assert.That(result.ByGrabId[grabId][1], Is.Null);
            Assert.That(result.RowByGrabId[grabId].RawMeanPeak,
                Is.EqualTo(0.2f).Within(0.0001f));
            Assert.That(result.RowByGrabId[grabId].RawMaxPeak,
                Is.EqualTo(0.7f).Within(0.0001f));
        }

        [Test]
        public void BuildSummaries_SecondReadUsesDailyProjectionIndex()
        {
            DateTime date = new DateTime(2026, 8, 11, 11, 0, 0);
            const string grabId = "260811-110000";
            var info = new GrabIdInfo
            {
                GrabId = grabId,
                Earliest = date,
                Latest = date.AddSeconds(9)
            };
            var summary = new SingleGrabCurveSummary(
                new[] { new[] { 25.5f, 51f }, (float[])null },
                new[] { new[] { 76.5f, 102f }, (float[])null },
                new[] { 25.5f, 51f },
                new[] { 76.5f, 102f },
                10);
            Assert.That(SingleGrabCurveSummaryStore.TrySave(
                _tempRoot, info, 2, summary), Is.True);

            var infos = new List<GrabIdInfo> { info };
            var configs = new Dictionary<string, CsvConfigSnapshot>();
            ColumnCurvePeakIndexResult first = ColumnCurvePeakIndex.BuildSummaries(
                _tempRoot, infos, new Dictionary<string, float>(), configs,
                2, CancellationToken.None);

            Assert.That(first.CacheGrabCount, Is.EqualTo(0));
            Assert.That(first.SummaryGrabCount, Is.EqualTo(1));
            Assert.That(File.Exists(
                CaptureStoragePaths.DailyCurvePeakIndex(_tempRoot, date)), Is.True);

            File.Delete(CaptureStoragePaths.GrabCurveSummary(
                _tempRoot, date, grabId));
            ColumnCurvePeakIndexResult second = ColumnCurvePeakIndex.BuildSummaries(
                _tempRoot, infos, new Dictionary<string, float>(), configs,
                2, CancellationToken.None);

            Assert.That(second.CacheGrabCount, Is.EqualTo(1));
            Assert.That(second.CacheDayCount, Is.EqualTo(1));
            Assert.That(second.SummaryGrabCount, Is.EqualTo(1));
            Assert.That(second.PendingBinGrabInfos, Is.Empty);
            Assert.That(second.ByGrabId[grabId][0].RawMeanPeak,
                Is.EqualTo(first.ByGrabId[grabId][0].RawMeanPeak));
            Assert.That(second.RowByGrabId[grabId].RawMaxPeak,
                Is.EqualTo(first.RowByGrabId[grabId].RawMaxPeak));
        }

        [Test]
        public void BuildSummaries_ChangedIdentityDoesNotUseStaleProjectionIndex()
        {
            DateTime date = new DateTime(2026, 8, 11, 11, 0, 0);
            const string grabId = "260811-110000";
            var original = new GrabIdInfo
            {
                GrabId = grabId,
                Earliest = date,
                Latest = date.AddSeconds(9)
            };
            var summary = new SingleGrabCurveSummary(
                new[] { new[] { 25.5f }, (float[])null },
                new[] { new[] { 51f }, (float[])null }, 1);
            Assert.That(SingleGrabCurveSummaryStore.TrySave(
                _tempRoot, original, 2, summary), Is.True);
            ColumnCurvePeakIndex.BuildSummaries(
                _tempRoot, new List<GrabIdInfo> { original },
                new Dictionary<string, float>(),
                new Dictionary<string, CsvConfigSnapshot>(),
                2, CancellationToken.None);

            var changed = new GrabIdInfo
            {
                GrabId = grabId,
                Earliest = date,
                Latest = date.AddSeconds(10)
            };
            ColumnCurvePeakIndexResult result = ColumnCurvePeakIndex.BuildSummaries(
                _tempRoot, new List<GrabIdInfo> { changed },
                new Dictionary<string, float>(),
                new Dictionary<string, CsvConfigSnapshot>(),
                2, CancellationToken.None);

            Assert.That(result.CacheGrabCount, Is.EqualTo(0));
            Assert.That(result.SummaryGrabCount, Is.EqualTo(0));
            Assert.That(result.PendingBinGrabInfos, Has.Count.EqualTo(1));
        }

        [Test]
        public void BuildAndStoreSummaryProjection_MakesFirstReportReadACacheHit()
        {
            DateTime date = new DateTime(2026, 8, 12, 9, 30, 0);
            const string grabId = "260812-093000";
            var info = new GrabIdInfo
            {
                GrabId = grabId,
                Earliest = date,
                Latest = date.AddSeconds(9)
            };
            var summary = new SingleGrabCurveSummary(
                new[] { new[] { 25.5f, 51f }, (float[])null },
                new[] { new[] { 76.5f, 102f }, (float[])null },
                new[] { 25.5f, 51f },
                new[] { 76.5f, 102f },
                10);

            ColumnCurvePeakIndexResult stored =
                ColumnCurvePeakIndex.BuildAndStoreSummaryProjection(
                    _tempRoot, info, null, summary, 2);

            Assert.That(stored.SummaryGrabCount, Is.EqualTo(1));
            Assert.That(File.Exists(
                CaptureStoragePaths.DailyCurvePeakIndex(_tempRoot, date)), Is.True);
            Assert.That(File.Exists(
                CaptureStoragePaths.GrabCurveSummary(_tempRoot, date, grabId)), Is.False,
                "The projection writer must not pretend that the asynchronous summary writer ran.");

            ColumnCurvePeakIndexResult loaded = ColumnCurvePeakIndex.BuildSummaries(
                _tempRoot,
                new List<GrabIdInfo> { info },
                new Dictionary<string, float>(),
                new Dictionary<string, CsvConfigSnapshot>(),
                2,
                CancellationToken.None);

            Assert.That(loaded.CacheGrabCount, Is.EqualTo(1));
            Assert.That(loaded.PendingBinGrabInfos, Is.Empty);
            Assert.That(loaded.ByGrabId[grabId][0].RawMaxPeak,
                Is.EqualTo(0.4f).Within(0.0001f));
            Assert.That(loaded.RowByGrabId[grabId].RawMeanPeak,
                Is.EqualTo(0.2f).Within(0.0001f));
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
            WriteCurveBin(Path.Combine(directory, baseName + CaptureFileNaming.MeanR),
                new[] { 25.5f, 51f });
            WriteCurveBin(Path.Combine(directory, baseName + CaptureFileNaming.MaxR),
                new[] { 127.5f, 178.5f });
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
