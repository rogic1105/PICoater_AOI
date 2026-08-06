using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class InspectionLogServiceTests
    {
        private string _tempRoot;

        [SetUp]
        public void SetUp()
        {
            _tempRoot = Path.Combine(Path.GetTempPath(), "LogTest_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tempRoot);
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_tempRoot, true); } catch { }
        }

        [Test]
        public void NextGrabId_ReturnsTimestampFormat()
        {
            var svc = new InspectionLogService(() => _tempRoot);
            string id = svc.NextGrabId();
            // 格式 yyMMdd-HHmmss，例如 "260401-130511"
            Assert.That(id.Length, Is.EqualTo(13));
            Assert.That(id[6], Is.EqualTo('-'));
        }

        [Test]
        public void FormatGrabId_ProducesCorrectFormat()
        {
            var dt = new DateTime(2026, 3, 30, 10, 15, 30);
            string id = InspectionLogService.FormatGrabId(dt);
            Assert.That(id, Is.EqualTo("260330-101530"));
        }

        [Test]
        public void AppendRecord_CreatesCsvWithHeaderAndData()
        {
            var svc = new InspectionLogService(() => _tempRoot);
            var ts = new DateTime(2026, 3, 30, 10, 15, 30, 500);
            var config = new CsvConfigSnapshot(
                new double[7], new double[7], null, null, null, 1.0f, 1.0f, 0.5f, 0.8f, 0.5f, 0.8f, 0.0, 0.0, ts);

            string grabId = InspectionLogService.FormatGrabId(ts);
            svc.AppendRecord(grabId, "20260330_101530.500-1",
                0.3f, 0.6f, 0.5f, 0.8f, 3001, 3001.0, 149.0, config, ts);

            string csvPath = Path.Combine(_tempRoot, "2026", "202603", "20260330.csv");
            Assert.That(File.Exists(csvPath), Is.True);

            string[] lines = File.ReadAllLines(csvPath);
            Assert.That(lines.Length, Is.GreaterThanOrEqualTo(3)); // #CFG + header + data
            Assert.That(lines[0], Does.StartWith("#CFG,"));
            Assert.That(lines[1], Does.Contain("Id,FileName,MaxExceed,MeanExceed"));
            Assert.That(lines[2], Does.StartWith("260330-101530,"));
        }

        [Test]
        public void AppendRecord_MultipleRecords_SameCfg_NoDuplicateCfg()
        {
            var svc = new InspectionLogService(() => _tempRoot);
            var ts = new DateTime(2026, 3, 30, 10, 0, 0, 0);
            var config = new CsvConfigSnapshot(
                new double[7], new double[7], null, null, null, 1.0f, 1.0f, 0.5f, 0.8f, 0.5f, 0.8f, 0.0, 0.0, ts);

            string grabId = InspectionLogService.FormatGrabId(ts);
            svc.AppendRecord(grabId, "20260330_100000.000-1",
                0.1f, 0.2f, 0.5f, 0.8f, 3001, 3001.0, 149.0, config, ts);
            svc.AppendRecord(grabId, "20260330_100000.000-2",
                0.3f, 0.4f, 0.5f, 0.8f, 3001, 3001.0, 149.0, config, ts);

            string csvPath = Path.Combine(_tempRoot, "2026", "202603", "20260330.csv");
            string[] lines = File.ReadAllLines(csvPath);

            int cfgCount = 0;
            foreach (string line in lines)
                if (line.StartsWith("#CFG,")) cfgCount++;
            Assert.That(cfgCount, Is.EqualTo(1), "Same config should not produce duplicate #CFG lines");
        }

        [Test]
        public void AppendRecord_CsvPathCannotBeCreated_RaisesWriteFailed()
        {
            string blockedRoot = Path.Combine(_tempRoot, "blocked");
            File.WriteAllText(blockedRoot, "not a directory");
            var svc = new InspectionLogService(() => blockedRoot);
            string failure = null;
            svc.WriteFailed += message => failure = message;
            var ts = new DateTime(2026, 3, 30, 10, 15, 30);

            svc.AppendRecord(
                InspectionLogService.FormatGrabId(ts),
                "20260330_101530.000-1",
                0.1f, 0.2f, 0.5f, 0.8f,
                3001, 3001.0, 149.0, null, ts);

            Assert.That(failure, Is.Not.Null.And.Not.Empty);
        }

        [Test]
        public void AppendRecord_ConfigChange_InsertNewCfg()
        {
            var svc = new InspectionLogService(() => _tempRoot);
            var ts = new DateTime(2026, 3, 30, 10, 0, 0, 0);
            var config1 = new CsvConfigSnapshot(
                new double[] { 1, 2, 3, 4, 5, 6, 7 }, new double[7], null, null, null, 1.0f, 1.0f, 0.5f, 0.8f, 0.5f, 0.8f, 0.0, 0.0, ts);
            var config2 = new CsvConfigSnapshot(
                new double[] { 10, 20, 30, 40, 50, 60, 70 }, new double[7], null, null, null, 2.0f, 2.0f, 0.6f, 0.9f, 0.6f, 0.9f, 0.0, 0.0, ts);

            string id1 = InspectionLogService.FormatGrabId(ts);
            string id2 = InspectionLogService.FormatGrabId(ts.AddSeconds(1));
            svc.AppendRecord(id1, "20260330_100000.000-1",
                0.1f, 0.2f, 0.5f, 0.8f, 3001, 3001.0, 149.0, config1, ts);
            svc.AppendRecord(id2, "20260330_100001.000-1",
                0.3f, 0.4f, 0.6f, 0.9f, 3001, 3001.0, 149.0, config2, ts);

            string csvPath = Path.Combine(_tempRoot, "2026", "202603", "20260330.csv");
            string[] lines = File.ReadAllLines(csvPath);

            int cfgCount = 0;
            foreach (string line in lines)
                if (line.StartsWith("#CFG,")) cfgCount++;
            Assert.That(cfgCount, Is.EqualTo(2), "Config change should insert new #CFG");
        }

        [Test]
        public void AppendRecord_RidgeSigmaChange_InsertNewCfg()
        {
            var svc = new InspectionLogService(() => _tempRoot);
            var ts = new DateTime(2026, 3, 30, 10, 0, 0, 0);
            var config1 = new CsvConfigSnapshot(
                new double[7], new double[7], null, null, null,
                1.0f, 1.0f, 8.0f, 0.5f, 0.8f, 0.5f, 0.8f, 0.0, 0.0, ts);
            var config2 = new CsvConfigSnapshot(
                new double[7], new double[7], null, null, null,
                1.0f, 1.0f, 9.0f, 0.5f, 0.8f, 0.5f, 0.8f, 0.0, 0.0, ts);

            svc.AppendRecord(
                InspectionLogService.FormatGrabId(ts), "20260330_100000.000-1",
                0.1f, 0.2f, 0.5f, 0.8f, 3001, 3001.0, 149.0, config1, ts);
            svc.AppendRecord(
                InspectionLogService.FormatGrabId(ts.AddSeconds(1)), "20260330_100001.000-1",
                0.1f, 0.2f, 0.5f, 0.8f, 3001, 3001.0, 149.0, config2, ts);

            string csvPath = Path.Combine(_tempRoot, "2026", "202603", "20260330.csv");
            string[] cfgLines = Array.FindAll(
                File.ReadAllLines(csvPath),
                line => line.StartsWith("#CFG,", StringComparison.Ordinal));

            Assert.That(cfgLines.Length, Is.EqualTo(2));
            Assert.That(cfgLines[0], Does.Contain("RidgeSigma=8.0000"));
            Assert.That(cfgLines[1], Does.Contain("RidgeSigma=9.0000"));
        }

        [Test]
        public void AppendRecord_PassFail_MaxExceedMeanExceed()
        {
            var svc = new InspectionLogService(() => _tempRoot);
            var ts = new DateTime(2026, 3, 30, 10, 0, 0, 0);
            var config = new CsvConfigSnapshot(
                new double[7], new double[7], null, null, null, 1.0f, 1.0f, 0.5f, 0.8f, 0.5f, 0.8f, 0.0, 0.0, ts);

            string id1 = InspectionLogService.FormatGrabId(ts);
            string id2 = InspectionLogService.FormatGrabId(ts.AddSeconds(1));

            // Pass: both peaks below thresholds
            svc.AppendRecord(id1, "20260330_100000.000-1",
                0.3f, 0.6f, 0.5f, 0.8f, 3001, 3001.0, 149.0, config, ts);

            // Fail: meanPeak > errMean
            svc.AppendRecord(id2, "20260330_100001.000-1",
                0.6f, 0.7f, 0.5f, 0.8f, 3001, 3001.0, 149.0, config, ts);

            string csvPath = Path.Combine(_tempRoot, "2026", "202603", "20260330.csv");
            string[] lines = File.ReadAllLines(csvPath);

            // Find data lines (skip #CFG and header)
            string passLine = null, failLine = null;
            foreach (string line in lines)
            {
                if (line.StartsWith(id1 + ",")) passLine = line;
                if (line.StartsWith(id2 + ",")) failLine = line;
            }

            Assert.That(passLine, Is.Not.Null);
            Assert.That(failLine, Is.Not.Null);

            // passLine: maxPeak=0.6 < errMax=0.8 → MaxExceed=0; meanPeak=0.3 < errMean=0.5 → MeanExceed=0
            Assert.That(passLine.Split(',')[2], Is.EqualTo("0"), "MaxExceed should be 0 (pass)");
            Assert.That(passLine.Split(',')[3], Is.EqualTo("0"), "MeanExceed should be 0 (pass)");

            // failLine: maxPeak=0.7 < errMax=0.8 → MaxExceed=0; meanPeak=0.6 > errMean=0.5 → MeanExceed=1
            Assert.That(failLine.Split(',')[2], Is.EqualTo("0"), "MaxExceed should be 0");
            Assert.That(failLine.Split(',')[3], Is.EqualTo("1"), "MeanExceed should be 1 (fail)");
        }

        [Test]
        public void AppendRecord_WritesMaxCMeanColumn()
        {
            var svc = new InspectionLogService(() => _tempRoot);
            var ts = new DateTime(2026, 3, 30, 11, 0, 0);
            var config = new CsvConfigSnapshot(
                new double[7], new double[7], null, null, null,
                1.0f, 1.0f, 0.5f, 0.8f, 0.5f, 0.8f, 0.0, 0.0, ts);

            svc.AppendRecord("260330-110000", "20260330_110000.000-1",
                0.3f, 0.6f, 0.412345f, 0.5f, 0.8f,
                3001, 3001.0, 149.0, config, ts);

            string csvPath = CaptureStoragePaths.DailyCsv(_tempRoot, ts);
            string[] lines = File.ReadAllLines(csvPath);
            Assert.That(lines[1], Does.EndWith(",MaxCMean,MeanRPeak,MaxRPeak"));
            Assert.That(lines[2].Split(',').Length, Is.EqualTo(12));
            Assert.That(lines[2].Split(',')[9], Is.EqualTo("0.412345"));
        }

        [Test]
        public void AppendRecord_WritesRowPeakColumns()
        {
            var svc = new InspectionLogService(() => _tempRoot);
            var ts = new DateTime(2026, 3, 30, 11, 5, 0);

            svc.AppendRecord("260330-110500", "20260330_110500.000-1",
                0.1f, 0.2f, 0.15f, 0.35f, 0.75f,
                0.2f, 0.6f, 3001, 3001, 149, null, ts);

            string[] columns = File.ReadAllLines(
                CaptureStoragePaths.DailyCsv(_tempRoot, ts))[1].Split(',');
            Assert.That(columns[10], Is.EqualTo("0.3500"));
            Assert.That(columns[11], Is.EqualTo("0.7500"));
        }

        [Test]
        public void AppendRecord_UpgradesExistingNineColumnHeader()
        {
            var ts = new DateTime(2026, 3, 30, 11, 30, 0);
            string csvPath = CaptureStoragePaths.DailyCsv(_tempRoot, ts);
            Directory.CreateDirectory(Path.GetDirectoryName(csvPath));
            File.WriteAllLines(csvPath, new[]
            {
                "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs",
                "260330-112959,20260330_112959.000-1,0,0,0.1,0.2,3001,3001.0,149.0"
            });

            var svc = new InspectionLogService(() => _tempRoot);
            svc.AppendRecord("260330-113000", "20260330_113000.000-1",
                0.1f, 0.2f, 0.3f, 0.5f, 0.8f, 3001, 3001, 149, null, ts);

            string[] lines = File.ReadAllLines(csvPath);
            Assert.That(lines[0], Does.EndWith(",MaxCMean,MeanRPeak,MaxRPeak"));
            Assert.That(lines[1].Split(',').Length, Is.EqualTo(9));
            Assert.That(lines[2].Split(',').Length, Is.EqualTo(12));
        }

        [Test]
        public void MuraProfileRepository_LoadRange_UsesEvenMeanAndTopScoredMaxRows()
        {
            var svc = new InspectionLogService(() => _tempRoot);
            var ts = new DateTime(2026, 3, 30, 12, 0, 0);
            var config = new CsvConfigSnapshot(
                new double[7], new double[7], null, null, null,
                1.0f, 1.0f, 0.5f, 0.8f, 0.5f, 0.8f, 0.0, 0.0, ts);

            string[] fileNames =
            {
                "20260330_120000.000-1",
                "20260330_120000.100-1",
                "20260330_120000.200-1"
            };
            svc.AppendRecord("260330-120000", fileNames[0],
                0.1f, 0.2f, 0.2f, 0.5f, 0.8f, 3001, 3001, 149, config, ts);
            svc.AppendRecord("260330-120000", fileNames[1],
                0.1f, 0.9f, 0.9f, 0.5f, 0.8f, 3001, 3001, 149, config, ts);
            svc.AppendRecord("260330-120000", fileNames[2],
                0.1f, 0.5f, 0.5f, 0.5f, 0.8f, 3001, 3001, 149, config, ts);

            string imageDir = CaptureStoragePaths.DateImageDir(_tempRoot, ts);
            Directory.CreateDirectory(imageDir);
            WriteCurveBin(Path.Combine(imageDir, fileNames[0] + CaptureFileNaming.MeanC), 10f);
            WriteCurveBin(Path.Combine(imageDir, fileNames[1] + CaptureFileNaming.MeanC), 100f);
            WriteCurveBin(Path.Combine(imageDir, fileNames[2] + CaptureFileNaming.MeanC), 30f);
            WriteCurveBin(Path.Combine(imageDir, fileNames[0] + CaptureFileNaming.MaxC), 10f);
            WriteCurveBin(Path.Combine(imageDir, fileNames[1] + CaptureFileNaming.MaxC), 90f);
            WriteCurveBin(Path.Combine(imageDir, fileNames[2] + CaptureFileNaming.MaxC), 50f);

            var range = new List<GrabIdInfo>
            {
                new GrabIdInfo { GrabId = "260330-120000", Earliest = ts }
            };
            var profiles = InspectionMuraProfileRepository.LoadRange(_tempRoot, range, 2);

            Assert.That(profiles.RankedCams, Is.EqualTo(1));
            Assert.That(profiles.TotalCams, Is.EqualTo(1));
            Assert.That(profiles.ScoredRows, Is.EqualTo(3));
            Assert.That(profiles.Mean[1][0], Is.EqualTo(20f).Within(0.001f));
            Assert.That(profiles.Max[1][0], Is.EqualTo(90f).Within(0.001f));
            Assert.That(profiles.IndexBuilds, Is.EqualTo(1));
            Assert.That(profiles.IndexHits, Is.EqualTo(0));
            Assert.That(profiles.CurveCacheHits, Is.EqualTo(0));
            Assert.That(profiles.CurveCacheMisses, Is.EqualTo(4));

            var cached = InspectionMuraProfileRepository.LoadRange(_tempRoot, range, 2);
            Assert.That(cached.IndexHits, Is.EqualTo(1));
            Assert.That(cached.IndexBuilds, Is.EqualTo(0));
            Assert.That(cached.CurveCacheHits, Is.EqualTo(4));
            Assert.That(cached.CurveCacheMisses, Is.EqualTo(0));

            const string appendedFile = "20260330_120000.300-1";
            svc.AppendRecord("260330-120000", appendedFile,
                0.1f, 0.7f, 0.7f, 0.5f, 0.8f, 3001, 3001, 149, config, ts);
            WriteCurveBin(Path.Combine(imageDir, appendedFile + CaptureFileNaming.MeanC), 70f);
            WriteCurveBin(Path.Combine(imageDir, appendedFile + CaptureFileNaming.MaxC), 70f);

            var refreshed = InspectionMuraProfileRepository.LoadRange(_tempRoot, range, 2);
            Assert.That(refreshed.IndexBuilds, Is.EqualTo(1));
            Assert.That(refreshed.IndexHits, Is.EqualTo(0));
            Assert.That(refreshed.TotalRows, Is.EqualTo(4));
        }

        [Test]
        public void MuraProfileRepository_LoadRange_RescalesCurvesAndRankingToCurrentHm()
        {
            var svc = new InspectionLogService(() => _tempRoot);
            var ts = new DateTime(2026, 3, 30, 12, 30, 0);
            var captureHm1 = new CsvConfigSnapshot(
                new double[7], new double[7], null, null, null,
                1.0f, 1.0f, 0.5f, 0.8f, 0.5f, 0.8f, 0.0, 0.0, ts);
            var captureHm2 = new CsvConfigSnapshot(
                new double[7], new double[7], null, null, null,
                2.0f, 1.0f, 0.5f, 0.8f, 0.5f, 0.8f, 0.0, 0.0, ts);
            const string firstFile = "20260330_123000.000-1";
            const string secondFile = "20260330_123000.100-1";

            svc.AppendRecord("260330-123000", firstFile,
                0.1f, 0.9f, 0.9f, 0.5f, 0.8f, 3001, 3001, 149, captureHm1, ts);
            svc.AppendRecord("260330-123000", secondFile,
                0.1f, 0.6f, 0.6f, 0.5f, 0.8f, 3001, 3001, 149, captureHm2, ts);

            string imageDir = CaptureStoragePaths.DateImageDir(_tempRoot, ts);
            Directory.CreateDirectory(imageDir);
            WriteCurveBin(Path.Combine(imageDir, firstFile + CaptureFileNaming.MeanC), 100f);
            WriteCurveBin(Path.Combine(imageDir, secondFile + CaptureFileNaming.MeanC), 100f);
            WriteCurveBin(Path.Combine(imageDir, firstFile + CaptureFileNaming.MaxC), 100f);
            WriteCurveBin(Path.Combine(imageDir, secondFile + CaptureFileNaming.MaxC), 100f);

            var profiles = InspectionMuraProfileRepository.LoadRange(
                _tempRoot,
                new List<GrabIdInfo>
                {
                    new GrabIdInfo { GrabId = "260330-123000", Earliest = ts }
                },
                1,
                currentHmV: 2.0f);

            Assert.That(profiles.HmRows, Is.EqualTo(2));
            Assert.That(profiles.Mean[1][0], Is.EqualTo(200f).Within(0.001f),
                "Mean 候選是第一筆，HM 1→2 後應線性放大兩倍");
            Assert.That(profiles.Max[1][0], Is.EqualTo(200f).Within(0.001f),
                "MaxCMean 也必先以相同正比公式換算後排名");
        }

        [Test]
        public void ComputeCurveMeanNormalized_ReturnsZeroToOneMean()
        {
            float value = CameraFrameSaver.ComputeCurveMeanNormalized(
                new[] { 0f, 127.5f, 255f });

            Assert.That(value, Is.EqualTo(0.5f).Within(0.000001f));
        }

        [Test]
        public void ComputeCurvePeakNormalized_ReturnsZeroToOnePeak()
        {
            float value = CameraFrameSaver.ComputeCurvePeakNormalized(
                new[] { 12f, 127.5f, 204f });

            Assert.That(value, Is.EqualTo(0.8f).Within(0.000001f));
        }

        [Test]
        public void CaptureFileNaming_ResolvesCurrentThenPreviousCurveFormats()
        {
            string basePath = Path.Combine(_tempRoot, "capture-1");
            string previous = basePath + CaptureFileNaming.MaxCPrevious;
            string current = basePath + CaptureFileNaming.MaxC;
            File.WriteAllText(previous, "previous");

            Assert.That(CaptureFileNaming.ResolveMaxC(basePath), Is.EqualTo(previous));

            File.WriteAllText(current, "current");
            Assert.That(CaptureFileNaming.ResolveMaxC(basePath), Is.EqualTo(current));
        }

        [Test]
        public void AppendColumnCurveSummary_WritesMergedPeaksForEachCamera()
        {
            var service = new InspectionLogService(() => _tempRoot);
            var captureDate = new DateTime(2026, 8, 4, 8, 55, 59);

            service.AppendColumnCurveSummary(
                "260804-085559",
                captureDate,
                0.5f,
                new[] { new[] { 12.75f, 38.25f }, new[] { 25.5f } },
                new[] { new[] { 76.5f, 127.5f }, new[] { 153f } });

            string csvPath = Path.Combine(
                _tempRoot, "2026", "202608", "20260804.csv");
            string[] lines = File.ReadAllLines(csvPath);

            Assert.That(lines, Has.Length.EqualTo(2));
            Assert.That(lines[0], Is.EqualTo(
                "#CURVE-C,1,260804-085559,1,0.5,0.15,0.5"));
            Assert.That(lines[1], Is.EqualTo(
                "#CURVE-C,1,260804-085559,2,0.5,0.1,0.6"));
        }

        private static void WriteCurveBin(string path, params float[] values)
        {
            using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(new byte[] { (byte)'M', (byte)'C', (byte)'B', (byte)'F' });
                bw.Write(1);
                bw.Write(1.0f);
                bw.Write(values.Length);
                foreach (float value in values) bw.Write(value);
            }
        }
    }
}
