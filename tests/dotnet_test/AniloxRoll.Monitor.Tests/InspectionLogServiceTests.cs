using System;
using System.IO;
using NUnit.Framework;
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
        public void NextGrabId_Increments()
        {
            var svc = new InspectionLogService(() => _tempRoot, startIdNum: 0);
            string id1 = svc.NextGrabId();
            string id2 = svc.NextGrabId();
            Assert.That(id1, Is.EqualTo("A00001"));
            Assert.That(id2, Is.EqualTo("A00002"));
            Assert.That(svc.LastIdNum, Is.EqualTo(2));
        }

        [Test]
        public void AppendRecord_CreatesCsvWithHeaderAndData()
        {
            var svc = new InspectionLogService(() => _tempRoot, startIdNum: 0);
            var ts = new DateTime(2026, 3, 30, 10, 15, 30, 500);
            var config = new CsvConfigSnapshot(
                new double[7], new double[7], 1.0f, 0.5f, 0.8f, ts);

            svc.AppendRecord("A00001", "20260330_101530.500-1",
                0.3f, 0.6f, 0.5f, 0.8f, 3001, 3001.0, 149.0, config, ts);

            string csvPath = Path.Combine(_tempRoot, "2026", "202603", "20260330.csv");
            Assert.That(File.Exists(csvPath), Is.True);

            string[] lines = File.ReadAllLines(csvPath);
            Assert.That(lines.Length, Is.GreaterThanOrEqualTo(3)); // #CFG + header + data
            Assert.That(lines[0], Does.StartWith("#CFG,"));
            Assert.That(lines[1], Does.Contain("Id,FileName,MaxExceed,MeanExceed"));
            Assert.That(lines[2], Does.StartWith("A00001,"));
        }

        [Test]
        public void AppendRecord_MultipleRecords_SameCfg_NoDuplicateCfg()
        {
            var svc = new InspectionLogService(() => _tempRoot, startIdNum: 0);
            var ts = new DateTime(2026, 3, 30, 10, 0, 0, 0);
            var config = new CsvConfigSnapshot(
                new double[7], new double[7], 1.0f, 0.5f, 0.8f, ts);

            svc.AppendRecord("A00001", "20260330_100000.000-1",
                0.1f, 0.2f, 0.5f, 0.8f, 3001, 3001.0, 149.0, config, ts);
            svc.AppendRecord("A00001", "20260330_100000.000-2",
                0.3f, 0.4f, 0.5f, 0.8f, 3001, 3001.0, 149.0, config, ts);

            string csvPath = Path.Combine(_tempRoot, "2026", "202603", "20260330.csv");
            string[] lines = File.ReadAllLines(csvPath);

            int cfgCount = 0;
            foreach (string line in lines)
                if (line.StartsWith("#CFG,")) cfgCount++;
            Assert.That(cfgCount, Is.EqualTo(1), "Same config should not produce duplicate #CFG lines");
        }

        [Test]
        public void AppendRecord_ConfigChange_InsertNewCfg()
        {
            var svc = new InspectionLogService(() => _tempRoot, startIdNum: 0);
            var ts = new DateTime(2026, 3, 30, 10, 0, 0, 0);
            var config1 = new CsvConfigSnapshot(
                new double[] { 1, 2, 3, 4, 5, 6, 7 }, new double[7], 1.0f, 0.5f, 0.8f, ts);
            var config2 = new CsvConfigSnapshot(
                new double[] { 10, 20, 30, 40, 50, 60, 70 }, new double[7], 2.0f, 0.6f, 0.9f, ts);

            svc.AppendRecord("A00001", "20260330_100000.000-1",
                0.1f, 0.2f, 0.5f, 0.8f, 3001, 3001.0, 149.0, config1, ts);
            svc.AppendRecord("A00002", "20260330_100001.000-1",
                0.3f, 0.4f, 0.6f, 0.9f, 3001, 3001.0, 149.0, config2, ts);

            string csvPath = Path.Combine(_tempRoot, "2026", "202603", "20260330.csv");
            string[] lines = File.ReadAllLines(csvPath);

            int cfgCount = 0;
            foreach (string line in lines)
                if (line.StartsWith("#CFG,")) cfgCount++;
            Assert.That(cfgCount, Is.EqualTo(2), "Config change should insert new #CFG");
        }

        [Test]
        public void AppendRecord_PassFail_MaxExceedMeanExceed()
        {
            var svc = new InspectionLogService(() => _tempRoot, startIdNum: 0);
            var ts = new DateTime(2026, 3, 30, 10, 0, 0, 0);
            var config = new CsvConfigSnapshot(
                new double[7], new double[7], 1.0f, 0.5f, 0.8f, ts);

            // Pass: both peaks below thresholds
            svc.AppendRecord("A00001", "20260330_100000.000-1",
                0.3f, 0.6f, 0.5f, 0.8f, 3001, 3001.0, 149.0, config, ts);

            // Fail: meanPeak > errMean
            svc.AppendRecord("A00002", "20260330_100001.000-1",
                0.6f, 0.7f, 0.5f, 0.8f, 3001, 3001.0, 149.0, config, ts);

            string csvPath = Path.Combine(_tempRoot, "2026", "202603", "20260330.csv");
            string[] lines = File.ReadAllLines(csvPath);

            // Find data lines (skip #CFG and header)
            string passLine = null, failLine = null;
            foreach (string line in lines)
            {
                if (line.StartsWith("A00001,")) passLine = line;
                if (line.StartsWith("A00002,")) failLine = line;
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
    }
}
