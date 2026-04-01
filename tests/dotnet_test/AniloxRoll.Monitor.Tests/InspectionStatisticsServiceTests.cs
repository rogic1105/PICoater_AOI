using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class InspectionStatisticsServiceTests
    {
        private string _tempRoot;

        [SetUp]
        public void SetUp()
        {
            _tempRoot = Path.Combine(Path.GetTempPath(), "StatTest_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tempRoot);
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_tempRoot, true); } catch { }
        }

        private string WriteCsv(string dateStr, string content)
        {
            string dir = Path.Combine(_tempRoot,
                dateStr.Substring(0, 4),
                dateStr.Substring(0, 6));
            Directory.CreateDirectory(dir);
            string path = Path.Combine(dir, dateStr + ".csv");
            File.WriteAllText(path, content);
            return path;
        }

        [Test]
        public void Compute_TimeRange_PassFail()
        {
            string csv =
                "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs\n" +
                "260330-100000,20260330_100000.000-1,0,0,0.3,0.6,3001,3001.0,149.0\n" +
                "260330-100000,20260330_100000.000-2,1,0,0.9,1.2,3001,3001.0,149.0\n" +
                "260330-100000,20260330_100000.000-3,0,1,0.7,0.5,3001,3001.0,149.0\n";
            WriteCsv("20260330", csv);

            var start = new DateTime(2026, 3, 30, 0, 0, 0);
            var end   = new DateTime(2026, 3, 30, 23, 59, 59);

            var stats = InspectionStatisticsService.Compute(_tempRoot, start, end);
            Assert.That(stats[1].Pass, Is.EqualTo(1));
            Assert.That(stats[1].Fail, Is.EqualTo(0));
            Assert.That(stats[2].Pass, Is.EqualTo(0));
            Assert.That(stats[2].Fail, Is.EqualTo(1));
            Assert.That(stats[3].Pass, Is.EqualTo(0));
            Assert.That(stats[3].Fail, Is.EqualTo(1));
        }

        [Test]
        public void ComputeByGrabIdRange_VetoLogic()
        {
            // Same grabId, same cam, one pass one fail → veto → Fail
            string csv =
                "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs\n" +
                "260330-100000,20260330_100000.000-1,0,0,0.3,0.6,3001,3001.0,149.0\n" +
                "260330-100000,20260330_100001.000-1,1,0,0.9,1.2,3001,3001.0,149.0\n";
            WriteCsv("20260330", csv);

            var stats = InspectionStatisticsService.ComputeByGrabIdRange(
                _tempRoot, "260330-100000", "260330-100000");
            Assert.That(stats[1].Fail, Is.EqualTo(1), "Veto: any fail in same grabId+cam → Fail");
            Assert.That(stats[1].Pass, Is.EqualTo(0));
        }

        [Test]
        public void ComputeDetailedByGrabIdRange_ReturnsPerGrabIdResult()
        {
            string csv =
                "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs\n" +
                "260330-100000,20260330_100000.000-1,0,0,0.3,0.6,3001,3001.0,149.0\n" +
                "260330-100000,20260330_100000.000-2,1,0,0.9,1.2,3001,3001.0,149.0\n" +
                "260330-100100,20260330_100100.000-1,0,0,0.2,0.4,3001,3001.0,149.0\n";
            WriteCsv("20260330", csv);

            var details = InspectionStatisticsService.ComputeDetailedByGrabIdRange(
                _tempRoot, "260330-100000", "260330-100100");
            Assert.That(details.Count, Is.EqualTo(2));

            Assert.That(details[0].GrabId, Is.EqualTo("260330-100000"));
            Assert.That(details[0].CamResult[0], Is.False, "CAM1 pass");
            Assert.That(details[0].CamResult[1], Is.True,  "CAM2 fail");

            Assert.That(details[1].GrabId, Is.EqualTo("260330-100100"));
            Assert.That(details[1].CamResult[0], Is.False, "CAM1 pass");
            Assert.That(details[1].CamResult[1], Is.Null,  "CAM2 no data");
        }

        [Test]
        public void LoadGrabIdInfos_ReturnsSortedByGrabId()
        {
            string csv =
                "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs\n" +
                "260330-100200,20260330_100200.000-1,0,0,0.1,0.2,3001,3001.0,149.0\n" +
                "260330-100000,20260330_100000.000-1,0,0,0.1,0.2,3001,3001.0,149.0\n" +
                "260330-100100,20260330_100100.000-1,0,0,0.1,0.2,3001,3001.0,149.0\n";
            WriteCsv("20260330", csv);

            var infos = InspectionStatisticsService.LoadGrabIdInfos(_tempRoot);
            Assert.That(infos.Count, Is.EqualTo(3));
            Assert.That(infos[0].GrabId, Is.EqualTo("260330-100000"));
            Assert.That(infos[1].GrabId, Is.EqualTo("260330-100100"));
            Assert.That(infos[2].GrabId, Is.EqualTo("260330-100200"));
        }

        [Test]
        public void Compute_EmptyDirectory_ReturnsDefault7CameraStats()
        {
            var stats = InspectionStatisticsService.Compute(
                _tempRoot, DateTime.MinValue, DateTime.MaxValue);
            Assert.That(stats.Count, Is.EqualTo(7));
            for (int i = 1; i <= 7; i++)
            {
                Assert.That(stats[i].Total, Is.EqualTo(0));
                Assert.That(stats[i].PassRate, Is.EqualTo(0f));
            }
        }

        [Test]
        public void Compute_CfgLinesSkipped()
        {
            string csv =
                "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs\n" +
                "#CFG,2026-03-30T10:00:00.000,Cam1_Ops=1.00\n" +
                "260330-100000,20260330_100000.000-1,0,0,0.3,0.6,3001,3001.0,149.0\n";
            WriteCsv("20260330", csv);

            var stats = InspectionStatisticsService.Compute(
                _tempRoot, new DateTime(2026, 3, 30), new DateTime(2026, 3, 31));
            Assert.That(stats[1].Pass, Is.EqualTo(1));
        }

        [Test]
        public void ComputeGroupedByHourOfDay_Returns24Entries()
        {
            string csv =
                "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs\n" +
                "260330-100000,20260330_100000.000-1,0,0,0.3,0.6,3001,3001.0,149.0\n" +
                "260330-143000,20260330_143000.000-1,1,0,0.9,1.2,3001,3001.0,149.0\n";
            WriteCsv("20260330", csv);

            var start = new DateTime(2026, 3, 30);
            var end   = new DateTime(2026, 3, 31);
            var periods = InspectionStatisticsService.ComputeGroupedByHourOfDay(_tempRoot, start, end);
            Assert.That(periods.Count, Is.EqualTo(24));
            Assert.That(periods[10].Pass, Is.EqualTo(1), "Hour 10 pass");
            Assert.That(periods[14].Fail, Is.EqualTo(1), "Hour 14 fail");
        }
    }
}
