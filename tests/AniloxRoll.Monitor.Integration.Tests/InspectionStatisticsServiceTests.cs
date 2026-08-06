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

            // 明細列表顯示序為新→舊：最新的 260330-100100 在前、最舊的 260330-100000 在後
            Assert.That(details[0].GrabId, Is.EqualTo("260330-100100"));
            Assert.That(details[0].CamResult[0], Is.False, "CAM1 pass");
            Assert.That(details[0].CamResult[1], Is.Null,  "CAM2 no data");

            Assert.That(details[1].GrabId, Is.EqualTo("260330-100000"));
            Assert.That(details[1].CamResult[0], Is.False, "CAM1 pass");
            Assert.That(details[1].CamResult[1], Is.True,  "CAM2 fail");

            var stats = InspectionStatisticsService.ComputeStatsFromDetails(details);
            Assert.That(stats[1].Pass, Is.EqualTo(2));
            Assert.That(stats[1].Fail, Is.EqualTo(0));
            Assert.That(stats[2].Pass, Is.EqualTo(0));
            Assert.That(stats[2].Fail, Is.EqualTo(1));
            Assert.That(stats[3].Total, Is.EqualTo(0));
        }

        [Test]
        public void ComputeDetailedByGrabIdRange_ColumnSummaryOverridesFrameVeto()
        {
            string csv =
                "#CFG,2026-08-04T08:55:59.000,HessianMaxFactorV=0.5000\n" +
                "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs\n" +
                "260804-085559,20260804_085559.000-1,0,1,0.47,0.49,3000,3000.0,50.0\n" +
                "#CURVE-C,1,260804-085559,1,0.5,0.15,0.49\n";
            WriteCsv("20260804", csv);

            var threshold = new ThresholdContext(0.5f, 0.2f, 0.6f);
            List<GrabDetail> details = InspectionStatisticsService.ComputeDetailedByGrabIdRange(
                _tempRoot, "260804-085559", "260804-085559", threshold);

            Assert.That(details, Has.Count.EqualTo(1));
            Assert.That(details[0].CamResult[0], Is.False,
                "Merged CurveMean is below the mean threshold and merged CurveMax is below the max threshold");
        }

        [Test]
        public void ComputeDetailedByGrabIdRange_ColumnSummaryUsesMaxThresholdForCurveMax()
        {
            string csv =
                "#CFG,2026-08-04T08:55:59.000,HessianMaxFactorV=0.5000\n" +
                "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs\n" +
                "260804-085559,20260804_085559.000-1,0,0,0.10,0.30,3000,3000.0,50.0\n" +
                "#CURVE-C,1,260804-085559,1,0.5,0.15,0.61\n";
            WriteCsv("20260804", csv);

            var threshold = new ThresholdContext(0.5f, 0.2f, 0.6f);
            List<GrabDetail> details = InspectionStatisticsService.ComputeDetailedByGrabIdRange(
                _tempRoot, "260804-085559", "260804-085559", threshold);

            Assert.That(details[0].CamResult[0], Is.True,
                "Merged CurveMax must cross the max threshold, not the mean threshold");
        }

        [Test]
        public void IsColumnCurveFail_UsesSeparateMeanAndMaxThresholds()
        {
            var threshold = new ThresholdContext(0.5f, 0.2f, 0.6f);

            bool? failed = threshold.IsColumnCurveFail(
                new[] { 25.5f }, new[] { 127.5f }, 0.5f,
                out float meanPeak, out float maxPeak);

            Assert.That(failed, Is.False);
            Assert.That(meanPeak, Is.EqualTo(0.1f).Within(0.0001f));
            Assert.That(maxPeak, Is.EqualTo(0.5f).Within(0.0001f));
        }

        [TestCase(0.21f, 0.50f, ColumnFailureCause.Mean)]
        [TestCase(0.10f, 0.61f, ColumnFailureCause.Max)]
        [TestCase(0.21f, 0.61f, ColumnFailureCause.Both)]
        [TestCase(0.20f, 0.60f, ColumnFailureCause.None)]
        public void GetColumnFailureCause_IdentifiesTheThresholdThatWasCrossed(
            float meanPeak, float maxPeak, ColumnFailureCause expected)
        {
            var threshold = new ThresholdContext(0.5f, 0.2f, 0.6f);

            ColumnFailureCause actual = threshold.GetColumnFailureCause(
                meanPeak, maxPeak, 0.5f);

            Assert.That(actual, Is.EqualTo(expected));
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
        public void LoadSnapshot_OnePassIndexesFeedPeriodCharts()
        {
            string csv =
                "#CFG,2026-03-30T10:00:00.000,HessianMaxFactorV=1.0000\n" +
                "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs\n" +
                "260330-100000,20260330_100000.000-1,0,0,0.1,0.2,3001,3001.0,149.0\n" +
                "260330-100000,20260330_100001.000-2,1,0,0.1,0.8,3001,3001.0,149.0\n" +
                "260330-140000,20260330_140000.000-1,0,0,0.1,0.2,3001,3001.0,149.0\n";
            WriteCsv("20260330", csv);

            var threshold = new ThresholdContext(1f, 0.2f, 0.5f);
            InspectionStatisticsSnapshot snapshot =
                InspectionStatisticsService.LoadSnapshot(_tempRoot, threshold);

            Assert.That(snapshot.CsvFileCount, Is.EqualTo(1));
            Assert.That(snapshot.RecordCount, Is.EqualTo(3));
            Assert.That(snapshot.GrabIdsDescending.Count, Is.EqualTo(2));
            Assert.That(snapshot.GrabIdsDescending[0].GrabId, Is.EqualTo("260330-140000"));
            Assert.That(snapshot.AvailableTimes.Count, Is.EqualTo(3));
            Assert.That(snapshot.DetailsByGrabId["260330-100000"].CamResult[0], Is.False);
            Assert.That(snapshot.DetailsByGrabId["260330-100000"].CamResult[1], Is.True);

            var hourly = InspectionStatisticsService.ComputeGroupedByHourOfDay(
                snapshot.GrabIdsDescending,
                snapshot.DetailsByGrabId,
                new DateTime(2026, 3, 30),
                new DateTime(2026, 3, 30, 23, 59, 59));
            Assert.That(hourly[10].Pass, Is.EqualTo(1));
            Assert.That(hourly[10].Fail, Is.EqualTo(1));
            Assert.That(hourly[14].Pass, Is.EqualTo(1));
        }

        [Test]
        public void LoadSnapshot_RowPeaksUseCurrentRowThresholdAndVtoHScale()
        {
            string csv =
                "#CFG,2026-03-30T10:00:00.000,HessianMaxFactorV=0.5000\n" +
                "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs,MaxCMean,MeanRPeak,MaxRPeak\n" +
                "260330-100000,20260330_100000.000-1,0,0,0.1,0.2,3001,3001.0,149.0,0.1,0.2,0.8\n" +
                "260330-100100,20260330_100100.000-1,0,0,0.1,0.2,3001,3001.0,149.0,0.1,0.1,0.1\n";
            WriteCsv("20260330", csv);

            var threshold = new ThresholdContext(
                0.5f, 0.2f, 0.6f,
                1.0f, 0.2f, 0.3f);
            InspectionStatisticsSnapshot snapshot =
                InspectionStatisticsService.LoadSnapshot(_tempRoot, threshold);

            Assert.That(snapshot.DetailsByGrabId["260330-100000"].RowResult, Is.True,
                "0.8 * (HM_H_current 1.0 / HM_V_capture 0.5) = 1.6 > 0.3");
            Assert.That(snapshot.DetailsByGrabId["260330-100100"].RowResult, Is.False);
        }

        [Test]
        public void LoadSnapshot_LegacyCsvWithoutRowPeaksKeepsRowUnknown()
        {
            string csv =
                "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs\n" +
                "260330-100000,20260330_100000.000-1,0,0,0.1,0.2,3001,3001.0,149.0\n";
            WriteCsv("20260330", csv);

            InspectionStatisticsSnapshot snapshot = InspectionStatisticsService.LoadSnapshot(
                _tempRoot, new ThresholdContext(1f, 0.2f, 0.6f, 1f, 0.2f, 0.6f));

            Assert.That(snapshot.DetailsByGrabId["260330-100000"].RowResult, Is.Null);
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
