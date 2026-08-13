using System;
using System.Collections.Generic;
using System.Linq;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Presenters;
using AniloxRoll.Monitor.UI.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class ReportCurveVerdictPresenterTests
    {
        [Test]
        public void ApplyCurrentIfNeeded_ThresholdChange_ReplacesVerdictsAndAuditsList()
        {
            var index = new ReportCurveVerdictIndex();
            var detail = new GrabDetail { GrabId = "g1" };
            ThresholdContext threshold = Context(1.0f);
            index.ReplaceDetails("D:\\data", Details(detail), threshold);
            index.ColumnPeaks["g1"] = Columns(0.8f, 0.9f);
            index.RowPeaks["g1"] = Row(0.8f, 0.9f);
            var logs = new List<string>();
            var presenter = Create(index, () => threshold, logs);

            threshold = Context(0.5f);
            bool changed = presenter.ApplyCurrentIfNeeded("g1");

            Assert.That(changed, Is.True);
            Assert.That(detail.CamResult[0], Is.True);
            Assert.That(detail.RowResult, Is.True);
            Assert.That(logs.Any(line => line ==
                "DT verdict refresh source=peak-index columns=1 rows=1"), Is.True);
            Assert.That(logs.Any(line =>
                line.StartsWith("DT verdict audit g1 trigger=settings cam=1 ") &&
                line.Contains("result=fail") && line.Contains("list=fail")), Is.True);
            Assert.That(logs.Any(line =>
                line.StartsWith("DT row verdict audit g1 trigger=settings ") &&
                line.Contains("result=fail") && line.Contains("list=fail")), Is.True);
        }

        [Test]
        public void ApplyVisibleCurves_MergedColumnAndRow_UpdateSameDetail()
        {
            var index = new ReportCurveVerdictIndex();
            var detail = new GrabDetail { GrabId = "g1" };
            ThresholdContext threshold = Context(0.5f);
            index.ReplaceDetails("D:\\data", Details(detail), threshold);
            var logs = new List<string>();
            var presenter = Create(index, () => threshold, logs);
            var data = new SingleGrabCurveData
            {
                Config = Config(),
                ColumnMean = new[] { new[] { 0f, 204f, 0f } },
                ColumnMax = new[] { new[] { 0f, 230f, 0f } },
                MergedRowMean = new[] { 0f, 204f, 0f },
                MergedRowMax = new[] { 0f, 230f, 0f }
            };

            bool applied = presenter.ApplyVisibleCurves("g1", data, detail);

            Assert.That(applied, Is.True, "visible curves should be accepted");
            Assert.That(detail.CamResult[0], Is.True, "column peak should fail");
            Assert.That(detail.RowResult, Is.True, "row peak should fail");
            Assert.That(index.ColumnPeaks.ContainsKey("g1"), Is.True);
            Assert.That(index.RowPeaks.ContainsKey("g1"), Is.True);
            Assert.That(logs.Any(line =>
                line.StartsWith("DT verdict g1 cam=1 ") &&
                line.EndsWith("source=visible-merged-curve")), Is.True);
            Assert.That(logs.Any(line =>
                line.StartsWith("DT row verdict g1 merged=1 ") &&
                line.EndsWith("source=visible-merged-curve")), Is.True);
        }

        [Test]
        public void AuditSelected_DvtDisabled_ProducesNoEvidence()
        {
            var index = new ReportCurveVerdictIndex();
            index.ReplaceDetails("D:\\data", Details(new GrabDetail { GrabId = "g1" }), Context(1f));
            var logs = new List<string>();
            var presenter = new ReportCurveVerdictPresenter(
                index, () => Context(1f), Config, () => 1, () => false, logs.Add);

            presenter.AuditSelected("g1", "settings");

            Assert.That(logs, Is.Empty);
        }

        private static ReportCurveVerdictPresenter Create(
            ReportCurveVerdictIndex index,
            Func<ThresholdContext> threshold,
            IList<string> logs)
        {
            return new ReportCurveVerdictPresenter(
                index, threshold, Config, () => 1, () => true, logs.Add);
        }

        private static Dictionary<string, GrabDetail> Details(GrabDetail detail)
        {
            return new Dictionary<string, GrabDetail> { [detail.GrabId] = detail };
        }

        private static ColumnCurvePeakRecord[] Columns(float mean, float max)
        {
            return new[]
            {
                new ColumnCurvePeakRecord
                {
                    GrabId = "g1",
                    CameraId = 1,
                    CaptureHmV = 1f,
                    RawMeanPeak = mean,
                    RawMaxPeak = max
                }
            };
        }

        private static RowCurvePeakRecord Row(float mean, float max)
        {
            return new RowCurvePeakRecord
            {
                GrabId = "g1",
                CaptureHmV = 1f,
                RawMeanPeak = mean,
                RawMaxPeak = max
            };
        }

        private static ThresholdContext Context(float threshold)
        {
            return new ThresholdContext(
                1f, threshold, threshold,
                1f, threshold, threshold,
                ColumnCurveDisplayMode.Both, RidgeDirection.Both);
        }

        private static CsvConfigSnapshot Config()
        {
            return new CsvConfigSnapshot(
                new[] { 100.0 }, new[] { 0.0 }, new[] { 3 },
                new[] { 50.0 }, new[] { 3000.0 },
                1f, 1f, 0f,
                0.5f, 0.5f, 0.5f, 0.5f,
                0, 0, DateTime.UtcNow);
        }
    }
}
