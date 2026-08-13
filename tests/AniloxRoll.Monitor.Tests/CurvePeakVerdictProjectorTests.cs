using System.Collections.Generic;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.Core.Data;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class CurvePeakVerdictProjectorTests
    {
        [Test]
        public void Apply_ThresholdMovesBothDirections_ReplacesPriorVerdict()
        {
            var detail = new GrabDetail { GrabId = "g1" };
            var details = Details(detail);
            var columns = Columns(0.8f, 0.9f);
            var rows = Rows(0.8f, 0.9f);

            CurvePeakVerdictProjector.Apply(
                details, columns, rows,
                Context(0.5f, 0.5f, ColumnCurveDisplayMode.Both));
            Assert.That(detail.CamResult[0], Is.True);
            Assert.That(detail.RowResult, Is.True);

            CurvePeakVerdictProjector.Apply(
                details, columns, rows,
                Context(1.0f, 1.0f, ColumnCurveDisplayMode.Both));
            Assert.That(detail.CamResult[0], Is.False);
            Assert.That(detail.RowResult, Is.False);

            CurvePeakVerdictProjector.Apply(
                details, columns, rows,
                Context(0.5f, 0.5f, ColumnCurveDisplayMode.Both));
            Assert.That(detail.CamResult[0], Is.True);
            Assert.That(detail.RowResult, Is.True);
        }

        [TestCase(ColumnCurveDisplayMode.Mean, 0.8f, 0.1f, true)]
        [TestCase(ColumnCurveDisplayMode.Mean, 0.1f, 0.8f, false)]
        [TestCase(ColumnCurveDisplayMode.Max, 0.8f, 0.1f, false)]
        [TestCase(ColumnCurveDisplayMode.Max, 0.1f, 0.8f, true)]
        [TestCase(ColumnCurveDisplayMode.Both, 0.8f, 0.1f, true)]
        [TestCase(ColumnCurveDisplayMode.Both, 0.1f, 0.8f, true)]
        public void Apply_DisplayMode_UsesOnlyEnabledCurve(
            ColumnCurveDisplayMode mode, float mean, float max, bool expectedFail)
        {
            var detail = new GrabDetail { GrabId = "g1" };

            CurvePeakVerdictProjector.Apply(
                Details(detail), Columns(mean, max), Rows(mean, max),
                Context(0.5f, 0.5f, mode));

            Assert.That(detail.CamResult[0], Is.EqualTo(expectedFail));
            Assert.That(detail.RowResult, Is.EqualTo(expectedFail));
        }

        [Test]
        public void Apply_DirectionChange_ClearsDisabledAxisVerdicts()
        {
            var detail = new GrabDetail { GrabId = "g1" };
            var details = Details(detail);
            var columns = Columns(0.8f, 0.8f);
            var rows = Rows(0.8f, 0.8f);

            CurvePeakVerdictProjector.Apply(
                details, columns, rows,
                Context(0.5f, 0.5f, ColumnCurveDisplayMode.Both, RidgeDirection.Both));
            Assert.That(detail.CamResult[0], Is.True);
            Assert.That(detail.RowResult, Is.True);

            CurvePeakVerdictProjector.Apply(
                details, columns, rows,
                Context(0.5f, 0.5f, ColumnCurveDisplayMode.Both, RidgeDirection.Vertical));
            Assert.That(detail.CamResult[0], Is.True);
            Assert.That(detail.RowResult, Is.Null);

            CurvePeakVerdictProjector.Apply(
                details, columns, rows,
                Context(0.5f, 0.5f, ColumnCurveDisplayMode.Both, RidgeDirection.Horizontal));
            Assert.That(detail.CamResult[0], Is.Null);
            Assert.That(detail.RowResult, Is.True);
        }

        [Test]
        public void SettingsSnapshot_AnyVerdictInputChanges_DoesNotMatch()
        {
            ThresholdContext baseline = Context(
                0.5f, 0.6f, ColumnCurveDisplayMode.Both, RidgeDirection.Both,
                0.7f, 0.8f, 0.9f);
            CurveVerdictSettingsSnapshot snapshot =
                CurveVerdictSettingsSnapshot.Capture(baseline);

            Assert.That(snapshot.Matches(baseline), Is.True);
            Assert.That(snapshot.Matches(Context(0.5f, 0.6f, ColumnCurveDisplayMode.Both,
                RidgeDirection.Both, 0.7f, 0.8f, 0.9f, 1.01f)), Is.False);
            Assert.That(snapshot.Matches(Context(0.51f, 0.6f, ColumnCurveDisplayMode.Both,
                RidgeDirection.Both, 0.7f, 0.8f, 0.9f)), Is.False);
            Assert.That(snapshot.Matches(Context(0.5f, 0.61f, ColumnCurveDisplayMode.Both,
                RidgeDirection.Both, 0.7f, 0.8f, 0.9f)), Is.False);
            Assert.That(snapshot.Matches(Context(0.5f, 0.6f, ColumnCurveDisplayMode.Max,
                RidgeDirection.Both, 0.7f, 0.8f, 0.9f)), Is.False);
            Assert.That(snapshot.Matches(Context(0.5f, 0.6f, ColumnCurveDisplayMode.Both,
                RidgeDirection.Vertical, 0.7f, 0.8f, 0.9f)), Is.False);
            Assert.That(snapshot.Matches(Context(0.5f, 0.6f, ColumnCurveDisplayMode.Both,
                RidgeDirection.Both, 0.71f, 0.8f, 0.9f)), Is.False);
            Assert.That(snapshot.Matches(Context(0.5f, 0.6f, ColumnCurveDisplayMode.Both,
                RidgeDirection.Both, 0.7f, 0.81f, 0.9f)), Is.False);
            Assert.That(snapshot.Matches(Context(0.5f, 0.6f, ColumnCurveDisplayMode.Both,
                RidgeDirection.Both, 0.7f, 0.8f, 0.91f)), Is.False);
        }

        private static Dictionary<string, GrabDetail> Details(GrabDetail detail)
        {
            return new Dictionary<string, GrabDetail> { [detail.GrabId] = detail };
        }

        private static Dictionary<string, ColumnCurvePeakRecord[]> Columns(float mean, float max)
        {
            return new Dictionary<string, ColumnCurvePeakRecord[]>
            {
                ["g1"] = new[]
                {
                    new ColumnCurvePeakRecord
                    {
                        GrabId = "g1",
                        CameraId = 1,
                        CaptureHmV = 1f,
                        RawMeanPeak = mean,
                        RawMaxPeak = max
                    }
                }
            };
        }

        private static Dictionary<string, RowCurvePeakRecord> Rows(float mean, float max)
        {
            return new Dictionary<string, RowCurvePeakRecord>
            {
                ["g1"] = new RowCurvePeakRecord
                {
                    GrabId = "g1",
                    CaptureHmV = 1f,
                    RawMeanPeak = mean,
                    RawMaxPeak = max
                }
            };
        }

        private static ThresholdContext Context(
            float meanThreshold, float maxThreshold,
            ColumnCurveDisplayMode mode,
            RidgeDirection direction = RidgeDirection.Both,
            float rowNormalization = 1f,
            float rowMeanThreshold = float.NaN,
            float rowMaxThreshold = float.NaN,
            float columnNormalization = 1f)
        {
            if (float.IsNaN(rowMeanThreshold)) rowMeanThreshold = meanThreshold;
            if (float.IsNaN(rowMaxThreshold)) rowMaxThreshold = maxThreshold;
            return new ThresholdContext(
                columnNormalization, meanThreshold, maxThreshold,
                rowNormalization, rowMeanThreshold, rowMaxThreshold,
                mode, direction);
        }
    }
}
