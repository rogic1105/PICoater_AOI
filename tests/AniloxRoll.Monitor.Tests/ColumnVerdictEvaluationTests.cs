using System;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class ColumnVerdictEvaluationTests
    {
        [Test]
        public void MeanMode_IgnoresMaximumThreshold()
        {
            ThresholdContext context = Create(ColumnCurveDisplayMode.Mean);

            ColumnVerdictEvaluation result = context.EvaluateColumn(0.10f, 0.90f, 1f);

            Assert.That(result.IsFail, Is.False);
            Assert.That(result.Cause, Is.EqualTo(ColumnFailureCause.None));
            Assert.That(result.MeanEnabled, Is.True);
            Assert.That(result.MaxEnabled, Is.False);
        }

        [Test]
        public void MaxMode_IgnoresMeanThreshold()
        {
            ThresholdContext context = Create(ColumnCurveDisplayMode.Max);

            ColumnVerdictEvaluation result = context.EvaluateColumn(0.90f, 0.40f, 1f);

            Assert.That(result.IsFail, Is.False);
            Assert.That(result.Cause, Is.EqualTo(ColumnFailureCause.None));
            Assert.That(result.MeanEnabled, Is.False);
            Assert.That(result.MaxEnabled, Is.True);
        }

        [TestCase(0.21f, 0.50f, ColumnFailureCause.Mean)]
        [TestCase(0.10f, 0.61f, ColumnFailureCause.Max)]
        [TestCase(0.21f, 0.61f, ColumnFailureCause.Both)]
        [TestCase(0.20f, 0.60f, ColumnFailureCause.None)]
        public void BothMode_UsesIndependentThresholds(
            float meanPeak, float maxPeak, ColumnFailureCause expectedCause)
        {
            ThresholdContext context = Create(ColumnCurveDisplayMode.Both);

            ColumnVerdictEvaluation result = context.EvaluateColumn(meanPeak, maxPeak, 1f);

            Assert.That(result.Cause, Is.EqualTo(expectedCause));
            Assert.That(result.IsFail, Is.EqualTo(expectedCause != ColumnFailureCause.None));
        }

        [Test]
        public void EvaluateColumn_RescalesCapturePeakToCurrentNormalization()
        {
            ThresholdContext context = Create(ColumnCurveDisplayMode.Max, currentHmV: 0.5f);

            ColumnVerdictEvaluation result = context.EvaluateColumn(0.10f, 0.40f, 1f);

            Assert.That(result.DisplayMaxPeak, Is.EqualTo(0.80f).Within(0.0001f));
            Assert.That(result.Cause, Is.EqualTo(ColumnFailureCause.Max));
        }

        [Test]
        public void EvaluateColumn_SingleMeanValueBelowThreshold_Passes()
        {
            var context = new ThresholdContext(
                1f, 0.80f, 2f,
                1f, 0.20f, 0.60f,
                ColumnCurveDisplayMode.Mean);

            ColumnVerdictEvaluation result = context.EvaluateColumn(
                0.70f, float.NaN, 1f);

            Assert.That(result.DisplayMeanPeak, Is.EqualTo(0.70f).Within(0.0001f));
            Assert.That(result.Cause, Is.EqualTo(ColumnFailureCause.None));
            Assert.That(result.IsFail, Is.False);
        }

        [Test]
        public void IsColumnCurveFail_WhenSelectedMetricIsMissing_ReturnsUnknown()
        {
            ThresholdContext context = Create(ColumnCurveDisplayMode.Mean);

            bool? result = context.IsColumnCurveFail(
                null,
                new[] { 255f },
                1f,
                out _,
                out _);

            Assert.That(result, Is.Null);
        }

        [TestCase(ColumnCurveDisplayMode.Mean, 0.21f, 0.10f, ColumnFailureCause.Mean)]
        [TestCase(ColumnCurveDisplayMode.Mean, 0.10f, 0.61f, ColumnFailureCause.None)]
        [TestCase(ColumnCurveDisplayMode.Max, 0.21f, 0.10f, ColumnFailureCause.None)]
        [TestCase(ColumnCurveDisplayMode.Max, 0.10f, 0.61f, ColumnFailureCause.Max)]
        [TestCase(ColumnCurveDisplayMode.Both, 0.21f, 0.61f, ColumnFailureCause.Both)]
        public void EvaluateColumnFailureCause_AppliesSelectedDetectionMetrics(
            ColumnCurveDisplayMode mode,
            float meanPeak,
            float maxPeak,
            ColumnFailureCause expected)
        {
            ColumnFailureCause result = ThresholdContext.EvaluateColumnFailureCause(
                meanPeak, maxPeak, 0.20f, 0.60f, mode);

            Assert.That(result, Is.EqualTo(expected));
        }

        [Test]
        public void ProjectVisibleRecords_IgnoresPeakHiddenByOverlapOwner()
        {
            var config = new CsvConfigSnapshot(
                new[] { 1000.0, 1000.0 },
                new[] { 0.0, 2.0 },
                new int[2], new double[2], new double[2],
                1f, 1f, 0.2f, 0.6f, 0.2f, 0.6f,
                0, 0, DateTime.UtcNow);
            float[][] means =
            {
                new[] { 10f, 20f, 30f, 250f },
                new[] { 40f, 50f, 60f, 70f }
            };
            float[][] maxes =
            {
                new[] { 10f, 20f, 30f, 250f },
                new[] { 40f, 50f, 60f, 70f }
            };

            ColumnCurvePeakRecord[] records = ColumnCurvePeakIndex.ProjectVisibleRecords(
                "grab", means, maxes, config, 1f, 2);

            Assert.That(records[0].MeanPeak, Is.EqualTo(30f / 255f).Within(0.0001f));
            Assert.That(records[1].MeanPeak, Is.EqualTo(70f / 255f).Within(0.0001f));
        }

        private static ThresholdContext Create(
            ColumnCurveDisplayMode mode,
            float currentHmV = 1f)
        {
            return new ThresholdContext(
                currentHmV, 0.20f, 0.60f,
                1f, 0.20f, 0.60f,
                mode);
        }
    }
}
