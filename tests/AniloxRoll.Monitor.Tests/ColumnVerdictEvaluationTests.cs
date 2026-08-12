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
            ThresholdContext context = Create(ColumnCurveDisplayMode.Max, currentHmV: 2f);

            ColumnVerdictEvaluation result = context.EvaluateColumn(0.10f, 0.40f, 1f);

            Assert.That(result.DisplayMaxPeak, Is.EqualTo(0.80f).Within(0.0001f));
            Assert.That(result.Cause, Is.EqualTo(ColumnFailureCause.Max));
        }

        [TestCase(0.5f, 0.5f, 0.4f)]
        [TestCase(0.5f, 1.0f, 0.8f)]
        [TestCase(0.5f, 1.5f, 1.2f)]
        public void EvaluateColumn_CurrentNormalizationScalesPeakLinearly(
            float captureNormalization,
            float currentNormalization,
            float expectedPeak)
        {
            var context = new ThresholdContext(
                currentNormalization, 2f, 2f,
                1f, 2f, 2f,
                ColumnCurveDisplayMode.Max);

            ColumnVerdictEvaluation result = context.EvaluateColumn(
                float.NaN, 0.4f, captureNormalization);

            Assert.That(result.DisplayMaxPeak, Is.EqualTo(expectedPeak).Within(0.0001f));
            Assert.That(result.IsFail, Is.False);
        }

        [Test]
        public void EvaluateRawColumnCurve_UsesSameScaleAsDisplayedCurve()
        {
            var context = new ThresholdContext(
                1f, 0.30f, 0.60f,
                1f, 0.20f, 0.60f,
                ColumnCurveDisplayMode.Both);

            ColumnVerdictEvaluation result = context.EvaluateRawColumnCurve(
                0.06982818f, 1.52019489f, 0.3f);

            Assert.That(result.DisplayMeanPeak, Is.EqualTo(0.02094845f).Within(0.0001f));
            Assert.That(result.DisplayMaxPeak, Is.EqualTo(0.45605847f).Within(0.0001f));
            Assert.That(result.IsFail, Is.False);
        }

        [TestCase(ColumnCurveDisplayMode.Mean, true, false)]
        [TestCase(ColumnCurveDisplayMode.Max, false, true)]
        [TestCase(ColumnCurveDisplayMode.Both, true, true)]
        public void EvaluateColumn_ReportsExactlyTheEnabledMetrics(
            ColumnCurveDisplayMode mode,
            bool meanEnabled,
            bool maxEnabled)
        {
            ColumnVerdictEvaluation result = Create(mode).EvaluateColumn(
                0.1f, 0.1f, 1f);

            Assert.That(result.MeanEnabled, Is.EqualTo(meanEnabled));
            Assert.That(result.MaxEnabled, Is.EqualTo(maxEnabled));
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

        [TestCase(ColumnCurveDisplayMode.Mean, 0.10f, 0.90f, false)]
        [TestCase(ColumnCurveDisplayMode.Mean, 0.30f, 0.10f, true)]
        [TestCase(ColumnCurveDisplayMode.Max, 0.90f, 0.50f, false)]
        [TestCase(ColumnCurveDisplayMode.Max, 0.10f, 0.70f, true)]
        [TestCase(ColumnCurveDisplayMode.Both, 0.30f, 0.10f, true)]
        [TestCase(ColumnCurveDisplayMode.Both, 0.10f, 0.70f, true)]
        public void IsRowFail_AppliesSelectedDetectionMetrics(
            ColumnCurveDisplayMode mode,
            float meanPeak,
            float maxPeak,
            bool expected)
        {
            bool? result = Create(mode).IsRowFail(meanPeak, maxPeak, 1f);

            Assert.That(result, Is.EqualTo(expected));
        }

        [Test]
        public void IsRowFail_WhenSelectedMetricIsMissing_ReturnsUnknown()
        {
            bool? result = Create(ColumnCurveDisplayMode.Mean).IsRowFail(
                float.NaN, 0.90f, 1f);

            Assert.That(result, Is.Null);
        }

        [Test]
        public void EvaluateRawRowCurve_NormalizationCanCrossThresholdBothDirections()
        {
            var low = new ThresholdContext(
                1f, 1f, 1f,
                0.5f, 0.7f, 0.6f,
                ColumnCurveDisplayMode.Max,
                RidgeDirection.Both);
            var high = new ThresholdContext(
                1f, 1f, 1f,
                3.4f, 0.7f, 0.6f,
                ColumnCurveDisplayMode.Max,
                RidgeDirection.Both);

            ColumnVerdictEvaluation lowResult = low.EvaluateRawRowCurve(
                float.NaN, 0.25f, 1f);
            ColumnVerdictEvaluation highResult = high.EvaluateRawRowCurve(
                float.NaN, 0.25f, 1f);

            Assert.That(lowResult.DisplayMaxPeak, Is.EqualTo(0.125f).Within(0.0001f));
            Assert.That(lowResult.IsFail, Is.False);
            Assert.That(highResult.DisplayMaxPeak, Is.EqualTo(0.85f).Within(0.0001f));
            Assert.That(highResult.IsFail, Is.True);
        }

        [TestCase(RidgeDirection.Vertical, true, false)]
        [TestCase(RidgeDirection.Horizontal, false, true)]
        [TestCase(RidgeDirection.Both, true, true)]
        public void DetectionDirection_EnablesOnlyRequestedListAxes(
            RidgeDirection direction, bool columnEnabled, bool rowEnabled)
        {
            var context = new ThresholdContext(
                1f, 0.2f, 0.6f,
                1f, 0.2f, 0.6f,
                ColumnCurveDisplayMode.Both,
                direction);

            Assert.That(context.ColumnDetectionEnabled, Is.EqualTo(columnEnabled));
            Assert.That(context.RowDetectionEnabled, Is.EqualTo(rowEnabled));
            Assert.That(context.EvaluateRawColumnCurve(1f, 1f, 1f).HasData,
                Is.EqualTo(columnEnabled));
            Assert.That(context.EvaluateRawRowCurve(1f, 1f, 1f).HasData,
                Is.EqualTo(rowEnabled));
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

            Assert.That(records[0].RawMeanPeak, Is.EqualTo(30f / 255f).Within(0.0001f));
            Assert.That(records[1].RawMeanPeak, Is.EqualTo(70f / 255f).Within(0.0001f));
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
