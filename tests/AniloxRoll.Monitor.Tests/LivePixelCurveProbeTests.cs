using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class LivePixelCurveProbeTests
    {
        [Test]
        public void Measure_SameFrameImageAndMaxCurves_ReportsMatch()
        {
            LivePixelCurveProbeResult result = LivePixelCurveProbe.Measure(
                new byte[] { 0, 40, 120 },
                new byte[] { 0, 80, 200 },
                new[] { 10f, 30f },
                new[] { 20f, 120f },
                new[] { 25f, 50f },
                new[] { 30f, 200f },
                captureHm: 1f,
                currentColumnHm: 0.5f,
                currentRowHm: 0.5f,
                columnSourceGain: 1f,
                rowSourceGain: 1f);

            Assert.That(result.Column.MaxMatches, Is.True);
            Assert.That(result.Row.MaxMatches, Is.True);
            Assert.That(result.Column.DisplayImagePeak, Is.EqualTo(60f / 255f).Within(0.0001f));
            Assert.That(result.Row.DisplayImagePeak, Is.EqualTo(100f / 255f).Within(0.0001f));
        }

        [Test]
        public void Measure_DifferentImageAndMaxCurve_ReportsMismatch()
        {
            LivePixelCurveProbeResult result = LivePixelCurveProbe.Measure(
                new byte[] { 10 },
                new byte[] { 20 },
                new[] { 5f },
                new[] { 200f },
                new[] { 5f },
                new[] { 20f },
                captureHm: 1f,
                currentColumnHm: 1f,
                currentRowHm: 1f,
                columnSourceGain: 1f,
                rowSourceGain: 1f);

            Assert.That(result.Column.MaxMatches, Is.False);
            Assert.That(result.Row.MaxMatches, Is.True);
        }

        [Test]
        public void Measure_FloatCurveWithinOneGrayStep_ReportsMatch()
        {
            LivePixelCurveProbeResult result = LivePixelCurveProbe.Measure(
                new byte[] { 100 },
                new byte[] { 100 },
                new[] { 20f },
                new[] { 100.75f },
                new[] { 20f },
                new[] { 100.75f },
                captureHm: 1f,
                currentColumnHm: 1f,
                currentRowHm: 1f,
                columnSourceGain: 1f,
                rowSourceGain: 1f);

            Assert.That(result.Column.MaxMatches, Is.True);
            Assert.That(result.Row.MaxMatches, Is.True);
            Assert.That(result.Column.MaxDelta, Is.GreaterThan(0.5f / 255f));
        }

        [Test]
        public void Measure_NeutralImageAndCaptureScaledCurve_CompareAtCurrentGain()
        {
            LivePixelCurveProbeResult result = LivePixelCurveProbe.Measure(
                new byte[] { 0, 204 },
                new byte[] { 0, 102 },
                new[] { 10f, 20f },
                new[] { 20f, 51f },
                new[] { 20f, 50f },
                new[] { 20f, 25.5f },
                captureHm: 2f,
                currentColumnHm: 0.5f,
                currentRowHm: 0.5f,
                columnSourceGain: 2f,
                rowSourceGain: 2f);

            Assert.That(result.Column.MaxMatches, Is.True);
            Assert.That(result.Row.MaxMatches, Is.True);
            Assert.That(result.Column.DisplayImagePeak, Is.EqualTo(51f / 255f).Within(0.0001f));
            Assert.That(result.Row.DisplayImagePeak, Is.EqualTo(26f / 255f).Within(0.0001f));
        }

        [Test]
        public void Measure_NonzeroCurveCollapsedToBlackImage_ReportsInformationLoss()
        {
            LivePixelCurveProbeResult result = LivePixelCurveProbe.Measure(
                new byte[] { 0 },
                new byte[] { 0 },
                new[] { 0.2f },
                new[] { 0.75f },
                new[] { 0.1f },
                new[] { 0.25f },
                captureHm: 10.3f,
                currentColumnHm: 10.3f,
                currentRowHm: 41.5f,
                columnSourceGain: 1f,
                rowSourceGain: 1f);

            Assert.That(result.Column.QuantizedToZero, Is.True);
            Assert.That(result.Row.QuantizedToZero, Is.True);
            Assert.That(result.Column.MaxMatches, Is.False);
            Assert.That(result.Row.MaxMatches, Is.False);
        }
    }
}
