using AniloxRoll.Monitor.UI.Managers;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class AcquisitionFramePeriodPolicyTests
    {
        [Test]
        public void IsWithinTolerance_MatchingHardwarePeriod_Passes()
        {
            bool aligned = AcquisitionFramePeriodPolicy.IsWithinTolerance(
                6000, 6000, 1000, 126000, 1, 125000,
                out double expectedMs, out double actualMs, out double toleranceMs);

            Assert.That(aligned, Is.True);
            Assert.That(expectedMs, Is.EqualTo(1000).Within(0.001));
            Assert.That(actualMs, Is.EqualTo(1000).Within(0.001));
            Assert.That(toleranceMs, Is.EqualTo(200).Within(0.001));
        }

        [Test]
        public void IsWithinTolerance_HalfSpeedCamera_Fails()
        {
            bool aligned = AcquisitionFramePeriodPolicy.IsWithinTolerance(
                6000, 6000, 1000, 251000, 1, 125000,
                out _, out _, out _);

            Assert.That(aligned, Is.False);
        }

        [Test]
        public void IsWithinTolerance_MultipleObservedFrames_UsesAveragePeriod()
        {
            bool aligned = AcquisitionFramePeriodPolicy.IsWithinTolerance(
                3000, 6000, 1000, 188500, 3, 125000,
                out _, out double actualMs, out _);

            Assert.That(aligned, Is.True);
            Assert.That(actualMs, Is.EqualTo(500).Within(0.001));
        }

        [Test]
        public void IsWithinTolerance_MissingMeasurement_Fails()
        {
            bool aligned = AcquisitionFramePeriodPolicy.IsWithinTolerance(
                6000, 6000, 1000, 1000, 0, 125000,
                out _, out _, out _);

            Assert.That(aligned, Is.False);
        }
    }
}
