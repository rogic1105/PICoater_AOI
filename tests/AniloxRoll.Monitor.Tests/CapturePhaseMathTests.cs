using AniloxRoll.Monitor.Core.Camera;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class CapturePhaseMathTests
    {
        [TestCase(100L, 102L, 2L)]
        [TestCase(998L, 2L, 4L)]
        [TestCase(100L, 1100L, 0L)]
        [TestCase(0L, 722L, 278L)]
        public void CircularSpread_UsesNearestFramePhase(
            long first, long second, long expected)
        {
            long spread;
            bool measured = CapturePhaseMath.TryGetCircularSpreadTicks(
                new[] { first, second }, 1000, out spread);

            Assert.That(measured, Is.True);
            Assert.That(spread, Is.EqualTo(expected));
        }

        [Test]
        public void CircularSpread_UsesSmallestArcForMultipleCameras()
        {
            long spread;
            bool measured = CapturePhaseMath.TryGetCircularSpreadTicks(
                new[] { 998L, 1L, 3L }, 1000, out spread);

            Assert.That(measured, Is.True);
            Assert.That(spread, Is.EqualTo(5));
        }

        [Test]
        public void CircularSpread_UsesCapturedWarmTicksWithoutLaterFrameDrift()
        {
            long spread;
            bool measured = CapturePhaseMath.TryGetCircularSpreadTicks(
                new[] { 83908918285779L, 83908918327483L },
                125000000L,
                out spread);

            Assert.That(measured, Is.True);
            Assert.That(spread, Is.EqualTo(41704L));
        }

        [Test]
        public void CircularSpread_RejectsInvalidInput()
        {
            long spread;
            Assert.That(
                CapturePhaseMath.TryGetCircularSpreadTicks(
                    new long[0], 1000, out spread),
                Is.False);
            Assert.That(
                CapturePhaseMath.TryGetCircularSpreadTicks(
                    new[] { 1L }, 0, out spread),
                Is.False);
        }
    }
}
