using AniloxRoll.Monitor.UI.Coordinators;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class HardwareReconnectPolicyTests
    {
        [TestCase(1, false)]
        [TestCase(4, false)]
        [TestCase(5, true)]
        [TestCase(6, false)]
        [TestCase(10, true)]
        public void LightReconnect_FullPortScanRunsEveryFiveAttempts(int attempt, bool expected)
        {
            Assert.That(
                LightConnectionCoordinator.ShouldRunFullPortScan(attempt),
                Is.EqualTo(expected));
        }
    }
}
