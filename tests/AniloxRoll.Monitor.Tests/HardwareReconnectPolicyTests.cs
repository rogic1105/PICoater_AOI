using System;
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

        [TestCase(@"\\192.168.10.20\Anilox\Captures", "192.168.10.20")]
        [TestCase(@"\\storage-pc\Anilox", "storage-pc")]
        [TestCase("//storage-pc/Anilox", "storage-pc")]
        [TestCase("", null)]
        public void StorageHealth_ParseUncHostReturnsServer(
            string path,
            string expected)
        {
            Assert.That(
                StorageHealthCoordinator.ParseUncHost(path),
                Is.EqualTo(expected));
        }

        [Test]
        public void StorageHealth_InvalidDrivePathDoesNotReportCapacity()
        {
            long freeBytes;
            long totalBytes;

            bool success = StorageHealthCoordinator.TryReadDriveCapacity(
                string.Empty,
                out freeBytes,
                out totalBytes);

            Assert.That(success, Is.False);
            Assert.That(freeBytes, Is.EqualTo(-1));
            Assert.That(totalBytes, Is.EqualTo(0));
        }

        [TestCase(true, 5, true)]
        [TestCase(true, 15, true)]
        [TestCase(true, 16, false)]
        [TestCase(false, 5, false)]
        public void StorageHealth_TransientHeartbeatReadUsesFreshLastSuccess(
            bool previousAlive,
            int ageSeconds,
            bool expected)
        {
            DateTime now = DateTime.UtcNow;

            Assert.That(
                StorageHealthCoordinator.ShouldKeepRemoteAppAlive(
                    previousAlive,
                    now.AddSeconds(-ageSeconds),
                    now),
                Is.EqualTo(expected));
        }
    }
}
