using System;
using System.IO;
using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class StorageAppHeartbeatServiceTests
    {
        private string _root;
        private string _config;
        private string _captures;

        [SetUp]
        public void SetUp()
        {
            _root = Path.Combine(Path.GetTempPath(), "StorageHeartbeat_" + Guid.NewGuid().ToString("N"));
            _config = Path.Combine(_root, "Config");
            _captures = Path.Combine(_root, "Captures");
            Directory.CreateDirectory(_config);
            Directory.CreateDirectory(_captures);
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_root, true); } catch { }
        }

        [Test]
        public void PublishNow_WritesReadableHeartbeat_ThenBecomesStale()
        {
            using (var service = new StorageAppHeartbeatService(() => _config, () => _captures))
            {
                service.PublishNow();

                StorageAppHeartbeatRecord record;
                string error;
                Assert.That(StorageAppHeartbeatService.TryRead(
                    _config, DateTime.UtcNow, out record, out error), Is.True, error);
                Assert.That(record.ProcessId, Is.EqualTo(System.Diagnostics.Process.GetCurrentProcess().Id));
                Assert.That(record.TotalBytes, Is.GreaterThan(0));

                Assert.That(StorageAppHeartbeatService.TryRead(
                    _config,
                    record.LastSeenUtc.ToUniversalTime() + StorageAppHeartbeatService.StaleAfter + TimeSpan.FromSeconds(1),
                    out record,
                    out error), Is.False);
                Assert.That(error, Does.Contain("stale"));
            }
        }

        [Test]
        public void RecordCleanup_PublishesCleanupResult()
        {
            using (var service = new StorageAppHeartbeatService(() => _config, () => _captures))
            {
                service.RecordCleanup(12345);

                StorageAppHeartbeatRecord record;
                string error;
                Assert.That(StorageAppHeartbeatService.TryRead(
                    _config, DateTime.UtcNow, out record, out error), Is.True, error);
                Assert.That(record.LastCleanupUtc, Is.Not.Null);
                Assert.That(record.LastCleanupFreedBytes, Is.EqualTo(12345));
            }
        }
    }
}
