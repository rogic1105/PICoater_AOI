using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using NUnit.Framework;
using StorageBridge.Core;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    [NonParallelizable]
    public class RemoteCopyServiceIntegrationTests
    {
        private string _tempRoot;
        private string _localRoot;
        private string _remoteRoot;

        [SetUp]
        public void SetUp()
        {
            _tempRoot = Path.Combine(
                Path.GetTempPath(), "RemoteCopyTest_" + Guid.NewGuid().ToString("N"));
            _localRoot = Path.Combine(_tempRoot, "local");
            _remoteRoot = Path.Combine(_tempRoot, "remote");
            Directory.CreateDirectory(_localRoot);
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_tempRoot, true); }
            catch (Exception ex) { TestContext.WriteLine("Cleanup failed: " + ex.Message); }
        }

        [Test]
        public void EnqueueFile_RemoteRecovers_PublishesAndClearsPendingMarker()
        {
            string blockedPath = CreateBlockedRemotePath();
            string currentRemote = blockedPath;
            string source = CreateCaptureFile("capture.bin", "payload-1");

            using (var service = new RemoteCopyService(() => currentRemote, () => _localRoot))
            {
                service.EnqueueFile(source);
                WaitUntil(() => service.TotalRetryAttempts > 0, 5000, "initial transfer failure");

                Directory.CreateDirectory(_remoteRoot);
                currentRemote = _remoteRoot;
                Assert.That(service.ProbeRemoteWritable(), Is.True);
                WaitUntil(() => service.QueueCount == 0, 5000, "recovered queue drain");

                string destination = Path.Combine(
                    _remoteRoot, "2026", "202607", "20260715", "capture.bin");
                Assert.That(File.ReadAllText(destination), Is.EqualTo("payload-1"));
                Assert.That(FindPendingMarkers(), Is.Empty);
                Assert.That(FindPartFiles(_remoteRoot), Is.Empty);
                Assert.That(service.IsRemoteWritable, Is.True);
            }
        }

        [Test]
        public void Restart_WithPendingMarker_RestoresAndCopiesFile()
        {
            string blockedPath = CreateBlockedRemotePath();
            string source = CreateCaptureFile("restart.bin", "payload-2");

            using (var first = new RemoteCopyService(() => blockedPath, () => _localRoot))
            {
                first.EnqueueFile(source);
                WaitUntil(() => first.QueueCount == 1, 2000, "pending marker creation");
            }

            Assert.That(FindPendingMarkers().Length, Is.EqualTo(1));
            Directory.CreateDirectory(_remoteRoot);

            using (var restarted = new RemoteCopyService(() => _remoteRoot, () => _localRoot))
            {
                WaitUntil(() => restarted.QueueCount == 0, 5000, "restored queue drain");
                Assert.That(restarted.TotalCopiedFiles, Is.EqualTo(1));
            }

            string destination = Path.Combine(
                _remoteRoot, "2026", "202607", "20260715", "restart.bin");
            Assert.That(File.ReadAllText(destination), Is.EqualTo("payload-2"));
            Assert.That(FindPendingMarkers(), Is.Empty);
        }

        [Test]
        public void ProbeRemoteWritable_RequiresUsableShareAndLeavesNoProbeFile()
        {
            string currentRemote = Path.Combine(_tempRoot, "missing");
            using (var service = new RemoteCopyService(() => currentRemote, () => _localRoot))
            {
                Assert.That(service.ProbeRemoteWritable(), Is.False);
                Assert.That(service.IsRemoteWritable, Is.False);

                Directory.CreateDirectory(_remoteRoot);
                currentRemote = _remoteRoot;
                Assert.That(service.ProbeRemoteWritable(), Is.True);
                Assert.That(service.IsRemoteWritable, Is.True);
                Assert.That(
                    Directory.GetFiles(_remoteRoot, ".picoater-write-probe-*"),
                    Is.Empty);
            }
        }

        [Test]
        public void Retention_PendingRemoteCopy_PreservesSourceUntilPublished()
        {
            string blockedPath = CreateBlockedRemotePath();
            string currentRemote = blockedPath;
            string source = CreateCaptureFile("protected.bin", "payload-3");

            using (var service = new RemoteCopyService(() => currentRemote, () => _localRoot))
            {
                service.EnqueueFile(source);
                WaitUntil(() => service.QueueCount == 1, 2000, "pending source protection");

                var retention = new StorageRetentionService(
                    () => _localRoot,
                    () => long.MaxValue,
                    service.HasPendingFilesUnder);
                retention.RunCleanup();
                Assert.That(File.Exists(source), Is.True, "pending source was deleted");

                Directory.CreateDirectory(_remoteRoot);
                currentRemote = _remoteRoot;
                Assert.That(service.ProbeRemoteWritable(), Is.True);
                WaitUntil(() => service.QueueCount == 0, 5000, "protected source publication");

                retention.RunCleanup();
                Assert.That(File.Exists(source), Is.False, "published source was not eligible for cleanup");
            }
        }

        private string CreateBlockedRemotePath()
        {
            string blocker = Path.Combine(_tempRoot, "blocked-parent");
            File.WriteAllText(blocker, "not a directory");
            return Path.Combine(blocker, "share");
        }

        private string CreateCaptureFile(string fileName, string content)
        {
            string dayDirectory = Path.Combine(_localRoot, "2026", "202607", "20260715");
            Directory.CreateDirectory(dayDirectory);
            string path = Path.Combine(dayDirectory, fileName);
            File.WriteAllText(path, content);
            return path;
        }

        private string[] FindPendingMarkers()
        {
            string directory = Path.Combine(_localRoot, ".remote-copy-pending");
            return Directory.Exists(directory)
                ? Directory.GetFiles(directory, "*.pending")
                : new string[0];
        }

        private static string[] FindPartFiles(string root)
        {
            return Directory.Exists(root)
                ? Directory.GetFiles(root, "*.part-*", SearchOption.AllDirectories)
                : new string[0];
        }

        private static void WaitUntil(
            Func<bool> condition, int timeoutMs, string description)
        {
            var stopwatch = Stopwatch.StartNew();
            while (stopwatch.ElapsedMilliseconds < timeoutMs)
            {
                if (condition()) return;
                Thread.Sleep(20);
            }
            Assert.Fail("Timed out waiting for " + description);
        }
    }
}
