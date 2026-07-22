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
        public void Restart_WithCorruptPendingMarker_MovesMarkerToQuarantine()
        {
            string pendingDirectory = Path.Combine(_localRoot, ".remote-copy-pending");
            Directory.CreateDirectory(pendingDirectory);
            string corruptMarker = Path.Combine(pendingDirectory, "broken.pending");
            File.WriteAllText(corruptMarker, "not-a-valid-pending-marker");

            using (var service = new RemoteCopyService(() => _remoteRoot, () => _localRoot))
            {
                Assert.That(service.QuarantinedMarkerCount, Is.EqualTo(1));
                Assert.That(service.QueueCount, Is.Zero);
            }

            Assert.That(File.Exists(corruptMarker), Is.False);
            string quarantine = Path.Combine(pendingDirectory, "quarantine");
            Assert.That(Directory.GetFiles(quarantine, "*broken.pending"), Has.Length.EqualTo(1));
        }

        [Test]
        public void EnqueueFile_RemoteHasStalePart_CleansPartAndRecordsLastSuccess()
        {
            string destinationDirectory = Path.Combine(
                _remoteRoot, "2026", "202607", "20260715");
            Directory.CreateDirectory(destinationDirectory);
            string stalePart = Path.Combine(destinationDirectory, "capture.bin.part-orphan");
            File.WriteAllText(stalePart, "incomplete");
            File.SetLastWriteTimeUtc(stalePart, DateTime.UtcNow.AddDays(-2));
            string source = CreateCaptureFile("capture.bin", "payload");

            using (var service = new RemoteCopyService(() => _remoteRoot, () => _localRoot))
            {
                service.EnqueueFile(source);
                WaitUntil(() => service.QueueCount == 0, 5000, "copy and stale part cleanup");

                Assert.That(File.Exists(stalePart), Is.False);
                Assert.That(service.LastSuccessfulCopyUtc, Is.Not.Null);
                Assert.That(
                    service.LastSuccessfulCopyUtc.Value,
                    Is.GreaterThan(DateTime.UtcNow.AddMinutes(-1)));
            }
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
        public void EnqueueMutableFile_UpdatedWhilePending_PublishesLatestSnapshot()
        {
            string blockedPath = CreateBlockedRemotePath();
            string currentRemote = blockedPath;
            string source = CreateCaptureFile("_ticks.csv", "frame-1,100\r\n");

            using (var service = new RemoteCopyService(() => currentRemote, () => _localRoot))
            {
                service.EnqueueFile(source);
                WaitUntil(() => service.QueueCount == 1, 2000, "mutable file pending");

                File.AppendAllText(source, "frame-2,200\r\n");
                service.EnqueueFile(source);

                Directory.CreateDirectory(_remoteRoot);
                currentRemote = _remoteRoot;
                Assert.That(service.ProbeRemoteWritable(), Is.True);
                WaitUntil(() => service.QueueCount == 0, 5000, "mutable file republish");

                string destination = Path.Combine(
                    _remoteRoot, "2026", "202607", "20260715", "_ticks.csv");
                Assert.That(File.ReadAllText(destination), Is.EqualTo(File.ReadAllText(source)));
                // An update before the first snapshot starts coalesces into one publish;
                // an in-flight snapshot requires one dirty follow-up publish.
                Assert.That(service.TotalCopiedFiles, Is.InRange(1L, 2L));
                Assert.That(FindPendingMarkers(), Is.Empty);
                Assert.That(FindPartFiles(_remoteRoot), Is.Empty);
            }
        }

        [Test]
        public void StageFiles_HoldsLatestContentUntilReleased()
        {
            Directory.CreateDirectory(_remoteRoot);
            string source = CreateCaptureFile("capture.acap", "frame-1");
            string destination = Path.Combine(
                _remoteRoot, "2026", "202607", "20260715", "capture.acap");

            using (var service = new RemoteCopyService(() => _remoteRoot, () => _localRoot))
            {
                service.StageFiles(new[] { source });
                File.AppendAllText(source, "-frame-2");
                service.StageFiles(new[] { source });

                Thread.Sleep(250);
                Assert.That(File.Exists(destination), Is.False);
                Assert.That(service.QueueCount, Is.EqualTo(1));
                Assert.That(FindPendingMarkers(), Has.Length.EqualTo(1));

                service.ReleaseStagedFiles(new[] { source });
                WaitUntil(() => service.QueueCount == 0, 5000, "staged file release");

                Assert.That(File.ReadAllText(destination), Is.EqualTo("frame-1-frame-2"));
                Assert.That(service.TotalCopiedFiles, Is.EqualTo(1));
                Assert.That(FindPendingMarkers(), Is.Empty);
            }
        }

        [Test]
        public void Restart_WithStagedMarker_ReleasesInterruptedCapture()
        {
            Directory.CreateDirectory(_remoteRoot);
            string source = CreateCaptureFile("interrupted.acap", "partial-capture");

            using (var first = new RemoteCopyService(() => _remoteRoot, () => _localRoot))
            {
                first.StageFiles(new[] { source });
                Assert.That(first.QueueCount, Is.EqualTo(1));
            }

            using (var restarted = new RemoteCopyService(() => _remoteRoot, () => _localRoot))
            {
                WaitUntil(() => restarted.QueueCount == 0, 5000, "staged restart recovery");
            }

            string destination = Path.Combine(
                _remoteRoot, "2026", "202607", "20260715", "interrupted.acap");
            Assert.That(File.ReadAllText(destination), Is.EqualTo("partial-capture"));
            Assert.That(FindPendingMarkers(), Is.Empty);
        }

        [Test]
        public void StageFiles_HoldsPathThatWasAlreadyQueuedByPreviousCapture()
        {
            string blockedPath = CreateBlockedRemotePath();
            string currentRemote = blockedPath;
            string source = CreateCaptureFile("20260715.csv", "previous-grab\r\n");
            string destination = Path.Combine(
                _remoteRoot, "2026", "202607", "20260715", "20260715.csv");

            using (var service = new RemoteCopyService(() => currentRemote, () => _localRoot))
            {
                service.EnqueueFile(source);
                WaitUntil(() => service.TotalRetryAttempts > 0, 5000, "previous capture retry");

                File.AppendAllText(source, "current-grab\r\n");
                service.StageFiles(new[] { source });
                Directory.CreateDirectory(_remoteRoot);
                currentRemote = _remoteRoot;
                Assert.That(service.ProbeRemoteWritable(), Is.True);

                Thread.Sleep(2500);
                Assert.That(File.Exists(destination), Is.False);
                Assert.That(service.QueueCount, Is.EqualTo(1));

                service.ReleaseStagedFiles(new[] { source });
                WaitUntil(() => service.QueueCount == 0, 5000, "current capture release");

                Assert.That(
                    File.ReadAllText(destination),
                    Is.EqualTo("previous-grab\r\ncurrent-grab\r\n"));
                Assert.That(service.TotalCopiedFiles, Is.EqualTo(1));
                Assert.That(FindPendingMarkers(), Is.Empty);
            }
        }

        [Test]
        public void CopySharedSnapshot_AllowsCaptureWriterToAppendWhileSourceIsOpen()
        {
            string source = CreateCaptureFile("capture.acap", "frame-1");
            string destination = Path.Combine(_tempRoot, "capture.acap.part-test");
            bool appended = false;

            RemoteCopyService.CopySharedSnapshot(
                source,
                destination,
                () =>
                {
                    using (var writer = new FileStream(
                        source,
                        FileMode.Append,
                        FileAccess.Write,
                        FileShare.Read))
                    {
                        writer.WriteByte(0x7F);
                        writer.Flush(true);
                    }
                    appended = true;
                });

            Assert.That(appended, Is.True);
            Assert.That(new FileInfo(source).Length, Is.EqualTo("frame-1".Length + 1));
            Assert.That(new FileInfo(destination).Length, Is.EqualTo("frame-1".Length));
        }

        [Test]
        public void Retention_LowSpace_DeletesCompleteOldestDayAndCancelsPending()
        {
            string blockedPath = CreateBlockedRemotePath();
            string currentRemote = blockedPath;
            string source = CreateCaptureFile("protected.bin", "payload-3");
            string dayDirectory = Path.GetDirectoryName(source);
            string summaryDirectory = Path.Combine(dayDirectory, "_curve_summary");
            Directory.CreateDirectory(summaryDirectory);
            string summary = Path.Combine(summaryDirectory, "260715-120000.mcsf");
            string ticks = Path.Combine(dayDirectory, "_ticks.csv");
            string dailyCsv = Path.Combine(Path.GetDirectoryName(dayDirectory), "20260715.csv");
            File.WriteAllText(summary, "derived");
            File.WriteAllText(ticks, "frame,100\r\n");
            File.WriteAllText(dailyCsv, "inspection-record");

            using (var service = new RemoteCopyService(() => currentRemote, () => _localRoot))
            {
                service.EnqueueFiles(new[] { source, dailyCsv });
                WaitUntil(() => service.QueueCount == 2, 2000, "pending sources");

                var retention = new StorageRetentionService(
                    () => _localRoot,
                    () => GetCleanupTriggerThreshold(_localRoot),
                    dayDirectoryToDelete =>
                    {
                        int canceled = service.CancelPendingFilesUnder(dayDirectoryToDelete);
                        string month = Path.GetDirectoryName(dayDirectoryToDelete);
                        string csv = Path.Combine(
                            month, Path.GetFileName(dayDirectoryToDelete) + ".csv");
                        if (service.CancelPendingFile(csv)) canceled++;
                        return canceled;
                    });
                retention.RunCleanup();
                Assert.That(File.Exists(source), Is.False, "oldest-day source was retained");
                Assert.That(File.Exists(summary), Is.False, "derived curve summary was retained");
                Assert.That(File.Exists(ticks), Is.False, "orphaned tick sidecar was retained");
                Assert.That(File.Exists(dailyCsv), Is.False, "daily inspection CSV was retained");
                Assert.That(service.QueueCount, Is.Zero);
                Assert.That(service.PendingBytes, Is.Zero);
                Assert.That(FindPendingMarkers(), Is.Empty);
                Assert.That(retention.LastCleanedDayFolders, Is.EqualTo(1));
            }
        }

        [Test]
        public void Retention_MinFreeAtOrAboveVolumeTotal_DoesNotDeleteCaptures()
        {
            string source = CreateCaptureFile("must-remain.bin", "payload");
            var retention = new StorageRetentionService(
                () => _localRoot,
                () => long.MaxValue);

            retention.RunCleanup();

            Assert.That(File.Exists(source), Is.True);
            Assert.That(retention.LastCleanedBytes, Is.Zero);
            Assert.That(retention.LastDriveTotalBytes, Is.GreaterThan(0));
        }

        private string CreateBlockedRemotePath()
        {
            string blocker = Path.Combine(_tempRoot, "blocked-parent");
            File.WriteAllText(blocker, "not a directory");
            return Path.Combine(blocker, "share");
        }

        private static long GetCleanupTriggerThreshold(string path)
        {
            string root = Path.GetPathRoot(Path.GetFullPath(path));
            return new DriveInfo(root).TotalSize - 1;
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
