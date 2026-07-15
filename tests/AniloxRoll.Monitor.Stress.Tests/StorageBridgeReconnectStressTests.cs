using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using NUnit.Framework;
using StorageBridge.Core;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    [Category("BridgeStress")]
    [NonParallelizable]
    public class StorageBridgeReconnectStressTests
    {
        [Test]
        [Timeout(120000)]
        public void Restart_WithOneThousandPendingFiles_RestoresAndDrainsWithoutPartials()
        {
            const int fileCount = 1000;
            string tempRoot = Path.Combine(
                Path.GetTempPath(), "StorageStress_" + Guid.NewGuid().ToString("N"));
            string localRoot = Path.Combine(tempRoot, "local");
            string remoteRoot = Path.Combine(tempRoot, "remote");
            string blocker = Path.Combine(tempRoot, "blocked-parent");
            string blockedRemote = Path.Combine(blocker, "share");

            Directory.CreateDirectory(localRoot);
            File.WriteAllText(blocker, "not a directory");

            try
            {
                using (var first = new RemoteCopyService(() => blockedRemote, () => localRoot))
                {
                    for (int i = 0; i < fileCount; i++)
                    {
                        string dayDirectory = Path.Combine(
                            localRoot, "2026", "202607", (15 + i % 2).ToString("20260700"));
                        Directory.CreateDirectory(dayDirectory);
                        string source = Path.Combine(dayDirectory, "capture-" + i + ".bin");
                        File.WriteAllText(source, "payload-" + i);
                        first.EnqueueFile(source);
                    }

                    Assert.That(first.QueueCount, Is.EqualTo(fileCount));
                }

                Directory.CreateDirectory(remoteRoot);
                var stopwatch = Stopwatch.StartNew();
                using (var restarted = new RemoteCopyService(() => remoteRoot, () => localRoot))
                {
                    WaitUntil(() => restarted.QueueCount == 0, 90000);
                    Assert.That(restarted.TotalCopiedFiles, Is.EqualTo(fileCount));
                }

                Assert.That(
                    Directory.GetFiles(remoteRoot, "*.bin", SearchOption.AllDirectories).Length,
                    Is.EqualTo(fileCount));
                Assert.That(
                    Directory.GetFiles(remoteRoot, "*.part-*", SearchOption.AllDirectories),
                    Is.Empty);
                Assert.That(
                    Directory.GetFiles(
                        Path.Combine(localRoot, ".remote-copy-pending"), "*.pending"),
                    Is.Empty);
                TestContext.WriteLine(
                    "Restored and published " + fileCount + " files in " +
                    stopwatch.ElapsedMilliseconds + " ms");
            }
            finally
            {
                try { Directory.Delete(tempRoot, true); }
                catch (Exception ex) { TestContext.WriteLine("Cleanup failed: " + ex.Message); }
            }
        }

        private static void WaitUntil(Func<bool> condition, int timeoutMs)
        {
            var stopwatch = Stopwatch.StartNew();
            while (stopwatch.ElapsedMilliseconds < timeoutMs)
            {
                if (condition()) return;
                Thread.Sleep(25);
            }
            Assert.Fail("Timed out waiting for durable storage queue to drain");
        }
    }
}
