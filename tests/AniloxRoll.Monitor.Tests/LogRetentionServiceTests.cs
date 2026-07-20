using System;
using System.IO;
using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class LogRetentionServiceTests
    {
        [Test]
        public void Cleanup_DeletesOnlyExpiredCatalogedLogs()
        {
            string root = Path.Combine(
                Path.GetTempPath(), "LogRetention_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);
            try
            {
                string expired = Path.Combine(root, "trace-20200101_000000.log");
                string expiredIo = Path.Combine(root, "io-20200101.log");
                string expiredCrash = Path.Combine(root, "AniloxRoll-crash.log");
                string current = Path.Combine(root, "trace-current.log");
                string unknown = Path.Combine(root, "operator-note.txt");
                File.WriteAllText(expired, "old");
                File.WriteAllText(expiredIo, "old io");
                File.WriteAllText(expiredCrash, "old crash");
                File.WriteAllText(current, "new");
                File.WriteAllText(unknown, "keep");
                foreach (string path in new[] { expired, expiredIo, expiredCrash })
                {
                    File.SetCreationTimeUtc(path, DateTime.UtcNow.AddDays(-30));
                    File.SetLastWriteTimeUtc(path, DateTime.UtcNow.AddDays(-30));
                }

                using (var service = new LogRetentionService(() => root, () => 168))
                    service.RunCleanup();

                Assert.That(File.Exists(expired), Is.False);
                Assert.That(File.Exists(expiredIo), Is.False);
                Assert.That(File.Exists(expiredCrash), Is.False);
                Assert.That(File.Exists(current), Is.True);
                Assert.That(File.Exists(unknown), Is.True);
            }
            finally
            {
                try { Directory.Delete(root, true); } catch { }
            }
        }
    }
}
