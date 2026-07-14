using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Integration.Tests
{
    [TestFixture]
    public class SingleGrabCurveSummaryStoreTests
    {
        private string _tempRoot;
        private GrabIdInfo _info;

        [SetUp]
        public void SetUp()
        {
            _tempRoot = Path.Combine(Path.GetTempPath(),
                "SingleGrabCurveSummaryStoreTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tempRoot);
            _info = new GrabIdInfo
            {
                GrabId = "260713-183000",
                Earliest = new DateTime(2026, 7, 13, 18, 30, 0, 100),
                Latest = new DateTime(2026, 7, 13, 18, 30, 3, 900)
            };
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_tempRoot, true); } catch { }
        }

        [Test]
        public void SaveThenLoad_RoundTripsAllCameraCurves()
        {
            var expected = new SingleGrabCurveSummary(
                new[] { new[] { 1f, 2f }, null, new[] { 3.5f } },
                new[] { new[] { 5f, 4f }, null, new[] { 9.5f } },
                17);

            Assert.That(SingleGrabCurveSummaryStore.TrySave(
                _tempRoot, _info, 3, expected), Is.True);
            Assert.That(SingleGrabCurveSummaryStore.TryLoad(
                _tempRoot, _info, 3, out var actual), Is.True);

            Assert.That(actual.CaptureCount, Is.EqualTo(17));
            Assert.That(actual.Mean[0], Is.EqualTo(new[] { 1f, 2f }));
            Assert.That(actual.Max[0], Is.EqualTo(new[] { 5f, 4f }));
            Assert.That(actual.Mean[1], Is.Null);
            Assert.That(actual.Max[1], Is.Null);
            Assert.That(actual.Mean[2], Is.EqualTo(new[] { 3.5f }));
            Assert.That(actual.Max[2], Is.EqualTo(new[] { 9.5f }));
            Assert.That(Directory.GetFiles(
                Path.GetDirectoryName(CaptureStoragePaths.GrabCurveSummary(
                    _tempRoot, _info.Earliest, _info.GrabId)), "*.tmp"), Is.Empty);
        }

        [Test]
        public void Load_WhenGrabTimeRangeChanged_RejectsStaleSummary()
        {
            var summary = new SingleGrabCurveSummary(
                new[] { new[] { 1f } }, new[] { new[] { 2f } }, 1);
            Assert.That(SingleGrabCurveSummaryStore.TrySave(
                _tempRoot, _info, 1, summary), Is.True);

            var changed = new GrabIdInfo
            {
                GrabId = _info.GrabId,
                Earliest = _info.Earliest,
                Latest = _info.Latest.AddMilliseconds(1)
            };

            Assert.That(SingleGrabCurveSummaryStore.TryLoad(
                _tempRoot, changed, 1, out _), Is.False);
        }

        [Test]
        public void Load_WhenFileIsCorrupt_ReturnsFalse()
        {
            string path = CaptureStoragePaths.GrabCurveSummary(
                _tempRoot, _info.Earliest, _info.GrabId);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllBytes(path, new byte[] { 1, 2, 3, 4, 5 });

            Assert.That(SingleGrabCurveSummaryStore.TryLoad(
                _tempRoot, _info, 1, out _), Is.False);
        }

        [Test]
        public void Save_WhenSummaryExists_ReplacesCompleteFile()
        {
            var first = new SingleGrabCurveSummary(
                new[] { new[] { 1f } }, new[] { new[] { 2f } }, 1);
            var second = new SingleGrabCurveSummary(
                new[] { new[] { 7f, 8f } }, new[] { new[] { 9f, 10f } }, 2);

            Assert.That(SingleGrabCurveSummaryStore.TrySave(
                _tempRoot, _info, 1, first), Is.True);
            Assert.That(SingleGrabCurveSummaryStore.TrySave(
                _tempRoot, _info, 1, second), Is.True);
            Assert.That(SingleGrabCurveSummaryStore.TryLoad(
                _tempRoot, _info, 1, out var actual), Is.True);

            Assert.That(actual.CaptureCount, Is.EqualTo(2));
            Assert.That(actual.Mean[0], Is.EqualTo(new[] { 7f, 8f }));
            Assert.That(actual.Max[0], Is.EqualTo(new[] { 9f, 10f }));
        }

        [Test]
        public void QueueSave_WaitsForReadIdleThenPersists()
        {
            var summary = new SingleGrabCurveSummary(
                new[] { new[] { 1f, 2f } }, new[] { new[] { 3f, 4f } }, 1);
            string path = CaptureStoragePaths.GrabCurveSummary(
                _tempRoot, _info.Earliest, _info.GrabId);

            SingleGrabCurveSummaryStore.NotifyReadActivity();
            Assert.That(SingleGrabCurveSummaryStore.QueueSave(
                _tempRoot, _info, 1, summary), Is.True);
            Assert.That(File.Exists(path), Is.False, "QueueSave must not block on disk persistence.");

            var wait = Stopwatch.StartNew();
            while (!File.Exists(path) && wait.ElapsedMilliseconds < 3000)
                Thread.Sleep(20);

            Assert.That(File.Exists(path), Is.True);
            Assert.That(SingleGrabCurveSummaryStore.TryLoad(
                _tempRoot, _info, 1, out var loaded), Is.True);
            Assert.That(loaded.Mean[0], Is.EqualTo(new[] { 1f, 2f }));
        }
    }
}
