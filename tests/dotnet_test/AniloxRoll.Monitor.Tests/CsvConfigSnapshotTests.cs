using System;
using NUnit.Framework;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class CsvConfigSnapshotTests
    {
        [Test]
        public void ToCsvLine_ThenTryParse_RoundTrip()
        {
            var ops = new double[] { 1.11, 2.22, 3.33, 4.44, 5.55, 6.66, 7.77 };
            var pos = new double[] { 10.5, 20.5, 30.5, 40.5, 50.5, 60.5, 70.5 };
            var ts = new DateTime(2026, 3, 30, 14, 30, 45, 123);
            var grabH = new int[] { 3001, 3001, 3001, 3001, 3001, 3001, 3001 };
            var expUs = new double[] { 149, 150, 151, 152, 153, 154, 155 };
            var lrHz = new double[] { 3001, 3002, 3003, 3004, 3005, 3006, 3007 };
            var snap = new CsvConfigSnapshot(ops, pos, grabH, expUs, lrHz, 1.2345f, 0.5678f, 0.9012f, ts);

            string csv = snap.ToCsvLine();
            Assert.That(csv.StartsWith("#CFG,"), Is.True);

            bool ok = CsvConfigSnapshot.TryParse(csv, out var parsed);
            Assert.That(ok, Is.True);

            Assert.That(parsed.Timestamp, Is.EqualTo(ts));
            Assert.That(parsed.HessianMaxFactor, Is.EqualTo(1.2345f).Within(0.001f));
            Assert.That(parsed.ErrorValueMean, Is.EqualTo(0.5678f).Within(0.001f));
            Assert.That(parsed.ErrorValueMax, Is.EqualTo(0.9012f).Within(0.001f));

            for (int i = 0; i < 7; i++)
            {
                Assert.That(parsed.CamOps[i], Is.EqualTo(ops[i]).Within(0.01));
                Assert.That(parsed.CamPos[i], Is.EqualTo(pos[i]).Within(0.01));
                Assert.That(parsed.CamGrabHeight[i], Is.EqualTo(grabH[i]));
                Assert.That(parsed.CamExposureUs[i], Is.EqualTo(expUs[i]).Within(0.01));
                Assert.That(parsed.CamLineRateHz[i], Is.EqualTo(lrHz[i]).Within(0.01));
            }
        }

        [Test]
        public void ContentKey_SameValues_Equal()
        {
            var ops = new double[] { 1, 2, 3, 4, 5, 6, 7 };
            var pos = new double[] { 10, 20, 30, 40, 50, 60, 70 };
            var a = new CsvConfigSnapshot(ops, pos, null, null, null, 1.0f, 2.0f, 3.0f, DateTime.Now);
            var b = new CsvConfigSnapshot(ops, pos, null, null, null, 1.0f, 2.0f, 3.0f, DateTime.Now.AddHours(1));
            Assert.That(a.ContentKey, Is.EqualTo(b.ContentKey));
        }

        [Test]
        public void ContentKey_DifferentValues_NotEqual()
        {
            var ops1 = new double[] { 1, 2, 3, 4, 5, 6, 7 };
            var ops2 = new double[] { 1, 2, 3, 4, 5, 6, 8 };
            var pos = new double[] { 10, 20, 30, 40, 50, 60, 70 };
            var a = new CsvConfigSnapshot(ops1, pos, null, null, null, 1.0f, 2.0f, 3.0f, DateTime.Now);
            var b = new CsvConfigSnapshot(ops2, pos, null, null, null, 1.0f, 2.0f, 3.0f, DateTime.Now);
            Assert.That(a.ContentKey, Is.Not.EqualTo(b.ContentKey));
        }

        [Test]
        public void TryParse_InvalidLine_ReturnsFalse()
        {
            Assert.That(CsvConfigSnapshot.TryParse(null, out _), Is.False);
            Assert.That(CsvConfigSnapshot.TryParse("", out _), Is.False);
            Assert.That(CsvConfigSnapshot.TryParse("not a cfg line", out _), Is.False);
            Assert.That(CsvConfigSnapshot.TryParse("#CFG,bad-timestamp", out _), Is.False);
        }

        [Test]
        public void Constructor_NullArrays_DefaultsToEmpty()
        {
            var snap = new CsvConfigSnapshot(null, null, null, null, null, 0, 0, 0, DateTime.Now);
            Assert.That(snap.CamOps.Length, Is.EqualTo(7));
            Assert.That(snap.CamPos.Length, Is.EqualTo(7));
            Assert.That(snap.CamGrabHeight.Length, Is.EqualTo(7));
            Assert.That(snap.CamExposureUs.Length, Is.EqualTo(7));
            Assert.That(snap.CamLineRateHz.Length, Is.EqualTo(7));
        }
    }
}
