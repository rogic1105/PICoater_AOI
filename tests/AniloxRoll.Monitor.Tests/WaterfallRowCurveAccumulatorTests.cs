using AniloxRoll.Monitor.UI.Managers;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class WaterfallRowCurveAccumulatorTests
    {
        [Test]
        public void Rescale_UpdatesExistingHistoryWithoutAdvancingWritePosition()
        {
            var accumulator = new WaterfallRowCurveAccumulator();
            accumulator.Append(
                new[] { 1f, 2f, 3f },
                new[] { 4f, 5f, 6f },
                capacity: 10,
                ring: false,
                displayFactor: 0.5f);
            int writeBefore = accumulator.WritePosition;

            accumulator.Rescale(2f);

            Assert.That(accumulator.WritePosition, Is.EqualTo(writeBefore));
            CollectionAssert.AreEqual(
                new[] { 2f, 4f, 6f },
                new[] { accumulator.Mean[0], accumulator.Mean[1], accumulator.Mean[2] });
            CollectionAssert.AreEqual(
                new[] { 8f, 10f, 12f },
                new[] { accumulator.Max[0], accumulator.Max[1], accumulator.Max[2] });

            accumulator.Rescale(3f);

            Assert.That(accumulator.WritePosition, Is.EqualTo(writeBefore));
            CollectionAssert.AreEqual(
                new[] { 3f, 6f, 9f },
                new[] { accumulator.Mean[0], accumulator.Mean[1], accumulator.Mean[2] },
                "連續改正規值必須每次從 neutral 資料重算，不能把前次倍率再乘一次。");
        }

        [Test]
        public void Append_AfterRescaleContinuesAtOriginalTail()
        {
            var accumulator = new WaterfallRowCurveAccumulator();
            accumulator.Append(
                new[] { 1f, 2f },
                new[] { 3f, 4f },
                capacity: 8,
                ring: false,
                displayFactor: 1f);
            accumulator.Rescale(2f);

            accumulator.Append(
                new[] { 5f },
                new[] { 6f },
                capacity: 8,
                ring: false,
                displayFactor: 2f);

            Assert.That(accumulator.WritePosition, Is.EqualTo(3));
            CollectionAssert.AreEqual(
                new[] { 2f, 4f, 10f },
                new[] { accumulator.Mean[0], accumulator.Mean[1], accumulator.Mean[2] });
            CollectionAssert.AreEqual(
                new[] { 6f, 8f, 12f },
                new[] { accumulator.Max[0], accumulator.Max[1], accumulator.Max[2] });
        }
    }
}
