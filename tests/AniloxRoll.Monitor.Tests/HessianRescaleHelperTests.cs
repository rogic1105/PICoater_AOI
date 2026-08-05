using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class HessianRescaleHelperTests
    {
        [Test]
        public void CloneAndRescale1D_LiveRowUsesCaptureColumnAndCurrentRowFactors()
        {
            float[] raw = { 0f, 64f, 128f, 255f };

            float[] display = HessianRescaleHelper.CloneAndRescale1D(
                raw, captureHm: 0.5f, currentHm: 1.0f);

            CollectionAssert.AreEqual(
                new[] { 0f, 32f, 64f, 127.5f },
                display);
            CollectionAssert.AreEqual(
                new[] { 0f, 64f, 128f, 255f },
                raw,
                "Live 顯示換算不可改寫後續存檔使用的原始 Curve。");
        }

        [Test]
        public void Ratio_InvalidFactorFallsBackToOne()
        {
            Assert.That(HessianRescaleHelper.Ratio(0.5f, 0f), Is.EqualTo(1f));
            Assert.That(HessianRescaleHelper.Ratio(0f, 0.5f), Is.EqualTo(1f));
        }
    }
}
