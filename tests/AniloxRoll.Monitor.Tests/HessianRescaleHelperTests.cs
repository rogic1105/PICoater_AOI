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
        public void Scale_InvalidFactorFallsBackToOne()
        {
            Assert.That(HessianRescaleHelper.RawCurveToDisplayScale(0.5f, 0f),
                Is.EqualTo(1f));
            Assert.That(HessianRescaleHelper.NormalizedValueToDisplayScale(0f, 0.5f),
                Is.EqualTo(1f));
        }

        [Test]
        public void RawCurveScale_CurrentNormalizationIsLinearGain()
        {
            Assert.That(HessianRescaleHelper.RawCurveToDisplayScale(0.5f, 0.5f),
                Is.EqualTo(0.25f));
            Assert.That(HessianRescaleHelper.RawCurveToDisplayScale(0.5f, 1.0f),
                Is.EqualTo(0.5f));
            Assert.That(HessianRescaleHelper.RawCurveToDisplayScale(0.5f, 1.5f),
                Is.EqualTo(0.75f));
        }

        [Test]
        public void NormalizedValueScale_PreservesCaptureValueAtSameNormalization()
        {
            Assert.That(HessianRescaleHelper.NormalizedValueToDisplayScale(0.5f, 0.5f),
                Is.EqualTo(1f));
            Assert.That(HessianRescaleHelper.NormalizedValueToDisplayScale(0.5f, 1.0f),
                Is.EqualTo(2f));
        }

    }
}
