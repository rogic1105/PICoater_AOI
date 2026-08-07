using NUnit.Framework;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class GrayIntensityTests
    {
        [Test]
        public void ScaleCopy_DoublesAndClampsWithoutChangingSource()
        {
            byte[] source = { 0, 10, 100, 200 };

            byte[] result = GrayIntensity.ScaleCopy(source, 2f);

            CollectionAssert.AreEqual(new byte[] { 0, 20, 200, 255 }, result);
            CollectionAssert.AreEqual(new byte[] { 0, 10, 100, 200 }, source);
        }

        [Test]
        public void ScaleCopy_RepeatedSettingChangesAlwaysUseOriginalSource()
        {
            byte[] source = { 40, 80, 120 };

            byte[] brighter = GrayIntensity.ScaleCopy(source, 2f);
            byte[] darker = GrayIntensity.ScaleCopy(source, 0.5f);

            CollectionAssert.AreEqual(new byte[] { 80, 160, 240 }, brighter);
            CollectionAssert.AreEqual(new byte[] { 20, 40, 60 }, darker);
        }
    }
}
