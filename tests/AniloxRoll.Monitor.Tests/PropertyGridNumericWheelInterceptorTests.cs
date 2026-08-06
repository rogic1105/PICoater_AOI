using System;
using AniloxRoll.Monitor.UI.Widgets;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public sealed class PropertyGridNumericWheelInterceptorTests
    {
        [TestCase(0.3f, 1, 0.4f)]
        [TestCase(0.3f, 3, 0.6f)]
        [TestCase(0.3f, -4, 0.1f)]
        public void DecimalStep_UsesPointOneAndClampsToPositive(float current, int notches, float expected)
        {
            bool changed = PropertyGridNumericWheelInterceptor.TryCalculateNext(
                current, typeof(float), 0.1m, notches, out object next);

            Assert.That(changed, Is.True);
            Assert.That((float)next, Is.EqualTo(expected).Within(0.0001f));
        }

        [Test]
        public void RidgeSigmaStep_UsesWholeNumbers()
        {
            bool changed = PropertyGridNumericWheelInterceptor.TryCalculateNext(
                9f, typeof(float), 1m, 2, out object next);

            Assert.That(changed, Is.True);
            Assert.That((float)next, Is.EqualTo(11f));
        }

        [Test]
        public void ZeroNotches_DoesNotChangeValue()
        {
            bool changed = PropertyGridNumericWheelInterceptor.TryCalculateNext(
                0.3f, typeof(float), 0.1m, 0, out object next);

            Assert.That(changed, Is.False);
            Assert.That(next, Is.EqualTo(0.3f));
        }

        [Test]
        public void ClickingSameNumericProperty_TogglesWheelEditingOff()
        {
            string armed = PropertyGridNumericWheelInterceptor.ResolveArmedProperty(null, "factor");
            Assert.That(armed, Is.EqualTo("factor"));

            armed = PropertyGridNumericWheelInterceptor.ResolveArmedProperty(armed, "factor");
            Assert.That(armed, Is.Null);
        }

        [Test]
        public void ClickingAnotherProperty_MovesWheelEditingArm()
        {
            string armed = PropertyGridNumericWheelInterceptor.ResolveArmedProperty("mean", "max");

            Assert.That(armed, Is.EqualTo("max"));
        }
    }
}
