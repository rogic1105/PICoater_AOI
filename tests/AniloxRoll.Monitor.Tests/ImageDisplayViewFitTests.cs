using System.Drawing;
using NUnit.Framework;
using TanukiCv.Controls;
using TanukiCv.Core;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class ImageDisplayViewFitTests
    {
        [Test]
        public void ComputeMergeFit_UsesMergedGeometryAndViewportLetterbox()
        {
            bool ok = ImageDisplayView.TryComputeMergeFitViewRange(
                new[] { 100, 100 }, new[] { 100, 100 },
                new[] { 0d, 100d }, new[] { 1000d, 1000d },
                1, 1d, true, MergeOverlap.Midline, false,
                new Size(400, 200), out ImageViewRange range);

            Assert.That(ok, Is.True);
            Assert.That(range.ContentWidth, Is.EqualTo(200));
            Assert.That(range.ContentHeight, Is.EqualTo(100));
            Assert.That(range.LeftMm, Is.EqualTo(-5.263).Within(0.001));
            Assert.That(range.RightMm, Is.EqualTo(205.263).Within(0.001));
            Assert.That(range.TopMm, Is.EqualTo(-2.632).Within(0.001));
            Assert.That(range.BottomMm, Is.EqualTo(102.632).Within(0.001));
        }

        [Test]
        public void ComputeMergeFit_BottomOriginMirrorsVerticalEdges()
        {
            bool ok = ImageDisplayView.TryComputeMergeFitViewRange(
                new[] { 100, 100 }, new[] { 100, 100 },
                new[] { 0d, 100d }, new[] { 1000d, 1000d },
                1, 1d, true, MergeOverlap.Midline, true,
                new Size(400, 200), out ImageViewRange range);

            Assert.That(ok, Is.True);
            Assert.That(range.TopMm, Is.EqualTo(102.632).Within(0.001));
            Assert.That(range.BottomMm, Is.EqualTo(-2.632).Within(0.001));
        }
    }
}
