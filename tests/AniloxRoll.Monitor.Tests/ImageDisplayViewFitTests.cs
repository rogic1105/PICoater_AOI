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

        [Test]
        public void GrayBitmap_ColdHeatmapPreservesIntensityInBlueChannel()
        {
            using (Bitmap bitmap = GrayBitmap.From(
                new byte[] { 0, 128, 255 }, 3, 1, false,
                IntensityColorMap.HeatmapCold))
            {
                Assert.That(bitmap.GetPixel(0, 0).ToArgb(), Is.EqualTo(Color.Black.ToArgb()));
                Assert.That(bitmap.GetPixel(1, 0).ToArgb(), Is.EqualTo(Color.FromArgb(0, 51, 128).ToArgb()));
                Assert.That(bitmap.GetPixel(2, 0).ToArgb(), Is.EqualTo(Color.White.ToArgb()));
                Assert.That(bitmap.GetPixel(1, 0).B, Is.EqualTo(128));
            }
        }

        [Test]
        public void GrayBitmap_WarmHeatmapPreservesIntensityInRedChannel()
        {
            using (Bitmap bitmap = GrayBitmap.From(
                new byte[] { 0, 128, 255 }, 3, 1, false,
                IntensityColorMap.HeatmapWarm))
            {
                Assert.That(bitmap.GetPixel(0, 0).ToArgb(), Is.EqualTo(Color.Black.ToArgb()));
                Assert.That(bitmap.GetPixel(1, 0).ToArgb(), Is.EqualTo(Color.FromArgb(128, 51, 0).ToArgb()));
                Assert.That(bitmap.GetPixel(2, 0).ToArgb(), Is.EqualTo(Color.White.ToArgb()));
                Assert.That(bitmap.GetPixel(1, 0).R, Is.EqualTo(128));
            }
        }

        [Test]
        public void GrayBitmap_BlueYellowRedHeatmapUsesExplicitAnchorsAndRoundTripsIntensity()
        {
            using (Bitmap bitmap = GrayBitmap.From(
                new byte[] { 0, 85, 128, 170, 255 }, 5, 1, false,
                IntensityColorMap.HeatmapBlueYellowRed))
            {
                Assert.That(bitmap.GetPixel(0, 0).ToArgb(), Is.EqualTo(Color.Black.ToArgb()));
                Assert.That(bitmap.GetPixel(1, 0).ToArgb(), Is.EqualTo(Color.Blue.ToArgb()));
                Assert.That(bitmap.GetPixel(3, 0).ToArgb(), Is.EqualTo(Color.Yellow.ToArgb()));
                Assert.That(bitmap.GetPixel(4, 0).ToArgb(), Is.EqualTo(Color.Red.ToArgb()));
                var selector = GrayBitmap.GetBrightnessSelector(IntensityColorMap.HeatmapBlueYellowRed);
                Assert.That(selector(bitmap.GetPixel(0, 0)), Is.EqualTo(0));
                Assert.That(selector(bitmap.GetPixel(1, 0)), Is.EqualTo(85));
                Assert.That(selector(bitmap.GetPixel(2, 0)), Is.EqualTo(128));
                Assert.That(selector(bitmap.GetPixel(3, 0)), Is.EqualTo(170));
                Assert.That(selector(bitmap.GetPixel(4, 0)), Is.EqualTo(255));
            }
        }
    }
}
