using System.Collections.Generic;
using System.Drawing;
using AniloxRoll.Monitor.UI.Widgets;
using NUnit.Framework;
using TanukiCv.Controls;
using TanukiCv.Core;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class HorizontalDisplayCropTests
    {
        [Test]
        public void ComputeMergeFit_CropsBeforeAspectFit()
        {
            bool ok = ImageDisplayView.TryComputeMergeFitViewRange(
                new[] { 100, 100 }, new[] { 100, 100 },
                new[] { 0d, 100d }, new[] { 1000d, 1000d },
                1, 1d, true, MergeOverlap.Midline,
                20d, 30d, false,
                new Size(300, 200), out ImageViewRange range);

            Assert.That(ok, Is.True);
            Assert.That(range.ContentWidth, Is.EqualTo(150));
            Assert.That(range.ContentHeight, Is.EqualTo(100));
            Assert.That((range.LeftMm + range.RightMm) / 2d, Is.EqualTo(95d).Within(0.001));
        }

        [Test]
        public void Apply_ClipsEdgeCamerasWithoutChangingSourceCoordinates()
        {
            var source = new List<CameraPlacement>
            {
                new CameraPlacement { CameraId = 1, XOffset = 0, SrcLeft = 0, SrcWidth = 100 },
                new CameraPlacement { CameraId = 2, XOffset = 100, SrcLeft = 0, SrcWidth = 100 }
            };
            HorizontalDisplayCrop crop = HorizontalDisplayCrop.Compute(
                200, 0d, 1d, 20d, 30d);

            List<CameraPlacement> visible = crop.Apply(source);

            Assert.That(visible.Count, Is.EqualTo(2));
            Assert.That(visible[0].CameraId, Is.EqualTo(1));
            Assert.That(visible[0].SrcLeft, Is.EqualTo(20));
            Assert.That(visible[0].SrcWidth, Is.EqualTo(80));
            Assert.That(visible[0].DestX, Is.EqualTo(0));
            Assert.That(visible[1].CameraId, Is.EqualTo(2));
            Assert.That(visible[1].SrcLeft, Is.EqualTo(0));
            Assert.That(visible[1].SrcWidth, Is.EqualTo(70));
            Assert.That(visible[1].DestX, Is.EqualTo(80));
        }

        [Test]
        public void Apply_MapsDisplayCropBackToScaledSourcePixels()
        {
            var source = new List<CameraPlacement>
            {
                new CameraPlacement
                {
                    CameraId = 1,
                    XOffset = 0,
                    SrcLeft = 0,
                    SrcWidth = 100,
                    DisplayLeft = 0,
                    DisplayWidth = 200
                }
            };
            HorizontalDisplayCrop crop = HorizontalDisplayCrop.Compute(
                200, 0d, 1d, 50d, 0d);

            List<CameraPlacement> visible = crop.Apply(source);

            Assert.That(visible.Count, Is.EqualTo(1));
            Assert.That(visible[0].SrcLeft, Is.EqualTo(25));
            Assert.That(visible[0].SrcWidth, Is.EqualTo(75));
            Assert.That(visible[0].DestX, Is.EqualTo(0));
            Assert.That(visible[0].DestWidth, Is.EqualTo(150));
        }

        [Test]
        public void ColumnChartFallbackRange_UsesCropWithoutMutatingCurveLength()
        {
            double left = double.NaN;
            double right = double.NaN;

            CurveMergeHelper.ResolveHorizontalDisplayRange(
                0d, 200, 1d, 20d, 30d, ref left, ref right);

            Assert.That(left, Is.EqualTo(20d));
            Assert.That(right, Is.EqualTo(170d));
        }

        [Test]
        public void ColumnChartExistingFullRange_IsClampedToDisplayCrop()
        {
            double left = 0d;
            double right = 200d;

            CurveMergeHelper.ResolveHorizontalDisplayRange(
                0d, 200, 1d, 20d, 30d, ref left, ref right);

            Assert.That(left, Is.EqualTo(20d));
            Assert.That(right, Is.EqualTo(170d));
        }

        [Test]
        public void ColumnChartExistingZoomInsideCrop_IsPreserved()
        {
            double left = 40d;
            double right = 100d;

            CurveMergeHelper.ResolveHorizontalDisplayRange(
                0d, 200, 1d, 20d, 30d, ref left, ref right);

            Assert.That(left, Is.EqualTo(40d));
            Assert.That(right, Is.EqualTo(100d));
        }
    }
}
