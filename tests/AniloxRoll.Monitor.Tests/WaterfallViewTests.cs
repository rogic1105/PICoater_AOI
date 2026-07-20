using System.Drawing;
using System.Threading;
using System.Windows.Forms;
using NUnit.Framework;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    [Apartment(ApartmentState.STA)]
    public class WaterfallViewTests
    {
        [Test]
        public void ConfiguredEmptyView_PublishesPropertyGridRangeAndKeepsItAcrossReset()
        {
            using (var host = new Panel { Size = new Size(1000, 500) })
            using (var view = new WaterfallView(
                host,
                camCount: 7,
                totalHeight: 30000,
                fullMode: WaterfallFullMode.Restart,
                screenMmPerPx: 0.264))
            {
                int rangeChangedCount = 0;
                double leftMm = 0, rightMm = 0, topMm = 0, bottomMm = 0;
                view.ViewRangeMmChanged += (left, right, top, bottom) =>
                {
                    rangeChangedCount++;
                    leftMm = left;
                    rightMm = right;
                    topMm = top;
                    bottomMm = bottom;
                };

                view.FlipVertical = true;
                view.SetRowPitch(0.22214);
                view.SetLayout(
                    new[] { 0.0, 345.0, 690.0, 1035.0, 1380.0, 1725.0, 2070.0 },
                    new[] { 24.4140625, 24.4140625, 24.4140625, 24.4140625, 24.4140625, 24.4140625, 24.4140625 },
                    refOpsMm: 0.0244140625);

                var canvas = (ImageCanvas)host.Controls[0];
                Assert.That(canvas.LodActive, Is.True);
                Assert.That(rangeChangedCount, Is.GreaterThan(0));
                Assert.That(rightMm - leftMm, Is.GreaterThan(2000));
                Assert.That(topMm - bottomMm, Is.GreaterThan(6000));

                float zoomBeforeReset = canvas.Zoom;
                PointF panBeforeReset = canvas.PanOffset;
                view.Reset();

                Assert.That(canvas.LodActive, Is.True);
                Assert.That(canvas.Zoom, Is.EqualTo(zoomBeforeReset));
                Assert.That(canvas.PanOffset, Is.EqualTo(panBeforeReset));
            }
        }
    }
}
