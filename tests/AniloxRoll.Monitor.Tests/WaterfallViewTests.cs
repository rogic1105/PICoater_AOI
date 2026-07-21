using System.Drawing;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;
using NUnit.Framework;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    [Apartment(ApartmentState.STA)]
    [NonParallelizable]
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

        [Test]
        public void DisplayLayerSwitch_PreservesHistoryWriteHeadAndView()
        {
            using (var host = new Panel { Size = new Size(320, 240) })
            using (var view = new WaterfallView(
                host,
                camCount: 1,
                totalHeight: 1000,
                fullMode: WaterfallFullMode.Restart,
                screenMmPerPx: 0.264))
            {
                view.PushFrameVariants(
                    1,
                    new byte[] { 10, 11, 12, 13 },
                    new byte[] { 20, 21, 22, 23 },
                    new byte[] { 30, 31, 32, 33 },
                    2, 2, 100);
                view.SetLayout(new[] { 0.0 }, new[] { 1000000.0 }, 1.0);
                view.PushFrameVariants(
                    1,
                    new byte[] { 14, 15, 16, 17 },
                    new byte[] { 24, 25, 26, 27 },
                    new byte[] { 34, 35, 36, 37 },
                    2, 2, 200);

                Thread.Sleep(200);
                Application.DoEvents();
                Assert.That(
                    SpinWait.SpinUntil(() => ReadPrivate<int>(view, "_writeRow") >= 4, 2000),
                    Is.True,
                    "waterfall writer did not flush the aligned bands");
                Assert.That(
                    SpinWait.SpinUntil(() =>
                    {
                        byte[][][] pendingLayers = ReadPrivate<byte[][][]>(view, "_layerChunks");
                        return pendingLayers[0][0] != null
                            && !ReadPrivate<bool>(view, "_writerRunning");
                    }, 2000),
                    Is.True,
                    "waterfall writer did not finish before verification");

                var canvas = (ImageCanvas)host.Controls[0];
                float zoomBefore = canvas.Zoom;
                PointF panBefore = canvas.PanOffset;
                int writeRowBefore = ReadPrivate<int>(view, "_writeRow");
                byte[][][] layers = ReadPrivate<byte[][][]>(view, "_layerChunks");

                Assert.That(layers[(int)WaterfallFrameLayer.Raw][0][0], Is.EqualTo(10));
                Assert.That(layers[(int)WaterfallFrameLayer.Column][0][0], Is.EqualTo(20));
                Assert.That(layers[(int)WaterfallFrameLayer.Row][0][0], Is.EqualTo(30));

                view.SetDisplayLayer(WaterfallFrameLayer.Column);
                view.SetDisplayLayer(WaterfallFrameLayer.Row);

                Assert.That(view.DisplayLayer, Is.EqualTo(WaterfallFrameLayer.Row));
                Assert.That(ReadPrivate<int>(view, "_writeRow"), Is.EqualTo(writeRowBefore));
                Assert.That(canvas.Zoom, Is.EqualTo(zoomBefore));
                Assert.That(canvas.PanOffset, Is.EqualTo(panBefore));
            }
        }

        private static T ReadPrivate<T>(object target, string fieldName)
        {
            FieldInfo field = target.GetType().GetField(
                fieldName,
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.That(field, Is.Not.Null, fieldName);
            return (T)field.GetValue(target);
        }
    }
}
