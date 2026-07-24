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

        [Test]
        public void ExpectedFramePeriod_AllowsFirstAlignedCameraSetToFlush()
        {
            using (var host = new Panel { Size = new Size(320, 240) })
            using (var view = new WaterfallView(
                host,
                camCount: 2,
                totalHeight: 1000,
                fullMode: WaterfallFullMode.Restart,
                screenMmPerPx: 0.264))
            {
                view.SetLayout(
                    new[] { 0.0, 2.0 },
                    new[] { 1000000.0, 1000000.0 },
                    refOpsMm: 1.0);
                view.SetExpectedFramePeriod(periodTicks: 500, periodMs: 500);
                view.Reset();

                view.PushFrame(1, new byte[] { 10, 11, 12, 13 }, 2, 2, tick: 1000);
                view.PushFrame(2, new byte[] { 20, 21, 22, 23 }, 2, 2, tick: 1002);

                Thread.Sleep(200);
                Application.DoEvents();
                Assert.That(
                    SpinWait.SpinUntil(() => ReadPrivate<int>(view, "_writeRow") >= 2, 1000),
                    Is.True,
                    "the first aligned camera set should not wait for a second frame to learn its period");
            }
        }

        [Test]
        public void MonitoringCanvas_WheelOutCanZoomBelowLegacyPointZeroOneFloor()
        {
            using (var canvas = new ImageCanvas { Size = new Size(1000, 500) })
            using (var image = new Bitmap(1000, 500))
            {
                canvas.Image = image;
                canvas.FitRelativeZoom = false;
                canvas.FitToScreen();

                MethodInfo onMouseWheel = typeof(ImageCanvas).GetMethod(
                    "OnMouseWheel",
                    BindingFlags.Instance | BindingFlags.NonPublic);
                Assert.That(onMouseWheel, Is.Not.Null);
                onMouseWheel.Invoke(
                    canvas,
                    new object[] { new MouseEventArgs(MouseButtons.None, 0, 500, 250, -12000) });

                Assert.That(canvas.Zoom, Is.LessThan(0.01f));
                Assert.That(canvas.Zoom, Is.EqualTo(1f / 500f).Within(0.000001f));
            }
        }

        [Test]
        public void LodTile_FromPreviousContentGeneration_IsNotInstalled()
        {
            using (var canvas = new ImageCanvas { Size = new Size(320, 240) })
            using (var staleTile = new Bitmap(32, 24))
            {
                WritePrivate(canvas, "_lodActive", true);
                WritePrivate(canvas, "_lodContentGeneration", 2L);
                int applied = 0;
                canvas.LodTileApplied += generation => applied++;

                MethodInfo applyLodTile = typeof(ImageCanvas).GetMethod(
                    "ApplyLodTile",
                    BindingFlags.Instance | BindingFlags.NonPublic);
                Assert.That(applyLodTile, Is.Not.Null);
                applyLodTile.Invoke(
                    canvas,
                    new object[] { staleTile, new Rectangle(0, 0, 32, 24), 1L });

                Assert.That(ReadPrivate<Bitmap>(canvas, "_lodTile"), Is.Null);
                Assert.That(applied, Is.Zero);
            }
        }

        [Test]
        public void RefreshLod_NewSequence_ClearsVisibleTileBeforeRecompute()
        {
            using (var canvas = new ImageCanvas { Size = new Size(320, 240) })
            using (var previousTile = new Bitmap(32, 24))
            {
                WritePrivate(canvas, "_lodActive", true);
                WritePrivate(canvas, "_lodContentGeneration", 1L);
                WritePrivate(canvas, "_lodTile", previousTile);

                canvas.RefreshLod(clearCurrentTile: true);

                Assert.That(ReadPrivate<Bitmap>(canvas, "_lodTile"), Is.Null);
                Assert.That(canvas.LodContentGeneration, Is.EqualTo(2L));
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

        private static void WritePrivate<T>(object target, string fieldName, T value)
        {
            FieldInfo field = target.GetType().GetField(
                fieldName,
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.That(field, Is.Not.Null, fieldName);
            field.SetValue(target, value);
        }
    }
}
