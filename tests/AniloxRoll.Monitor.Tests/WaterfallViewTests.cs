using System.Collections.Generic;
using System.Drawing;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;
using NUnit.Framework;
using TanukiCv.Controls;
using TanukiCv.Core;

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
        public void Layout_UsesEachCameraOpsForPhysicalWidth()
        {
            using (var host = new Panel { Size = new Size(600, 200) })
            using (var view = new WaterfallView(
                host,
                camCount: 2,
                totalHeight: 1000,
                fullMode: WaterfallFullMode.Restart,
                screenMmPerPx: 0.264))
            {
                WritePrivate(view, "_defaultFrameW", 100);
                view.SetLayout(
                    new[] { 0d, 100d },
                    new[] { 1000d, 2000d },
                    refOpsMm: 1d);

                List<CameraPlacement> placements =
                    ReadPrivate<List<CameraPlacement>>(view, "_cameraPlacements");
                Assert.That(ReadPrivate<int>(view, "_fullW"), Is.EqualTo(300));
                Assert.That(placements[0].DestWidth, Is.EqualTo(100));
                Assert.That(placements[1].DestWidth, Is.EqualTo(200));
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
                        byte[][][][] pendingLayers = ReadPrivate<byte[][][][]>(view, "_cameraLayerChunks");
                        return pendingLayers[0][0][0] != null
                            && !ReadPrivate<bool>(view, "_writerRunning");
                    }, 2000),
                    Is.True,
                    "waterfall writer did not finish before verification");

                var canvas = (ImageCanvas)host.Controls[0];
                float zoomBefore = canvas.Zoom;
                PointF panBefore = canvas.PanOffset;
                int writeRowBefore = ReadPrivate<int>(view, "_writeRow");
                byte[][][][] layers = ReadPrivate<byte[][][][]>(view, "_cameraLayerChunks");

                Assert.That(layers[(int)WaterfallFrameLayer.Raw][0][0][0], Is.EqualTo(10));
                Assert.That(layers[(int)WaterfallFrameLayer.Column][0][0][0], Is.EqualTo(20));
                Assert.That(layers[(int)WaterfallFrameLayer.Row][0][0][0], Is.EqualTo(30));

                view.SetDisplayLayer(WaterfallFrameLayer.Column);
                view.SetDisplayLayer(WaterfallFrameLayer.Row);

                Assert.That(view.DisplayLayer, Is.EqualTo(WaterfallFrameLayer.Row));
                Assert.That(ReadPrivate<int>(view, "_writeRow"), Is.EqualTo(writeRowBefore));
                Assert.That(canvas.Zoom, Is.EqualTo(zoomBefore));
                Assert.That(canvas.PanOffset, Is.EqualTo(panBefore));
            }
        }

        [Test]
        public void ColumnIntensityScale_RecolorsStoredHistoryUsingItsFrameSourceGain()
        {
            using (var host = new Panel { Size = new Size(320, 240) })
            using (var view = new WaterfallView(
                host,
                camCount: 1,
                totalHeight: 1000,
                fullMode: WaterfallFullMode.Restart,
                screenMmPerPx: 0.264))
            {
                view.SetLayout(new[] { 0.0 }, new[] { 1000000.0 }, 1.0);
                view.SetExpectedFramePeriod(periodTicks: 100, periodMs: 100);
                view.SetDisplayLayer(WaterfallFrameLayer.Column);
                view.SetLayerIntensityScale(WaterfallFrameLayer.Column, 0.5f);

                byte[] raw = { 0, 0, 0, 0 };
                byte[] column = { 255, 255, 255, 255 };
                view.PushFrameVariants(
                    1, raw, column, raw, 2, 2, tick: 100,
                    columnSourceGain: 2f, rowSourceGain: 0f);
                view.PushFrameVariants(
                    1, raw, column, raw, 2, 2, tick: 200,
                    columnSourceGain: 2f, rowSourceGain: 0f);

                Thread.Sleep(200);
                Application.DoEvents();
                Assert.That(
                    SpinWait.SpinUntil(
                        () => ReadPrivate<int>(view, "_writeRow") >= 2 &&
                              !ReadPrivate<bool>(view, "_writerRunning"),
                        2000),
                    Is.True,
                    "waterfall writer did not persist the source gain with the aligned band");

                using (Bitmap halfScale = ReadRegion(view, width: 2))
                    Assert.That(halfScale.GetPixel(0, 0).R, Is.EqualTo(64).Within(1));

                view.SetLayerIntensityScale(WaterfallFrameLayer.Column, 1f);

                using (Bitmap fullScale = ReadRegion(view, width: 2))
                    Assert.That(fullScale.GetPixel(0, 0).R, Is.EqualTo(128).Within(1));
            }
        }

        [Test]
        public void HorizontalCrop_PreservesWaterfallHistoryAndWriteHead()
        {
            using (var host = new Panel { Size = new Size(320, 240) })
            using (var view = new WaterfallView(
                host,
                camCount: 1,
                totalHeight: 1000,
                fullMode: WaterfallFullMode.Restart,
                screenMmPerPx: 0.264))
            {
                view.SetLayout(new[] { 0.0 }, new[] { 1000000.0 }, 1.0);
                view.SetExpectedFramePeriod(periodTicks: 100, periodMs: 100);
                view.PushFrame(1, new byte[] { 10, 11, 12, 13 }, 2, 2, tick: 100);
                view.PushFrame(1, new byte[] { 14, 15, 16, 17 }, 2, 2, tick: 200);

                Thread.Sleep(200);
                Application.DoEvents();
                Assert.That(
                    SpinWait.SpinUntil(() => ReadPrivate<int>(view, "_writeRow") >= 2, 1000),
                    Is.True);
                Assert.That(
                    SpinWait.SpinUntil(
                        () => !ReadPrivate<bool>(view, "_writerRunning"), 1000),
                    Is.True);

                int writeRow = ReadPrivate<int>(view, "_writeRow");
                byte[][][][] history = ReadPrivate<byte[][][][]>(view, "_cameraLayerChunks");
                byte firstPixel = history[(int)WaterfallFrameLayer.Raw][0][0][0];

                view.SetHorizontalDisplayCrop(0.25, 0.25);

                Assert.That(
                    ReadPrivate<byte[][][][]>(view, "_cameraLayerChunks"),
                    Is.SameAs(history));
                Assert.That(ReadPrivate<int>(view, "_writeRow"), Is.EqualTo(writeRow));
                Assert.That(
                    history[(int)WaterfallFrameLayer.Raw][0][0][0],
                    Is.EqualTo(firstPixel));
            }
        }

        [Test]
        public void StartChange_RepositionsAlreadyWrittenWaterfallHistory()
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
                    new[] { 1000.0, 1000.0 },
                    refOpsMm: 1.0);
                view.SetExpectedFramePeriod(periodTicks: 100, periodMs: 100);
                view.PushFrame(1, new byte[] { 10, 11 }, 2, 1, tick: 100);
                view.PushFrame(2, new byte[] { 20, 21 }, 2, 1, tick: 100);

                Thread.Sleep(200);
                Application.DoEvents();
                Assert.That(
                    SpinWait.SpinUntil(
                        () => ReadPrivate<int>(view, "_writeRow") >= 1 &&
                              !ReadPrivate<bool>(view, "_writerRunning"),
                        2000),
                    Is.True);

                using (Bitmap before = ReadRegion(view, width: 4))
                {
                    Assert.That(before.GetPixel(0, 0).R, Is.EqualTo(10));
                    Assert.That(before.GetPixel(1, 0).R, Is.EqualTo(11));
                    Assert.That(before.GetPixel(2, 0).R, Is.EqualTo(20));
                    Assert.That(before.GetPixel(3, 0).R, Is.EqualTo(21));
                }

                int historyRows = ReadPrivate<int>(view, "_writeRow");
                view.SetLayout(
                    new[] { 0.0, 4.0 },
                    new[] { 1000.0, 1000.0 },
                    refOpsMm: 1.0);

                Assert.That(ReadPrivate<int>(view, "_writeRow"), Is.EqualTo(historyRows));
                using (Bitmap after = ReadRegion(view, width: 6))
                {
                    Assert.That(after.GetPixel(0, 0).R, Is.EqualTo(10));
                    Assert.That(after.GetPixel(1, 0).R, Is.EqualTo(11));
                    Assert.That(after.GetPixel(2, 0).R, Is.Zero);
                    Assert.That(after.GetPixel(3, 0).R, Is.Zero);
                    Assert.That(after.GetPixel(4, 0).R, Is.EqualTo(20));
                    Assert.That(after.GetPixel(5, 0).R, Is.EqualTo(21));
                }
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
                view.Reset(new[] { 1, 2 });

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
        public void ExpectedCameraSet_DoesNotFlushFirstBandWhileAnotherCameraIsCopying()
        {
            using (var host = new Panel { Size = new Size(320, 240) })
            using (var view = new WaterfallView(
                host,
                camCount: 2,
                totalHeight: 1000,
                fullMode: WaterfallFullMode.Restart,
                screenMmPerPx: 0.264))
            {
                var flow = new List<string>();
                view.FlowLog = flow.Add;
                view.SetLayout(
                    new[] { 0.0, 2.0 },
                    new[] { 1000000.0, 1000000.0 },
                    refOpsMm: 1.0);
                view.SetExpectedFramePeriod(periodTicks: 500, periodMs: 500);
                view.Reset(new[] { 1, 2 });

                view.PushFrame(2, new byte[] { 20, 21, 22, 23 }, 2, 2, tick: 1002);
                Thread.Sleep(250);
                Application.DoEvents();
                Assert.That(ReadPrivate<int>(view, "_writeRow"), Is.EqualTo(0));

                view.PushFrame(1, new byte[] { 10, 11, 12, 13 }, 2, 2, tick: 1000);
                Assert.That(
                    SpinWait.SpinUntil(() => ReadPrivate<int>(view, "_writeRow") >= 2, 1000),
                    Is.True);
                Assert.That(ReadPrivate<int>(view, "_writeRow"), Is.EqualTo(2));
                Assert.That(
                    flow.Exists(line => line.Contains(
                        "band first generation=1 seq=0 cams=1,2 expected=1,2")),
                    Is.True);
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

        private static Bitmap ReadRegion(WaterfallView view, int width)
        {
            MethodInfo method = typeof(WaterfallView).GetMethod(
                "ProvideRegion",
                BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.That(method, Is.Not.Null);
            return (Bitmap)method.Invoke(
                view,
                new object[] { new Rectangle(0, 0, width, 1), new Size(width, 1) });
        }
    }
}
