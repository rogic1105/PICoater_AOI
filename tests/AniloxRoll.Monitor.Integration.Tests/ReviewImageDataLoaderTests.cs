using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Integration.Tests
{
    [TestFixture]
    public class ReviewImageDataLoaderTests
    {
        private string _tempRoot;

        [SetUp]
        public void SetUp()
        {
            _tempRoot = Path.Combine(Path.GetTempPath(),
                "ReviewImageDataLoaderTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tempRoot);
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_tempRoot, true); } catch { }
        }

        [Test]
        public void Load_OneCapture_ReturnsImageCurvesGrayFrameAndConfig()
        {
            DateTime date = new DateTime(2026, 7, 21);
            string grabId = "260721-080000";
            string fileName = "20260721_080000.000-1";
            string imageDir = CaptureStoragePaths.DateImageDir(_tempRoot, date);
            Directory.CreateDirectory(imageDir);
            WriteJpeg(Path.Combine(imageDir, fileName + CaptureFileNaming.RawJpg));
            WriteCurveBin(Path.Combine(imageDir, fileName + CaptureFileNaming.MeanC), new[] { 1f, 3f });
            WriteCurveBin(Path.Combine(imageDir, fileName + CaptureFileNaming.MaxC), new[] { 2f, 4f });
            WriteCurveBin(Path.Combine(imageDir, fileName + CaptureFileNaming.MeanR), new[] { 5f, 6f });
            WriteCurveBin(Path.Combine(imageDir, fileName + CaptureFileNaming.MaxR), new[] { 7f, 8f });
            WriteCsv(date, grabId, fileName);

            var loader = new ReviewImageDataLoader();
            ReviewImageLoadPlan plan = loader.Prepare(
                _tempRoot, grabId, date, date, 2, false, "v");
            Assert.That(plan.ExpectedWidths[0], Is.EqualTo(4));
            Assert.That(plan.ExpectedHeights[0], Is.EqualTo(3));
            Assert.That(plan.ExpectedWidths[1], Is.Zero);
            Assert.That(plan.ExpectedHeights[1], Is.Zero);

            ReviewImageData result = loader.Load(plan, 2, false, "v");

            try
            {
                Assert.That(result.TotalImageCount, Is.EqualTo(1));
                Assert.That(result.Config, Is.Not.Null);
                Assert.That(result.Images[0], Is.Not.Null);
                Assert.That(result.Images[1], Is.Null);
                Assert.That(result.GrayWidths[0], Is.EqualTo(4));
                Assert.That(result.GrayHeights[0], Is.EqualTo(3));
                Assert.That(result.GrayFrames[0], Has.Length.EqualTo(12));
                Assert.That(result.ColumnMean[0], Is.EqualTo(new[] { 1f, 3f }));
                Assert.That(result.ColumnMax[0], Is.EqualTo(new[] { 2f, 4f }));
                Assert.That(result.RowMean[0], Is.EqualTo(new[] { 5f, 6f }));
                Assert.That(result.RowMax[0], Is.EqualTo(new[] { 7f, 8f }));
            }
            finally
            {
                result.DisposeImages();
            }
        }

        [Test]
        public void Load_ImagesOnly_SkipsCurveBins()
        {
            DateTime date = new DateTime(2026, 7, 21);
            string grabId = "260721-081000";
            string fileName = "20260721_081000.000-1";
            string imageDir = CaptureStoragePaths.DateImageDir(_tempRoot, date);
            Directory.CreateDirectory(imageDir);
            WriteJpeg(Path.Combine(imageDir, fileName + CaptureFileNaming.RawJpg));
            WriteCsv(date, grabId, fileName);

            var loader = new ReviewImageDataLoader();
            ReviewImageLoadPlan plan = loader.Prepare(
                _tempRoot, grabId, date, date, 1, false, "v");
            ReviewImageData result = loader.Load(
                plan, 1, false, "v", includeCurves: false);

            try
            {
                Assert.That(result.Images[0], Is.Not.Null);
                Assert.That(result.GrayFrames[0], Has.Length.EqualTo(12));
                Assert.That(result.ColumnMean, Is.Null);
                Assert.That(result.ColumnMax, Is.Null);
                Assert.That(result.RowMean, Is.Null);
                Assert.That(result.RowMax, Is.Null);
            }
            finally
            {
                result.DisposeImages();
            }
        }

        [Test]
        public void Load_EnhancedThumbnailWithStandardMap_UsesCurrentGain()
        {
            DateTime date = new DateTime(2026, 7, 21);
            string grabId = "260721-082000";
            string fileName = "20260721_082000.000-1";
            string imageDir = CaptureStoragePaths.DateImageDir(_tempRoot, date);
            Directory.CreateDirectory(imageDir);
            string rawPath = Path.Combine(
                imageDir, fileName + CaptureFileNaming.RawJpg);
            WriteJpeg(rawPath);
            WriteJpeg(Path.Combine(imageDir, fileName + CaptureFileNaming.ProcC));
            File.WriteAllBytes(
                CaptureFileNaming.StripRawJpg(rawPath) + CaptureFileNaming.HessianC,
                HessianStandardMapCodec.Encode(
                    new byte[]
                    {
                        0x00, 0x00, 0x00, 0x38,
                        0x00, 0x3c, 0x00, 0x40
                    },
                    2, 2));
            WriteCsv(date, grabId, fileName);

            var loader = new ReviewImageDataLoader();
            ReviewImageLoadPlan plan = loader.Prepare(
                _tempRoot, grabId, date, date, 1, true, "c");
            ReviewImageData result = loader.Load(
                plan, 1, true, "c", includeCurves: false,
                useThumbnail: true, standardDisplayGain: 0.5f);

            try
            {
                Assert.That(result.PreviewSource, Is.EqualTo("hessian"));
                Assert.That(result.Images[0].Size, Is.EqualTo(new Size(2, 2)));
                Assert.That(result.Images[0].GetPixel(1, 0).R,
                    Is.EqualTo(64).Within(1));
                Assert.That(result.Images[0].GetPixel(0, 1).R,
                    Is.EqualTo(128).Within(1));
                Assert.That(result.Images[0].GetPixel(1, 1).R, Is.EqualTo(255));
            }
            finally
            {
                result.DisposeImages();
            }
        }

        private void WriteCsv(DateTime date, string grabId, string fileName)
        {
            string csvPath = CaptureStoragePaths.DailyCsv(_tempRoot, date);
            Directory.CreateDirectory(Path.GetDirectoryName(csvPath));
            var config = new CsvConfigSnapshot(
                new double[7], new double[7], new int[7], new double[7], new double[7],
                0.75f, 1f, 0f, 0f, 0f, 0f, 0d, 0d, date.AddHours(8));
            using (var writer = new StreamWriter(csvPath))
            {
                writer.WriteLine("Id,FileName,MaxExceed,MeanExceed");
                writer.WriteLine(config.ToCsvLine());
                writer.WriteLine($"{grabId},{fileName},0,0");
            }
        }

        private static void WriteJpeg(string path)
        {
            using (var bitmap = new Bitmap(4, 3, PixelFormat.Format24bppRgb))
            using (Graphics graphics = Graphics.FromImage(bitmap))
            {
                graphics.Clear(Color.FromArgb(40, 80, 120));
                bitmap.Save(path, ImageFormat.Jpeg);
            }
        }

        private static void WriteCurveBin(string path, float[] values)
        {
            using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write))
            using (var writer = new BinaryWriter(stream))
            {
                writer.Write(new[] { (byte)'M', (byte)'C', (byte)'B', (byte)'F' });
                writer.Write(1);
                writer.Write(1.0f);
                writer.Write(values.Length);
                foreach (float value in values) writer.Write(value);
            }
        }
    }
}
