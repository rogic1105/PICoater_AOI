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
            ReviewImageData result = loader.Load(
                _tempRoot, grabId, date, date, 2, false, "v");

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
