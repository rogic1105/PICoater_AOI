using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Integration.Tests
{
    [TestFixture]
    public class ReviewPeriodDataLoaderTests
    {
        private string _tempDirectory;

        [SetUp]
        public void SetUp()
        {
            _tempDirectory = Path.Combine(Path.GetTempPath(),
                "ReviewPeriodDataLoaderTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tempDirectory);
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_tempDirectory, true); } catch { }
        }

        [Test]
        public void Loaders_OnePeriod_ReturnFramesAndColumnAndMergedRowCurves()
        {
            string baseName = Path.Combine(_tempDirectory, "20260721_080000.000-1");
            string imagePath = baseName + CaptureFileNaming.RawJpg;
            WriteJpeg(imagePath);
            WriteCurveBin(baseName + CaptureFileNaming.MeanC, new[] { 1f, 2f });
            WriteCurveBin(baseName + CaptureFileNaming.MaxC, new[] { 3f, 4f });
            WriteCurveBin(baseName + CaptureFileNaming.MeanR, new[] { 5f, 6f });
            WriteCurveBin(baseName + CaptureFileNaming.MaxR, new[] { 7f, 8f });
            var images = new Dictionary<int, string> { { 1, imagePath } };
            var loader = new ReviewPeriodDataLoader();

            ReviewPeriodFrames frames = loader.LoadFrames(images, 2, 1, null, false, "v");
            ReviewPeriodColumnCurves column = loader.LoadColumnCurves(images, 2);
            ReviewPeriodRowCurves row = loader.LoadMergedRowCurves(images, 2);

            Assert.That(frames.Widths[0], Is.EqualTo(4));
            Assert.That(frames.Heights[0], Is.EqualTo(3));
            Assert.That(frames.GrayFrames[0], Has.Length.EqualTo(12));
            Assert.That(frames.GrayFrames[1], Is.Null);
            Assert.That(column.Mean[0], Is.EqualTo(new[] { 1f, 2f }));
            Assert.That(column.Max[0], Is.EqualTo(new[] { 3f, 4f }));
            Assert.That(column.Mean[1], Is.Null);
            Assert.That(row.Mean, Is.EqualTo(new[] { 5f, 6f }));
            Assert.That(row.Max, Is.EqualTo(new[] { 7f, 8f }));
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
