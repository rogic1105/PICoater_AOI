using System.Drawing;
using AniloxRoll.Monitor.UI.State;
using NUnit.Framework;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public sealed class ReviewDisplayContentTests
    {
        [Test]
        public void ReplaceImages_PreservesCurvesForReportReusePath()
        {
            var content = new ReviewDisplayContent();
            var columnMean = new[] { new[] { 0.1f } };
            content.SetCurves(columnMean, null, null, null, null, null);

            var image = new Bitmap(2, 2);
            content.ReplaceImages(new[] { image });

            Assert.That(content.ColumnMean, Is.SameAs(columnMean));
            Assert.That(content.Images[0], Is.SameAs(image));
            content.ClearAll();
            BitmapPool.Rent(2, 2).Dispose();
        }

        [Test]
        public void ClearAll_ReturnsCurrentImagesAndClearsEveryCurveFamily()
        {
            var content = new ReviewDisplayContent();
            var image = new Bitmap(3, 3);
            content.ReplaceImages(new[] { image });
            content.SetCurves(
                new[] { new[] { 0.1f } }, new[] { new[] { 0.2f } },
                new[] { new[] { 0.3f } }, new[] { new[] { 0.4f } },
                new[] { 0.5f }, new[] { 0.6f });

            content.ClearAll();

            Assert.That(content.HasImages, Is.False);
            Assert.That(content.ColumnMean, Is.Null);
            Assert.That(content.ColumnMax, Is.Null);
            Assert.That(content.RowMean, Is.Null);
            Assert.That(content.RowMax, Is.Null);
            Assert.That(content.MergedRowMean, Is.Null);
            Assert.That(content.MergedRowMax, Is.Null);
            Bitmap pooled = BitmapPool.Rent(3, 3);
            Assert.That(pooled, Is.SameAs(image));
            pooled.Dispose();
        }
    }
}
