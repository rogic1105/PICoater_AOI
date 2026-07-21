using System.Drawing;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.State
{
    /// <summary>
    /// Owns the currently displayed Review images and curve snapshots.
    /// Loading order belongs to ReviewStitchCoordinator; rendering belongs to ReviewChartPresenter.
    /// </summary>
    internal sealed class ReviewDisplayContent
    {
        public Bitmap[] Images { get; private set; }
        public float[][] ColumnMean { get; private set; }
        public float[][] ColumnMax { get; private set; }
        public float[][] RowMean { get; private set; }
        public float[][] RowMax { get; private set; }
        public float[] MergedRowMean { get; private set; }
        public float[] MergedRowMax { get; private set; }

        public bool HasImages => Images != null;

        public void ReplaceImages(Bitmap[] images)
        {
            ClearImages();
            Images = images;
        }

        public void SetCurves(
            float[][] columnMean,
            float[][] columnMax,
            float[][] rowMean,
            float[][] rowMax,
            float[] mergedRowMean,
            float[] mergedRowMax)
        {
            ColumnMean = columnMean;
            ColumnMax = columnMax;
            RowMean = rowMean;
            RowMax = rowMax;
            MergedRowMean = mergedRowMean;
            MergedRowMax = mergedRowMax;
        }

        public void ClearImages()
        {
            Bitmap[] images = Images;
            Images = null;
            if (images == null) return;

            foreach (Bitmap bitmap in images)
                BitmapPool.Return(bitmap);
        }

        public void ClearAll()
        {
            ClearImages();
            ColumnMean = null;
            ColumnMax = null;
            RowMean = null;
            RowMax = null;
            MergedRowMean = null;
            MergedRowMax = null;
        }
    }
}
