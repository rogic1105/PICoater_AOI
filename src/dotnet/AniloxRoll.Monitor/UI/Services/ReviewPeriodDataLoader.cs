using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Widgets;

namespace AniloxRoll.Monitor.UI.Services
{
    internal sealed class ReviewPeriodFrames
    {
        public byte[][] GrayFrames { get; set; }
        public int[] Widths { get; set; }
        public int[] Heights { get; set; }
    }

    internal sealed class ReviewPeriodColumnCurves
    {
        public float[][] Mean { get; set; }
        public float[][] Max { get; set; }
    }

    internal sealed class ReviewPeriodRowCurves
    {
        public float[] Mean { get; set; }
        public float[] Max { get; set; }
    }

    /// <summary>
    /// Loads the image and curve projections for one Review period. It owns persisted-file
    /// interpretation and conversion, but does not know about settings, controls, or chart state.
    /// </summary>
    internal sealed class ReviewPeriodDataLoader
    {
        public ReviewPeriodFrames LoadFrames(
            IReadOnlyDictionary<int, string> images,
            int cameraCount,
            int scale,
            Func<string, Bitmap> bitmapLoader,
            bool useProcessed,
            string ridgeDirection)
        {
            var result = new ReviewPeriodFrames
            {
                GrayFrames = new byte[cameraCount][],
                Widths = new int[cameraCount],
                Heights = new int[cameraCount]
            };

            for (int index = 0; index < cameraCount; index++)
            {
                if (!images.TryGetValue(index + 1, out string path)) continue;
                Bitmap bitmap = null;
                try
                {
                    bitmap = GrabImageStitcher.LoadCameraImage(
                        path, scale, bitmapLoader, useProcessed, ridgeDirection);
                    if (bitmap != null)
                    {
                        result.GrayFrames[index] = BitmapGrayConverter.ToGray8(
                            bitmap, out result.Widths[index], out result.Heights[index]);
                    }
                }
                catch (Exception ex)
                {
                    Trace.WriteLine(
                        $"[GlobalMerge] CAM{index + 1}: {ex.GetType().Name}: {ex.Message}");
                }
                finally
                {
                    bitmap?.Dispose();
                }
            }

            return result;
        }

        public ReviewPeriodColumnCurves LoadColumnCurves(
            IReadOnlyDictionary<int, string> images,
            int cameraCount)
        {
            var result = new ReviewPeriodColumnCurves
            {
                Mean = new float[cameraCount][],
                Max = new float[cameraCount][]
            };

            for (int index = 0; index < cameraCount; index++)
            {
                if (!images.TryGetValue(index + 1, out string path)) continue;
                string basePath = CurveMergeHelper.GetCurveBasePath(path);
                result.Mean[index] = CurveBinFile.Load(CaptureFileNaming.ResolveMeanC(basePath));
                result.Max[index] = CurveBinFile.Load(CaptureFileNaming.ResolveMaxC(basePath));
            }

            return result;
        }

        public ReviewPeriodRowCurves LoadMergedRowCurves(
            IReadOnlyDictionary<int, string> images,
            int cameraCount)
        {
            var mean = new float[cameraCount][];
            var max = new float[cameraCount][];
            for (int index = 0; index < cameraCount; index++)
            {
                if (!images.TryGetValue(index + 1, out string path)) continue;
                string basePath = CurveMergeHelper.GetCurveBasePath(path);
                mean[index] = CurveBinFile.Load(CaptureFileNaming.ResolveMeanR(basePath));
                max[index] = CurveBinFile.Load(CaptureFileNaming.ResolveMaxR(basePath));
            }

            CurveMergeHelper.MergeRowCurvesOverlap(
                mean, max, cameraCount, out float[] mergedMean, out float[] mergedMax);
            return new ReviewPeriodRowCurves { Mean = mergedMean, Max = mergedMax };
        }
    }
}
