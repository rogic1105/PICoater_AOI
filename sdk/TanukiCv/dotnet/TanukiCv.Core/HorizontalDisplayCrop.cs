using System;
using System.Collections.Generic;

namespace TanukiCv.Core
{
    /// <summary>
    /// Describes a horizontal display-only crop in merged-image coordinates.
    /// The source placements and source image data remain unchanged.
    /// </summary>
    public struct HorizontalDisplayCrop
    {
        public int SourceWidthPx;
        public int SourceLeftPx;
        public int VisibleWidthPx;
        public double VisibleStartMm;

        public bool IsCropped => VisibleWidthPx < SourceWidthPx;

        public static HorizontalDisplayCrop Compute(
            int sourceWidthPx,
            double sourceStartMm,
            double mmPerSourcePixel,
            double trimHeadMm,
            double trimTailMm)
        {
            int sourceWidth = Math.Max(1, sourceWidthPx);
            double pitch = mmPerSourcePixel > 0 ? mmPerSourcePixel : 1.0;
            int headPx = Math.Max(0, (int)Math.Round(Math.Max(0, trimHeadMm) / pitch));
            int tailPx = Math.Max(0, (int)Math.Round(Math.Max(0, trimTailMm) / pitch));

            headPx = Math.Min(headPx, sourceWidth - 1);
            tailPx = Math.Min(tailPx, sourceWidth - headPx - 1);
            return new HorizontalDisplayCrop
            {
                SourceWidthPx = sourceWidth,
                SourceLeftPx = headPx,
                VisibleWidthPx = Math.Max(1, sourceWidth - headPx - tailPx),
                VisibleStartMm = sourceStartMm + headPx * pitch
            };
        }

        public List<CameraPlacement> Apply(IList<CameraPlacement> source)
        {
            var result = new List<CameraPlacement>();
            if (source == null || VisibleWidthPx <= 0) return result;

            int cropLeft = SourceLeftPx;
            int cropRight = cropLeft + VisibleWidthPx;
            for (int i = 0; i < source.Count; i++)
            {
                CameraPlacement placement = source[i];
                int sourceDestLeft = placement.DestX;
                int sourceDestRight = sourceDestLeft + placement.SrcWidth;
                int clippedLeft = Math.Max(sourceDestLeft, cropLeft);
                int clippedRight = Math.Min(sourceDestRight, cropRight);
                if (clippedRight <= clippedLeft) continue;

                int clippedSourceLeft =
                    placement.SrcLeft + clippedLeft - sourceDestLeft;
                int shiftedDestLeft = clippedLeft - cropLeft;
                result.Add(new CameraPlacement
                {
                    CameraId = placement.CameraId,
                    SrcLeft = clippedSourceLeft,
                    SrcWidth = clippedRight - clippedLeft,
                    XOffset = shiftedDestLeft - clippedSourceLeft
                });
            }
            return result;
        }
    }
}
