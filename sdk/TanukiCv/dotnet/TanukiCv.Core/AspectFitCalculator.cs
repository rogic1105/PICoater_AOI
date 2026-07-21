using System;

namespace TanukiCv.Core
{
    /// <summary>Aspect-fit transform shared by canvases and pre-layout callers.</summary>
    public struct AspectFitTransform
    {
        public float Zoom;
        public float PanX;
        public float PanY;

        public double ContentX(double viewportX) => (viewportX - PanX) / Zoom;
        public double ContentY(double viewportY) => (viewportY - PanY) / Zoom;
    }

    public static class AspectFitCalculator
    {
        public const float ContentFill = 0.95f;

        public static bool TryCompute(
            int contentWidth, int contentHeight,
            int viewportWidth, int viewportHeight,
            out AspectFitTransform transform)
        {
            transform = default(AspectFitTransform);
            if (contentWidth <= 0 || contentHeight <= 0 ||
                viewportWidth <= 0 || viewportHeight <= 0)
                return false;

            float ratioW = (float)viewportWidth / contentWidth;
            float ratioH = (float)viewportHeight / contentHeight;
            float zoom = Math.Min(ratioW, ratioH) * ContentFill;
            if (zoom <= 0 || float.IsNaN(zoom) || float.IsInfinity(zoom))
                return false;

            float drawW = contentWidth * zoom;
            float drawH = contentHeight * zoom;
            transform = new AspectFitTransform
            {
                Zoom = zoom,
                PanX = (viewportWidth - drawW) / 2f,
                PanY = (viewportHeight - drawH) / 2f
            };
            return true;
        }
    }
}
