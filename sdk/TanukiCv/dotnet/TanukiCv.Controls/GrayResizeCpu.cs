using System;

namespace TanukiCv.Controls
{
    /// <summary>
    /// 純 CPU 的灰階縮放（雙線性內插，無 GPU、無 System.Drawing）——LOD provider 的 CPU 版單一來源。
    /// 簽章吻合 <see cref="GrayResize"/> 委派，可直接：<c>view.EnableLod(GrayResizeCpu.Resize)</c>。
    ///
    /// 沒 GPU（或刻意不用 GPU）的機器走這條；有 GPU 的呼叫端可改餵 GPU 版（如 core_cv 的 TanukiCv_Resize_GPU），
    /// 兩者由 <c>chkLodCPU</c>/<c>chkLodGPU</c> 分開跑測。雙線性對縮小（LOD 縮到 panel）平滑、對放大也堪用。
    /// </summary>
    public static class GrayResizeCpu
    {
        /// <summary>8-bit 灰階 src(srcW×srcH) → dst(dstW×dstH)，雙線性內插。回傳長度 dstW*dstH 的灰階 bytes。</summary>
        public static byte[] Resize(byte[] src, int srcW, int srcH, int dstW, int dstH)
        {
            if (src == null || srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) return null;
            if (src.Length < srcW * srcH) return null;

            var dst = new byte[dstW * dstH];
            double sx = (double)srcW / dstW;
            double sy = (double)srcH / dstH;

            for (int y = 0; y < dstH; y++)
            {
                double fy = (y + 0.5) * sy - 0.5;          // 像素中心對齊（half-pixel），避免整體偏移
                int y0 = (int)Math.Floor(fy);
                double wy = fy - y0;
                int y0c = Clamp(y0, srcH);
                int y1c = Clamp(y0 + 1, srcH);
                int row0 = y0c * srcW, row1 = y1c * srcW, drow = y * dstW;

                for (int x = 0; x < dstW; x++)
                {
                    double fx = (x + 0.5) * sx - 0.5;
                    int x0 = (int)Math.Floor(fx);
                    double wx = fx - x0;
                    int x0c = Clamp(x0, srcW);
                    int x1c = Clamp(x0 + 1, srcW);

                    double top = src[row0 + x0c] * (1 - wx) + src[row0 + x1c] * wx;
                    double bot = src[row1 + x0c] * (1 - wx) + src[row1 + x1c] * wx;
                    int v = (int)(top * (1 - wy) + bot * wy + 0.5);
                    dst[drow + x] = (byte)(v < 0 ? 0 : (v > 255 ? 255 : v));
                }
            }
            return dst;
        }

        private static int Clamp(int v, int len) => v < 0 ? 0 : (v >= len ? len - 1 : v);
    }
}
