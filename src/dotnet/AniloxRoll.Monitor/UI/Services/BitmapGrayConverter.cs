using System.Drawing;
using System.Drawing.Imaging;

namespace AniloxRoll.Monitor.UI.Services
{
    internal static class BitmapGrayConverter
    {
        public static byte[] ToGray8(Bitmap bitmap, out int width, out int height)
        {
            width = bitmap.Width;
            height = bitmap.Height;
            if (width <= 0 || height <= 0) return null;

            var rect = new Rectangle(0, 0, width, height);
            var destination = new byte[width * height];
            if (bitmap.PixelFormat == PixelFormat.Format8bppIndexed)
            {
                var data = bitmap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);
                try
                {
                    for (int y = 0; y < height; y++)
                    {
                        System.Runtime.InteropServices.Marshal.Copy(
                            data.Scan0 + y * data.Stride, destination, y * width, width);
                    }
                }
                finally { bitmap.UnlockBits(data); }
                return destination;
            }

            var rgb = bitmap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            try
            {
                int stride = rgb.Stride;
                var row = new byte[stride];
                for (int y = 0; y < height; y++)
                {
                    System.Runtime.InteropServices.Marshal.Copy(
                        rgb.Scan0 + y * stride, row, 0, stride);
                    int offset = y * width;
                    for (int x = 0; x < width; x++)
                        destination[offset + x] = row[x * 3 + 1];
                }
            }
            finally { bitmap.UnlockBits(rgb); }
            return destination;
        }
    }
}
