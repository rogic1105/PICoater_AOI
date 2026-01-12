// PICoater_AOI\src_dotnet\AniloxRoll.Monitor\Utils\BitmapHelper.cs

using System.Drawing;
using System.Drawing.Drawing2D;

namespace AniloxRoll.Monitor.Utils
{
    public static class BitmapHelper
    {
        /// <summary>
        /// 使用 GDI+ 建立高品質縮圖
        /// </summary>
        public static Bitmap MakeThumbnail(Bitmap src, int width)
        {
            if (src == null) return null;
            int height = (int)((float)src.Height / src.Width * width);
            Bitmap thumb = new Bitmap(width, height);
            using (Graphics g = Graphics.FromImage(thumb))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.DrawImage(src, 0, 0, width, height);
            }
            return thumb;
        }
    }
}