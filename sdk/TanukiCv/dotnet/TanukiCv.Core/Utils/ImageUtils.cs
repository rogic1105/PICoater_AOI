using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace TanukiCv.Utils
{
    public static class ImageUtils
    {
        public static readonly ColorPalette GrayScalePalette;

        static ImageUtils()
        {
            using (var tempBmp = new Bitmap(1, 1, PixelFormat.Format8bppIndexed))
            {
                GrayScalePalette = tempBmp.Palette;
            }
            for (int i = 0; i < 256; i++) GrayScalePalette.Entries[i] = Color.FromArgb(i, i, i);
        }

        // 保持 byte[] 版本兼容性，但轉發到 IntPtr 版本
        public static Bitmap Create8bppBitmap(byte[] data, int width, int height)
        {
            if (width <= 0 || height <= 0 || data == null) return null;
            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                return Create8bppBitmap(handle.AddrOfPinnedObject(), width, height);
            }
            finally
            {
                handle.Free();
            }
        }

        /// <summary>
        /// [修改] 增加 flipY 參數，預設為 false (保持相容)
        /// 當 flipY = true 時，執行上下翻轉複製
        /// </summary>
        public static Bitmap Create8bppBitmap(IntPtr srcPtr, int width, int height, bool flipY = false)
        {
            if (width <= 0 || height <= 0 || srcPtr == IntPtr.Zero) return null;

            Bitmap bmp = new Bitmap(width, height, PixelFormat.Format8bppIndexed);
            bmp.Palette = GrayScalePalette;

            BitmapData bData = bmp.LockBits(new Rectangle(0, 0, width, height),
                ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);

            try
            {
                unsafe
                {
                    byte* pSrc = (byte*)srcPtr;
                    byte* pDst = (byte*)bData.Scan0;
                    int stride = bData.Stride;

                    // 若不需翻轉且 Stride 對齊，走極速複製
                    if (!flipY && stride == width)
                    {
                        long bufferSize = (long)width * height;
                        Buffer.MemoryCopy(pSrc, pDst, bufferSize, bufferSize);
                    }
                    else
                    {
                        // 處理 [翻轉] 或 [Padding] 的逐行複製
                        // Parallel.For 雖然快，但對於這種記憶體密集操作，簡單迴圈通常足夠且穩定
                        for (int y = 0; y < height; y++)
                        {
                            // [關鍵邏輯]
                            // 寫入(Dst)：永遠從第 0 行寫到第 h-1 行 (由上而下)
                            // 讀取(Src)：
                            //   若 flipY=true，讀取第 (h-1-y) 行 (由下而上) -> 達成翻轉
                            //   若 flipY=false，讀取第 y 行 (由上而下)
                            int srcRowIndex = flipY ? (height - 1 - y) : y;

                            byte* srcRow = pSrc + (long)srcRowIndex * width;
                            byte* dstRow = pDst + (long)y * stride;

                            Buffer.MemoryCopy(srcRow, dstRow, width, width);
                        }
                    }
                }
            }
            finally
            {
                bmp.UnlockBits(bData);
            }

            return bmp;
        }

        public static byte[] BitmapTo8bppArray(Bitmap src, out int w, out int h)
        {
            // 保持原本的 BitmapTo8bppArray 代碼不變...
            // (為了節省篇幅，這裡省略，請保留原檔案內容)
            w = src.Width;
            h = src.Height;
            byte[] result = new byte[w * h];
            // ...
            return result;
        }
    }
}