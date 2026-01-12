using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.IO;

namespace AniloxRoll.Monitor.Core
{
    public class PICoaterWrapper : IDisposable
    {
        // === 1. DllImport 定義 (對應 export_api.h) ===
        private const string DllName = "picoater_api.dll";

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr PICoater_Create();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void PICoater_Destroy(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int PICoater_Initialize(IntPtr handle, int width, int height);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr PICoater_AllocPinned(ulong size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void PICoater_FreePinned(IntPtr ptr);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int PICoater_Run(
            IntPtr handle,
            IntPtr h_in,
            IntPtr h_bg_out,
            IntPtr h_mura_out,
            IntPtr h_ridge_out,
            float bgSigmaFactor,
            float ridgeSigma,
            string ridgeMode,
            IntPtr stream
        );

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int PICoater_LoadThumbnail(
            string path,
            int targetWidth,
            IntPtr outBuffer,
            out int outRealW,
            out int outRealH
        );

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern bool PICoater_FastReadBMP(string filepath, out int w, out int h, IntPtr outBuffer, int bufferSize);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern bool PICoater_FastWriteBMP(string filepath, int w, int h, IntPtr inBuffer);


        // === 2. 成員變數 ===
        private IntPtr _handle = IntPtr.Zero;
        private bool _disposed = false;

        // === 3. 建構與解構 ===
        public PICoaterWrapper()
        {
            _handle = PICoater_Create();
            if (_handle == IntPtr.Zero) throw new Exception("Failed to create PICoater instance.");
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero) { PICoater_Destroy(_handle); _handle = IntPtr.Zero; }
                _disposed = true;
            }
        }

        // === 4. 功能 A: 快速讀取縮圖 (Static) ===
        public static Bitmap LoadThumbnail(string path, int targetWidth)
        {
            if (!File.Exists(path)) return null;

            // 預估 Buffer 大小 (假設長寬比不會超過 1:20，給大一點沒關係)
            // 8-bit gray: width * height * 1 byte
            int maxBufferSize = targetWidth * (targetWidth * 20);

            // 申請非託管記憶體來接資料
            IntPtr pBuffer = Marshal.AllocHGlobal(maxBufferSize);

            try
            {
                int w, h;
                int ret = PICoater_LoadThumbnail(path, targetWidth, pBuffer, out w, out h);

                if (ret != 0) return null; // 讀取失敗

                // 建立 C# Bitmap (8bppIndexed)
                Bitmap bmp = new Bitmap(w, h, PixelFormat.Format8bppIndexed);

                // 設定灰階調色盤 (必須！)
                ColorPalette pal = bmp.Palette;
                for (int i = 0; i < 256; i++) pal.Entries[i] = Color.FromArgb(i, i, i);
                bmp.Palette = pal;

                // 拷貝像素資料 (Unmanaged -> Managed Bitmap)
                BitmapData bData = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
                unsafe
                {
                    Buffer.MemoryCopy((void*)pBuffer, (void*)bData.Scan0, maxBufferSize, w * h);
                }
                bmp.UnlockBits(bData);

                return bmp;
            }
            finally
            {
                Marshal.FreeHGlobal(pBuffer);
            }
        }

        // === 5. 功能 B: 執行檢測 (使用 Pinned Memory) ===
        // 這裡回傳處理後的 Mura 圖，你也可以改回傳結構包含 Ridge/BG
        public Bitmap RunInspectionFast(string filePath)
        {
            if (_handle == IntPtr.Zero) throw new ObjectDisposedException("PICoaterWrapper");
            if (!File.Exists(filePath)) return null;

            // 1. 預估緩衝區大小 (假設最大 16384x10000)
            // 為了安全，也可以先讀 Header，但為了極速我們先給個夠大的
            int maxW = 16384;
            int maxH = 10000;
            ulong maxBytes = (ulong)(maxW * maxH);

            // 2. 申請 Pinned Memory (Input)
            IntPtr pIn = PICoater_AllocPinned(maxBytes);
            IntPtr pBg = IntPtr.Zero;
            IntPtr pMura = IntPtr.Zero;
            IntPtr pRidge = IntPtr.Zero;

            try
            {
                // 3. 極速讀圖 (SSD -> Pinned Memory)
                int w, h;
                if (!PICoater_FastReadBMP(filePath, out w, out h, pIn, (int)maxBytes))
                {
                    return null; // 讀取失敗 (可能不是 BMP)
                }

                ulong imgSize = (ulong)(w * h);

                // 4. 申請輸出 Pinned Memory
                // pBg = PICoater_AllocPinned(imgSize);
                pMura = PICoater_AllocPinned(imgSize);
                // pRidge = PICoater_AllocPinned(imgSize);

                // 5. 初始化演算法 (若尺寸沒變會跳過)
                PICoater_Initialize(_handle, w, h);

                // 6. 執行演算法
                int ret = PICoater_Run(
                    _handle,
                    pIn, pBg, pMura, pRidge,
                    2.0f, 9.0f, "vertical", IntPtr.Zero
                );

                if (ret != 0) throw new Exception($"PICoater_Run failed: {ret}");

                // 7. 轉成 Bitmap 回傳 (這裡回傳 Mura 圖)
                // 注意：這裡會發生一次 Memory Copy (Pinned -> Managed Heap)
                Bitmap resultBmp = new Bitmap(w, h, PixelFormat.Format8bppIndexed);
                ColorPalette pal = resultBmp.Palette;
                for (int i = 0; i < 256; i++) pal.Entries[i] = Color.FromArgb(i, i, i);
                resultBmp.Palette = pal;

                BitmapData dstData = resultBmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
                unsafe
                {
                    Buffer.MemoryCopy((void*)pMura, (void*)dstData.Scan0, imgSize, imgSize);
                }
                resultBmp.UnlockBits(dstData);

                return resultBmp;
            }
            finally
            {
                // 8. 釋放所有 Pinned Memory
                if (pIn != IntPtr.Zero) PICoater_FreePinned(pIn);
                if (pBg != IntPtr.Zero) PICoater_FreePinned(pBg);
                if (pMura != IntPtr.Zero) PICoater_FreePinned(pMura);
                if (pRidge != IntPtr.Zero) PICoater_FreePinned(pRidge);
            }
        }


    }
}