using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.IO;

namespace AniloxRoll.Monitor
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

        // === 2. 成員變數 ===
        private IntPtr _handle = IntPtr.Zero;
        private bool _disposed = false;

        // === 3. 建構與解構 ===
        public PICoaterWrapper()
        {
            // 建立 C++ 物件
            _handle = PICoater_Create();
            if (_handle == IntPtr.Zero)
            {
                throw new Exception("Failed to create PICoater instance.");
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    PICoater_Destroy(_handle);
                    _handle = IntPtr.Zero;
                }
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
        public Bitmap RunInspection(Bitmap srcBmp)
        {
            if (_handle == IntPtr.Zero) throw new ObjectDisposedException("PICoaterWrapper");

            int w = srcBmp.Width;
            int h = srcBmp.Height;
            ulong size = (ulong)(w * h);

            // 1. 初始化 (分配 GPU 記憶體，若尺寸沒變 C++ 內部會跳過)
            PICoater_Initialize(_handle, w, h);

            // 2. 申請 Pinned Memory (加速傳輸)
            // 這次我們需要四塊：Input, BG, Mura, Ridge
            IntPtr pIn = PICoater_AllocPinned(size);
            IntPtr pBg = PICoater_AllocPinned(size);   // 如果不想看結果，可傳 IntPtr.Zero (需修改 C++ 允許 null)
            IntPtr pMura = PICoater_AllocPinned(size);
            IntPtr pRidge = PICoater_AllocPinned(size);

            Bitmap resultBmp = null;

            try
            {
                // 3. 複製 Input: Bitmap -> Pinned Memory
                BitmapData srcData = srcBmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);
                unsafe
                {
                    Buffer.MemoryCopy((void*)srcData.Scan0, (void*)pIn, size, size);
                }
                srcBmp.UnlockBits(srcData);

                // 4. 執行演算法
                // 參數可拉出去變成函式參數
                float bgSigma = 2.0f;
                float ridgeSigma = 9.0f;
                string ridgeMode = "vertical";

                int ret = PICoater_Run(
                    _handle,
                    pIn, pBg, pMura, pRidge,
                    bgSigma, ridgeSigma, ridgeMode,
                    IntPtr.Zero // Stream
                );

                if (ret != 0) throw new Exception($"PICoater_Run failed with code {ret}");

                // 5. 複製 Output: Pinned Memory -> Result Bitmap (這裡只取 Mura 做示範)
                resultBmp = new Bitmap(w, h, PixelFormat.Format8bppIndexed);
                ColorPalette pal = resultBmp.Palette;
                for (int i = 0; i < 256; i++) pal.Entries[i] = Color.FromArgb(i, i, i);
                resultBmp.Palette = pal;

                BitmapData dstData = resultBmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
                unsafe
                {
                    // 這裡選擇拷貝 pMura (瑕疵圖)，如果要看背景改 pBg，看脊線改 pRidge
                    Buffer.MemoryCopy((void*)pMura, (void*)dstData.Scan0, size, size);
                }
                resultBmp.UnlockBits(dstData);
            }
            finally
            {
                // 6. 釋放 Pinned Memory (一定要做！)
                if (pIn != IntPtr.Zero) PICoater_FreePinned(pIn);
                if (pBg != IntPtr.Zero) PICoater_FreePinned(pBg);
                if (pMura != IntPtr.Zero) PICoater_FreePinned(pMura);
                if (pRidge != IntPtr.Zero) PICoater_FreePinned(pRidge);
            }

            return resultBmp;
        }
    }
}