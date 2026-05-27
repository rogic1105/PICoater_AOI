// AOI_SDK\src_dotnet\AOI.SDK\Core\CoreCVWrapper.cs

using System;
using System.Runtime.InteropServices;

namespace AOI.SDK.Core
{
    /// <summary>
    /// 封裝 core_cv_api.dll 的原生函式庫
    /// </summary>
    public static class CoreCVWrapper
    {
        private const string DLL_NAME = "core_cv_api.dll";

        // =========================================================
        // 0. GPU 暖身 (Warm-Up)
        // =========================================================
        // 強迫 CUDA context / driver 提早初始化。暖身的底層細節
        // (malloc + kernel + free) 全封裝在 native，C# 只呼叫一次。
        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_WarmUp();

        // =========================================================
        // 1. 記憶體管理 (Pinned Memory) - 用於極速 IO
        // =========================================================
        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr CoreCV_AllocPinned(ulong size);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void CoreCV_FreePinned(IntPtr ptr);

        // =========================================================
        // 2. 極速 IO (Fast IO) - 繞過 GDI+ 直接讀寫 BMP
        // =========================================================
        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)] // C++ bool 轉 C# bool
        public static extern bool CoreCV_FastReadBMP(string filepath, out int w, out int h, IntPtr outBuffer, int bufferSize);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool CoreCV_FastWriteBMP(string filepath, int w, int h, IntPtr inBuffer);

        // =========================================================
        // 3. 影像處理演算法
        // =========================================================

        // 亮度調整
        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_Brighten(IntPtr src, int width, int height, int value, IntPtr dst);

        // 二值化
        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_Threshold(IntPtr src, int width, int height, byte threshold, IntPtr dst);

        // 反轉
        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_Invert(IntPtr src, int width, int height, IntPtr dst);

        // 卷積 (預留)
        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_Convolution(IntPtr src, int width, int height, IntPtr mask, int maskSize, IntPtr dst);

        // =========================================================
        // 4. 進階 GPU 操作 (GPU Resident Mode)
        // =========================================================

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_MallocGPU(out IntPtr d_ptr, int width, int height);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_FreeGPU(IntPtr d_ptr);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_Upload(IntPtr h_src, IntPtr d_dst, int width, int height);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_Download(IntPtr d_src, IntPtr h_dst, int width, int height);

        // 純 GPU 版本
        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_Brighten_GPU(IntPtr d_src, int width, int height, int value, IntPtr d_dst);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_Threshold_GPU(IntPtr d_src, int width, int height, byte threshold, IntPtr d_dst);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_Invert_GPU(IntPtr d_src, int width, int height, IntPtr d_dst);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_Convolution_GPU(IntPtr d_src, int width, int height, IntPtr d_mask, int maskSize, IntPtr d_dst);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_MallocGPU_Float(out IntPtr d_ptr, int count);

        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_FreeGPU_Float(IntPtr d_ptr);

        // h_src 傳 float[] 即可，.NET 會自動 Marshal
        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_Upload_Float(float[] h_src, IntPtr d_dst, int count);

        // =========================================================
        // 5. GPU 縮圖 (Thumbnail)
        // =========================================================
        // h_src / h_dst 若為 CoreCV_AllocPinned 分配，H<->D 傳輸自動走 DMA 加速。
        [DllImport(DLL_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_Resize_GPU(
            IntPtr hSrc, int srcW, int srcH,
            IntPtr hDst, int dstW, int dstH);

    }
}