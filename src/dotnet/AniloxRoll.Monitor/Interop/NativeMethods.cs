using System;
using System.Runtime.InteropServices;

namespace AniloxRoll.Monitor.Core.Interop
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct AoiInputImageNative
    {
        public int Width;
        public int Height;
        public IntPtr Data;
        public IntPtr Stream;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct AoiOutputBuffersNative
    {
        public int Width;
        public int Height;
        public IntPtr BackgroundData;
        public IntPtr MuraData;
        public IntPtr RidgeData;
        public IntPtr MuraCurveMean;
        public IntPtr MuraCurveMax;
        public IntPtr MuraRowCurveMean;
        public IntPtr MuraRowCurveMax;
        public IntPtr Stream;
    }

    internal static class NativeMethods
    {
        private const string DllName = "tanuki_pipeline_api.dll";
        private const string TanukiCvDllName = "tanuki_cv_api.dll";

        // =====================================================
        // tanuki_cv_api.dll — Pinned Memory (CUDA cudaMallocHost)
        // =====================================================
        [DllImport(TanukiCvDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr TanukiCv_AllocPinned(ulong size);

        [DllImport(TanukiCvDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TanukiCv_FreePinned(IntPtr ptr);

        // =====================================================
        // tanuki_cv_api.dll — Fast IO (繞過 GDI+ 直讀 8-bit BMP)
        // =====================================================
        [DllImport(TanukiCvDllName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool TanukiCv_FastReadBMP(
            [MarshalAs(UnmanagedType.LPStr)] string filePath,
            out int width, out int height,
            IntPtr outBuffer, int bufferSize);

        // =====================================================
        // tanuki_cv_api.dll — GPU Thumbnail Resize
        // h_src / h_dst 若為 Pinned Memory，H<->D 走 DMA 加速。
        // =====================================================
        [DllImport(TanukiCvDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int TanukiCv_Resize_GPU(
            IntPtr hSrc, int srcW, int srcH,
            IntPtr hDst, int dstW, int dstH);

        // =====================================================
        // tanuki_pipeline_api.dll — 檢測 pipeline（4b 定版：run(name, json)）
        // 演算法參數走 json 字串（純 ASCII，LPStr 安全）；指標類走 struct / 獨立引數。
        // =====================================================
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr TanukiPipeline_Create(
            [MarshalAs(UnmanagedType.LPStr)] string pipelineName,
            [MarshalAs(UnmanagedType.LPStr)] string jsonOptions);  // 可 null

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int TanukiPipeline_Process(
            IntPtr handle,
            ref AoiInputImageNative input,
            [MarshalAs(UnmanagedType.LPStr)] string jsonParams,
            IntPtr precomputedColMean,   // host float*, size = width. IntPtr.Zero = 每幀自算
            ref AoiOutputBuffersNative output);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr TanukiPipeline_GetLastError(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TanukiPipeline_Destroy(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int TanukiPipeline_ComputeColumnMean(
            IntPtr handle,
            ref AoiInputImageNative input,
            float bgSigmaFactor,
            IntPtr outColMean);  // host float*, size = width

        // =====================================================
        // user32.dll — 捲軸位置讀寫（PropertyGrid 保持捲軸用）
        // =====================================================
        [DllImport("user32.dll")]
        public static extern int GetScrollPos(IntPtr hWnd, int nBar);

        [DllImport("user32.dll", CharSet = CharSet.Auto)]
        public static extern IntPtr SendMessage(IntPtr hWnd, int msg, IntPtr wParam, IntPtr lParam);
    }
}
