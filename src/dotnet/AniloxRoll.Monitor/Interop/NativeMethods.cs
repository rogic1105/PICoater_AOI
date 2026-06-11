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

    [StructLayout(LayoutKind.Sequential)]
    internal struct AoiAlgorithmParamsNative
    {
        public float BgSigmaFactor;
        public float RidgeSigma;
        public float HessianMaxFactor;
        public IntPtr RidgeMode;
        public IntPtr PrecomputedColMean;  // host float*, size = width. IntPtr.Zero = per-frame mode.
    }

    internal static class NativeMethods
    {
        private const string DllName = "picoater_api.dll";
        private const string CoreCVDllName = "tanuki_cv_api.dll";

        // =====================================================
        // tanuki_cv_api.dll — Pinned Memory (CUDA cudaMallocHost)
        // =====================================================
        [DllImport(CoreCVDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr TanukiCv_AllocPinned(ulong size);

        [DllImport(CoreCVDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TanukiCv_FreePinned(IntPtr ptr);

        // =====================================================
        // tanuki_cv_api.dll — Fast IO (繞過 GDI+ 直讀 8-bit BMP)
        // =====================================================
        [DllImport(CoreCVDllName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool TanukiCv_FastReadBMP(
            [MarshalAs(UnmanagedType.LPStr)] string filePath,
            out int width, out int height,
            IntPtr outBuffer, int bufferSize);

        // =====================================================
        // tanuki_cv_api.dll — GPU Thumbnail Resize
        // h_src / h_dst 若為 Pinned Memory，H<->D 走 DMA 加速。
        // =====================================================
        [DllImport(CoreCVDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int TanukiCv_Resize_GPU(
            IntPtr hSrc, int srcW, int srcH,
            IntPtr hDst, int dstW, int dstH);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr PICoaterAPI_CreatePipeline();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int PICoaterAPI_ProcessPipeline(
            IntPtr handle,
            ref AoiInputImageNative input,
            ref AoiAlgorithmParamsNative parameters,
            ref AoiOutputBuffersNative output);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr PICoaterAPI_GetLastError(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void PICoaterAPI_DestroyPipeline(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int PICoaterAPI_ComputeColumnMean(
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
