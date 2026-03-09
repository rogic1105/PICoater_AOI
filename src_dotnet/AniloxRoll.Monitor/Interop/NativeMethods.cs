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
        public IntPtr Stream;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct AoiAlgorithmParamsNative
    {
        public float BgSigmaFactor;
        public float RidgeSigma;
        public float HessianMaxFactor;
        public IntPtr RidgeMode;
    }

    internal static class NativeMethods
    {
        private const string DllName = "picoater_api.dll";
        private const string CoreCVDllName = "core_cv_api.dll";

        // =====================================================
        // core_cv_api.dll — Pinned Memory (CUDA cudaMallocHost)
        // =====================================================
        [DllImport(CoreCVDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr CoreCV_AllocPinned(ulong size);

        [DllImport(CoreCVDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void CoreCV_FreePinned(IntPtr ptr);

        // =====================================================
        // core_cv_api.dll — Fast IO (繞過 GDI+ 直讀 8-bit BMP)
        // =====================================================
        [DllImport(CoreCVDllName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool CoreCV_FastReadBMP(
            [MarshalAs(UnmanagedType.LPStr)] string filePath,
            out int width, out int height,
            IntPtr outBuffer, int bufferSize);

        // =====================================================
        // core_cv_api.dll — GPU Thumbnail Resize
        // h_src / h_dst 若為 Pinned Memory，H<->D 走 DMA 加速。
        // =====================================================
        [DllImport(CoreCVDllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int CoreCV_Resize_GPU(
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
        public static extern IntPtr PICoaterAPI_CreateMockPlc();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void PICoaterAPI_DestroyPlc(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int PICoaterAPI_PlcConnect(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int PICoaterAPI_PlcReadBit(
            IntPtr handle,
            int address,
            [MarshalAs(UnmanagedType.I1)] out bool value);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int PICoaterAPI_PlcWriteBit(
            IntPtr handle,
            int address,
            [MarshalAs(UnmanagedType.I1)] bool value);
    }
}
