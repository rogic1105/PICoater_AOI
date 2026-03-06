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
