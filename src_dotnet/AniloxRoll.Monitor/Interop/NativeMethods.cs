using System;
using System.Runtime.InteropServices;

namespace AniloxRoll.Monitor.Core.Interop
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct AoiImageNative
    {
        public int Width;
        public int Height;
        public IntPtr Data;
        public IntPtr BackgroundData;
        public IntPtr MuraData;
        public IntPtr RidgeData;
        public IntPtr MuraCurveMean;
        public IntPtr MuraCurveMax;
        public float BgSigmaFactor;
        public float RidgeSigma;
        public float HessianMaxFactor;
        public IntPtr RidgeMode;
        public IntPtr Stream;
    }

    internal static class NativeMethods
    {
        private const string DllName = "picoater_api.dll";

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr PICoaterAPI_CreatePipeline();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int PICoaterAPI_ProcessPipeline(
            IntPtr handle,
            int width,
            int height,
            IntPtr d_input,
            IntPtr d_background_output,
            IntPtr d_mura_output,
            IntPtr d_ridge_output,
            IntPtr d_mura_curve_mean_output,
            IntPtr d_mura_curve_max_output,
            float bg_sigma_factor,
            float ridge_sigma,
            float hessian_max_factor,
            [MarshalAs(UnmanagedType.LPStr)] string ridge_mode,
            IntPtr stream);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr PICoaterAPI_GetLastError(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void PICoaterAPI_DestroyPipeline(IntPtr handle);
    }
}
