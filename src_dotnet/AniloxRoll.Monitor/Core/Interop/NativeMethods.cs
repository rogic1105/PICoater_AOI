using System;
using System.Runtime.InteropServices;

namespace AniloxRoll.Monitor.Core.Interop
{
    internal static class NativeMethods
    {
        private const string DllName = "picoater_api.dll";

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr PICoater_Create();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void PICoater_Destroy(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int PICoater_Initialize(IntPtr handle, int width, int height);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr PICoater_AllocPinned(ulong size);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void PICoater_FreePinned(IntPtr ptr);

        // [核心修改] 這是新的執行介面
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int PICoater_Run(
            IntPtr handle,
            IntPtr h_img_in,
            IntPtr h_bg_out,
            IntPtr h_mura_out,
            IntPtr h_ridge_out,
            IntPtr h_mura_curve_out,
            float bgSigma,
            float ridgeSigma,
            int heatmap_lower_thres,
            float heatmap_alpha,
            [MarshalAs(UnmanagedType.LPStr)] string rMode
        );

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool PICoater_FastReadBMP(
            [MarshalAs(UnmanagedType.LPStr)] string filepath,
            out int width,
            out int height,
            IntPtr outBuffer,
            ulong bufferSize // 這裡改用 ulong 對應 C++ size_t
        );

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int PICoater_LoadThumbnail(
            [MarshalAs(UnmanagedType.LPStr)] string path,
            int targetWidth,
            IntPtr outBuffer,
            out int outRealW,
            out int outRealH
        );
    }
}