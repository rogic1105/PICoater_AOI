using System;
using System.Runtime.InteropServices;

namespace AniloxRoll.Monitor.Core.Interop
{
    internal static class PICoaterNative
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

        // 用於執行檢測並回傳縮圖 (GPU 運算模式)
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int PICoater_Run_And_GetThumbnail(
            IntPtr handle,
            IntPtr inputHandle,
            IntPtr thumbOutHandle,
            int thumbWidth,
            int thumbHeight,
            float bgSigma,
            float ridgeSigma,
            [MarshalAs(UnmanagedType.LPStr)] string ridgeMode
        );

        // 快速讀取 BMP 到記憶體
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool PICoater_FastReadBMP(
            [MarshalAs(UnmanagedType.LPStr)] string filepath,
            out int width,
            out int height,
            IntPtr outBuffer,
            int bufferSize
        );

        // 純粹讀取縮圖 (IO 模式，不跑檢測)
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