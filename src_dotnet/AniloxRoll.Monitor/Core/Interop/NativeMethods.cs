using System;
using System.Runtime.InteropServices;

namespace AniloxRoll.Monitor.Core.Interop
{
    /// <summary>
    /// [Interop] 定義與 C++ (picoater_api.dll) 對接的 P/Invoke 介面。
    /// 包含記憶體配置 (AllocPinned)、演算法執行 (Run) 與資源釋放 (Destroy) 的原始入口點。
    /// </summary>
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

        // 用於執行檢測並回傳縮圖 (GPU 運算模式)
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int PICoater_Run_And_GetThumbnail(
            IntPtr handle,
            IntPtr h_in,             // 輸入大圖
            IntPtr h_thumb_out,      // 輸出縮圖
            IntPtr h_mura_curve_out, // [新增] 輸出曲線 (float*)
            int thumb_w,
            int thumb_h,
            float bgSigma,
            float ridgeSigma,
            [MarshalAs(UnmanagedType.LPStr)] string rMode
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