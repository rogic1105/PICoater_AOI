using System;
using System.Runtime.InteropServices;

namespace MilGrabber.Monitor
{
    /// <summary>
    /// 縮圖走 tanuki_cv_api.dll 既有的 GPU resize（與主程式 AniloxRoll.Monitor 同一份 native）。
    /// 主程式宣告在 Interop/NativeMethods.cs；本測試 sample 自帶最小宣告（throwaway 工具，不套主程式的「P/Invoke 只在 NativeMethods」規則）。
    /// 需要 GPU（檢測機跑 pipeline 故有）；tanuki_cv_api.dll 在共用 bin\x64\Release\。
    /// </summary>
    internal static class NativeResize
    {
        private const string Dll = "tanuki_cv_api.dll";

        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr TanukiCv_AllocPinned(ulong size);

        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TanukiCv_FreePinned(IntPtr ptr);

        // h_src / h_dst 為 8-bit 灰階 host buffer（pinned 走 DMA 加速）。輸出為 bottom-up（Y 反轉）。
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int TanukiCv_Resize_GPU(
            IntPtr hSrc, int srcW, int srcH,
            IntPtr hDst, int dstW, int dstH);
    }
}
