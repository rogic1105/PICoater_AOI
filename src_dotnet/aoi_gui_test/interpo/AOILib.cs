// AOILib.cs

using System;
using System.Runtime.InteropServices;

namespace AOIUI.Interop
{
    public static class AOILib
    {
        [DllImport("ExportCDLL.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int BuildFullMaskFromFFT_C(
        byte[] img_gray, int H, int W,
        float fft_th, int bw_th, int border_t,
        byte[] full_mask_out);



    }
}
