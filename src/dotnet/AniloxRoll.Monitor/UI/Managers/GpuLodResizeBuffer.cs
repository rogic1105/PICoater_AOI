using System;
using System.Runtime.InteropServices;
using AniloxRoll.Monitor.Core.Interop; // NativeMethods（P/Invoke 唯一宣告點）

namespace AniloxRoll.Monitor.UI.Managers
{
    /// <summary>
    /// 動態 LOD 的 GPU 縮放 buffer 池 —— **資源擁有者**（CUDA pinned memory）：自己擁有 src/dst pinned ptr +
    /// 容量 + 鎖 + use-after-free 旗標。只裁可見小區、依需求成長重用（非每幀全幀）。
    ///
    /// 從 LiveCameraManager 提取（2026-06-26 重構）：把「GPU pinned buffer 生命週期 + resize + 安全釋放」
    /// 這個職責收成自己一個類別。背景 provider 呼叫 <see cref="Resize"/> 與 UI 釋放 <see cref="Release"/> 互斥
    /// （_lock + _released 旗標防 use-after-free：釋放後背景仍在跑也回 null）。CPU LOD 走 GrayResizeCpu 無 pinned，不經此。
    /// </summary>
    internal sealed class GpuLodResizeBuffer
    {
        private IntPtr _srcPinned, _dstPinned;
        private int _srcCap, _dstCap;
        private readonly object _lock = new object();
        private volatile bool _released = true;   // 預設 released；GPU LOD 啟用時 Arm() 才允許 Resize 配 pinned

        /// <summary>啟用 GPU LOD 時呼叫 → 解除 released 旗標（允許 Resize 重新配 pinned）。</summary>
        public void Arm() { lock (_lock) _released = false; }

        /// <summary>GPU 縮放（LiveDisplayView 背景執行緒呼叫；只縮可見區一塊）：src 灰階 → dst 灰階。
        /// 與 Release 互斥；已釋放 / pinned 配置失敗回 null。</summary>
        public byte[] Resize(byte[] src, int sw, int sh, int dw, int dh)
        {
            int srcPix = sw * sh, dstPix = dw * dh;
            byte[] dst;
            lock (_lock)
            {
                if (_released) return null;
                if (_srcCap < srcPix)
                {
                    if (_srcPinned != IntPtr.Zero) NativeMethods.TanukiCv_FreePinned(_srcPinned);
                    _srcPinned = NativeMethods.TanukiCv_AllocPinned((ulong)srcPix); _srcCap = srcPix;
                }
                if (_dstCap < dstPix)
                {
                    if (_dstPinned != IntPtr.Zero) NativeMethods.TanukiCv_FreePinned(_dstPinned);
                    _dstPinned = NativeMethods.TanukiCv_AllocPinned((ulong)dstPix); _dstCap = dstPix;
                }
                if (_srcPinned == IntPtr.Zero || _dstPinned == IntPtr.Zero) return null;
                Marshal.Copy(src, 0, _srcPinned, srcPix);
                NativeMethods.TanukiCv_Resize_GPU(_srcPinned, sw, sh, _dstPinned, dw, dh);
                dst = new byte[dstPix];
                Marshal.Copy(_dstPinned, dst, 0, dstPix);
            }
            return dst;
        }

        /// <summary>釋放 pinned（鎖內 + 設 released 旗標，等背景 Resize 用完防 use-after-free）。可重複呼叫（冪等）。</summary>
        public void Release()
        {
            lock (_lock)
            {
                _released = true;
                if (_srcPinned != IntPtr.Zero) { NativeMethods.TanukiCv_FreePinned(_srcPinned); _srcPinned = IntPtr.Zero; _srcCap = 0; }
                if (_dstPinned != IntPtr.Zero) { NativeMethods.TanukiCv_FreePinned(_dstPinned); _dstPinned = IntPtr.Zero; _dstCap = 0; }
            }
        }
    }
}
