using System;
using System.Runtime.InteropServices;

namespace TanukiCv.Core
{
    /// <summary>
    /// Reusable GPU gray-image resize provider for LOD paths.
    /// Owns CUDA pinned host buffers, grows them on demand, and serializes resize/release
    /// so background LOD work cannot use freed buffers.
    /// </summary>
    public sealed class GpuGrayResizeProvider : IDisposable
    {
        private readonly Func<ulong, IntPtr> _allocPinned;
        private readonly Action<IntPtr> _freePinned;
        private readonly Func<IntPtr, int, int, IntPtr, int, int, int> _resizeGpu;
        private readonly object _lock = new object();

        private IntPtr _srcPinned;
        private IntPtr _dstPinned;
        private int _srcCap;
        private int _dstCap;
        private volatile bool _released = true;

        /// <summary>
        /// Creates a provider with explicit native delegates. Use this when the application
        /// owns its P/Invoke declarations and wants this class to own only the buffer lifecycle.
        /// </summary>
        public GpuGrayResizeProvider(
            Func<ulong, IntPtr> allocPinned,
            Action<IntPtr> freePinned,
            Func<IntPtr, int, int, IntPtr, int, int, int> resizeGpu)
        {
            _allocPinned = allocPinned ?? throw new ArgumentNullException(nameof(allocPinned));
            _freePinned = freePinned ?? throw new ArgumentNullException(nameof(freePinned));
            _resizeGpu = resizeGpu ?? throw new ArgumentNullException(nameof(resizeGpu));
        }

        /// <summary>
        /// Creates a provider backed by <see cref="TanukiCvWrapper"/> and tanuki_cv_api.dll.
        /// </summary>
        public static GpuGrayResizeProvider CreateTanukiCv()
        {
            return new GpuGrayResizeProvider(
                TanukiCvWrapper.TanukiCv_AllocPinned,
                TanukiCvWrapper.TanukiCv_FreePinned,
                TanukiCvWrapper.TanukiCv_Resize_GPU);
        }

        /// <summary>
        /// Allows subsequent <see cref="Resize"/> calls to allocate/reuse pinned buffers.
        /// Call when GPU LOD is enabled or re-enabled after <see cref="Release"/>.
        /// </summary>
        public void Arm()
        {
            lock (_lock) _released = false;
        }

        /// <summary>
        /// Resizes an 8-bit gray buffer. Returns null when released, allocation fails, or dimensions are invalid.
        /// Signature matches TanukiCv.Controls.GrayResize and can be passed to LiveDisplayView.EnableLod.
        /// </summary>
        public byte[] Resize(byte[] src, int srcW, int srcH, int dstW, int dstH)
        {
            if (src == null || srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0)
                return null;

            int srcPix = srcW * srcH;
            int dstPix = dstW * dstH;
            if (srcPix <= 0 || dstPix <= 0 || src.Length < srcPix) return null;

            lock (_lock)
            {
                if (_released) return null;

                if (!EnsureCapacity(ref _srcPinned, ref _srcCap, srcPix)) return null;
                if (!EnsureCapacity(ref _dstPinned, ref _dstCap, dstPix)) return null;

                Marshal.Copy(src, 0, _srcPinned, srcPix);
                int ret = _resizeGpu(_srcPinned, srcW, srcH, _dstPinned, dstW, dstH);
                if (ret != 0) return null;

                var dst = new byte[dstPix];
                Marshal.Copy(_dstPinned, dst, 0, dstPix);
                return dst;
            }
        }

        /// <summary>
        /// Releases pinned buffers and makes future <see cref="Resize"/> calls return null until <see cref="Arm"/> is called.
        /// Safe to call repeatedly.
        /// </summary>
        public void Release()
        {
            lock (_lock)
            {
                _released = true;
                Free(ref _srcPinned, ref _srcCap);
                Free(ref _dstPinned, ref _dstCap);
            }
        }

        public void Dispose()
        {
            Release();
        }

        private bool EnsureCapacity(ref IntPtr ptr, ref int cap, int needed)
        {
            if (cap >= needed && ptr != IntPtr.Zero) return true;
            Free(ref ptr, ref cap);
            ptr = _allocPinned((ulong)needed);
            cap = ptr == IntPtr.Zero ? 0 : needed;
            return ptr != IntPtr.Zero;
        }

        private void Free(ref IntPtr ptr, ref int cap)
        {
            if (ptr != IntPtr.Zero)
            {
                _freePinned(ptr);
                ptr = IntPtr.Zero;
            }
            cap = 0;
        }
    }
}
