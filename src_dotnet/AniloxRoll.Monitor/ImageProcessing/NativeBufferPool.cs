using System;
using AniloxRoll.Monitor.Core.Interop;

namespace AniloxRoll.Monitor.Core.Services
{
    public sealed class NativeBufferPool : IDisposable
    {
        private IntPtr _inputBuffer = IntPtr.Zero;
        private IntPtr _thumbnailBuffer = IntPtr.Zero;
        private IntPtr _muraBuffer = IntPtr.Zero;
        private IntPtr _ridgeBuffer = IntPtr.Zero;
        private IntPtr _curveMeanBuffer = IntPtr.Zero;
        private IntPtr _curveMaxBuffer = IntPtr.Zero;
        private IntPtr _curveRowMeanBuffer = IntPtr.Zero;
        private IntPtr _curveRowMaxBuffer = IntPtr.Zero;

        public IntPtr InputBuffer => _inputBuffer;
        public IntPtr ThumbnailBuffer => _thumbnailBuffer;
        public IntPtr MuraBuffer => _muraBuffer;
        public IntPtr RidgeBuffer => _ridgeBuffer;
        public IntPtr CurveMeanBuffer => _curveMeanBuffer;
        public IntPtr CurveMaxBuffer => _curveMaxBuffer;
        public IntPtr CurveRowMeanBuffer => _curveRowMeanBuffer;
        public IntPtr CurveRowMaxBuffer => _curveRowMaxBuffer;

        public ulong ImageBufferSize { get; }
        public int ThumbnailBufferSize { get; }
        public int CurveBufferSize { get; }
        public int CurveRowBufferSize { get; }

        private bool _isDisposed;

        public NativeBufferPool(int maxWidth, int maxHeight, int maxThumbnailSide)
        {
            ImageBufferSize = (ulong)(maxWidth * maxHeight);
            ThumbnailBufferSize = maxThumbnailSide * maxThumbnailSide;
            CurveBufferSize = maxWidth * sizeof(float);
            CurveRowBufferSize = maxHeight * sizeof(float);

            // 使用 CUDA Pinned Memory（cudaMallocHost），讓 H<->D memcpy 走非同步 DMA，
            // 對應 de24f715 版本的 PICoater_AllocPinned 設計。
            _inputBuffer        = AllocatePinned(ImageBufferSize);
            _muraBuffer         = AllocatePinned(ImageBufferSize);
            _ridgeBuffer        = AllocatePinned(ImageBufferSize);
            _thumbnailBuffer    = AllocatePinned((ulong)ThumbnailBufferSize);
            _curveMeanBuffer    = AllocatePinned((ulong)CurveBufferSize);
            _curveMaxBuffer     = AllocatePinned((ulong)CurveBufferSize);
            _curveRowMeanBuffer = AllocatePinned((ulong)CurveRowBufferSize);
            _curveRowMaxBuffer  = AllocatePinned((ulong)CurveRowBufferSize);
        }

        public void Dispose()
        {
            if (_isDisposed) return;
            _isDisposed = true; // 先設旗標，即使後續 Free 拋例外也不會重複釋放

            FreePinned(ref _inputBuffer);
            FreePinned(ref _thumbnailBuffer);
            FreePinned(ref _muraBuffer);
            FreePinned(ref _ridgeBuffer);
            FreePinned(ref _curveMeanBuffer);
            FreePinned(ref _curveMaxBuffer);
            FreePinned(ref _curveRowMeanBuffer);
            FreePinned(ref _curveRowMaxBuffer);
        }

        private static IntPtr AllocatePinned(ulong size)
        {
            IntPtr ptr = NativeMethods.CoreCV_AllocPinned(size);
            if (ptr == IntPtr.Zero)
            {
                throw new OutOfMemoryException($"CUDA pinned buffer allocation failed. Requested size={size}.");
            }

            return ptr;
        }

        private static void FreePinned(ref IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
            {
                return;
            }

            NativeMethods.CoreCV_FreePinned(ptr);
            ptr = IntPtr.Zero;
        }
    }
}
