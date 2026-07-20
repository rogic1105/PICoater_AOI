using System;
using AniloxRoll.Monitor.Core.Interop;

namespace AniloxRoll.Monitor.Core.Services
{
    public sealed class NativeBufferPool : IDisposable
    {
        private const ulong Alignment = 64;

        private IntPtr _slabBuffer = IntPtr.Zero;
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
        public ulong PinnedBytes { get; }

        private readonly Action<IntPtr> _freePinned;
        private bool _isDisposed;

        public NativeBufferPool(int maxWidth, int maxHeight, int maxThumbnailSide)
            : this(maxWidth, maxHeight, maxThumbnailSide, AllocatePinned, FreePinned)
        {
        }

        internal NativeBufferPool(
            int maxWidth,
            int maxHeight,
            int maxThumbnailSide,
            Func<ulong, IntPtr> allocatePinned,
            Action<IntPtr> freePinned)
        {
            if (maxWidth <= 0) throw new ArgumentOutOfRangeException(nameof(maxWidth));
            if (maxHeight <= 0) throw new ArgumentOutOfRangeException(nameof(maxHeight));
            if (maxThumbnailSide < 0) throw new ArgumentOutOfRangeException(nameof(maxThumbnailSide));
            if (allocatePinned == null) throw new ArgumentNullException(nameof(allocatePinned));
            if (freePinned == null) throw new ArgumentNullException(nameof(freePinned));

            ImageBufferSize = checked((ulong)maxWidth * (ulong)maxHeight);
            ThumbnailBufferSize = checked(maxThumbnailSide * maxThumbnailSide);
            CurveBufferSize = checked(maxWidth * sizeof(float));
            CurveRowBufferSize = checked(maxHeight * sizeof(float));
            _freePinned = freePinned;

            ulong total = 0;
            ulong inputOffset = Reserve(ref total, ImageBufferSize);
            ulong muraOffset = Reserve(ref total, ImageBufferSize);
            ulong ridgeOffset = Reserve(ref total, ImageBufferSize);
            ulong thumbnailOffset = Reserve(ref total, (ulong)ThumbnailBufferSize);
            ulong curveMeanOffset = Reserve(ref total, (ulong)CurveBufferSize);
            ulong curveMaxOffset = Reserve(ref total, (ulong)CurveBufferSize);
            ulong curveRowMeanOffset = Reserve(ref total, (ulong)CurveRowBufferSize);
            ulong curveRowMaxOffset = Reserve(ref total, (ulong)CurveRowBufferSize);
            PinnedBytes = total;

            // One cudaMallocHost call per pool avoids repeating CUDA context/OS page-lock overhead.
            _slabBuffer = allocatePinned(PinnedBytes);
            if (_slabBuffer == IntPtr.Zero)
                throw new OutOfMemoryException(
                    $"CUDA pinned slab allocation failed. Requested size={PinnedBytes}.");

            _inputBuffer = Add(_slabBuffer, inputOffset);
            _muraBuffer = Add(_slabBuffer, muraOffset);
            _ridgeBuffer = Add(_slabBuffer, ridgeOffset);
            _thumbnailBuffer = Add(_slabBuffer, thumbnailOffset);
            _curveMeanBuffer = Add(_slabBuffer, curveMeanOffset);
            _curveMaxBuffer = Add(_slabBuffer, curveMaxOffset);
            _curveRowMeanBuffer = Add(_slabBuffer, curveRowMeanOffset);
            _curveRowMaxBuffer = Add(_slabBuffer, curveRowMaxOffset);
        }

        public void Dispose()
        {
            if (_isDisposed) return;
            _isDisposed = true; // 先設旗標，即使後續 Free 拋例外也不會重複釋放

            if (_slabBuffer != IntPtr.Zero)
            {
                _freePinned(_slabBuffer);
                _slabBuffer = IntPtr.Zero;
            }

            _inputBuffer = IntPtr.Zero;
            _thumbnailBuffer = IntPtr.Zero;
            _muraBuffer = IntPtr.Zero;
            _ridgeBuffer = IntPtr.Zero;
            _curveMeanBuffer = IntPtr.Zero;
            _curveMaxBuffer = IntPtr.Zero;
            _curveRowMeanBuffer = IntPtr.Zero;
            _curveRowMaxBuffer = IntPtr.Zero;
        }

        private static IntPtr AllocatePinned(ulong size)
        {
            IntPtr ptr = NativeMethods.TanukiCv_AllocPinned(size);
            if (ptr == IntPtr.Zero)
            {
                throw new OutOfMemoryException($"CUDA pinned buffer allocation failed. Requested size={size}.");
            }

            return ptr;
        }

        private static void FreePinned(IntPtr ptr)
        {
            if (ptr != IntPtr.Zero)
                NativeMethods.TanukiCv_FreePinned(ptr);
        }

        private static ulong Reserve(ref ulong total, ulong size)
        {
            total = Align(total);
            ulong offset = total;
            total = checked(total + size);
            return offset;
        }

        private static ulong Align(ulong value)
        {
            return checked((value + Alignment - 1) & ~(Alignment - 1));
        }

        private static IntPtr Add(IntPtr basePointer, ulong offset)
        {
            return new IntPtr(checked(basePointer.ToInt64() + (long)offset));
        }
    }
}
