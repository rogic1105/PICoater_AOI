using System;
using System.Runtime.InteropServices;

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

        public IntPtr InputBuffer => _inputBuffer;
        public IntPtr ThumbnailBuffer => _thumbnailBuffer;
        public IntPtr MuraBuffer => _muraBuffer;
        public IntPtr RidgeBuffer => _ridgeBuffer;
        public IntPtr CurveMeanBuffer => _curveMeanBuffer;
        public IntPtr CurveMaxBuffer => _curveMaxBuffer;

        public ulong ImageBufferSize { get; }
        public int ThumbnailBufferSize { get; }
        public int CurveBufferSize { get; }

        private bool _isDisposed;

        public NativeBufferPool(int maxWidth, int maxHeight, int maxThumbnailSide)
        {
            ImageBufferSize = (ulong)(maxWidth * maxHeight);
            ThumbnailBufferSize = maxThumbnailSide * maxThumbnailSide;
            CurveBufferSize = maxWidth * sizeof(float);

            _inputBuffer = Allocate((IntPtr)ImageBufferSize);
            _muraBuffer = Allocate((IntPtr)ImageBufferSize);
            _ridgeBuffer = Allocate((IntPtr)ImageBufferSize);
            _thumbnailBuffer = Allocate((IntPtr)ThumbnailBufferSize);
            _curveMeanBuffer = Allocate((IntPtr)CurveBufferSize);
            _curveMaxBuffer = Allocate((IntPtr)CurveBufferSize);
        }

        public void Dispose()
        {
            if (_isDisposed)
            {
                return;
            }

            Free(ref _inputBuffer);
            Free(ref _thumbnailBuffer);
            Free(ref _muraBuffer);
            Free(ref _ridgeBuffer);
            Free(ref _curveMeanBuffer);
            Free(ref _curveMaxBuffer);

            _isDisposed = true;
        }

        private static IntPtr Allocate(IntPtr size)
        {
            IntPtr ptr = Marshal.AllocHGlobal(size);
            if (ptr == IntPtr.Zero)
            {
                throw new OutOfMemoryException($"Native buffer allocation failed. Requested size={size}.");
            }

            return ptr;
        }

        private static void Free(ref IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
            {
                return;
            }

            Marshal.FreeHGlobal(ptr);
            ptr = IntPtr.Zero;
        }
    }
}
