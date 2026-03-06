using System;
using System.Runtime.InteropServices;

namespace AniloxRoll.Monitor.Core.Services
{
    public sealed class NativeBufferPool : IDisposable
    {
        public IntPtr InputBuffer { get; private set; } = IntPtr.Zero;
        public IntPtr ThumbnailBuffer { get; private set; } = IntPtr.Zero;
        public IntPtr MuraBuffer { get; private set; } = IntPtr.Zero;
        public IntPtr RidgeBuffer { get; private set; } = IntPtr.Zero;
        public IntPtr CurveMeanBuffer { get; private set; } = IntPtr.Zero;
        public IntPtr CurveMaxBuffer { get; private set; } = IntPtr.Zero;

        public ulong ImageBufferSize { get; }
        public int ThumbnailBufferSize { get; }
        public int CurveBufferSize { get; }

        private bool _isDisposed;

        public NativeBufferPool(int maxWidth, int maxHeight, int maxThumbnailSide)
        {
            ImageBufferSize = (ulong)(maxWidth * maxHeight);
            ThumbnailBufferSize = maxThumbnailSide * maxThumbnailSide;
            CurveBufferSize = maxWidth * sizeof(float);

            InputBuffer = Allocate((IntPtr)ImageBufferSize);
            MuraBuffer = Allocate((IntPtr)ImageBufferSize);
            RidgeBuffer = Allocate((IntPtr)ImageBufferSize);
            ThumbnailBuffer = Allocate(ThumbnailBufferSize);
            CurveMeanBuffer = Allocate(CurveBufferSize);
            CurveMaxBuffer = Allocate(CurveBufferSize);
        }

        public void Dispose()
        {
            if (_isDisposed)
            {
                return;
            }

            Free(ref InputBuffer);
            Free(ref ThumbnailBuffer);
            Free(ref MuraBuffer);
            Free(ref RidgeBuffer);
            Free(ref CurveMeanBuffer);
            Free(ref CurveMaxBuffer);

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
