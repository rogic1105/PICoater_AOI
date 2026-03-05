using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using AniloxRoll.Monitor.Core.Data;
using AOI.SDK.Core.Models;

namespace AniloxRoll.Monitor.Core.Services
{
    public partial class InspectionEngine : IDisposable
    {
        private readonly object _lock = new object();
        private readonly AoiService _aoiService;

        private IntPtr _inputBuffer = IntPtr.Zero;
        private IntPtr _thumbnailBuffer = IntPtr.Zero;
        private IntPtr _muraBuffer = IntPtr.Zero;
        private IntPtr _ridgeBuffer = IntPtr.Zero;
        private IntPtr _curveMeanBuffer = IntPtr.Zero;
        private IntPtr _curveMaxBuffer = IntPtr.Zero;

        private ulong _imgBufferSize = 0;
        private int _thumbnailBufferSize = 0;
        private int _curveBufferSize = 0;

        private bool _isDisposed = false;

        public InspectionEngine()
        {
            _aoiService = new AoiService();
            InitializeNativeResources();
        }

        private void InitializeNativeResources()
        {
            _aoiService.Initialize();

            _imgBufferSize = (ulong)(InspectionEngineConfig.MaxWidth * InspectionEngineConfig.MaxHeight);
            _inputBuffer = Marshal.AllocHGlobal((IntPtr)_imgBufferSize);
            _muraBuffer = Marshal.AllocHGlobal((IntPtr)_imgBufferSize);
            _ridgeBuffer = Marshal.AllocHGlobal((IntPtr)_imgBufferSize);

            _thumbnailBufferSize = InspectionEngineConfig.MaxThumbnailSide * InspectionEngineConfig.MaxThumbnailSide;
            _thumbnailBuffer = Marshal.AllocHGlobal(_thumbnailBufferSize);

            _curveBufferSize = InspectionEngineConfig.MaxWidth * sizeof(float);
            _curveMeanBuffer = Marshal.AllocHGlobal(_curveBufferSize);
            _curveMaxBuffer = Marshal.AllocHGlobal(_curveBufferSize);
        }

        public void WarmUp()
        {
            if (_isDisposed) return;

            lock (_lock)
            {
                try
                {
                    _aoiService.ProcessImage(
                        64,
                        64,
                        _inputBuffer,
                        IntPtr.Zero,
                        _muraBuffer,
                        _ridgeBuffer,
                        _curveMeanBuffer,
                        _curveMaxBuffer,
                        InspectionEngineConfig.DefaultBgSigma,
                        InspectionEngineConfig.DefaultRidgeSigma,
                        1.0f,
                        InspectionEngineConfig.DefaultRidgeMode,
                        IntPtr.Zero);
                }
                catch
                {
                    // Ignore warm-up errors.
                }
            }
        }

        private TimedResult<T> ExecuteTimedOperation<T>(
                    string filePath,
                    Func<Stopwatch, (T data, long io, long gpu, long bmp)> operation)
        {
            lock (_lock)
            {
                var result = new TimedResult<T>();
                if (!File.Exists(filePath)) return result;

                try
                {
                    var sw = new Stopwatch();
                    var (data, io, gpu, bmp) = operation(sw);
                    result.Data = data;
                    result.IoDurationMs = io;
                    result.ComputeDurationMs = gpu;
                    result.BitmapDurationMs = bmp;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"InspectionEngine Error: {ex.Message}");
                }
                return result;
            }
        }

        public void Dispose()
        {
            if (_isDisposed)
            {
                return;
            }

            lock (_lock)
            {
                if (_inputBuffer != IntPtr.Zero) Marshal.FreeHGlobal(_inputBuffer);
                if (_thumbnailBuffer != IntPtr.Zero) Marshal.FreeHGlobal(_thumbnailBuffer);
                if (_muraBuffer != IntPtr.Zero) Marshal.FreeHGlobal(_muraBuffer);
                if (_ridgeBuffer != IntPtr.Zero) Marshal.FreeHGlobal(_ridgeBuffer);
                if (_curveMeanBuffer != IntPtr.Zero) Marshal.FreeHGlobal(_curveMeanBuffer);
                if (_curveMaxBuffer != IntPtr.Zero) Marshal.FreeHGlobal(_curveMaxBuffer);

                _inputBuffer = IntPtr.Zero;
                _thumbnailBuffer = IntPtr.Zero;
                _muraBuffer = IntPtr.Zero;
                _ridgeBuffer = IntPtr.Zero;
                _curveMeanBuffer = IntPtr.Zero;
                _curveMaxBuffer = IntPtr.Zero;

                _aoiService.Dispose();
                _isDisposed = true;
            }
        }
    }
}
