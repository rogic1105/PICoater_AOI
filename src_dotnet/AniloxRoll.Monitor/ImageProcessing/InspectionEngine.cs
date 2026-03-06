using System;
using System.Diagnostics;
using System.IO;
using AniloxRoll.Monitor.Core.Data;
using AOI.SDK.Core.Models;

namespace AniloxRoll.Monitor.Core.Services
{
    public partial class InspectionEngine : IDisposable
    {
        private readonly object _lock = new object();
        private readonly AoiService _aoiService;
        private readonly NativeBufferPool _bufferPool;

        private IntPtr _inputBuffer => _bufferPool.InputBuffer;
        private IntPtr _thumbnailBuffer => _bufferPool.ThumbnailBuffer;
        private IntPtr _muraBuffer => _bufferPool.MuraBuffer;
        private IntPtr _ridgeBuffer => _bufferPool.RidgeBuffer;
        private IntPtr _curveMeanBuffer => _bufferPool.CurveMeanBuffer;
        private IntPtr _curveMaxBuffer => _bufferPool.CurveMaxBuffer;

        private ulong _imgBufferSize => _bufferPool.ImageBufferSize;
        private int _thumbnailBufferSize => _bufferPool.ThumbnailBufferSize;
        private int _curveBufferSize => _bufferPool.CurveBufferSize;

        private bool _isDisposed = false;

        public InspectionEngine()
        {
            _aoiService = new AoiService();
            _bufferPool = new NativeBufferPool(
                InspectionEngineConfig.MaxWidth,
                InspectionEngineConfig.MaxHeight,
                InspectionEngineConfig.MaxThumbnailSide);
            InitializeNativeResources();
        }

        private void InitializeNativeResources()
        {
            _aoiService.Initialize();
        }

        public void WarmUp()
        {
            if (_isDisposed) return;

            lock (_lock)
            {
                try
                {
                    _aoiService.ProcessImage(new AoiProcessRequest
                    {
                        Input = new AoiProcessRequest.InputImage
                        {
                            Width = 64,
                            Height = 64,
                            Data = _inputBuffer,
                            Stream = IntPtr.Zero
                        },
                        Output = new AoiProcessRequest.OutputBuffers
                        {
                            BackgroundData = IntPtr.Zero,
                            MuraData = _muraBuffer,
                            RidgeData = _ridgeBuffer,
                            MuraCurveMean = _curveMeanBuffer,
                            MuraCurveMax = _curveMaxBuffer,
                            Stream = IntPtr.Zero
                        },
                        Params = new AoiProcessRequest.AlgorithmParams
                        {
                            BgSigmaFactor = InspectionEngineConfig.DefaultBgSigma,
                            RidgeSigma = InspectionEngineConfig.DefaultRidgeSigma,
                            HessianMaxFactor = 1.0f,
                            RidgeMode = InspectionEngineConfig.DefaultRidgeMode
                        }
                    });
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
                _bufferPool.Dispose();
                _aoiService.Dispose();
                _isDisposed = true;
            }
        }
    }
}
