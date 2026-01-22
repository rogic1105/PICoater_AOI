using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Interop;
using AOI.SDK.Core.Models;
using AOI.SDK.Utils;

namespace AniloxRoll.Monitor.Core.Services
{
    public class InspectionEngine : IDisposable
    {
        // [新增] 執行緒鎖：防止多執行緒同時存取共用 Buffer
        private readonly object _lock = new object();

        private IntPtr _handle = IntPtr.Zero;

        // Pinned Buffers
        private IntPtr _inputBuffer = IntPtr.Zero;
        private IntPtr _thumbnailBuffer = IntPtr.Zero;
        private IntPtr _muraBuffer = IntPtr.Zero;
        private IntPtr _ridgeBuffer = IntPtr.Zero;
        private IntPtr _curveMeanBuffer = IntPtr.Zero;
        private IntPtr _curveMaxBuffer = IntPtr.Zero; // [修正] 確保這個有被 Alloc

        private ulong _imgBufferSize = 0;
        private int _thumbnailBufferSize = 0;
        private int _curveBufferSize = 0;

        private const int MaxWidth = 16384;
        private const int MaxHeight = 10000;
        private const int MaxThumbnailSide = 2000;

        // ... (常數保持不變) ...
        private const float DefaultBgSigma = 2.0f;
        private const float DefaultRidgeSigma = 9.0f;
        private const int DefaultHeatmapThres = 20;
        private const float DefaultHeatmapAlpha = 0.6f;
        private const string DefaultRidgeMode = "vertical";

        private bool _isDisposed = false;

        public InspectionEngine() { InitializeNativeResources(); }

        private void InitializeNativeResources()
        {
            _handle = NativeMethods.PICoater_Create();

            _imgBufferSize = (ulong)(MaxWidth * MaxHeight);
            _inputBuffer = NativeMethods.PICoater_AllocPinned(_imgBufferSize);
            _muraBuffer = NativeMethods.PICoater_AllocPinned(_imgBufferSize);
            _ridgeBuffer = NativeMethods.PICoater_AllocPinned(_imgBufferSize);

            _thumbnailBufferSize = MaxThumbnailSide * MaxThumbnailSide;
            _thumbnailBuffer = NativeMethods.PICoater_AllocPinned((ulong)_thumbnailBufferSize);

            _curveBufferSize = MaxWidth * sizeof(float);
            _curveMeanBuffer = NativeMethods.PICoater_AllocPinned((ulong)_curveBufferSize);

            // [新增] 補上 Max Buffer 的記憶體配置
            _curveMaxBuffer = NativeMethods.PICoater_AllocPinned((ulong)_curveBufferSize);
        }

        public void WarmUp()
        {
            if (_isDisposed) return;
            // WarmUp 也要鎖
            lock (_lock)
            {
                try
                {
                    int w = 14288;
                    int h = 9003;
                    NativeMethods.PICoater_Initialize(_handle, w, h);
                    NativeMethods.PICoater_RunThumbnail_GPU(_handle, _inputBuffer, 1000, _thumbnailBuffer, out _, out _);
                }
                catch { /* ignore */ }
            }
        }

        private TimedResult<T> ExecuteTimedOperation<T>(
                    string filePath,
                    Func<Stopwatch, (T data, long io, long gpu, long bmp)> operation)
        {
            // [關鍵] 加入 lock：確保同一時間只有一個執行緒能使用 Engine 和 Buffers
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

        public TimedResult<InspectionData> ProcessImage(string filePath, int targetThumbWidth)
        {
            if (_isDisposed) throw new ObjectDisposedException(nameof(InspectionEngine));

            // ExecuteTimedOperation 已經有 lock 了，這裡不用再加
            return ExecuteTimedOperation<InspectionData>(filePath, (stopwatch) =>
            {
                stopwatch.Start();
                bool readSuccess = NativeMethods.PICoater_FastReadBMP(
                    filePath, out int w, out int h, _inputBuffer, _imgBufferSize);
                stopwatch.Stop();
                long ioTime = stopwatch.ElapsedMilliseconds;

                if (!readSuccess) return (null, ioTime, 0, 0);

                stopwatch.Restart();
                NativeMethods.PICoater_Initialize(_handle, w, h);

                int thumbH = (int)((float)h / w * targetThumbWidth);

                int ret = NativeMethods.PICoater_Run_WithThumb(
                    _handle, _inputBuffer, _thumbnailBuffer, targetThumbWidth, thumbH,
                    _curveMeanBuffer, _curveMaxBuffer, // 這裡現在安全了
                    DefaultBgSigma, DefaultRidgeSigma, DefaultHeatmapThres, DefaultHeatmapAlpha, DefaultRidgeMode
                );

                stopwatch.Stop();
                long algoTime = stopwatch.ElapsedMilliseconds;
                if (ret != 0) throw new Exception($"Algo Error: {ret}");

                stopwatch.Restart();

                var thumb = ImageUtils.Create8bppBitmap(_thumbnailBuffer, targetThumbWidth, thumbH);

                float[] curveData = new float[w];
                Marshal.Copy(_curveMeanBuffer, curveData, 0, w);

                var data = new InspectionData { Image = thumb, MuraCurve = curveData };

                stopwatch.Stop();
                long bmpTime = stopwatch.ElapsedMilliseconds;

                return (data, ioTime, algoTime, bmpTime);
            });
        }

        public TimedResult<InspectionData> LoadThumbnailOnly(string filePath, int targetThumbWidth)
        {
            return ExecuteTimedOperation<InspectionData>(filePath, (stopwatch) =>
            {
                stopwatch.Start();
                bool readSuccess = NativeMethods.PICoater_FastReadBMP(
                    filePath, out int w, out int h, _inputBuffer, _imgBufferSize);
                stopwatch.Stop();
                long ioTime = stopwatch.ElapsedMilliseconds;

                if (!readSuccess) return (null, ioTime, 0, 0);

                stopwatch.Restart();
                NativeMethods.PICoater_Initialize(_handle, w, h);

                int ret = NativeMethods.PICoater_RunThumbnail_GPU(
                    _handle, _inputBuffer, targetThumbWidth, _thumbnailBuffer,
                    out int realW, out int realH
                );

                stopwatch.Stop();
                long gpuTime = stopwatch.ElapsedMilliseconds;

                if (ret != 0) throw new Exception($"GPU Resize Error: {ret}");

                stopwatch.Restart();

                var bitmap = ImageUtils.Create8bppBitmap(_thumbnailBuffer, realW, realH);

                stopwatch.Stop();
                long bmpTime = stopwatch.ElapsedMilliseconds;

                var data = new InspectionData { Image = bitmap, MuraCurve = null };
                return (data, ioTime, gpuTime, bmpTime);
            });
        }

        public Bitmap RunInspectionFullRes(string filePath, bool isProcessedMode)
        {
            if (_isDisposed) return null;
            if (!File.Exists(filePath)) return null;

            // [關鍵] 這裡沒有呼叫 ExecuteTimedOperation，所以必須手動加 lock
            lock (_lock)
            {
                bool readSuccess = NativeMethods.PICoater_FastReadBMP(
                     filePath, out int w, out int h, _inputBuffer, _imgBufferSize);
                if (!readSuccess) return null;

                if (isProcessedMode)
                {
                    NativeMethods.PICoater_Initialize(_handle, w, h);
                    int ret = NativeMethods.PICoater_Run(
                        _handle, _inputBuffer, IntPtr.Zero, IntPtr.Zero, _ridgeBuffer, IntPtr.Zero, IntPtr.Zero,
                        DefaultBgSigma, DefaultRidgeSigma, DefaultHeatmapThres, DefaultHeatmapAlpha, DefaultRidgeMode
                    );
                    if (ret != 0) return null;

                    return ImageUtils.Create8bppBitmap(_ridgeBuffer, w, h, flipY: false);
                }
                else
                {
                    return ImageUtils.Create8bppBitmap(_inputBuffer, w, h, flipY: false);
                }
            }
        }

        public void Dispose()
        {
            if (!_isDisposed)
            {
                lock (_lock) // Dispose 時也要鎖
                {
                    if (_inputBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_inputBuffer);
                    if (_thumbnailBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_thumbnailBuffer);
                    if (_muraBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_muraBuffer);
                    if (_ridgeBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_ridgeBuffer);
                    if (_curveMeanBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_curveMeanBuffer);

                    // [新增] 釋放 Max Buffer
                    if (_curveMaxBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_curveMaxBuffer);

                    if (_handle != IntPtr.Zero) NativeMethods.PICoater_Destroy(_handle);
                    _isDisposed = true;
                }
            }
        }
    }
}