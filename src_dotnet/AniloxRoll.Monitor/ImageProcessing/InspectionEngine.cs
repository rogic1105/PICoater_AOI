using System;
using System.Diagnostics;
using System.IO;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Interop;
using AOI.SDK.Core.Models;

namespace AniloxRoll.Monitor.Core.Services
{
    public partial class InspectionEngine : IDisposable
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

        private bool _isDisposed = false;

        public InspectionEngine() { InitializeNativeResources(); }

        private void InitializeNativeResources()
        {
            _handle = NativeMethods.PICoater_Create();

            _imgBufferSize = (ulong)(InspectionEngineConfig.MaxWidth * InspectionEngineConfig.MaxHeight);
            _inputBuffer = NativeMethods.PICoater_AllocPinned(_imgBufferSize);
            _muraBuffer = NativeMethods.PICoater_AllocPinned(_imgBufferSize);
            _ridgeBuffer = NativeMethods.PICoater_AllocPinned(_imgBufferSize);

            _thumbnailBufferSize = InspectionEngineConfig.MaxThumbnailSide * InspectionEngineConfig.MaxThumbnailSide;
            _thumbnailBuffer = NativeMethods.PICoater_AllocPinned((ulong)_thumbnailBufferSize);

            _curveBufferSize = InspectionEngineConfig.MaxWidth * sizeof(float);
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