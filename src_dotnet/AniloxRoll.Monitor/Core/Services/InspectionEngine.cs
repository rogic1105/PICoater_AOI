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
        private IntPtr _handle = IntPtr.Zero;

        // Pinned Buffers (這些是給演算法用的，永遠保持 Raw 狀態，不要去翻轉它們)
        private IntPtr _inputBuffer = IntPtr.Zero;
        private IntPtr _thumbnailBuffer = IntPtr.Zero;
        private IntPtr _muraBuffer = IntPtr.Zero;
        private IntPtr _ridgeBuffer = IntPtr.Zero;
        private IntPtr _curveBuffer = IntPtr.Zero;

        private ulong _imgBufferSize = 0;
        private int _thumbnailBufferSize = 0;
        private int _curveBufferSize = 0;

        private const int MaxWidth = 16384;
        private const int MaxHeight = 10000;
        private const int MaxThumbnailSide = 2000;

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
            _curveBuffer = NativeMethods.PICoater_AllocPinned((ulong)_curveBufferSize);
        }

        public void WarmUp()
        {
            if (_isDisposed) return;
            try
            {
                int w = 14288;
                int h = 9003;
                NativeMethods.PICoater_Initialize(_handle, w, h);
                NativeMethods.PICoater_RunThumbnail_GPU(_handle, _inputBuffer, 1000, _thumbnailBuffer, out _, out _);
            }
            catch { /* ignore */ }
        }

        private TimedResult<T> ExecuteTimedOperation<T>(
                    string filePath,
                    Func<Stopwatch, (T data, long io, long gpu, long bmp)> operation)
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

        // ----------------------------------------------------------------------
        // 1. [Processed Mode] 
        // ----------------------------------------------------------------------
        public TimedResult<InspectionData> ProcessImage(string filePath, int targetThumbWidth)
        {
            if (_isDisposed) throw new ObjectDisposedException(nameof(InspectionEngine));

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
                    _curveBuffer, DefaultBgSigma, DefaultRidgeSigma, DefaultHeatmapThres, DefaultHeatmapAlpha, DefaultRidgeMode
                );

                stopwatch.Stop();
                long algoTime = stopwatch.ElapsedMilliseconds;
                if (ret != 0) throw new Exception($"Algo Error: {ret}");

                stopwatch.Restart();

                // [Process Mode] 直接拷貝 (保持 Raw 的顛倒狀態)
                var thumb = ImageUtils.Create8bppBitmap(_thumbnailBuffer, targetThumbWidth, thumbH);

                float[] curveData = new float[w];
                Marshal.Copy(_curveBuffer, curveData, 0, w);

                var data = new InspectionData { Image = thumb, MuraCurve = curveData };

                stopwatch.Stop();
                long bmpTime = stopwatch.ElapsedMilliseconds;

                return (data, ioTime, algoTime, bmpTime);
            });
        }

        // ----------------------------------------------------------------------
        // 2. [Original Mode]
        // ----------------------------------------------------------------------
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

                // [Original Mode] 為了讓人眼看著正常，必須手動翻轉 Bitmap (CreateFlippedBitmap)
                // 這只會改變顯示用的 Bitmap，不會影響 _inputBuffer
                var bitmap = ImageUtils.Create8bppBitmap(_thumbnailBuffer, realW, realH);

                stopwatch.Stop();
                long bmpTime = stopwatch.ElapsedMilliseconds;

                var data = new InspectionData { Image = bitmap, MuraCurve = null };
                return (data, ioTime, gpuTime, bmpTime);
            });
        }

        // ----------------------------------------------------------------------
        // 3. [Full Res Mode] 大圖檢視
        // ----------------------------------------------------------------------
        public Bitmap RunInspectionFullRes(string filePath, bool isProcessedMode)
        {
            if (_isDisposed) return null;
            if (!File.Exists(filePath)) return null;

            bool readSuccess = NativeMethods.PICoater_FastReadBMP(
                 filePath, out int w, out int h, _inputBuffer, _imgBufferSize);
            if (!readSuccess) return null;

            if (isProcessedMode)
            {
                // [Processed Mode] 演算法結果 -> 顯示顛倒 (Raw) -> 直接拷貝
                NativeMethods.PICoater_Initialize(_handle, w, h);
                int ret = NativeMethods.PICoater_Run(
                    _handle, _inputBuffer, IntPtr.Zero, IntPtr.Zero, _ridgeBuffer, IntPtr.Zero,
                    DefaultBgSigma, DefaultRidgeSigma, DefaultHeatmapThres, DefaultHeatmapAlpha, DefaultRidgeMode
                );
                if (ret != 0) return null;

                // 直接使用 _ridgeBuffer (不翻轉)
                return ImageUtils.Create8bppBitmap(_ridgeBuffer, w, h, flipY: false);
            }
            else
            {
                // [Original Mode] 原圖 -> 顯示正常 (Upright) -> 需要翻轉
                // 使用 _inputBuffer 並進行翻轉
                return ImageUtils.Create8bppBitmap(_inputBuffer, w, h, flipY: false);
            }
        }

        public void Dispose()
        {
            if (!_isDisposed)
            {
                if (_inputBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_inputBuffer);
                if (_thumbnailBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_thumbnailBuffer);
                if (_muraBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_muraBuffer);
                if (_ridgeBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_ridgeBuffer);
                if (_curveBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_curveBuffer);
                if (_handle != IntPtr.Zero) NativeMethods.PICoater_Destroy(_handle);
                _isDisposed = true;
            }
        }
    }
}