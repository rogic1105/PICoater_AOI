using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging; // 用於 PixelFormat
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

        // Pinned Buffers
        private IntPtr _inputBuffer = IntPtr.Zero;
        private IntPtr _thumbnailBuffer = IntPtr.Zero;
        private IntPtr _muraBuffer = IntPtr.Zero; 
        private IntPtr _ridgeBuffer = IntPtr.Zero;
        private IntPtr _curveBuffer = IntPtr.Zero;

        // Buffer Sizes
        private ulong _inputBufferSize = 0;
        private int _thumbnailBufferSize = 0;
        private ulong _imgBufferSize = 0; 
        private int _curveBufferSize = 0;

        // Constants
        private const int MaxWidth = 16384;
        private const int MaxHeight = 10000;
        private const int MaxThumbnailSide = 2000;

        // Default Params
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

            // 1. Image Buffers (Gray 8bit) - Input, Mura, Ridge 都是一樣大
            _imgBufferSize = (ulong)(MaxWidth * MaxHeight);

            _inputBuffer = NativeMethods.PICoater_AllocPinned(_imgBufferSize);
            _muraBuffer = NativeMethods.PICoater_AllocPinned(_imgBufferSize);
            _ridgeBuffer = NativeMethods.PICoater_AllocPinned(_imgBufferSize); 

            // 2. Thumbnail Buffer
            _thumbnailBufferSize = MaxThumbnailSide * MaxThumbnailSide;
            _thumbnailBuffer = NativeMethods.PICoater_AllocPinned((ulong)_thumbnailBufferSize);

            // 3. Curve Buffer
            _curveBufferSize = MaxWidth * sizeof(float);
            _curveBuffer = NativeMethods.PICoater_AllocPinned((ulong)_curveBufferSize);

            // 4. Curve Buffer
            _curveBufferSize = MaxWidth * sizeof(float);
            _curveBuffer = NativeMethods.PICoater_AllocPinned((ulong)_curveBufferSize);
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

        /// <summary>
        /// 批次處理：現在改為顯示 Ridge 的縮圖
        /// </summary>
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

                // [修改] 請求 Ridge 輸出 (_ridgeBuffer)
                int ret = NativeMethods.PICoater_Run(
                    _handle,
                    _inputBuffer,
                    IntPtr.Zero,    // BG
                    IntPtr.Zero,    // Mura (如果不需要顯示 Mura 就不傳)
                    _ridgeBuffer,   // Ridge (Output) <--- 這次我們要這個
                    _curveBuffer,   // Curve
                    DefaultBgSigma, DefaultRidgeSigma, DefaultHeatmapThres, DefaultHeatmapAlpha, DefaultRidgeMode
                );

                stopwatch.Stop();
                long algoTime = stopwatch.ElapsedMilliseconds;
                if (ret != 0) throw new Exception($"Algo Error: {ret}");

                stopwatch.Restart();

                int thumbH = (int)((float)h / w * targetThumbWidth);

                // [修改] 從 _ridgeBuffer 建立縮圖
                // ImageUtils.Create8bppBitmap 應該支援 IntPtr 來源
                using (var fullRidge = ImageUtils.Create8bppBitmap(_ridgeBuffer, w, h))
                {
                    var thumb = new Bitmap(fullRidge, targetThumbWidth, thumbH);

                    float[] curveData = new float[w];
                    Marshal.Copy(_curveBuffer, curveData, 0, w);

                    var data = new InspectionData { Image = thumb, MuraCurve = curveData };

                    stopwatch.Stop();
                    long bmpTime = stopwatch.ElapsedMilliseconds;
                    return (data, ioTime, algoTime, bmpTime);
                }
            });
        }


        /// <summary>
        /// 單張檢視：回傳 Ridge 原圖 (8-bit Gray)
        /// </summary>
        public Bitmap RunInspectionFullRes(string filePath)
        {
            if (_isDisposed) return null;
            if (!File.Exists(filePath)) return null;

            bool readSuccess = NativeMethods.PICoater_FastReadBMP(
                 filePath, out int w, out int h, _inputBuffer, _imgBufferSize);
            if (!readSuccess) return null;

            NativeMethods.PICoater_Initialize(_handle, w, h);

            // [修改] 請求 Ridge 輸出
            int ret = NativeMethods.PICoater_Run(
                _handle,
                _inputBuffer,
                IntPtr.Zero,
                IntPtr.Zero,
                _ridgeBuffer,   // Ridge Output
                IntPtr.Zero,
                DefaultBgSigma,
                DefaultRidgeSigma,
                DefaultHeatmapThres,
                DefaultHeatmapAlpha,
                DefaultRidgeMode
            );

            if (ret != 0) return null;

            // [修改] 回傳 8-bit Bitmap
            // ImageUtils.Create8bppBitmap 會負責設定 Grayscale Palette
            return ImageUtils.Create8bppBitmap(_ridgeBuffer, w, h);
        }

        public TimedResult<InspectionData> LoadThumbnailOnly(string filePath, int targetThumbWidth)
        {
            // 保持不變
            return ExecuteTimedOperation<InspectionData>(filePath, (stopwatch) =>
            {
                stopwatch.Start();
                int ret = NativeMethods.PICoater_LoadThumbnail(
                    filePath, targetThumbWidth, _thumbnailBuffer, out int realW, out int realH);
                stopwatch.Stop();
                long ioTime = stopwatch.ElapsedMilliseconds;

                if (ret != 0) return (null, ioTime, 0, 0);

                stopwatch.Restart();
                var bitmap = ImageUtils.Create8bppBitmap(_thumbnailBuffer, realW, realH);
                stopwatch.Stop();
                long bmpTime = stopwatch.ElapsedMilliseconds;

                var data = new InspectionData { Image = bitmap, MuraCurve = null };
                return (data, ioTime, 0, bmpTime);
            });
        }

        public void Dispose()
        {
            if (!_isDisposed)
            {
                if (_inputBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_inputBuffer);
                if (_thumbnailBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_thumbnailBuffer);
                if (_muraBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_muraBuffer);
                if (_ridgeBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_ridgeBuffer); // [新增]
                if (_curveBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_curveBuffer);
                // [移除] _heatmapBuffer free

                if (_handle != IntPtr.Zero) NativeMethods.PICoater_Destroy(_handle);
                _isDisposed = true;
            }
        }


    }
}