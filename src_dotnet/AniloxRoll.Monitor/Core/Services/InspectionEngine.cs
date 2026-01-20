using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using AniloxRoll.Monitor.Core.Interop;
using AOI.SDK.Core.Models;
using AOI.SDK.Utils;
using System.Runtime.InteropServices;
using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.Core.Services
{
    public class InspectionEngine : IDisposable
    {
        // ... (成員變數與 InitializeNativeResources 保持不變) ...
        private IntPtr _handle = IntPtr.Zero;
        private IntPtr _inputBuffer = IntPtr.Zero;
        private IntPtr _thumbnailBuffer = IntPtr.Zero;
        private ulong _inputBufferSize = 0;
        private int _thumbnailBufferSize = 0;
        private const int MaxWidth = 16384;
        private const int MaxHeight = 10000;
        private const int MaxThumbnailSide = 2000;
        private bool _isDisposed = false;
        private IntPtr _curveBuffer = IntPtr.Zero;
        private int _curveBufferSize = 0;

        public InspectionEngine() { InitializeNativeResources(); }

        private void InitializeNativeResources()
        {
            _handle = NativeMethods.PICoater_Create();
            _inputBufferSize = (ulong)(MaxWidth * MaxHeight);
            _inputBuffer = NativeMethods.PICoater_AllocPinned(_inputBufferSize);
            _thumbnailBufferSize = MaxThumbnailSide * MaxThumbnailSide;
            _thumbnailBuffer = NativeMethods.PICoater_AllocPinned((ulong)_thumbnailBufferSize);
            _curveBufferSize = MaxWidth * sizeof(float);
            _curveBuffer = NativeMethods.PICoater_AllocPinned((ulong)_curveBufferSize);
        }

        // [修正] 泛型化 ExecuteTimedOperation
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

        public TimedResult<InspectionData> ProcessImage(string filePath, int targetThumbWidth)
        {
            if (_isDisposed) throw new ObjectDisposedException(nameof(InspectionEngine));

            // [修正] 明確指定泛型型別 <InspectionData>
            return ExecuteTimedOperation<InspectionData>(filePath, (stopwatch) =>
            {
                stopwatch.Start();
                bool readSuccess = NativeMethods.PICoater_FastReadBMP(
                    filePath, out int w, out int h, _inputBuffer, (int)_inputBufferSize);
                stopwatch.Stop();

                if (!readSuccess) return (null, stopwatch.ElapsedMilliseconds, 0, 0);
                long ioTime = stopwatch.ElapsedMilliseconds;

                stopwatch.Restart();
                int thumbH = (int)((float)h / w * targetThumbWidth);
                NativeMethods.PICoater_Initialize(_handle, w, h);

                int ret = NativeMethods.PICoater_Run_And_GetThumbnail(
                    _handle, _inputBuffer, _thumbnailBuffer,
                    _curveBuffer,
                    targetThumbWidth, thumbH, 2.0f, 9.0f, "vertical");

                stopwatch.Stop();
                if (ret != 0) throw new Exception($"Algo Error: {ret}");
                long algoTime = stopwatch.ElapsedMilliseconds;

                stopwatch.Restart();
                var bitmap = ImageUtils.Create8bppBitmap(_thumbnailBuffer, targetThumbWidth, thumbH);
                float[] curveData = new float[w];
                Marshal.Copy(_curveBuffer, curveData, 0, w);

                var data = new InspectionData { Image = bitmap, MuraCurve = curveData };
                stopwatch.Stop();
                long bmpTime = stopwatch.ElapsedMilliseconds;

                return (data, ioTime, algoTime, bmpTime);
            });
        }

        // [修正] 回傳型別改為 TimedResult<InspectionData>
        public TimedResult<InspectionData> LoadThumbnailOnly(string filePath, int targetThumbWidth)
        {
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

                // [修正] 即使沒有 Curve，也要回傳 InspectionData 結構
                var data = new InspectionData { Image = bitmap, MuraCurve = null };
                return (data, ioTime, 0, bmpTime);
            });
        }

        public Bitmap RunInspectionFullRes(string filePath)
        {
            var res = ProcessImage(filePath, 2000);
            if (res?.Data == null) return null;

            // [修正] 從 InspectionData 取出 Image
            var bmp = res.Data.Image;

            // 防止 Dispose 時把 Bitmap 釋放掉 (因為我们要回傳它)
            res.Data.Image = null;
            res.Dispose();

            return bmp;
        }

        public void Dispose()
        {
            if (!_isDisposed)
            {
                if (_inputBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_inputBuffer);
                if (_thumbnailBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_thumbnailBuffer);
                if (_handle != IntPtr.Zero) NativeMethods.PICoater_Destroy(_handle);
                if (_curveBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_curveBuffer);
                _isDisposed = true;
            }
        }
    }
}