using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using AniloxRoll.Monitor.Core.Interop;
using AOI.SDK.Core.Models;
using AOI.SDK.Utils; // [Refactor] 引用 SDK Utils

namespace AniloxRoll.Monitor.Core.Services
{
    public class PICoaterProcessor : IDisposable
    {
        private IntPtr _handle = IntPtr.Zero;

        // Pinned Memory (重複使用以避免 GC 與碎片化)
        private IntPtr _inputBuffer = IntPtr.Zero;
        private IntPtr _thumbnailBuffer = IntPtr.Zero;
        private ulong _inputBufferSize = 0;
        private int _thumbnailBufferSize = 0;

        private const int MaxWidth = 16384;
        private const int MaxHeight = 10000;
        private const int MaxThumbnailSide = 2000;

        private bool _isDisposed = false;

        public PICoaterProcessor()
        {
            InitializeNativeResources();
        }

        private void InitializeNativeResources()
        {
            _handle = PICoaterNative.PICoater_Create();
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create PICoater instance.");

            // 預先分配最大記憶體
            _inputBufferSize = (ulong)(MaxWidth * MaxHeight);
            _inputBuffer = PICoaterNative.PICoater_AllocPinned(_inputBufferSize);

            _thumbnailBufferSize = MaxThumbnailSide * MaxThumbnailSide;
            _thumbnailBuffer = PICoaterNative.PICoater_AllocPinned((ulong)_thumbnailBufferSize);
        }

        /// <summary>
        /// 完整檢測模式 (計算 IO 與 GPU 時間)
        /// </summary>
        public TimedResult<Bitmap> ProcessImage(string filePath, int targetThumbWidth)
        {
            if (_isDisposed) throw new ObjectDisposedException(nameof(PICoaterProcessor));

            // 使用本地函數封裝核心邏輯，方便計時
            return ExecuteTimedOperation(filePath, (stopwatch) =>
            {
                // 1. IO 階段
                stopwatch.Start();
                bool readSuccess = PICoaterNative.PICoater_FastReadBMP(
                    filePath, out int w, out int h, _inputBuffer, (int)_inputBufferSize);
                stopwatch.Stop();

                if (!readSuccess) return (null, stopwatch.ElapsedMilliseconds, 0);

                long ioTime = stopwatch.ElapsedMilliseconds;

                // 2. 運算階段
                stopwatch.Restart();
                int thumbH = (int)((float)h / w * targetThumbWidth);

                if (targetThumbWidth * thumbH > _thumbnailBufferSize)
                    return (null, ioTime, 0); // Buffer 不足

                PICoaterNative.PICoater_Initialize(_handle, w, h);

                int ret = PICoaterNative.PICoater_Run_And_GetThumbnail(
                    _handle, _inputBuffer, _thumbnailBuffer,
                    targetThumbWidth, thumbH, 2.0f, 9.0f, "vertical");

                if (ret != 0) throw new Exception($"Algo Error: {ret}");

                // [Refactor] 使用 SDK 的 ImageUtils 直接從指標建立 Bitmap
                var bitmap = ImageUtils.Create8bppBitmap(_thumbnailBuffer, targetThumbWidth, thumbH);

                stopwatch.Stop();
                long computeTime = stopwatch.ElapsedMilliseconds;

                return (bitmap, ioTime, computeTime);
            });
        }

        /// <summary>
        /// 純讀取縮圖模式 (僅 IO，不進行檢測)
        /// </summary>
        public TimedResult<Bitmap> LoadThumbnailOnly(string filePath, int targetThumbWidth)
        {
            if (_isDisposed) throw new ObjectDisposedException(nameof(PICoaterProcessor));

            return ExecuteTimedOperation(filePath, (stopwatch) =>
            {
                stopwatch.Start();

                int ret = PICoaterNative.PICoater_LoadThumbnail(
                    filePath, targetThumbWidth, _thumbnailBuffer, out int realW, out int realH);

                Bitmap bitmap = null;
                if (ret == 0)
                {
                    // [Refactor] 使用 SDK 的 ImageUtils
                    bitmap = ImageUtils.Create8bppBitmap(_thumbnailBuffer, realW, realH);
                }
                stopwatch.Stop();

                return (bitmap, stopwatch.ElapsedMilliseconds, 0);
            });
        }

        /// <summary>
        /// 輔助方法：統一處理計時、例外與結果封裝
        /// </summary>
        private TimedResult<Bitmap> ExecuteTimedOperation(string filePath, Func<Stopwatch, (Bitmap data, long io, long gpu)> operation)
        {
            var result = new TimedResult<Bitmap>();
            if (!File.Exists(filePath)) return result;

            try
            {
                var sw = new Stopwatch();
                var (data, io, gpu) = operation(sw);

                result.Data = data;
                result.IoDurationMs = io;
                result.ComputeDurationMs = gpu;
            }
            catch (Exception)
            {
                // 可在此加入 Log 機制
            }
            return result;
        }

        public Bitmap RunInspectionFullRes(string filePath)
        {
            // 實務上請改為回傳 1:1 的結果
            var res = ProcessImage(filePath, 2000);
            var bmp = res.Data;
            res.Data = null; // 轉移擁有權
            return bmp;
        }

        public void Dispose()
        {
            if (!_isDisposed)
            {
                if (_inputBuffer != IntPtr.Zero) PICoaterNative.PICoater_FreePinned(_inputBuffer);
                if (_thumbnailBuffer != IntPtr.Zero) PICoaterNative.PICoater_FreePinned(_thumbnailBuffer);
                if (_handle != IntPtr.Zero) PICoaterNative.PICoater_Destroy(_handle);
                _handle = IntPtr.Zero;
                _isDisposed = true;
            }
        }
    }
}