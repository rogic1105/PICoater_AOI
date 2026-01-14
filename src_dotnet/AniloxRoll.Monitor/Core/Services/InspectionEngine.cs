using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using AniloxRoll.Monitor.Core.Interop;
using AOI.SDK.Core.Models;
using AOI.SDK.Utils; // [Refactor] 引用 SDK Utils

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// [Wrapper] 封裝單一相機通道的檢測核心。
    /// 負責管理 Unmanaged 資源 (Pinned Memory) 的生命週期，
    /// 並計算 IO 與 GPU 的執行時間，提供上層安全的 Bitmap 輸出。
    /// </summary>
    public class InspectionEngine : IDisposable
    {
        private IntPtr _handle = IntPtr.Zero;
        private IntPtr _inputBuffer = IntPtr.Zero;
        private IntPtr _thumbnailBuffer = IntPtr.Zero;
        private ulong _inputBufferSize = 0;
        private int _thumbnailBufferSize = 0;
        private const int MaxWidth = 16384;
        private const int MaxHeight = 10000;
        private const int MaxThumbnailSide = 2000;
        private bool _isDisposed = false;

        public InspectionEngine()
        {
            InitializeNativeResources();
        }

        private void InitializeNativeResources()
        {
            // ... (保持原本的初始化程式碼) ...
            _handle = NativeMethods.PICoater_Create();
            if (_handle == IntPtr.Zero) throw new InvalidOperationException("Failed to create PICoater instance.");
            _inputBufferSize = (ulong)(MaxWidth * MaxHeight);
            _inputBuffer = NativeMethods.PICoater_AllocPinned(_inputBufferSize);
            _thumbnailBufferSize = MaxThumbnailSide * MaxThumbnailSide;
            _thumbnailBuffer = NativeMethods.PICoater_AllocPinned((ulong)_thumbnailBufferSize);
        }

        /// <summary>
        /// 完整檢測模式 (計算 IO 與 GPU 時間)
        /// </summary>
        public TimedResult<Bitmap> ProcessImage(string filePath, int targetThumbWidth)
        {
            if (_isDisposed) throw new ObjectDisposedException(nameof(InspectionEngine));

            return ExecuteTimedOperation(filePath, (stopwatch) =>
            {
                // 1. IO 階段 (讀取)
                stopwatch.Start();
                bool readSuccess = NativeMethods.PICoater_FastReadBMP(
                    filePath, out int w, out int h, _inputBuffer, (int)_inputBufferSize);
                stopwatch.Stop();

                if (!readSuccess) return (null, stopwatch.ElapsedMilliseconds, 0, 0);
                long ioTime = stopwatch.ElapsedMilliseconds;

                // 2. 運算階段 (GPU/Algo)
                stopwatch.Restart();
                int thumbH = (int)((float)h / w * targetThumbWidth);

                if (targetThumbWidth * thumbH > _thumbnailBufferSize)
                    return (null, ioTime, 0, 0); // Buffer 不足

                NativeMethods.PICoater_Initialize(_handle, w, h);

                int ret = NativeMethods.PICoater_Run_And_GetThumbnail(
                    _handle, _inputBuffer, _thumbnailBuffer,
                    targetThumbWidth, thumbH, 2.0f, 9.0f, "vertical");

                stopwatch.Stop();
                if (ret != 0) throw new Exception($"Algo Error: {ret}");
                long algoTime = stopwatch.ElapsedMilliseconds;

                // 3. 圖片生成階段 (Bitmap Creation)
                stopwatch.Restart();
                // 從 Pointer 複製資料建立 C# Bitmap 物件
                var bitmap = ImageUtils.Create8bppBitmap(_thumbnailBuffer, targetThumbWidth, thumbH);
                stopwatch.Stop();
                long bmpTime = stopwatch.ElapsedMilliseconds;

                return (bitmap, ioTime, algoTime, bmpTime);
            });
        }

        /// <summary>
        /// 純讀取縮圖模式
        /// </summary>
        public TimedResult<Bitmap> LoadThumbnailOnly(string filePath, int targetThumbWidth)
        {
            if (_isDisposed) throw new ObjectDisposedException(nameof(InspectionEngine));

            return ExecuteTimedOperation(filePath, (stopwatch) =>
            {
                // 1. IO 階段 (包含縮圖解碼)
                stopwatch.Start();
                int ret = NativeMethods.PICoater_LoadThumbnail(
                    filePath, targetThumbWidth, _thumbnailBuffer, out int realW, out int realH);
                stopwatch.Stop();
                long ioTime = stopwatch.ElapsedMilliseconds;

                if (ret != 0) return (null, ioTime, 0, 0);

                // 2. 圖片生成階段
                stopwatch.Restart();
                var bitmap = ImageUtils.Create8bppBitmap(_thumbnailBuffer, realW, realH);
                stopwatch.Stop();
                long bmpTime = stopwatch.ElapsedMilliseconds;

                return (bitmap, ioTime, 0, bmpTime);
            });
        }

        /// <summary>
        /// [修改] 輔助方法：支援 4 個回傳值 (Result, IO, Algo, Bitmap)
        /// </summary>
        private TimedResult<Bitmap> ExecuteTimedOperation(
            string filePath,
            Func<Stopwatch, (Bitmap data, long io, long gpu, long bmp)> operation)
        {
            var result = new TimedResult<Bitmap>();
            if (!File.Exists(filePath)) return result;

            try
            {
                var sw = new Stopwatch();
                // 執行傳入的邏輯
                var (data, io, gpu, bmp) = operation(sw);

                result.Data = data;
                result.IoDurationMs = io;
                result.ComputeDurationMs = gpu;
                result.BitmapDurationMs = bmp; // [新增]
            }
            catch (Exception ex)
            {
                // 建議補上 ex.Message 以便除錯
                Console.WriteLine($"InspectionEngine Error: {ex.Message}");
            }
            return result;
        }

        public Bitmap RunInspectionFullRes(string filePath)
        {
            // 呼叫 ProcessImage 並只回傳 Bitmap
            // 注意：這裡如果覺得慢，可以在外部改接收 TimedResult 來觀察 BitmapDurationMs
            var res = ProcessImage(filePath, 2000);
            var bmp = res.Data;
            res.Data = null;
            return bmp;
        }

        public void Dispose()
        {
            if (!_isDisposed)
            {
                if (_inputBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_inputBuffer);
                if (_thumbnailBuffer != IntPtr.Zero) NativeMethods.PICoater_FreePinned(_thumbnailBuffer);
                if (_handle != IntPtr.Zero) NativeMethods.PICoater_Destroy(_handle);
                _handle = IntPtr.Zero;
                _isDisposed = true;
            }
        }
    }
}