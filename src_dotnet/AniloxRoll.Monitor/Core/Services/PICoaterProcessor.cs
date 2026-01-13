using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

using AniloxRoll.Monitor.Core.Interop;
using AOI.SDK.Core.Models; 

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

        private static readonly ColorPalette _grayPalette;

        static PICoaterProcessor()
        {
            // 透過一個暫時的 Bitmap 取得標準 Palette 結構
            using (var bmp = new Bitmap(1, 1, PixelFormat.Format8bppIndexed))
            {
                _grayPalette = bmp.Palette;
            }
            // 填入灰階
            for (int i = 0; i < 256; i++)
            {
                _grayPalette.Entries[i] = Color.FromArgb(i, i, i);
            }
        }
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

            var result = new TimedResult<Bitmap>();
            var stopwatch = new Stopwatch();

            if (!File.Exists(filePath)) return result;

            try
            {
                // 1. IO 階段
                stopwatch.Start();
                bool readSuccess = PICoaterNative.PICoater_FastReadBMP(
                    filePath, out int w, out int h, _inputBuffer, (int)_inputBufferSize);
                stopwatch.Stop();
                result.IoDurationMs = stopwatch.ElapsedMilliseconds;

                if (!readSuccess) return result;

                // 2. 運算階段
                stopwatch.Restart();
                int thumbH = (int)((float)h / w * targetThumbWidth);

                // 檢查緩衝區
                if (targetThumbWidth * thumbH > _thumbnailBufferSize) return result;

                PICoaterNative.PICoater_Initialize(_handle, w, h);

                int ret = PICoaterNative.PICoater_Run_And_GetThumbnail(
                    _handle, _inputBuffer, _thumbnailBuffer,
                    targetThumbWidth, thumbH, 2.0f, 9.0f, "vertical");

                if (ret != 0) throw new Exception($"Algo Error: {ret}");

                result.Data = CopyToBitmap(targetThumbWidth, thumbH, _thumbnailBuffer);
                stopwatch.Stop();
                result.ComputeDurationMs = stopwatch.ElapsedMilliseconds;
            }
            catch (Exception) { /* Log error */ }
            return result;
        }

        /// <summary>
        /// 純讀取縮圖模式 (僅 IO，不進行檢測)
        /// </summary>
        public TimedResult<Bitmap> LoadThumbnailOnly(string filePath, int targetThumbWidth)
        {
            if (_isDisposed) throw new ObjectDisposedException(nameof(PICoaterProcessor));
            var result = new TimedResult<Bitmap>();
            var stopwatch = new Stopwatch();

            if (!File.Exists(filePath)) return result;

            try
            {
                stopwatch.Start();
                int realW, realH;
                // 使用 Native 快速縮放讀取
                int ret = PICoaterNative.PICoater_LoadThumbnail(
                    filePath, targetThumbWidth, _thumbnailBuffer, out realW, out realH);

                if (ret == 0)
                {
                    result.Data = CopyToBitmap(realW, realH, _thumbnailBuffer);
                }
                stopwatch.Stop();

                result.IoDurationMs = stopwatch.ElapsedMilliseconds;
                result.ComputeDurationMs = 0; // 無運算
            }
            catch (Exception) { /* Log error */ }
            return result;
        }

        /// <summary>
        /// 點擊放大時使用，產生高解析度結果
        /// </summary>
        public Bitmap RunInspectionFullRes(string filePath)
        {
            // 這裡應實作完整尺寸的 PICoater_Run
            // 為求簡潔，此處僅示意呼叫 ProcessImage 產生較大縮圖
            // 實務上請改為回傳 1:1 的結果
            var res = ProcessImage(filePath, 2000);
            var bmp = res.Data;
            res.Data = null; // 轉移擁有權，避免被 TimedResult Dispose
            return bmp;
        }

        private Bitmap CopyToBitmap(int w, int h, IntPtr ptr)
        {
            var bmp = new Bitmap(w, h, PixelFormat.Format8bppIndexed);

            // 直接指定快取好的 Palette，不用再跑迴圈
            bmp.Palette = _grayPalette;

            BitmapData bData = bmp.LockBits(new Rectangle(0, 0, w, h),
                                            ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
            unsafe
            {
                Buffer.MemoryCopy((void*)ptr, (void*)bData.Scan0, _thumbnailBufferSize, w * h);
            }
            bmp.UnlockBits(bData);
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