// AOI_SDK\src_dotnet\AOI.SDK\Core\GPUProcessor.cs

using System;
using System.Runtime.InteropServices;

namespace AOI.SDK.Core
{
    // 定義 GPU 運算委派
    public delegate int GPUKernelFunc(IntPtr d_src, IntPtr d_dst, int w, int h);

    public class GPUProcessor : IDisposable
    {
        private IntPtr _d_origin = IntPtr.Zero;
        private IntPtr _d_current = IntPtr.Zero;
        private IntPtr _d_temp = IntPtr.Zero;

        // 額外資源 (如 Mask) 可以用 Dictionary 管理，這裡先簡單處理
        private IntPtr _d_mask = IntPtr.Zero;

        private int _width;
        private int _height;

        public bool IsInitialized => _d_current != IntPtr.Zero;
        public int Width => _width;
        public int Height => _height;

        // 1. 初始化 GPU 資源
        public void Initialize(IntPtr pHostSrc, int w, int h)
        {
            Dispose(); // 清除舊的

            _width = w;
            _height = h;

            CoreCVWrapper.CoreCV_MallocGPU(out _d_origin, w, h);
            CoreCVWrapper.CoreCV_MallocGPU(out _d_current, w, h);
            CoreCVWrapper.CoreCV_MallocGPU(out _d_temp, w, h);

            // 上傳原圖到 Origin 和 Current
            CoreCVWrapper.CoreCV_Upload(pHostSrc, _d_origin, w, h);
            CoreCVWrapper.CoreCV_Upload(pHostSrc, _d_current, w, h);
        }

        // 2. 執行運算 (Ping-Pong)
        // 回傳運算耗時 (ms)
        public double Process(GPUKernelFunc kernelFunc, out int errorCode)
        {
            if (!IsInitialized) { errorCode = -1; return 0; }

            // 計時
            var sw = System.Diagnostics.Stopwatch.StartNew();

            // 執行 Kernel (Current -> Temp)
            errorCode = kernelFunc(_d_current, _d_temp, _width, _height);

            sw.Stop();

            if (errorCode == 0)
            {
                // 交換指標
                IntPtr swap = _d_current;
                _d_current = _d_temp;
                _d_temp = swap;
            }
            return sw.Elapsed.TotalMilliseconds;
        }

        // 3. 下載結果回 Host
        public void DownloadResult(byte[] hostBuffer)
        {
            if (!IsInitialized) return;

            GCHandle h = GCHandle.Alloc(hostBuffer, GCHandleType.Pinned);
            try
            {
                CoreCVWrapper.CoreCV_Download(_d_current, h.AddrOfPinnedObject(), _width, _height);
            }
            finally { h.Free(); }
        }

        // 4. 重置回原圖
        public void Reset()
        {
            if (!IsInitialized) return;
            // 這裡我們需要 DeviceToDevice Copy，但目前只能用 Upload
            // 或者：我們如果之前保留了 _d_origin，我們可以新增一個 Kernel 做 Copy
            // 暫時解法：假設外部會傳入原圖的 Host Buffer 重新 Upload
            // 更好的解法是 C++ 新增 CoreCV_CopyGPU(_d_origin, _d_current)
            // 這裡先留空，由外部 Upload 處理，或是補上 C++ API
        }

        // 重置回原圖 (使用 Host Buffer)
        public void Reset(byte[] originalData)
        {
            if (!IsInitialized) return;
            GCHandle h = GCHandle.Alloc(originalData, GCHandleType.Pinned);
            try
            {
                CoreCVWrapper.CoreCV_Upload(h.AddrOfPinnedObject(), _d_current, _width, _height);
            }
            finally { h.Free(); }
        }

        // Mask 管理
        public IntPtr GetConvolutionMask(float[] maskData)
        {
            if (_d_mask == IntPtr.Zero)
            {
                CoreCVWrapper.CoreCV_MallocGPU_Float(out _d_mask, maskData.Length);
                CoreCVWrapper.CoreCV_Upload_Float(maskData, _d_mask, maskData.Length);
            }
            return _d_mask;
        }

        public void Dispose()
        {
            if (_d_origin != IntPtr.Zero) { CoreCVWrapper.CoreCV_FreeGPU(_d_origin); _d_origin = IntPtr.Zero; }
            if (_d_current != IntPtr.Zero) { CoreCVWrapper.CoreCV_FreeGPU(_d_current); _d_current = IntPtr.Zero; }
            if (_d_temp != IntPtr.Zero) { CoreCVWrapper.CoreCV_FreeGPU(_d_temp); _d_temp = IntPtr.Zero; }
            if (_d_mask != IntPtr.Zero) { CoreCVWrapper.CoreCV_FreeGPU_Float(_d_mask); _d_mask = IntPtr.Zero; }
            if (_d_mask != IntPtr.Zero) { CoreCVWrapper.CoreCV_FreeGPU_Float(_d_mask); _d_mask = IntPtr.Zero; }
        }
    }
}