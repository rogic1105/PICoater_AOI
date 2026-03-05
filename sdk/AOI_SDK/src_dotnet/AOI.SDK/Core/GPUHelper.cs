// AOI_SDK\src_dotnet\AOI.SDK\Core\GPUHelper.cs

using System;
using System.Diagnostics;
using System.Threading.Tasks;

namespace AOI.SDK.Core
{
    public static class GPUHelper
    {
        /// <summary>
        /// 執行 GPU 暖身 (非同步執行，避免卡住 UI)
        /// </summary>
        public static async Task WarmUpAsync()
        {
            await Task.Run(() =>
            {
                try
                {
                    Debug.WriteLine("[GPU] Starting WarmUp...");
                    Stopwatch sw = Stopwatch.StartNew();

                    // 1. 分配極小的 GPU 記憶體
                    IntPtr d_src, d_dst;
                    int ret1 = CoreCVWrapper.CoreCV_MallocGPU(out d_src, 64, 64);
                    int ret2 = CoreCVWrapper.CoreCV_MallocGPU(out d_dst, 64, 64);

                    if (ret1 == 0 && ret2 == 0)
                    {
                        // 2. 隨便跑一個 Kernel (例如 Threshold)
                        // 這會強迫 CUDA Context 初始化並載入 Driver
                        CoreCVWrapper.CoreCV_Threshold_GPU(d_src, 64, 64, 128, d_dst);

                        // 3. 釋放
                        CoreCVWrapper.CoreCV_FreeGPU(d_src);
                        CoreCVWrapper.CoreCV_FreeGPU(d_dst);
                    }

                    sw.Stop();
                    Debug.WriteLine($"[GPU] WarmUp Completed in {sw.ElapsedMilliseconds} ms");
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[GPU] WarmUp Failed: {ex.Message}");
                }
            });
        }
    }
}