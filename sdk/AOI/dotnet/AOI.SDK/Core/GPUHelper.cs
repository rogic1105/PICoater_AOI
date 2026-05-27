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

                    // 暖身細節 (malloc + kernel + free) 封裝在 native，
                    // 「怎麼暖身」是 GPU 的內部事，不在 C# 端組底層 CUDA 呼叫。
                    int ret = CoreCVWrapper.CoreCV_WarmUp();

                    sw.Stop();
                    Debug.WriteLine($"[GPU] WarmUp Completed (ret={ret}) in {sw.ElapsedMilliseconds} ms");
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[GPU] WarmUp Failed: {ex.Message}");
                }
            });
        }
    }
}