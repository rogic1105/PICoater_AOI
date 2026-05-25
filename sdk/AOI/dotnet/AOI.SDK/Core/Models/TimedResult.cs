// AOI_SDK\src_dotnet\AOI.SDK\Core\Models\TimedResult.cs


using System;

namespace AOI.SDK.Core.Models
{
    /// <summary>
    /// 通用的計時結果容器，用於封裝資料與 IO/運算/生成耗時。
    /// </summary>
    /// <typeparam name="T">資料型別 (如 Bitmap)</typeparam>
    public class TimedResult<T> : IDisposable
    {
        public T Data { get; set; }
        public long IoDurationMs { get; set; }      // IO 耗時 (讀檔)
        public long ComputeDurationMs { get; set; } // 運算耗時 (GPU/Algo)
        public long BitmapDurationMs { get; set; }  // [新增] Bitmap 生成耗時 (Memory Copy)

        public TimedResult() { }

        public TimedResult(T data, long ioMs, long computeMs, long bitmapMs)
        {
            Data = data;
            IoDurationMs = ioMs;
            ComputeDurationMs = computeMs;
            BitmapDurationMs = bitmapMs;
        }

        public void Dispose()
        {
            if (Data is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
    }
}