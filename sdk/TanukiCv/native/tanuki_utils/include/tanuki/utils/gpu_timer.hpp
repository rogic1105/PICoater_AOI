#pragma once
// GPU kernel 計時器（cudaEvent）—— tanuki_utils 共用唯一來源。
// 為何用 cudaEvent 而非 host 計時：kernel 是 async，host 端 std::chrono 只量到「launch 回傳」
// 不是 kernel 真正跑完的時間。cudaEvent 在 GPU stream 上打點，量的是 device 實際耗時。
#include <cuda_runtime.h>

namespace tanuki { namespace utils {

class GpuTimer {
public:
    GpuTimer()  { cudaEventCreate(&a_); cudaEventCreate(&b_); }
    ~GpuTimer() { cudaEventDestroy(a_); cudaEventDestroy(b_); }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t s = 0) { cudaEventRecord(a_, s); }

    /// 結束打點 + 等該事件完成 → 回傳這段 GPU 毫秒。
    float stop_ms(cudaStream_t s = 0) {
        cudaEventRecord(b_, s);
        cudaEventSynchronize(b_);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a_, b_);
        return ms;
    }

private:
    cudaEvent_t a_, b_;
};

}} // namespace tanuki::utils
