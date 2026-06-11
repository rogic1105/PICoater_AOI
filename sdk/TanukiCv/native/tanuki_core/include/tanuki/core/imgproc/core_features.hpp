
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace tanuki { namespace core {
    // Ridge 偵測方向（對外列舉）。
    enum class detectionMode {VERTICAL = 0, HORIZONTAL = 1, BOTH = 2 };

    void sobel_u8_gpu(const uint8_t* d_in, uint8_t* d_out, int W, int H, cudaStream_t s = 0);

    void computeHessianResponse_gpu(const float* d_src, float* d_dst, int width, int height, detectionMode mode, cudaStream_t stream);
    // 註：hessianRidge_u8_gpu（blur+hessian+scale 組合）已移除 → ridge_hessian module（tanuki_pipeline）。

}}  // namespace core, tanuki