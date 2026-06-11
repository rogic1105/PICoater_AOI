
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace tanuki { namespace core {
    // �w�q Ridge Mode (���}���ϥΪ̬�)
    enum class detectionMode {VERTICAL = 0, HORIZONTAL = 1, BOTH = 2 };

    void sobel_u8_gpu(const uint8_t* d_in, uint8_t* d_out, int W, int H, cudaStream_t s = 0);

    void computeHessianResponse_gpu(const float* d_src, float* d_dst, int width, int height, detectionMode mode, cudaStream_t stream);
    
    void hessianRidge_u8_gpu(
        const uint8_t* d_in,
        uint8_t* d_out,
        int W, int H,
        float sigma,
        const char* mode_str,
        float fixed_max_val,
        float* d_temp_blur_f32,
        float* d_temp_response,
        cudaStream_t s,
        void* d_workspace = nullptr
    );

}}  // namespace core, tanuki