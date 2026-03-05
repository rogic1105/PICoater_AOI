// AOI_SDK\core_cv\include\core_cv\imgproc\core_utils.hpp
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace core {
    // 這些 helper 通常不需暴露給外部使用者，但若其他模組會用到，可以放這
    // 若只有內部 kernel 用，則不必在此宣告

    void zero_border_u8_gpu(uint8_t* d_gray, int roiW, int roiH, int t, cudaStream_t s = 0);

    void convert_f32_to_u8_clamp_gpu(const float* d_in, uint8_t* d_out, int N, cudaStream_t s = 0);
    
    void scale_clamp_f32_to_u8_gpu(const float* d_in, uint8_t* d_out, int num_pixels, float scale_factor, cudaStream_t stream = 0);

    void convert_u8_to_f32_gpu(const uint8_t* d_in, float* d_out, int N, cudaStream_t s = 0);
    
    void normalize_minmax_f32_u8_gpu(const float* d_in, uint8_t* d_out, int N, cudaStream_t s = 0);

    void overlay_heatmap_gpu(
        const uint8_t* d_src,
        const uint8_t* d_overlay,
        uint8_t* d_out_bgr,
        int width, int height,
        int lower_limit = 0,
        float alpha = 0.5f,
        cudaStream_t stream = 0
    );
}

