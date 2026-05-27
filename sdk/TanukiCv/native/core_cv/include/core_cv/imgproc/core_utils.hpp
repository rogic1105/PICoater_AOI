#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace core {
    // �o�� helper �q�`���ݼ��S���~���ϥΪ̡A���Y��L�Ҳշ|�Ψ�A�i�H��o
    // �Y�u������ kernel �ΡA�h�����b���ŧi

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

