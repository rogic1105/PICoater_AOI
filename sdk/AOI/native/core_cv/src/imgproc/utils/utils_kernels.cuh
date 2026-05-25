// AOI_SDK\core_cv\src\imgproc\utils\utils_kernels.cuh

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace core {
    __global__ void k_zeroBorder_u8(uint8_t* __restrict__ in, int roiW, int roiH, int t);

    __global__ void k_f32_to_u8_clamp(const float* __restrict__ in, uint8_t* __restrict__ out, int N);

    __global__ void k_scale_clamp_f32_to_u8(const float* src, uint8_t* dst, int num_pixels, float scale_factor);

    __global__ void k_u8_to_f32(const uint8_t* __restrict__ in, float* __restrict__ out, int N);

    __global__ void k_normalizeMinMax_f32_u8(const float* __restrict__ in, uint8_t* __restrict__ out, int N, float minVal, float maxVal);

    __global__ void k_overlay_heatmap(
        const uint8_t* __restrict__ src,      // 原始灰階圖
        const uint8_t* __restrict__ overlay,  // Heatmap 來源 (0-255)
        uint8_t* __restrict__ dst,            // 輸出 BGR 彩圖 (size * 3)
        int width, int height,
        int lower_limit,
        float alpha
    );

}