// AOI_SDK\core_cv\src\imgproc\filters\filters_kernels.cuh

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace core {
    // 一般 2D 卷積
    __global__ void k_convolution_u8(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int W, int H, const float* __restrict__ d_mask, int maskSize);

    // Float 版本的卷積 (中途運算用)
    __global__ void k_convolution_f32(const float* __restrict__ in, float* __restrict__ out, int W, int H, const float* __restrict__ d_mask, int maskSize);

    // 分離式高斯模糊 (Pass 1 & 2)
    __global__ void k_gaussianBlurRow(const float* __restrict__ in, float* __restrict__ out, int W, int H, const float* __restrict__ d_mask, int ksize);
    
    __global__ void k_gaussianBlurCol(const float* __restrict__ in, float* __restrict__ out, int W, int H, const float* __restrict__ d_mask, int ksize);
}