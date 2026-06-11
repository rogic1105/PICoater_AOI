
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace tanuki { namespace core {
    // �@�� 2D ���n
    __global__ void k_convolution_u8(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int W, int H, const float* __restrict__ d_mask, int maskSize);

    // Float ���������n (���~�B���)
    __global__ void k_convolution_f32(const float* __restrict__ in, float* __restrict__ out, int W, int H, const float* __restrict__ d_mask, int maskSize);

    // �����������ҽk (Pass 1 & 2)
    __global__ void k_gaussianBlurRow(const float* __restrict__ in, float* __restrict__ out, int W, int H, const float* __restrict__ d_mask, int ksize);
    
    __global__ void k_gaussianBlurCol(const float* __restrict__ in, float* __restrict__ out, int W, int H, const float* __restrict__ d_mask, int ksize);
}}  // namespace core, tanuki