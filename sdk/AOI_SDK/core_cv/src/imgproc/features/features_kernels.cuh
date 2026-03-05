// AOI_SDK\core_cv\src\imgproc\features\features_kernels.cuh

#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "core_cv/imgproc/core_features.hpp" // 為了引用 RidgeMode

namespace core {

    __global__ void k_sobelMagnitude_u8(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int W, int H);

    __global__ void k_hessianResponse(const float* __restrict__ in, float* __restrict__ out, int W, int H, detectionMode mode);
}