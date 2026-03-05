// AOI_SDK\core_cv\src\imgproc\enhance\enhance_kernels.cuh

#pragma once
#include <cuda_runtime.h>
#include "core_cv/base/cuda_utils.hpp"
#include <cstdint>

namespace core {
    __global__ void k_brighten_u8(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int N, int bright);
    __global__ void k_threshold_u8(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int N, uint8_t thresh);
    __global__ void k_invert_u8(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int N);

}