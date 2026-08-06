
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace tanuki { namespace core {


    __global__ void k_resize_nearest_u8(const uint8_t* src, int src_w, int src_h,
        uint8_t* dst, int dst_w, int dst_h);

    __global__ void k_downsample_max_f32_to_f16(const float* src, int src_w, int src_h,
        uint16_t* dst, int dst_w, int dst_h);


}}  // namespace core, tanuki
