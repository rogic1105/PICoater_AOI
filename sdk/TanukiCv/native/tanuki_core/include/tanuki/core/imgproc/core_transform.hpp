#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace tanuki { namespace core {

    void resize_u8_gpu(const uint8_t* d_src, int src_w, int src_h,
        uint8_t* d_dst, int dst_w, int dst_h,
        cudaStream_t stream);

}}  // namespace core, tanuki