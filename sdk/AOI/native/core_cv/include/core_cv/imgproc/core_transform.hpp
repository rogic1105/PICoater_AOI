// AOI_SDK\core_cv\include\core_cv\imgproc\core_transform.hpp
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace core {

    void resize_u8_gpu(const uint8_t* d_src, int src_w, int src_h,
        uint8_t* d_dst, int dst_w, int dst_h,
        cudaStream_t stream);

}