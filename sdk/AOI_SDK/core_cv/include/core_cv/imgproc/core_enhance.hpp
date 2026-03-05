// AOI_SDK\core_cv\include\core_cv\imgproc\core_enhance.hpp

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace core {

    void brighten_u8_gpu(const uint8_t* d_in, uint8_t* d_out, int W, int H, int bright, cudaStream_t s = 0);
    void threshold_u8_gpu(const uint8_t* d_in, uint8_t* d_out, int W, int H, uint8_t thresh, cudaStream_t s = 0);
    void invert_u8_gpu(const uint8_t* d_in, uint8_t* d_out, int W, int H, cudaStream_t s = 0);

}