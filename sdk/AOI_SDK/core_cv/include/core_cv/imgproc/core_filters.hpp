// AOI_SDK\core_cv\include\core_cv\imgproc\core_filters.hpp

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace core {

    void convolution_u8_gpu(const uint8_t* d_in, uint8_t* d_out, int W, int H, const float* d_mask, int maskSize, cudaStream_t s = 0);
    
    void gaussianBlur_u8_gpu(const uint8_t* d_in, uint8_t* d_out, int W, int H, float sigma, int ksize, cudaStream_t s, void* d_workspace);

    template <typename T_in, typename T_out>
    void gaussianBlur_gpu(const T_in* d_in, T_out* d_out, int width, int height,
        float sigma, int ksize, cudaStream_t stream, void* d_workspace = nullptr);

}