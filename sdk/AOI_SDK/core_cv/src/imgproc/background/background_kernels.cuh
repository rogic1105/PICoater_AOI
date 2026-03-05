// AOI_SDK\core_cv\src\imgproc\background\background_kernels.cuh

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace core {

    // 一般平均 (泛型)
    template <typename T>
    __global__ void k_calcColumnMeans(
        const T* __restrict__ src,
        float* __restrict__ dst,
        int W, int H
    );

    template <typename T>
    __global__ void k_calcColumnMax(
        const T* __restrict__ src,
        float* __restrict__ dst,
        int W, int H
    );

    // [修改] 去除離群值平均 (改為泛型，支援 uint8 和 float)
    template <typename T>
    __global__ void k_calcColumnMeans_RemoveOutliers(
        const T* __restrict__ src,
        float* __restrict__ dst,
        int W, int H,
        float sigma_threshold
    );

    // 背景相減 (輸入 uint8, 輸出 uint8) - 保持不變
    __global__ void k_calcColumnBackground(
        const uint8_t* __restrict__ input_image,
        const float* __restrict__ column_means,
        uint8_t* __restrict__ output_image,
        int width, int height
    );
}