
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace tanuki { namespace core {

    // �@�륭�� (�x��)
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

    // [�ק�] �h�����s�ȥ��� (�אּ�x���A�䴩 uint8 �M float)
    template <typename T>
    __global__ void k_calcColumnMeans_RemoveOutliers(
        const T* __restrict__ src,
        float* __restrict__ dst,
        int W, int H,
        float sigma_threshold
    );

    // Row-wise mean (1 thread per row)
    template <typename T>
    __global__ void k_calcRowMeans(
        const T* __restrict__ src,
        float* __restrict__ dst,
        int W, int H
    );

    // Row-wise max (1 thread per row)
    template <typename T>
    __global__ void k_calcRowMax(
        const T* __restrict__ src,
        float* __restrict__ dst,
        int W, int H
    );

    // �I���۴� (��J uint8, ��X uint8) - �O������
    __global__ void k_calcColumnBackground(
        const uint8_t* __restrict__ input_image,
        const float* __restrict__ column_means,
        uint8_t* __restrict__ output_image,
        int width, int height
    );
}}  // namespace core, tanuki