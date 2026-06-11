
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace tanuki { namespace core {

    template <typename T>
    void calcColumnMeans_gpu(const T* d_in, float* d_out, int W, int H, cudaStream_t stream, void* d_workspace);

    template <typename T>
    void calcColumnMax_gpu(const T* d_in, float* d_out, int W, int H, cudaStream_t stream);

    void calcColumnBackground_u8_gpu(const uint8_t* d_in, const float* d_mean, uint8_t* d_out, int W, int H, cudaStream_t s);

    template <typename T>
    void calcRowMeans_gpu(const T* d_in, float* d_out, int W, int H, cudaStream_t stream);

    template <typename T>
    void calcRowMax_gpu(const T* d_in, float* d_out, int W, int H, cudaStream_t stream);

    template <typename T>
    void calcColumnMeans_RemoveOutliers_gpu(const T* d_in, float* d_out, int W, int H, float sigma_threshold, cudaStream_t stream);


    // ==========================================
    // CPU Host Implementations 
    // ==========================================

    // �p��C���� (�������s��) - CPU ����
    // stride �w�]�� W�A�p�G�Ϥ��� padding �i�H�ǤJ��� stride
    void calcColumnMeans_RemoveOutliers_cpu(
        const uint8_t* h_in,
        float* h_out,
        int W,
        int H,
        int stride,
        float sigma_threshold
    );

    // �p��I�� (Mura) - CPU ����
    void calcColumnBackground_u8_cpu(
        const uint8_t* h_in,
        const float* h_mean,
        uint8_t* h_out,
        int W,
        int H,
        int stride
    );

}}  // namespace core, tanuki