// AOI_SDK\core_cv\src\imgproc\background\background_ops.cu

#include "core_cv/base/cuda_utils.hpp"
#include "core_cv/imgproc/core_background.hpp"
#include "background_kernels.cuh"
#include <cstdint>

namespace core {

    // 1. 一般平均 Host Function
    template <typename T>
    void calcColumnMeans_gpu(const T* d_in, float* d_out, int W, int H, cudaStream_t stream, void* d_workspace) {
        int gridSize, blockSize;
        get_optimal_launch_1d(k_calcColumnMeans<T>, W, gridSize, blockSize);
        k_calcColumnMeans<T> << <gridSize, blockSize, 0, stream >> > (d_in, d_out, W, H);
        CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    void calcColumnMax_gpu(const T* d_in, float* d_out, int W, int H, cudaStream_t stream) {
        int gridSize, blockSize;
        get_optimal_launch_1d(k_calcColumnMax<T>, W, gridSize, blockSize);
        k_calcColumnMax<T> << <gridSize, blockSize, 0, stream >> > (d_in, d_out, W, H);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2. [修改] 去除離群值 Host Function (改為泛型)
    template <typename T>
    void calcColumnMeans_RemoveOutliers_gpu(const T* d_in, float* d_out, int W, int H, float sigma, cudaStream_t stream) {
        int gridSize, blockSize;
        get_optimal_launch_1d(k_calcColumnMeans_RemoveOutliers<T>, W, gridSize, blockSize);
        k_calcColumnMeans_RemoveOutliers<T> << <gridSize, blockSize, 0, stream >> > (d_in, d_out, W, H, sigma);
        CUDA_CHECK(cudaGetLastError());
    }

    // 3. 背景相減 Host Function
    void calcColumnBackground_u8_gpu(const uint8_t* d_in, const float* d_mean, uint8_t* d_out, int W, int H, cudaStream_t s) {
        dim3 gridDim, blockDim;
        get_optimal_launch_2d(k_calcColumnBackground, W, H, gridDim, blockDim);
        k_calcColumnBackground << <gridDim, blockDim, 0, s >> > (d_in, d_mean, d_out, W, H);
        CUDA_CHECK(cudaGetLastError());
    }

    // =========================================================
    // 顯式實例化 (Explicit Instantiation)
    // 確保 Linker 找得到 uint8_t 和 float 版本的實作
    // =========================================================

    // 一般平均
    template void calcColumnMeans_gpu<uint8_t>(const uint8_t*, float*, int, int, cudaStream_t, void*);
    template void calcColumnMeans_gpu<float>(const float*, float*, int, int, cudaStream_t, void*);

    template void calcColumnMax_gpu<uint8_t>(const uint8_t*, float*, int, int, cudaStream_t);
    template void calcColumnMax_gpu<float>(const float*, float*, int, int, cudaStream_t);

    // 去除離群值平均
    template void calcColumnMeans_RemoveOutliers_gpu<uint8_t>(const uint8_t*, float*, int, int, float, cudaStream_t);
    template void calcColumnMeans_RemoveOutliers_gpu<float>(const float*, float*, int, int, float, cudaStream_t);
}