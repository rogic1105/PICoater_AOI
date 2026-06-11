
#include "tanuki/core/base/cuda_utils.hpp"
#include "tanuki/core/imgproc/core_background.hpp"
#include "background_kernels.cuh"
#include <cstdint>

namespace tanuki { namespace core {


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


    template <typename T>
    void calcColumnMeans_RemoveOutliers_gpu(const T* d_in, float* d_out, int W, int H, float sigma, cudaStream_t stream) {
        int gridSize, blockSize;
        get_optimal_launch_1d(k_calcColumnMeans_RemoveOutliers<T>, W, gridSize, blockSize);
        k_calcColumnMeans_RemoveOutliers<T> << <gridSize, blockSize, 0, stream >> > (d_in, d_out, W, H, sigma);
        CUDA_CHECK(cudaGetLastError());
    }


    void calcColumnBackground_u8_gpu(const uint8_t* d_in, const float* d_mean, uint8_t* d_out, int W, int H, cudaStream_t s) {
        dim3 gridDim, blockDim;
        get_optimal_launch_2d(k_calcColumnBackground, W, H, gridDim, blockDim);
        k_calcColumnBackground << <gridDim, blockDim, 0, s >> > (d_in, d_mean, d_out, W, H);
        CUDA_CHECK(cudaGetLastError());
    }

    // Row-wise mean
    template <typename T>
    void calcRowMeans_gpu(const T* d_in, float* d_out, int W, int H, cudaStream_t stream) {
        int gridSize, blockSize;
        get_optimal_launch_1d(k_calcRowMeans<T>, H, gridSize, blockSize);
        k_calcRowMeans<T> <<<gridSize, blockSize, 0, stream>>>(d_in, d_out, W, H);
        CUDA_CHECK(cudaGetLastError());
    }

    // Row-wise max
    template <typename T>
    void calcRowMax_gpu(const T* d_in, float* d_out, int W, int H, cudaStream_t stream) {
        int gridSize, blockSize;
        get_optimal_launch_1d(k_calcRowMax<T>, H, gridSize, blockSize);
        k_calcRowMax<T> <<<gridSize, blockSize, 0, stream>>>(d_in, d_out, W, H);
        CUDA_CHECK(cudaGetLastError());
    }

    // =========================================================
    // Explicit Instantiation
    // =========================================================

    // Column means
    template void calcColumnMeans_gpu<uint8_t>(const uint8_t*, float*, int, int, cudaStream_t, void*);
    template void calcColumnMeans_gpu<float>(const float*, float*, int, int, cudaStream_t, void*);

    template void calcColumnMax_gpu<uint8_t>(const uint8_t*, float*, int, int, cudaStream_t);
    template void calcColumnMax_gpu<float>(const float*, float*, int, int, cudaStream_t);

    // Column means (outlier removal)
    template void calcColumnMeans_RemoveOutliers_gpu<uint8_t>(const uint8_t*, float*, int, int, float, cudaStream_t);
    template void calcColumnMeans_RemoveOutliers_gpu<float>(const float*, float*, int, int, float, cudaStream_t);

    // Row means
    template void calcRowMeans_gpu<uint8_t>(const uint8_t*, float*, int, int, cudaStream_t);
    template void calcRowMeans_gpu<float>(const float*, float*, int, int, cudaStream_t);

    // Row max
    template void calcRowMax_gpu<uint8_t>(const uint8_t*, float*, int, int, cudaStream_t);
    template void calcRowMax_gpu<float>(const float*, float*, int, int, cudaStream_t);
}}  // namespace core, tanuki