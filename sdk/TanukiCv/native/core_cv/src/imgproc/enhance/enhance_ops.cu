
#include "core_cv/base/cuda_utils.hpp"
#include "enhance_kernels.cuh"
#include <vector>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


namespace tanuki { namespace core {

    void brighten_u8_gpu(const uint8_t* d_in, uint8_t* d_out, int W, int H, int bright, cudaStream_t s) {
        int N = W * H;
        int gridSize, blockSize;
        get_optimal_launch_1d(k_brighten_u8, N, gridSize, blockSize);
        // �`�N�G�o�̶� N �Ӥ��O W, H
        k_brighten_u8 << <gridSize, blockSize, 0, s >> > (d_in, d_out, N, bright);
        CUDA_CHECK(cudaGetLastError());
    }

    void threshold_u8_gpu(const uint8_t* d_in, uint8_t* d_out, int W, int H, uint8_t thresh, cudaStream_t s) {
        int N = W * H;
        int gridSize, blockSize;
        get_optimal_launch_1d(k_threshold_u8, N, gridSize, blockSize);
        k_threshold_u8 << <gridSize, blockSize, 0, s >> > (d_in, d_out, N, thresh);
        CUDA_CHECK(cudaGetLastError());
    }

    void invert_u8_gpu(const uint8_t* d_in, uint8_t* d_out, int W, int H, cudaStream_t s) {
        int N = W * H;
        int gridSize, blockSize;
        get_optimal_launch_1d(k_invert_u8, N, gridSize, blockSize);
        k_invert_u8 << <gridSize, blockSize, 0, s >> > (d_in, d_out, N);
        CUDA_CHECK(cudaGetLastError());
    }

}}  // namespace core, tanuki