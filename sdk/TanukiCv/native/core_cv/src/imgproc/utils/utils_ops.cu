
#include "core_cv/base/cuda_utils.hpp"
#include "utils_kernels.cuh"
#include <vector>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


namespace core {

    // 2D �B��: Zero Border
    void zero_border_u8_gpu(uint8_t* d_gray, int roiW, int roiH, int t, cudaStream_t s) {
        dim3 grid, block;
        get_optimal_launch_2d(k_zeroBorder_u8, roiW, roiH, grid, block);
        k_zeroBorder_u8 << <grid, block, 0, s >> > (d_gray, roiW, roiH, t);
        CUDA_CHECK(cudaGetLastError());
    }

    // ����ഫ: Float -> Uint8 (��ºI�_ Clamp)
    void convert_f32_to_u8_clamp_gpu(const float* d_in, uint8_t* d_out, int N, cudaStream_t s) {
        int gridSize, blockSize;
        get_optimal_launch_1d(k_f32_to_u8_clamp, N, gridSize, blockSize);
        k_f32_to_u8_clamp << <gridSize, blockSize, 0, s >> > (d_in, d_out, N);
        CUDA_CHECK(cudaGetLastError());
    }

    void scale_clamp_f32_to_u8_gpu(const float* d_in, uint8_t* d_out, int num_pixels, float scale_factor, cudaStream_t stream) {
        int gridSize, blockSize;

        get_optimal_launch_1d(k_scale_clamp_f32_to_u8, num_pixels, gridSize, blockSize);
        k_scale_clamp_f32_to_u8 << < gridSize, blockSize, 0, stream >> > (d_in, d_out, num_pixels, scale_factor);
        CUDA_CHECK(cudaGetLastError());
    }

    // ����ഫ: Uint8 -> Float
    void convert_u8_to_f32_gpu(const uint8_t* d_in, float* d_out, int N, cudaStream_t s) {
        int gridSize, blockSize;
        get_optimal_launch_1d(k_u8_to_f32, N, gridSize, blockSize);
        k_u8_to_f32 << <gridSize, blockSize, 0, s >> > (d_in, d_out, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // ���W��: Float -> Uint8 (MinMax 0-255)
    void normalize_minmax_f32_u8_gpu(const float* d_in, uint8_t* d_out, int N, cudaStream_t s) {
        // 1. �ϥ� Thrust ��X�̤j�ȻP�̤p��
        // �إ� Thrust Device Pointer (���|Ĳ�o�ƻs�A�u�O�]�˫���)
        thrust::device_ptr<const float> d_ptr(d_in);

        // ���� minmax_element
        // �ϥ� thrust::cuda::par.on(s) �T�O�b���w�� Stream �W����
        auto result = thrust::minmax_element(thrust::cuda::par.on(s), d_ptr, d_ptr + N);

        // 2. �N���G�q Device �Ǧ^ Host
        float min_val, max_val;
        // result.first �M second �O device iterator�A�ݭn���X����V����
        // �ϥ� Async �ƻs�H�t�X Stream
        CUDA_CHECK(cudaMemcpyAsync(&min_val, result.first.get(), sizeof(float), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(&max_val, result.second.get(), sizeof(float), cudaMemcpyDeviceToHost, s));

        // [���n] �o�̥����P�B Stream�A�]�����U�Ӫ� Kernel Launch �ݭn�Ψ� CPU �ݪ� min_val/max_val
        CUDA_CHECK(cudaStreamSynchronize(s));

        // 3. ���楿�W�� Kernel
        int gridSize, blockSize;
        get_optimal_launch_1d(k_normalizeMinMax_f32_u8, N, gridSize, blockSize);
        k_normalizeMinMax_f32_u8 << <gridSize, blockSize, 0, s >> > (d_in, d_out, N, min_val, max_val);

        CUDA_CHECK(cudaGetLastError());
    }

    void overlay_heatmap_gpu(
        const uint8_t* d_src,
        const uint8_t* d_overlay,
        uint8_t* d_out_bgr,
        int width, int height,
        int lower_limit,
        float alpha,
        cudaStream_t stream
    ) {
        dim3 grid, block;
        get_optimal_launch_2d(k_overlay_heatmap, width, height, grid, block);
        k_overlay_heatmap << < grid, block, 0, stream >> > (
            d_src,
            d_overlay,
            d_out_bgr,
            width, height,
            lower_limit,
            alpha
            );
        CUDA_CHECK(cudaGetLastError());
    }

}