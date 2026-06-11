
#include "core_cv/base/cuda_utils.hpp"
#include "background_kernels.cuh"
#include <cmath>
#include <cuda_fp16.h>

namespace tanuki { namespace core {

    // [Traits] �M�w�֥[�����O
    template <typename T> struct AccumulatorTraits { using Type = float; }; // �w�] float (�� float ��J��)
    template <> struct AccumulatorTraits<uint8_t> { using Type = uint32_t; }; // �S�� uint8 -> uint32
    template <> struct AccumulatorTraits<double> { using Type = double; };

    // 1. �@�륭�� Kernel
    template <typename T>
    __global__ void k_calcColumnMeans(
        const T* __restrict__ src,
        float* __restrict__ dst,
        int W, int H
    ) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        // [�w���ˬd 1] X �b�V�ɫO�@
        if (col >= W) return;

        using SumType = typename AccumulatorTraits<T>::Type;
        SumType sum = 0;

        // [�w���ˬd 2] ����ū��� (�����`������ϩR)
        if (src == nullptr || dst == nullptr) return;

        // ��Ϊ������ޭp��A�קK���в֥[�y������b����
        // �o�ؼg�k��sĶ���u�ƨӻ��O�@�˪��A�� debug ���[
        for (int y = 0; y < H; ++y) {
            // [�w���ˬd 3] �p����ޡA�T�O�b�޿�d��
            // size_t ���� int * int ���� (���M 1.2���������|����A���n�ߺD)
            size_t idx = (size_t)y * W + col;

            // Ū��
            sum += src[idx];
        }

        // �g�J���G
        dst[col] = (float)sum / (float)H;
    }

    template <typename T>
    __global__ void k_calcColumnMax(
        const T* __restrict__ src,
        float* __restrict__ dst,
        int W, int H
    ) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= W) return;
        if (src == nullptr || dst == nullptr) return;

        // ���b�G�p�G���׬�0�A�]��0
        if (H <= 0) {
            dst[col] = 0.0f;
            return;
        }

        // 1. �H�Ĥ@�� row ���ȧ@����l�̤j��
        // �o�̪����૬�� float �i�����A�P dst �����@�P
        float max_val = (float)src[col];

        // 2. �M���ѤU�� row
        for (int y = 1; y < H; ++y) {
            size_t idx = (size_t)y * W + col;
            float val = (float)src[idx];
            if (val > max_val) {
                max_val = val;
            }
        }

        // 3. �g�J���G
        dst[col] = max_val;
    }



    // 2. [�ק�] �h�����s�� Kernel (�x����)
    template <typename T>
    __global__ void k_calcColumnMeans_RemoveOutliers(
        const T* __restrict__ src,
        float* __restrict__ dst,
        int W, int H,
        float sigma_threshold
    ) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= W) return;

        // �w�q�֥[���O
        using SumType = typename AccumulatorTraits<T>::Type;

        // --- [Pass 1] �p�� Mean & StdDev ---
        SumType sum = 0;
        SumType sq_sum = 0; // ����M

        const T* col_ptr = src + col;

        for (int y = 0; y < H; ++y) {
            // �૬�� SumType ����� (�Ҧp uint8 -> uint32)
            SumType val = (SumType)(*col_ptr);
            sum += val;
            sq_sum += val * val;
            col_ptr += W;
        }

        float mean = (float)sum / (float)H;

        // �ܲ��� = E[X^2] - (E[X])^2
        float variance = ((float)sq_sum / (float)H) - (mean * mean);
        if (variance < 0.0f) variance = 0.0f;
        float std_dev = sqrtf(variance);

        // --- [Pass 2] �L�o ---
        float limit = sigma_threshold * std_dev;
        float lower_bound = mean - limit;
        float upper_bound = mean + limit;

        float clean_sum = 0.0f;
        int clean_count = 0;

        col_ptr = src + col; // Reset pointer

        for (int y = 0; y < H; ++y) {
            float val = (float)(*col_ptr);
            if (val >= lower_bound && val <= upper_bound) {
                clean_sum += val;
                clean_count++;
            }
            col_ptr += W;
        }

        if (clean_count > 0) {
            dst[col] = clean_sum / (float)clean_count;
        }
        else {
            dst[col] = mean; // ���b
        }
    }

    // 3. �I���۴� (�O������)
    __global__ void k_calcColumnBackground(
        const uint8_t* __restrict__ input_image,
        const float* __restrict__ column_means,
        uint8_t* __restrict__ output_image,
        int width, int height
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) return;

        int idx = y * width + x;
        float pixel_val = (float)__ldg(&input_image[idx]);
        float bg_val = column_means[x];
        float result = pixel_val - bg_val + 127.0f;

        if (result < 0.0f) result = 0.0f;
        else if (result > 255.0f) result = 255.0f;

        output_image[idx] = (uint8_t)result;
    }

    // Row-wise mean: 1 thread per row, iterate columns (contiguous memory → good coalescing)
    template <typename T>
    __global__ void k_calcRowMeans(
        const T* __restrict__ src,
        float* __restrict__ dst,
        int W, int H
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= H) return;
        if (src == nullptr || dst == nullptr) return;

        using SumType = typename AccumulatorTraits<T>::Type;
        SumType sum = 0;
        size_t base = (size_t)row * W;
        for (int x = 0; x < W; ++x) {
            sum += src[base + x];
        }
        dst[row] = (float)sum / (float)W;
    }

    // Row-wise max: 1 thread per row
    template <typename T>
    __global__ void k_calcRowMax(
        const T* __restrict__ src,
        float* __restrict__ dst,
        int W, int H
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= H) return;
        if (src == nullptr || dst == nullptr) return;
        if (W <= 0) { dst[row] = 0.0f; return; }

        size_t base = (size_t)row * W;
        float max_val = (float)src[base];
        for (int x = 1; x < W; ++x) {
            float val = (float)src[base + x];
            if (val > max_val) max_val = val;
        }
        dst[row] = max_val;
    }

    template __global__ void k_calcRowMeans<uint8_t>(const uint8_t* __restrict__, float* __restrict__, int, int);
    template __global__ void k_calcRowMeans<float>(const float* __restrict__, float* __restrict__, int, int);

    template __global__ void k_calcRowMax<uint8_t>(const uint8_t* __restrict__, float* __restrict__, int, int);
    template __global__ void k_calcRowMax<float>(const float* __restrict__, float* __restrict__, int, int);

    template __global__ void k_calcColumnMeans<uint8_t>(const uint8_t* __restrict__, float* __restrict__, int, int);
    template __global__ void k_calcColumnMeans<float>(const float* __restrict__, float* __restrict__, int, int);

    template __global__ void k_calcColumnMax<uint8_t>(const uint8_t* __restrict__, float* __restrict__, int, int);
    template __global__ void k_calcColumnMax<float>(const float* __restrict__, float* __restrict__, int, int);

    template __global__ void k_calcColumnMeans_RemoveOutliers<uint8_t>(const uint8_t* __restrict__, float* __restrict__, int, int, float);
    template __global__ void k_calcColumnMeans_RemoveOutliers<float>(const float* __restrict__, float* __restrict__, int, int, float);
}}  // namespace core, tanuki