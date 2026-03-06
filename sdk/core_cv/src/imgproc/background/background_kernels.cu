// AOI_SDK\core_cv\src\imgproc\background\background_kernels.cu

#include "core_cv/base/cuda_utils.hpp"
#include "background_kernels.cuh"
#include <cmath>
#include <cuda_fp16.h>

namespace core {

    // [Traits] 決定累加器型別
    template <typename T> struct AccumulatorTraits { using Type = float; }; // 預設 float (給 float 輸入用)
    template <> struct AccumulatorTraits<uint8_t> { using Type = uint32_t; }; // 特化 uint8 -> uint32
    template <> struct AccumulatorTraits<double> { using Type = double; };

    // 1. 一般平均 Kernel
    template <typename T>
    __global__ void k_calcColumnMeans(
        const T* __restrict__ src,
        float* __restrict__ dst,
        int W, int H
    ) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        // [安全檢查 1] X 軸越界保護
        if (col >= W) return;

        using SumType = typename AccumulatorTraits<T>::Type;
        SumType sum = 0;

        // [安全檢查 2] 防止空指標 (雖不常見但能救命)
        if (src == nullptr || dst == nullptr) return;

        // 改用直接索引計算，避免指標累加造成的潛在錯位
        // 這種寫法對編譯器優化來說是一樣的，但 debug 更直觀
        for (int y = 0; y < H; ++y) {
            // [安全檢查 3] 計算索引，確保在邏輯範圍內
            // size_t 防止 int * int 溢位 (雖然 1.2億像素不會溢位，但好習慣)
            size_t idx = (size_t)y * W + col;

            // 讀取
            sum += src[idx];
        }

        // 寫入結果
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

        // 防呆：如果高度為0，設為0
        if (H <= 0) {
            dst[col] = 0.0f;
            return;
        }

        // 1. 以第一個 row 的值作為初始最大值
        // 這裡直接轉型為 float 進行比較，與 dst 類型一致
        float max_val = (float)src[col];

        // 2. 遍歷剩下的 row
        for (int y = 1; y < H; ++y) {
            size_t idx = (size_t)y * W + col;
            float val = (float)src[idx];
            if (val > max_val) {
                max_val = val;
            }
        }

        // 3. 寫入結果
        dst[col] = max_val;
    }



    // 2. [修改] 去除離群值 Kernel (泛型化)
    template <typename T>
    __global__ void k_calcColumnMeans_RemoveOutliers(
        const T* __restrict__ src,
        float* __restrict__ dst,
        int W, int H,
        float sigma_threshold
    ) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= W) return;

        // 定義累加型別
        using SumType = typename AccumulatorTraits<T>::Type;

        // --- [Pass 1] 計算 Mean & StdDev ---
        SumType sum = 0;
        SumType sq_sum = 0; // 平方和

        const T* col_ptr = src + col;

        for (int y = 0; y < H; ++y) {
            // 轉型為 SumType 防止溢位 (例如 uint8 -> uint32)
            SumType val = (SumType)(*col_ptr);
            sum += val;
            sq_sum += val * val;
            col_ptr += W;
        }

        float mean = (float)sum / (float)H;

        // 變異數 = E[X^2] - (E[X])^2
        float variance = ((float)sq_sum / (float)H) - (mean * mean);
        if (variance < 0.0f) variance = 0.0f;
        float std_dev = sqrtf(variance);

        // --- [Pass 2] 過濾 ---
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
            dst[col] = mean; // 防呆
        }
    }

    // 3. 背景相減 (保持不變)
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

    template __global__ void k_calcColumnMeans<uint8_t>(const uint8_t* __restrict__, float* __restrict__, int, int);
    template __global__ void k_calcColumnMeans<float>(const float* __restrict__, float* __restrict__, int, int);

    template __global__ void k_calcColumnMax<uint8_t>(const uint8_t* __restrict__, float* __restrict__, int, int);
    template __global__ void k_calcColumnMax<float>(const float* __restrict__, float* __restrict__, int, int);

    template __global__ void k_calcColumnMeans_RemoveOutliers<uint8_t>(const uint8_t* __restrict__, float* __restrict__, int, int, float);
    template __global__ void k_calcColumnMeans_RemoveOutliers<float>(const float* __restrict__, float* __restrict__, int, int, float);
}