// AOI_SDK\core_cv\src\imgproc\filters\filters_ops.cu

#include "filters_kernels.cuh"
#include "core_cv/base/cuda_utils.hpp"
#include <cmath>


namespace core {

    // 卷積 (維持 2D)
    __global__ void k_convolution_u8(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int W, int H, const float* __restrict__ d_mask, int maskSize) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= W || y >= H) return;

        int r = maskSize / 2;
        float sum = 0.0f;

        for (int ky = -r; ky <= r; ++ky) {
            for (int kx = -r; kx <= r; ++kx) {
                int curX = x + kx;
                int curY = y + ky;

                // 邊緣鏡射 101
                if (curX < 0) curX = -curX; else if (curX >= W) curX = 2 * (W - 1) - curX;
                if (curY < 0) curY = -curY; else if (curY >= H) curY = 2 * (H - 1) - curY;

                // Clamp 確保安全
                if (curX < 0) curX = 0; else if (curX >= W) curX = W - 1;
                if (curY < 0) curY = 0; else if (curY >= H) curY = H - 1;

                sum += (float)in[curY * W + curX] * d_mask[(ky + r) * maskSize + (kx + r)];
            }
        }
        float abs_sum = fabsf(sum); // 負數變正數，讓黑色變白色的邊緣也能顯示
        if (abs_sum > 255.0f) abs_sum = 255.0f;

        out[y * W + x] = (uint8_t)(abs_sum + 0.5f);
    }

    // Float 版本的卷積 (用於高斯模糊)
    __global__ void k_convolution_f32(const float* __restrict__ in, float* __restrict__ out, int W, int H, const float* __restrict__ d_mask, int maskSize) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= W || y >= H) return;

        int r = maskSize / 2;
        float sum = 0.0f;

        // 簡單的 2D 卷積迴圈
        for (int ky = -r; ky <= r; ++ky) {
            for (int kx = -r; kx <= r; ++kx) {
                int curX = x + kx;
                int curY = y + ky;

                // 邊緣處理：Clamp (複製邊緣像素)
                if (curX < 0) curX = 0; else if (curX >= W) curX = W - 1;
                if (curY < 0) curY = 0; else if (curY >= H) curY = H - 1;

                sum += in[curY * W + curX] * d_mask[(ky + r) * maskSize + (kx + r)];
            }
        }
        out[y * W + x] = sum;
    }

    // Pass 1: 水平卷積 (Row)
    __global__ void k_gaussianBlurRow(const float* __restrict__ in, float* __restrict__ out, int W, int H, const float* __restrict__ d_mask, int ksize) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y >= H || x >= W) return;

        int r = ksize / 2;
        float sum = 0.0f;

        for (int k = -r; k <= r; ++k) {
            int curX = x + k;

            if (curX < 0) curX = 0;
            else if (curX >= W) curX = W - 1;

            // 直接讀取 Global Memory (有 L1/L2 Cache 輔助，速度依然很快)
            sum += in[y * W + curX] * d_mask[k + r];
        }

        out[y * W + x] = sum;
    }

    // Pass 2: 垂直卷積 (Column)
    __global__ void k_gaussianBlurCol(const float* __restrict__ in, float* __restrict__ out, int W, int H, const float* __restrict__ d_mask, int ksize) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (y >= H || x >= W) return;

        int r = ksize / 2;
        float sum = 0.0f;

        for (int k = -r; k <= r; ++k) {
            int curY = y + k;

            if (curY < 0) curY = 0;
            else if (curY >= H) curY = H - 1;

            sum += in[curY * W + x] * d_mask[k + r];
        }

        out[y * W + x] = sum;
    }


}