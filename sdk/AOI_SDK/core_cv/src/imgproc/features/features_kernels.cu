// AOI_SDK\core_cv\src\imgproc\features\features_ops.cu

#include "features_kernels.cuh"
#include "core_cv/base/cuda_utils.hpp"
#include <cmath>

namespace core {

    __global__ void k_sobelMagnitude_u8(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int W, int H) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= W || y >= H) return;

        // 邊界處理：邊緣設為 0
        if (x < 1 || x >= W - 1 || y < 1 || y >= H - 1) {
            out[y * W + x] = 0;
            return;
        }

        // 讀取 3x3
        // 00 01 02
        // 10 11 12
        // 20 21 22
        int idx = y * W + x;
        int i00 = in[idx - W - 1]; int i01 = in[idx - W]; int i02 = in[idx - W + 1];
        int i10 = in[idx - 1];     /* i11 */              int i12 = in[idx + 1];
        int i20 = in[idx + W - 1]; int i21 = in[idx + W]; int i22 = in[idx + W + 1];

        // Sobel X
        // -1  0  1
        // -2  0  2
        // -1  0  1
        float dx = (float)(i02 + 2 * i12 + i22 - (i00 + 2 * i10 + i20));

        // Sobel Y
        // -1 -2 -1
        //  0  0  0
        //  1  2  1
        float dy = (float)(i20 + 2 * i21 + i22 - (i00 + 2 * i01 + i02));

        // Magnitude
        float mag = sqrtf(dx * dx + dy * dy);

        if (mag > 255.0f) mag = 255.0f;
        out[idx] = (uint8_t)(mag + 0.5f);
    }

    __global__ void k_hessianResponse(const float* __restrict__ img, float* __restrict__ out, int W, int H, detectionMode mode) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= W * H) return;

        int x = idx % W;
        int y = idx / W;

        if (x < 1 || x >= W - 1 || y < 1 || y >= H - 1) {
            out[idx] = 0.0f;
            return;
        }

        // 這裡需要讀 float 類型的輸入
        float p00 = img[(y - 1) * W + (x - 1)];
        float p01 = img[(y - 1) * W + x];
        float p02 = img[(y - 1) * W + (x + 1)];
        float p10 = img[y * W + (x - 1)];
        float p11 = img[y * W + x];
        float p12 = img[y * W + (x + 1)];
        float p20 = img[(y + 1) * W + (x - 1)];
        float p21 = img[(y + 1) * W + x];
        float p22 = img[(y + 1) * W + (x + 1)];

        float val_xx = 0.0f;
        float val_yy = 0.0f;

        // Lxx (Vertical features)
        if (mode == detectionMode::VERTICAL || mode == detectionMode::BOTH) {
            val_xx = (p00 + p02 + p20 + p22) + 2.0f * (p10 + p12) - 2.0f * (p01 + p21) - 4.0f * p11;
            // 注意：上面的 mask 簡化寫法其實等於 Sobel(2,0)，標準是 1, -2, 1 卷積
            val_xx = (p00 - 2 * p01 + p02) + 2 * (p10 - 2 * p11 + p12) + (p20 - 2 * p21 + p22);
        }

        // Lyy (Horizontal features)
        if (mode == detectionMode::HORIZONTAL || mode == detectionMode::BOTH) {
            val_yy = (p00 - 2 * p10 + p20) + 2 * (p01 - 2 * p11 + p21) + (p02 - 2 * p12 + p22);
        }

        float response = 0.0f;
        if (mode == detectionMode::VERTICAL) response = fabsf(val_xx);
        else if (mode == detectionMode::HORIZONTAL) response = fabsf(val_yy);
        else response = fabsf(val_xx) + fabsf(val_yy);

        out[idx] = response;
    }

}