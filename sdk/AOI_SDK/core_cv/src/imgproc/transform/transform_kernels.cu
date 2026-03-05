// AOI_SDK\core_cv\src\imgproc\transform\transform_kernels.cu

#include "transform_kernels.cuh"
#include "core_cv/base/cuda_utils.hpp"
#include <cmath>



namespace core {

    __global__ void k_resize_nearest_u8(const uint8_t* src, int src_w, int src_h,
        uint8_t* dst, int dst_w, int dst_h) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= dst_w || y >= dst_h) return;

        // 計算對應的原圖座標 (浮點數計算)
        // 為了避免浮點誤差導致訪問越界，最後要 clamp
        float scale_x = (float)src_w / (float)dst_w;
        float scale_y = (float)src_h / (float)dst_h;

        int src_x = (int)(x * scale_x);
        int src_y = (int)(y * scale_y);

        // 邊界檢查 (Clamp)
        if (src_x >= src_w) src_x = src_w - 1;
        if (src_y >= src_h) src_y = src_h - 1;

        dst[y * dst_w + x] = src[src_y * src_w + src_x];
    }

}