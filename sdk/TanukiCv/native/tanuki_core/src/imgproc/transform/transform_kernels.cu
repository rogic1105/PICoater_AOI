
#include "transform_kernels.cuh"
#include "tanuki/core/base/cuda_utils.hpp"
#include <cuda_fp16.h>
#include <cmath>



namespace tanuki { namespace core {

    __global__ void k_resize_nearest_u8(const uint8_t* src, int src_w, int src_h,
        uint8_t* dst, int dst_w, int dst_h) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= dst_w || y >= dst_h) return;



        float scale_x = (float)src_w / (float)dst_w;
        float scale_y = (float)src_h / (float)dst_h;

        int src_x = (int)(x * scale_x);
        int src_y = (int)(y * scale_y);


        if (src_x >= src_w) src_x = src_w - 1;
        if (src_y >= src_h) src_y = src_h - 1;

        dst[y * dst_w + x] = src[src_y * src_w + src_x];
    }

    __global__ void k_downsample_max_f32_to_f16(const float* src, int src_w, int src_h,
        uint16_t* dst, int dst_w, int dst_h) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= dst_w || y >= dst_h) return;

        int x0 = (int)(((long long)x * src_w) / dst_w);
        int x1 = (int)((((long long)(x + 1) * src_w) + dst_w - 1) / dst_w);
        int y0 = (int)(((long long)y * src_h) / dst_h);
        int y1 = (int)((((long long)(y + 1) * src_h) + dst_h - 1) / dst_h);
        x1 = min(src_w, max(x0 + 1, x1));
        y1 = min(src_h, max(y0 + 1, y1));

        float peak = 0.0f;
        for (int sy = y0; sy < y1; ++sy) {
            const float* row = src + sy * src_w;
            for (int sx = x0; sx < x1; ++sx)
                peak = fmaxf(peak, row[sx]);
        }
        reinterpret_cast<__half*>(dst)[y * dst_w + x] = __float2half_rn(peak);
    }

}}  // namespace core, tanuki
