
#include "transform_kernels.cuh"
#include "tanuki/core/base/cuda_utils.hpp"
#include <cmath>



namespace tanuki { namespace core {

    __global__ void k_resize_nearest_u8(const uint8_t* src, int src_w, int src_h,
        uint8_t* dst, int dst_w, int dst_h) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= dst_w || y >= dst_h) return;

        // �p���������Ϯy�� (�B�I�ƭp��)
        // ���F�קK�B�I�~�t�ɭP�X�ݶV�ɡA�̫�n clamp
        float scale_x = (float)src_w / (float)dst_w;
        float scale_y = (float)src_h / (float)dst_h;

        int src_x = (int)(x * scale_x);
        int src_y = (int)(y * scale_y);

        // ����ˬd (Clamp)
        if (src_x >= src_w) src_x = src_w - 1;
        if (src_y >= src_h) src_y = src_h - 1;

        dst[y * dst_w + x] = src[src_y * src_w + src_x];
    }

}}  // namespace core, tanuki