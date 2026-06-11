
#include "enhance_kernels.cuh"
#include <cmath>

namespace tanuki { namespace core {

    //  �G�� (1D)
    __global__ void k_brighten_u8(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int N, int bright) {
        // ���� 1D ����
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < N) {
            int v = (int)in[idx] + bright;
            if (v < 0) v = 0; else if (v > 255) v = 255;
            out[idx] = (uint8_t)v;
        }
    }

    // �G�Ȥ� (1D)
    __global__ void k_threshold_u8(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int N, uint8_t thresh) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            out[idx] = (in[idx] >= thresh) ? 255 : 0;
        }
    }

    // ���� (1D)
    __global__ void k_invert_u8(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            out[idx] = (uint8_t)(255 - in[idx]);
        }
    }

}}  // namespace core, tanuki