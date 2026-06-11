
#include "utils_kernels.cuh"
#include "tanuki/core/base/cuda_utils.hpp"
#include <cmath>




namespace tanuki { namespace core {

    __global__ void k_zeroBorder_u8(uint8_t* __restrict__ in, int roiW, int roiH, int t) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= roiW || y >= roiH) return;
        if (x < t || x >= roiW - t || y < t || y >= roiH - t) {
            in[y * roiW + x] = 0;
        }
    }

    __global__ void k_f32_to_u8_clamp(const float* __restrict__ in, uint8_t* __restrict__ out, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;

        float val = in[idx];
        if (val < 0.0f) val = 0.0f;
        else if (val > 255.0f) val = 255.0f;

        out[idx] = (uint8_t)(val + 0.5f); // �|�����J
    }

    __global__ void k_scale_clamp_f32_to_u8(const float* src, uint8_t* dst, int num_pixels, float scale_factor) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_pixels) return;

        float val = src[idx] * scale_factor;

        // Clipping (0.0 ~ 255.0)
        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;

        // astype(np.uint8) �欰 (Truncation)
        dst[idx] = (uint8_t)val;
    }

    __global__ void k_u8_to_f32(const uint8_t* __restrict__ in, float* __restrict__ out, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) out[idx] = (float)in[idx];
    }

    __global__ void k_normalizeMinMax_f32_u8(const float* __restrict__ in, uint8_t* __restrict__ out, int N, float minVal, float maxVal) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;

        float range = maxVal - minVal;
        if (range < 1e-6f) {
            out[idx] = 0;
        }
        else {
            float val = (in[idx] - minVal) / range * 255.0f;
            if (val < 0.0f) val = 0.0f;
            if (val > 255.0f) val = 255.0f;
            out[idx] = (uint8_t)val;
        }
    }

    __device__ inline void get_jet_color(uint8_t v, float& b, float& g, float& r) {
        float val = v / 255.0f;

        // Jet �������
        // Base value = 1.5 - |4*val - shift|
        float b_val = 1.5f - fabsf(4.0f * val - 1.0f);
        float g_val = 1.5f - fabsf(4.0f * val - 2.0f);
        float r_val = 1.5f - fabsf(4.0f * val - 3.0f);

        // Clamp to [0, 1]
        b = fmaxf(0.0f, fminf(1.0f, b_val));
        g = fmaxf(0.0f, fminf(1.0f, g_val));
        r = fmaxf(0.0f, fminf(1.0f, r_val));

    }


    __global__ void k_overlay_heatmap(
        const uint8_t* __restrict__ src,
        const uint8_t* __restrict__ overlay,
        uint8_t* __restrict__ dst,
        int width, int height,
        int lower_limit,
        float alpha
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) return;

        int idx = y * width + x;

        uint8_t src_val = src[idx];
        uint8_t ov_val = overlay[idx];

        // �޿�: mask_indices = (overlay_image <= lower_limit)
        // �Y�b mask ���A�u��ܭ�� (�� BGR)
        if (ov_val <= lower_limit) {
            dst[idx * 3 + 0] = src_val; // B
            dst[idx * 3 + 1] = src_val; // G
            dst[idx * 3 + 2] = src_val; // R
        }
        else {
            // �p�� Heatmap �C��
            float h_r, h_g, h_b;
            get_jet_color(ov_val, h_r, h_g, h_b); // 0.0~1.0

            // �V�X: result = src * alpha + heatmap * (1 - alpha)
            // Python�N�X�� src_bgr �O�Ƕ��ন��BGR�A�ҥH src_r = src_g = src_b = src_val
            float beta = 1.0f - alpha;
            float s_v = (float)src_val;

            // B channel
            float out_b = s_v * alpha + (h_b * 255.0f) * beta;
            // G channel
            float out_g = s_v * alpha + (h_g * 255.0f) * beta;
            // R channel
            float out_r = s_v * alpha + (h_r * 255.0f) * beta;

            // �g�^ (Clamp 0-255)
            dst[idx * 3 + 0] = (uint8_t)fminf(255.0f, fmaxf(0.0f, out_b));
            dst[idx * 3 + 1] = (uint8_t)fminf(255.0f, fmaxf(0.0f, out_g));
            dst[idx * 3 + 2] = (uint8_t)fminf(255.0f, fmaxf(0.0f, out_r));
        }

    }




}}  // namespace core, tanuki