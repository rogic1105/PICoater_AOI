
#include "tanuki/core/base/cuda_utils.hpp"
#include "tanuki/core/imgproc/core_utils.hpp"


#include "filters_kernels.cuh"
#include <vector>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


namespace tanuki { namespace core {

    void convolution_u8_gpu(const uint8_t* d_in, uint8_t* d_out, int W, int H, const float* d_mask, int maskSize, cudaStream_t s) {
        dim3 grid, block;
        get_optimal_launch_2d(k_convolution_u8, W, H, grid, block);
        k_convolution_u8 << <grid, block, 0, s >> > (d_in, d_out, W, H, d_mask, maskSize);
        CUDA_CHECK(cudaGetLastError());
    }

    void gaussianBlur_u8_gpu(const uint8_t* d_in, uint8_t* d_out, int W, int H, float sigma, int ksize, cudaStream_t s, void* d_workspace) {
        int num_pixels = W * H;
        dim3 grid2d, block2d;

        float* d_f32_in = nullptr;
        float* d_f32_temp = nullptr;
        float* d_f32_out = nullptr;
        float* d_mask = nullptr;

        bool need_free = false;

        // �T�O ksize ���_��
        if (ksize % 2 == 0) ksize++;

        if (d_workspace != nullptr) {
            uint8_t* ptr = (uint8_t*)d_workspace;
            d_f32_in = (float*)(ptr);
            d_f32_temp = (float*)(ptr + num_pixels * sizeof(float));
            d_f32_out = (float*)(ptr + 2 * num_pixels * sizeof(float));
            d_mask = (float*)(ptr + 3 * num_pixels * sizeof(float));
        }
        else {
            need_free = true;
            CUDA_CHECK(cudaMalloc(&d_f32_in, num_pixels * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_f32_temp, num_pixels * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_f32_out, num_pixels * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_mask, ksize * sizeof(float))); // �o�̪������t ksize �j�p�Y�i
        }

        // �ǳ� Mask Host Data
        std::vector<float> h_kernel(ksize);
        float sum = 0.0f;
        int r = ksize / 2;
        float two_sigma_sq = 2.0f * sigma * sigma;

        for (int i = 0; i < ksize; ++i) {
            int x = i - r;
            h_kernel[i] = expf(-(x * x) / two_sigma_sq);
            sum += h_kernel[i];
        }
        for (int i = 0; i < ksize; ++i) h_kernel[i] /= sum;

        // [�ץ�] �����쥻�o�̿��~�� cudaMalloc(&d_mask...)
        // �u�� Memcpy�C�p�G d_mask �O workspace�A�N copy �� workspace�F�p�G�O malloc�A�N copy �� malloc�C
        CUDA_CHECK(cudaMemcpyAsync(d_mask, h_kernel.data(), ksize * sizeof(float), cudaMemcpyHostToDevice, s));

        // 3. �� Float
        tanuki::core::convert_u8_to_f32_gpu(d_in, d_f32_in, num_pixels, s);

        // 4. ���������n
        get_optimal_launch_2d(k_gaussianBlurRow, W, H, grid2d, block2d);
        k_gaussianBlurRow << <grid2d, block2d, 0, s >> > (d_f32_in, d_f32_temp, W, H, d_mask, ksize);
        k_gaussianBlurCol << <grid2d, block2d, 0, s >> > (d_f32_temp, d_f32_out, W, H, d_mask, ksize);

        // 5. ��^ Uint8
        tanuki::core::convert_f32_to_u8_clamp_gpu(d_f32_out, d_out, num_pixels, s);

        // 6. �M�z
        if (need_free) {
            CUDA_CHECK(cudaFree(d_f32_in));
            CUDA_CHECK(cudaFree(d_f32_temp));
            CUDA_CHECK(cudaFree(d_f32_out));
            CUDA_CHECK(cudaFree(d_mask));
        }
    }

    template <typename T_in, typename T_out>
    void gaussianBlur_gpu(const T_in* d_in, T_out* d_out, int width, int height,
        float sigma, int ksize, cudaStream_t stream, void* d_workspace) {

        // 1. �Ѽ��ˬd�P��l��
        int num_pixels = width * height;
        dim3 grid2d, block2d;

        // �T�O ksize ���_��
        if (ksize % 2 == 0) ksize++;

        // 2. �w�q�֤߹B��ݭn�� float ����
        const float* d_f32_src = nullptr;
        float* d_f32_dst = nullptr;
        float* d_f32_temp = nullptr;
        float* d_mask = nullptr;

        // 3. �O����޲z (Workspace) 
        // �ϥ� uint8_t* ��K�i�� byte-level �����йB��
        uint8_t* ptr_ws = (uint8_t*)d_workspace;
        std::vector<float*> temp_allocs;

        // [����ץ�] �j�� 256-byte �����t��
        // �T�O�C�@��O���骺�_�l��m���O 256 �����ơA���� float4 Ū������
        auto get_aligned_buffer = [&](size_t count) -> float* {
            size_t size_bytes = count * sizeof(float);

            // �p����᪺ size (�V�W����� 256 bytes)
            const size_t ALIGNMENT = 256;
            size_t aligned_size = (size_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

            if (ptr_ws) {
                // �T�O��e���Ф]�O���� (�p�G�O�Ĥ@��A�q�`�Ƕi�Ӫ��N�O����)
                // �o��²�氲�] ptr_ws ��l��m�O�n���A�ڭ̥u�t�d����
                float* p = (float*)ptr_ws;
                ptr_ws += aligned_size; // ���ʹ��᪺�Z��
                return p;
            }
            else {
                float* p = nullptr;
                CUDA_CHECK(cudaMalloc(&p, aligned_size)); // cudaMalloc �w�]�N�O 256 byte aligned
                temp_allocs.push_back(p);
                return p;
            }
            };

        // 4. ���t�O���� (�`�N���ǡI)

        // (A) ���B�z��J�ӷ� (Input Source)
        // �p�G��J�O u8�A�ڭ̻ݭn�@�� float buffer �ө��ഫ�᪺���
        float* alloc_in = nullptr;
        if constexpr (std::is_same<T_in, float>::value) {
            d_f32_src = (const float*)d_in;
        }
        else {
            alloc_in = get_aligned_buffer(num_pixels); // [�ץ�����] �����t�o��
            d_f32_src = alloc_in;
        }

        // (B) �A�B�z�����Ȧs (Temp Buffer)
        // [�ץ�����] �ĤG���t�o�ӡC
        // �p�G d_out �M d_workspace ���|�A�����t src �A���t temp �i�H�קK�g�J�Ĭ�
        d_f32_temp = get_aligned_buffer(num_pixels);

        // (C) �B�z��X�ؼ� (Output Destination)
        float* alloc_out = nullptr;
        if constexpr (std::is_same<T_out, float>::value) {
            d_f32_dst = (float*)d_out;
        }
        else {
            alloc_out = get_aligned_buffer(num_pixels);
            d_f32_dst = alloc_out;
        }

        // (D) �̫�~���t Mask
        // Mask �ܤp�A��b�̫᭱�̦w���A���|�v�T�e���j��v�������
        d_mask = get_aligned_buffer(ksize);


        // 5. �ǳ� Gaussian Kernel (Host -> Device)
        std::vector<float> h_kernel(ksize);
        float sum = 0.0f;
        int r = ksize / 2;
        float two_sigma_sq = 2.0f * sigma * sigma;
        for (int i = 0; i < ksize; ++i) {
            int x = i - r;
            h_kernel[i] = expf(-(x * x) / two_sigma_sq);
            sum += h_kernel[i];
        }
        for (int i = 0; i < ksize; ++i) h_kernel[i] /= sum;

        CUDA_CHECK(cudaMemcpyAsync(d_mask, h_kernel.data(), ksize * sizeof(float), cudaMemcpyHostToDevice, stream));

        // 6. ��J�ഫ (u8 -> f32)
        if constexpr (std::is_same<T_in, uint8_t>::value) {
            tanuki::core::convert_u8_to_f32_gpu((const uint8_t*)d_in, (float*)d_f32_src, num_pixels, stream);
        }

        // 7. ����֤� (Row & Col)
        // �T�O Kernel �Ѽƶ��ǥ��T: (Input, Output, Width, Height, ...)
        get_optimal_launch_2d(k_gaussianBlurRow, width, height, grid2d, block2d);

        k_gaussianBlurRow << < grid2d, block2d, 0, stream >> > (d_f32_src, d_f32_temp, width, height, d_mask, ksize);
        k_gaussianBlurCol << < grid2d, block2d, 0, stream >> > (d_f32_temp, d_f32_dst, width, height, d_mask, ksize);

        // 8. ��X�ഫ (f32 -> u8)
        if constexpr (std::is_same<T_out, uint8_t>::value) {
            tanuki::core::convert_f32_to_u8_clamp_gpu(d_f32_dst, (uint8_t*)d_out, num_pixels, stream);
        }

        // 9. �M�z (�Y�ϥ� cudaMalloc)
        for (float* p : temp_allocs) {
            CUDA_CHECK(cudaFree(p));
        }
    }
    // =========================================================
    // [����] �㦡��Ҥ� (Explicit Instantiation)
    // �o�|�j��sĶ���ͦ��o�|�زզX���N�X�A�� Linker ��o��C
    // =========================================================
    template void gaussianBlur_gpu<uint8_t, uint8_t>(const uint8_t*, uint8_t*, int, int, float, int, cudaStream_t, void*);
    template void gaussianBlur_gpu<uint8_t, float>(const uint8_t*, float*, int, int, float, int, cudaStream_t, void*);
    template void gaussianBlur_gpu<float, float>(const float*, float*, int, int, float, int, cudaStream_t, void*);
    template void gaussianBlur_gpu<float, uint8_t>(const float*, uint8_t*, int, int, float, int, cudaStream_t, void*);



}}  // namespace core, tanuki