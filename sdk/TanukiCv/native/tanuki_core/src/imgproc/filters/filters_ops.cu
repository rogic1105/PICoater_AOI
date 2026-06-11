
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
            CUDA_CHECK(cudaMalloc(&d_mask, ksize * sizeof(float)));
        }


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



        CUDA_CHECK(cudaMemcpyAsync(d_mask, h_kernel.data(), ksize * sizeof(float), cudaMemcpyHostToDevice, s));


        tanuki::core::convert_u8_to_f32_gpu(d_in, d_f32_in, num_pixels, s);


        get_optimal_launch_2d(k_gaussianBlurRow, W, H, grid2d, block2d);
        k_gaussianBlurRow << <grid2d, block2d, 0, s >> > (d_f32_in, d_f32_temp, W, H, d_mask, ksize);
        k_gaussianBlurCol << <grid2d, block2d, 0, s >> > (d_f32_temp, d_f32_out, W, H, d_mask, ksize);


        tanuki::core::convert_f32_to_u8_clamp_gpu(d_f32_out, d_out, num_pixels, s);


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


        int num_pixels = width * height;
        dim3 grid2d, block2d;


        if (ksize % 2 == 0) ksize++;


        const float* d_f32_src = nullptr;
        float* d_f32_dst = nullptr;
        float* d_f32_temp = nullptr;
        float* d_mask = nullptr;



        uint8_t* ptr_ws = (uint8_t*)d_workspace;
        std::vector<float*> temp_allocs;



        auto get_aligned_buffer = [&](size_t count) -> float* {
            size_t size_bytes = count * sizeof(float);


            const size_t ALIGNMENT = 256;
            size_t aligned_size = (size_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

            if (ptr_ws) {


                float* p = (float*)ptr_ws;
                ptr_ws += aligned_size;
                return p;
            }
            else {
                float* p = nullptr;
                CUDA_CHECK(cudaMalloc(&p, aligned_size));
                temp_allocs.push_back(p);
                return p;
            }
            };





        float* alloc_in = nullptr;
        if constexpr (std::is_same<T_in, float>::value) {
            d_f32_src = (const float*)d_in;
        }
        else {
            alloc_in = get_aligned_buffer(num_pixels);
            d_f32_src = alloc_in;
        }




        d_f32_temp = get_aligned_buffer(num_pixels);


        float* alloc_out = nullptr;
        if constexpr (std::is_same<T_out, float>::value) {
            d_f32_dst = (float*)d_out;
        }
        else {
            alloc_out = get_aligned_buffer(num_pixels);
            d_f32_dst = alloc_out;
        }



        d_mask = get_aligned_buffer(ksize);



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


        if constexpr (std::is_same<T_in, uint8_t>::value) {
            tanuki::core::convert_u8_to_f32_gpu((const uint8_t*)d_in, (float*)d_f32_src, num_pixels, stream);
        }



        get_optimal_launch_2d(k_gaussianBlurRow, width, height, grid2d, block2d);

        k_gaussianBlurRow << < grid2d, block2d, 0, stream >> > (d_f32_src, d_f32_temp, width, height, d_mask, ksize);
        k_gaussianBlurCol << < grid2d, block2d, 0, stream >> > (d_f32_temp, d_f32_dst, width, height, d_mask, ksize);


        if constexpr (std::is_same<T_out, uint8_t>::value) {
            tanuki::core::convert_f32_to_u8_clamp_gpu(d_f32_dst, (uint8_t*)d_out, num_pixels, stream);
        }


        for (float* p : temp_allocs) {
            CUDA_CHECK(cudaFree(p));
        }
    }
    // =========================================================


    // =========================================================
    template void gaussianBlur_gpu<uint8_t, uint8_t>(const uint8_t*, uint8_t*, int, int, float, int, cudaStream_t, void*);
    template void gaussianBlur_gpu<uint8_t, float>(const uint8_t*, float*, int, int, float, int, cudaStream_t, void*);
    template void gaussianBlur_gpu<float, float>(const float*, float*, int, int, float, int, cudaStream_t, void*);
    template void gaussianBlur_gpu<float, uint8_t>(const float*, uint8_t*, int, int, float, int, cudaStream_t, void*);



}}  // namespace core, tanuki