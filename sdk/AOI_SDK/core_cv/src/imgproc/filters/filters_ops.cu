// AOI_SDK\core_cv\src\imgproc\filters\filters_ops.cu

#include "core_cv/base/cuda_utils.hpp"
#include "core_cv/imgproc/core_utils.hpp"


#include "filters_kernels.cuh"
#include <vector>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


namespace core {

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

        // 確保 ksize 為奇數
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
            CUDA_CHECK(cudaMalloc(&d_mask, ksize * sizeof(float))); // 這裡直接分配 ksize 大小即可
        }

        // 準備 Mask Host Data
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

        // [修正] 移除原本這裡錯誤的 cudaMalloc(&d_mask...)
        // 只做 Memcpy。如果 d_mask 是 workspace，就 copy 到 workspace；如果是 malloc，就 copy 到 malloc。
        CUDA_CHECK(cudaMemcpyAsync(d_mask, h_kernel.data(), ksize * sizeof(float), cudaMemcpyHostToDevice, s));

        // 3. 轉 Float
        core::convert_u8_to_f32_gpu(d_in, d_f32_in, num_pixels, s);

        // 4. 分離式卷積
        get_optimal_launch_2d(k_gaussianBlurRow, W, H, grid2d, block2d);
        k_gaussianBlurRow << <grid2d, block2d, 0, s >> > (d_f32_in, d_f32_temp, W, H, d_mask, ksize);
        k_gaussianBlurCol << <grid2d, block2d, 0, s >> > (d_f32_temp, d_f32_out, W, H, d_mask, ksize);

        // 5. 轉回 Uint8
        core::convert_f32_to_u8_clamp_gpu(d_f32_out, d_out, num_pixels, s);

        // 6. 清理
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

        // 1. 參數檢查與初始化
        int num_pixels = width * height;
        dim3 grid2d, block2d;

        // 確保 ksize 為奇數
        if (ksize % 2 == 0) ksize++;

        // 2. 定義核心運算需要的 float 指標
        const float* d_f32_src = nullptr;
        float* d_f32_dst = nullptr;
        float* d_f32_temp = nullptr;
        float* d_mask = nullptr;

        // 3. 記憶體管理 (Workspace) 
        // 使用 uint8_t* 方便進行 byte-level 的指標運算
        uint8_t* ptr_ws = (uint8_t*)d_workspace;
        std::vector<float*> temp_allocs;

        // [關鍵修正] 強制 256-byte 對齊分配器
        // 確保每一塊記憶體的起始位置都是 256 的倍數，防止 float4 讀取錯位
        auto get_aligned_buffer = [&](size_t count) -> float* {
            size_t size_bytes = count * sizeof(float);

            // 計算對齊後的 size (向上取整到 256 bytes)
            const size_t ALIGNMENT = 256;
            size_t aligned_size = (size_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

            if (ptr_ws) {
                // 確保當前指標也是對齊的 (如果是第一塊，通常傳進來的就是對齊的)
                // 這裡簡單假設 ptr_ws 初始位置是好的，我們只負責移動
                float* p = (float*)ptr_ws;
                ptr_ws += aligned_size; // 移動對齊後的距離
                return p;
            }
            else {
                float* p = nullptr;
                CUDA_CHECK(cudaMalloc(&p, aligned_size)); // cudaMalloc 預設就是 256 byte aligned
                temp_allocs.push_back(p);
                return p;
            }
            };

        // 4. 分配記憶體 (注意順序！)

        // (A) 先處理輸入來源 (Input Source)
        // 如果輸入是 u8，我們需要一塊 float buffer 來放轉換後的資料
        float* alloc_in = nullptr;
        if constexpr (std::is_same<T_in, float>::value) {
            d_f32_src = (const float*)d_in;
        }
        else {
            alloc_in = get_aligned_buffer(num_pixels); // [修正順序] 先分配這個
            d_f32_src = alloc_in;
        }

        // (B) 再處理中間暫存 (Temp Buffer)
        // [修正順序] 第二分配這個。
        // 如果 d_out 和 d_workspace 重疊，先分配 src 再分配 temp 可以避免寫入衝突
        d_f32_temp = get_aligned_buffer(num_pixels);

        // (C) 處理輸出目標 (Output Destination)
        float* alloc_out = nullptr;
        if constexpr (std::is_same<T_out, float>::value) {
            d_f32_dst = (float*)d_out;
        }
        else {
            alloc_out = get_aligned_buffer(num_pixels);
            d_f32_dst = alloc_out;
        }

        // (D) 最後才分配 Mask
        // Mask 很小，放在最後面最安全，不會影響前面大塊影像的對齊
        d_mask = get_aligned_buffer(ksize);


        // 5. 準備 Gaussian Kernel (Host -> Device)
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

        // 6. 輸入轉換 (u8 -> f32)
        if constexpr (std::is_same<T_in, uint8_t>::value) {
            core::convert_u8_to_f32_gpu((const uint8_t*)d_in, (float*)d_f32_src, num_pixels, stream);
        }

        // 7. 執行核心 (Row & Col)
        // 確保 Kernel 參數順序正確: (Input, Output, Width, Height, ...)
        get_optimal_launch_2d(k_gaussianBlurRow, width, height, grid2d, block2d);

        k_gaussianBlurRow << < grid2d, block2d, 0, stream >> > (d_f32_src, d_f32_temp, width, height, d_mask, ksize);
        k_gaussianBlurCol << < grid2d, block2d, 0, stream >> > (d_f32_temp, d_f32_dst, width, height, d_mask, ksize);

        // 8. 輸出轉換 (f32 -> u8)
        if constexpr (std::is_same<T_out, uint8_t>::value) {
            core::convert_f32_to_u8_clamp_gpu(d_f32_dst, (uint8_t*)d_out, num_pixels, stream);
        }

        // 9. 清理 (若使用 cudaMalloc)
        for (float* p : temp_allocs) {
            CUDA_CHECK(cudaFree(p));
        }
    }
    // =========================================================
    // [關鍵] 顯式實例化 (Explicit Instantiation)
    // 這會強制編譯器生成這四種組合的代碼，讓 Linker 找得到。
    // =========================================================
    template void gaussianBlur_gpu<uint8_t, uint8_t>(const uint8_t*, uint8_t*, int, int, float, int, cudaStream_t, void*);
    template void gaussianBlur_gpu<uint8_t, float>(const uint8_t*, float*, int, int, float, int, cudaStream_t, void*);
    template void gaussianBlur_gpu<float, float>(const float*, float*, int, int, float, int, cudaStream_t, void*);
    template void gaussianBlur_gpu<float, uint8_t>(const float*, uint8_t*, int, int, float, int, cudaStream_t, void*);



}