// AOI_SDK\core_cv_tests\src\test_core.cpp

#include "core_cv/imgcodecs/core_imgcodecs_fast.hpp" // [新增] Fast IO
#include "core_cv/base/cuda_memory.hpp"             // [新增] Pinned Memory
#include "core_cv/base/cuda_utils.hpp"

#include "core_cv/imgproc/core_background.hpp"
#include "core_cv/imgproc/core_enhance.hpp"
#include "core_cv/imgproc/core_features.hpp"
#include "core_cv/imgproc/core_filters.hpp"
#include "core_cv/imgproc/core_utils.hpp"

#include "cpp_utils/timer_utils.hpp"
#include "cpp_utils/terminal_colors.hpp"
#include "framework/test_utils.hpp"

#include <vector>
#include <future> // [新增] 平行存檔

void RunCoreTests(const std::string& imgPath) {
    std::cout << Color::CYAN << "\n========= Running Core Tests (Fast IO) =========" << Color::RESET << "\n";

    // Host Pinned Pointers
    uint8_t* h_pinned_in = nullptr;
    uint8_t* h_pinned_out_bright = nullptr;
    uint8_t* h_pinned_out_thresh = nullptr;
    uint8_t* h_pinned_out_conv = nullptr;

    // Device Pointers
    uint8_t* d_in = nullptr;
    uint8_t* d_out = nullptr;
    float* d_mask = nullptr;

    try {
        int W = 0, H = 0;

        // 1. Fast Load (SSD -> Pinned Memory)
        {
            TIME_SCOPE_MS("Fast Load BMP");
            // 預估最大可能大小 (例如 16K * 10K)，直接分配
            size_t max_alloc_size = 16384 * 10000;
            h_pinned_in = (uint8_t*)core::alloc_pinned_memory(max_alloc_size);

            if (!core::fast_read_bmp_8bit(imgPath, W, H, h_pinned_in, max_alloc_size)) {
                throw std::runtime_error("Fast load failed");
            }
            std::cout << "Loaded: " << W << "x" << H << "\n";
        }

        size_t size = W * H;

        // 2. Allocate Output Pinned Memory
        h_pinned_out_bright = (uint8_t*)core::alloc_pinned_memory(size);
        h_pinned_out_thresh = (uint8_t*)core::alloc_pinned_memory(size);
        h_pinned_out_conv = (uint8_t*)core::alloc_pinned_memory(size);

        // 3. GPU Alloc & Upload
        checkCudaErrors(cudaMalloc(&d_in, size));
        checkCudaErrors(cudaMalloc(&d_out, size));

        {
            TIME_SCOPE_MS("Memcpy H2D (Pinned)");
            // Pinned Memory 會觸發 DMA 傳輸，速度極快
            checkCudaErrors(cudaMemcpy(d_in, h_pinned_in, size, cudaMemcpyHostToDevice));
        }

        // --- Test 1: Brighten ---
        {
            TIME_SCOPE_MS_SYNC("Core: Brighten (GPU)", cudaDeviceSynchronize());
            core::brighten_u8_gpu(d_in, d_out, W, H, 50, 0);
        }
        checkCudaErrors(cudaMemcpy(h_pinned_out_bright, d_out, size, cudaMemcpyDeviceToHost));

        // --- Test 2: Threshold ---
        {
            TIME_SCOPE_MS_SYNC("Core: Threshold (GPU)", cudaDeviceSynchronize());
            core::threshold_u8_gpu(d_in, d_out, W, H, 128, 0);
        }
        checkCudaErrors(cudaMemcpy(h_pinned_out_thresh, d_out, size, cudaMemcpyDeviceToHost));

        // --- Test 3: Convolution (Sharpen) ---
        float h_mask[] = { 0, 0, 0, -1, 2, -1, 0, 0, 0 };
        checkCudaErrors(cudaMalloc(&d_mask, 9 * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_mask, h_mask, 9 * sizeof(float), cudaMemcpyHostToDevice));

        {
            TIME_SCOPE_MS_SYNC("Core: Convolution 3x3 (GPU)", cudaDeviceSynchronize());
            core::convolution_u8_gpu(d_in, d_out, W, H, d_mask, 3, 0);
        }
        checkCudaErrors(cudaMemcpy(h_pinned_out_conv, d_out, size, cudaMemcpyDeviceToHost));

        // --- 4. Fast Save (Parallel) ---
        {
            TIME_SCOPE_MS("Fast Save (Parallel)");

            std::string outPath1 = framework::GetOutputPath("core_cv_tests", "out_core_brighten.bmp");
            std::string outPath2 = framework::GetOutputPath("core_cv_tests", "out_core_threshold.bmp");
            std::string outPath3 = framework::GetOutputPath("core_cv_tests", "out_core_convolution.bmp");

            auto f1 = std::async(std::launch::async, [&] { core::fast_write_bmp_8bit(outPath1, W, H, h_pinned_out_bright); });
            auto f2 = std::async(std::launch::async, [&] { core::fast_write_bmp_8bit(outPath2, W, H, h_pinned_out_thresh); });
            auto f3 = std::async(std::launch::async, [&] { core::fast_write_bmp_8bit(outPath3, W, H, h_pinned_out_conv); });

            f1.get(); f2.get(); f3.get();
        }

        std::cout << Color::GREEN << "Core Tests Completed." << Color::RESET << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Core Test Failed: " << e.what() << "\n";
    }

    // Cleanup
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (d_mask) cudaFree(d_mask);

    core::free_pinned_memory(h_pinned_in);
    core::free_pinned_memory(h_pinned_out_bright);
    core::free_pinned_memory(h_pinned_out_thresh);
    core::free_pinned_memory(h_pinned_out_conv);
}