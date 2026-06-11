
#include "tanuki/core/imgcodecs/core_imgcodecs_fast.hpp" // [�s�W] Fast IO
#include "tanuki/core/base/cuda_memory.hpp"             // [�s�W] Pinned Memory
#include "tanuki/core/base/cuda_utils.hpp"

#include "tanuki/core/imgproc/core_background.hpp"
#include "tanuki/core/imgproc/core_enhance.hpp"
#include "tanuki/core/imgproc/core_features.hpp"
#include "tanuki/core/imgproc/core_filters.hpp"
#include "tanuki/core/imgproc/core_utils.hpp"

#include "tanuki/utils/timer_utils.hpp"
#include "tanuki/utils/terminal_colors.hpp"
#include "bench_framework/test_utils.hpp"

#include <vector>
#include <future> // [�s�W] ����s��

void RunCoreTests(const std::string& imgPath) {
    std::cout << tanuki::utils::CYAN << "\n========= Running Core Tests (Fast IO) =========" << tanuki::utils::RESET << "\n";

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
            // �w���̤j�i��j�p (�Ҧp 16K * 10K)�A�������t
            size_t max_alloc_size = 16384 * 10000;
            h_pinned_in = (uint8_t*)tanuki::core::alloc_pinned_memory(max_alloc_size);

            if (!tanuki::core::fast_read_bmp_8bit(imgPath, W, H, h_pinned_in, max_alloc_size)) {
                throw std::runtime_error("Fast load failed");
            }
            std::cout << "Loaded: " << W << "x" << H << "\n";
        }

        size_t size = W * H;

        // 2. Allocate Output Pinned Memory
        h_pinned_out_bright = (uint8_t*)tanuki::core::alloc_pinned_memory(size);
        h_pinned_out_thresh = (uint8_t*)tanuki::core::alloc_pinned_memory(size);
        h_pinned_out_conv = (uint8_t*)tanuki::core::alloc_pinned_memory(size);

        // 3. GPU Alloc & Upload
        checkCudaErrors(cudaMalloc(&d_in, size));
        checkCudaErrors(cudaMalloc(&d_out, size));

        {
            TIME_SCOPE_MS("Memcpy H2D (Pinned)");
            // Pinned Memory �|Ĳ�o DMA �ǿ�A�t�׷���
            checkCudaErrors(cudaMemcpy(d_in, h_pinned_in, size, cudaMemcpyHostToDevice));
        }

        // --- Test 1: Brighten ---
        {
            TIME_SCOPE_MS_SYNC("Core: Brighten (GPU)", cudaDeviceSynchronize());
            tanuki::core::brighten_u8_gpu(d_in, d_out, W, H, 50, 0);
        }
        checkCudaErrors(cudaMemcpy(h_pinned_out_bright, d_out, size, cudaMemcpyDeviceToHost));

        // --- Test 2: Threshold ---
        {
            TIME_SCOPE_MS_SYNC("Core: Threshold (GPU)", cudaDeviceSynchronize());
            tanuki::core::threshold_u8_gpu(d_in, d_out, W, H, 128, 0);
        }
        checkCudaErrors(cudaMemcpy(h_pinned_out_thresh, d_out, size, cudaMemcpyDeviceToHost));

        // --- Test 3: Convolution (Sharpen) ---
        float h_mask[] = { 0, 0, 0, -1, 2, -1, 0, 0, 0 };
        checkCudaErrors(cudaMalloc(&d_mask, 9 * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_mask, h_mask, 9 * sizeof(float), cudaMemcpyHostToDevice));

        {
            TIME_SCOPE_MS_SYNC("Core: Convolution 3x3 (GPU)", cudaDeviceSynchronize());
            tanuki::core::convolution_u8_gpu(d_in, d_out, W, H, d_mask, 3, 0);
        }
        checkCudaErrors(cudaMemcpy(h_pinned_out_conv, d_out, size, cudaMemcpyDeviceToHost));

        // --- 4. Fast Save (Parallel) ---
        {
            TIME_SCOPE_MS("Fast Save (Parallel)");

            std::string outPath1 = bench_framework::GetOutputPath("core_cv_benchmark", "out_core_brighten.bmp");
            std::string outPath2 = bench_framework::GetOutputPath("core_cv_benchmark", "out_core_threshold.bmp");
            std::string outPath3 = bench_framework::GetOutputPath("core_cv_benchmark", "out_core_convolution.bmp");

            auto f1 = std::async(std::launch::async, [&] { tanuki::core::fast_write_bmp_8bit(outPath1, W, H, h_pinned_out_bright); });
            auto f2 = std::async(std::launch::async, [&] { tanuki::core::fast_write_bmp_8bit(outPath2, W, H, h_pinned_out_thresh); });
            auto f3 = std::async(std::launch::async, [&] { tanuki::core::fast_write_bmp_8bit(outPath3, W, H, h_pinned_out_conv); });

            f1.get(); f2.get(); f3.get();
        }

        std::cout << tanuki::utils::GREEN << "Core Tests Completed." << tanuki::utils::RESET << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Core Test Failed: " << e.what() << "\n";
    }

    // Cleanup
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (d_mask) cudaFree(d_mask);

    tanuki::core::free_pinned_memory(h_pinned_in);
    tanuki::core::free_pinned_memory(h_pinned_out_bright);
    tanuki::core::free_pinned_memory(h_pinned_out_thresh);
    tanuki::core::free_pinned_memory(h_pinned_out_conv);
}