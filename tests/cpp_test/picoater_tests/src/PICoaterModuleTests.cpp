// PICoater_AOI\tests\cpp_test\picoater_tests\src\PICoaterModuleTests.cpp

#include "framework/test_utils.hpp"

#include "core_cv_tests/image_utils.hpp"
#include "core_cv/base/cuda_utils.hpp"
#include "core_cv/base/cuda_memory.hpp"

#include "cpp_utils/timer_utils.hpp"
#include "cpp_utils/terminal_colors.hpp"

#include "Module_GetPICoaterBackground.hpp"


void PICoaterModuleTests(const std::string& imgPath) {
    std::cout << Color::CYAN << "\n========= Running PICoater Module Tests =========" << Color::RESET << "\n";

    // 定義 Pinned Memory 指標 (CPU 端)
    uint8_t* h_pinned_bg = nullptr;
    uint8_t* h_pinned_mura = nullptr;
    uint8_t* h_pinned_ridge = nullptr;

    // 定義 Device Memory 指標 (GPU 端)
    uint8_t* d_in = nullptr;
    uint8_t* d_bg = nullptr;
    uint8_t* d_mura = nullptr;
    uint8_t* d_ridge = nullptr;

    try {
        TestImage src = LoadImageRaw(imgPath);
        std::vector<uint8_t> gray = ConvertToGray(src);
        int width = src.w;
        int height = src.h;
        size_t size = width * height;

        // 1. 準備 GPU 記憶體 (這部分維持 cudaMalloc)
        checkCudaErrors(cudaMalloc(&d_in, size));
        checkCudaErrors(cudaMalloc(&d_bg, size));
        checkCudaErrors(cudaMalloc(&d_mura, size));
        checkCudaErrors(cudaMalloc(&d_ridge, size));

        // 2. 準備 CPU Pinned Memory (用來快速接圖)
        h_pinned_bg = (uint8_t*)core::alloc_pinned_memory(size);
        h_pinned_mura = (uint8_t*)core::alloc_pinned_memory(size);
        h_pinned_ridge = (uint8_t*)core::alloc_pinned_memory(size);

        // Input 也可以優化，但為了簡單先略過，只做 Output 優化
        checkCudaErrors(cudaMemcpy(d_in, gray.data(), size, cudaMemcpyHostToDevice));

        // 3. 執行模組
        float bgSigma = 2.0f;
        float ridgeSigma = 9.0f;
        const char* ridgeMode = "vertical";

        {
            picoater::PICoaterDetector detector;
            detector.Initialize(width, height);

            TIME_SCOPE_MS_SYNC("Module: PICoater Detector Run (GPU)", cudaDeviceSynchronize());

            detector.Run(d_in, d_bg, d_mura, d_ridge, bgSigma, ridgeSigma, ridgeMode, 0);
        }

        // 4. 下載結果 (這一步會變快！)
        // 因為 h_pinned_bg 是鎖頁記憶體，cudaMemcpy 會跑出全速 DMA
        checkCudaErrors(cudaMemcpy(h_pinned_bg, d_bg, size, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_pinned_mura, d_mura, size, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_pinned_ridge, d_ridge, size, cudaMemcpyDeviceToHost));

        // 存圖
        std::string outPath1 = framework::GetOutputPath("picoater_tests", "picoater_background.bmp");
        stbi_write_bmp(outPath1.c_str(), width, height, 1, h_pinned_bg); // 直接用 pinned pointer

        std::string outPath2 = framework::GetOutputPath("picoater_tests", "picoater_mura.bmp");
        stbi_write_bmp(outPath2.c_str(), width, height, 1, h_pinned_mura);

        std::string outPath3 = framework::GetOutputPath("picoater_tests", "picoater_ridge.bmp");
        stbi_write_bmp(outPath3.c_str(), width, height, 1, h_pinned_ridge);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    // 清理 GPU
    if (d_in) cudaFree(d_in);
    if (d_bg) cudaFree(d_bg);
    if (d_mura) cudaFree(d_mura);
    if (d_ridge) cudaFree(d_ridge);

    // [修改] 清理 CPU Pinned Memory
    core::free_pinned_memory(h_pinned_bg);
    core::free_pinned_memory(h_pinned_mura);
    core::free_pinned_memory(h_pinned_ridge);

    std::cout << Color::GREEN << "Modules Tests Completed." << Color::RESET << "\n";
}