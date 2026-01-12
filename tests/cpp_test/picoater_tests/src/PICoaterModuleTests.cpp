// PICoater_AOI\tests\cpp_test\picoater_tests\src\PICoaterModuleTests.cpp

#include "framework/test_utils.hpp"
#include "core_cv/base/cuda_utils.hpp"
#include "core_cv/base/cuda_memory.hpp"
#include "core_cv/imgcodecs/core_imgcodecs.hpp" 

#include "cpp_utils/timer_utils.hpp"
#include "cpp_utils/terminal_colors.hpp"
#include "Module_GetPICoaterBackground.hpp"

#include <future> 

void PICoaterModuleTests(const std::string& imgPath) {
    std::cout << Color::CYAN << "\n========= Running PICoater Module Tests =========" << Color::RESET << "\n";

    uint8_t* h_pinned_bg = nullptr;
    uint8_t* h_pinned_mura = nullptr;
    uint8_t* h_pinned_ridge = nullptr;

    uint8_t* d_in = nullptr;
    uint8_t* d_bg = nullptr;
    uint8_t* d_mura = nullptr;
    uint8_t* d_ridge = nullptr;

    try {
        // [修改] 使用新的 core::Image
        core::Image src;

        {
            TIME_SCOPE_MS("Total Loading Time");

            // [關鍵優化] 參數 '1' 代表強制讀取為灰階 (1 channel)
            // 這樣 STB 內部解碼時就直接處理好，我們不需要再跑 ConvertToGray
            // 速度會快很多，且資料量直接少 2/3
            src = core::imread(imgPath, 1);

            if (src.empty()) throw std::runtime_error("Load failed");

            std::cout << "[Load Info] " << src.w << "x" << src.h << ", channels=" << src.c << "\n";
        }

        size_t size = src.w * src.h; // 因為是灰階，所以 size = w*h

        checkCudaErrors(cudaMalloc(&d_in, size));
        checkCudaErrors(cudaMalloc(&d_bg, size));
        checkCudaErrors(cudaMalloc(&d_mura, size));
        checkCudaErrors(cudaMalloc(&d_ridge, size));

        h_pinned_bg = (uint8_t*)core::alloc_pinned_memory(size);
        h_pinned_mura = (uint8_t*)core::alloc_pinned_memory(size);
        h_pinned_ridge = (uint8_t*)core::alloc_pinned_memory(size);

        // Upload Input
        {
            TIME_SCOPE_MS("  -> cudaMemcpy (Host to Device)");
            checkCudaErrors(cudaMemcpy(d_in, src.data.data(), size, cudaMemcpyHostToDevice));
        }

        // Run Module
        float bgSigma = 2.0f;
        float ridgeSigma = 9.0f;
        const char* ridgeMode = "vertical";

        {
            picoater::PICoaterDetector detector;
            detector.Initialize(src.w, src.h);

            TIME_SCOPE_MS_SYNC("Module: PICoater Detector Run (GPU)", cudaDeviceSynchronize());
            detector.Run(d_in, d_bg, d_mura, d_ridge, bgSigma, ridgeSigma, ridgeMode, 0);
        }

        // Download Result
        {
            TIME_SCOPE_MS("Download Result (Device -> Host Pinned)");
            checkCudaErrors(cudaMemcpy(h_pinned_bg, d_bg, size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_pinned_mura, d_mura, size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_pinned_ridge, d_ridge, size, cudaMemcpyDeviceToHost));
        }

        // [關鍵優化] 平行存檔
        // 因為你要存原圖(大圖)，這非常吃 I/O，平行化可以讓多個 SSD 通道同時運作
        {
            TIME_SCOPE_MS("Total Saving Time (Parallel)");

            std::string outPath1 = framework::GetOutputPath("picoater_tests", "picoater_background.bmp");
            std::string outPath2 = framework::GetOutputPath("picoater_tests", "picoater_mura.bmp");
            std::string outPath3 = framework::GetOutputPath("picoater_tests", "picoater_ridge.bmp");

            // 使用 std::async 啟動三個非同步任務
            // 注意：imwrite 第三個參數是 1，代表存成 8-bit Gray BMP (速度最快)
            auto f1 = std::async(std::launch::async, [&] {
                core::imwrite(outPath1, src.w, src.h, 1, h_pinned_bg);
                });

            auto f2 = std::async(std::launch::async, [&] {
                core::imwrite(outPath2, src.w, src.h, 1, h_pinned_mura);
                });

            auto f3 = std::async(std::launch::async, [&] {
                core::imwrite(outPath3, src.w, src.h, 1, h_pinned_ridge);
                });

            // 等待三個任務都完成
            f1.get();
            f2.get();
            f3.get();
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    if (d_in) cudaFree(d_in);
    if (d_bg) cudaFree(d_bg);
    if (d_mura) cudaFree(d_mura);
    if (d_ridge) cudaFree(d_ridge);

    core::free_pinned_memory(h_pinned_bg);
    core::free_pinned_memory(h_pinned_mura);
    core::free_pinned_memory(h_pinned_ridge);

    std::cout << Color::GREEN << "Modules Tests Completed." << Color::RESET << "\n";

}