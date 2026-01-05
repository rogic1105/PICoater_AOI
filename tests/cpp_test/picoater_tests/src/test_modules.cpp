// PICoater_AOI\tests\cpp_test\picoater_tests\src\test_modules.cpp

#include "framework/test_utils.hpp"
#include "core_cv_tests/image_utils.hpp"
#include "framework/test_utils.hpp" 
#include "Module_GetPICoaterBackground.hpp"

#include "core_cv/base/cuda_utils.hpp"
#include "cpp_utils/timer_utils.hpp" 
#include "cpp_utils/terminal_colors.hpp"

void RunModuleTests(const std::string& imgPath) {
    std::cout << Color::CYAN << "\n========= Running Module Tests =========" << Color::RESET << "\n";

    try {
        TestImage src = LoadImageRaw(imgPath);
        std::vector<uint8_t> gray = ConvertToGray(src);
        int width = src.w;
        int height = src.h;
        size_t size = width * height;

        // 1. 準備 GPU 記憶體
        // 新增 d_ridge 用於存放脊線偵測結果
        uint8_t* d_in = nullptr;
        uint8_t* d_bg = nullptr;
        uint8_t* d_mura = nullptr;
        uint8_t* d_ridge = nullptr;

        checkCudaErrors(cudaMalloc(&d_in, size));
        checkCudaErrors(cudaMalloc(&d_bg, size));
        checkCudaErrors(cudaMalloc(&d_mura, size));
        checkCudaErrors(cudaMalloc(&d_ridge, size)); // Allocate Ridge

        checkCudaErrors(cudaMemcpy(d_in, gray.data(), size, cudaMemcpyHostToDevice));

        // 2. 設定參數
        float bgSigma = 2.0f;       // 背景計算 Sigma
        float ridgeSigma = 9.0f;    // Ridge 偵測 Sigma
        const char* ridgeMode = "vertical"; // 偵測方向 vertical = 0, horizontal = 1, both = 2
        // 3. 執行模組
        {
            // 測量模組整體演算法耗時 (含 Sync)
            TIME_SCOPE_MS_SYNC("Module: PICoater Background + Ridge (GPU)", cudaDeviceSynchronize());

            picoater::GetPICoaterBackground_gpu(
                d_in,
                d_bg,
                d_mura,
                d_ridge,    // 新增輸出參數
                width,
                height,
                bgSigma,
                ridgeSigma, // 新增參數
                ridgeMode,  // 新增參數
                0           // Stream
            );
        }

        // 4. 下載並存圖 - 背景 (Background)
        std::vector<uint8_t> h_bg(size);
        checkCudaErrors(cudaMemcpy(h_bg.data(), d_bg, size, cudaMemcpyDeviceToHost));

        std::string outPath1 = framework::GetOutputPath("picoater_tests", "picoater_background.bmp");
        stbi_write_bmp(outPath1.c_str(), width, height, 1, h_bg.data());
        std::cout << "[Save] " << outPath1 << "\n";

        // 5. 下載並存圖 - Mura (AbsDiff)
        std::vector<uint8_t> h_mura(size);
        checkCudaErrors(cudaMemcpy(h_mura.data(), d_mura, size, cudaMemcpyDeviceToHost));

        std::string outPath2 = framework::GetOutputPath("picoater_tests", "picoater_mura.bmp");
        stbi_write_bmp(outPath2.c_str(), width, height, 1, h_mura.data());
        std::cout << "[Save] " << outPath2 << "\n";

        // 6. 下載並存圖 - Ridge (New!)
        std::vector<uint8_t> h_ridge(size);
        checkCudaErrors(cudaMemcpy(h_ridge.data(), d_ridge, size, cudaMemcpyDeviceToHost));

        std::string outPath3 = framework::GetOutputPath("picoater_tests", "picoater_ridge.bmp");
        stbi_write_bmp(outPath3.c_str(), width, height, 1, h_ridge.data());
        std::cout << "[Save] " << outPath3 << "\n";

        // 7. 清理記憶體
        cudaFree(d_in);
        cudaFree(d_bg);
        cudaFree(d_mura);
        cudaFree(d_ridge); // Free Ridge

        std::cout << Color::GREEN << "Modules Tests Completed." << Color::RESET << "\n";

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}