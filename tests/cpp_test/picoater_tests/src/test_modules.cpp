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
        int W = src.w;
        int H = src.h;
        size_t size = W * H;

        // 準備 GPU 記憶體
        uint8_t* d_in = nullptr, * d_bg = nullptr, * d_mura = nullptr;
        checkCudaErrors(cudaMalloc(&d_in, size));
        checkCudaErrors(cudaMalloc(&d_bg, size));
        checkCudaErrors(cudaMalloc(&d_mura, size));

        checkCudaErrors(cudaMemcpy(d_in, gray.data(), size, cudaMemcpyHostToDevice));

        // 執行模組 (Sigma=2.0)
        {
            // 這是最重要的部分：測量模組整體演算法耗時
            // 因為模組內部可能有多個 Kernel，加上 Sync 才能確保全部跑完
            TIME_SCOPE_MS_SYNC("Module: PICoater Background (GPU)", cudaDeviceSynchronize());

            picoater::GetPICoaterBackground_gpu(d_in, d_bg, d_mura, W, H, 2.0f, 0);
        }

        // 下載並存圖 - 背景
        std::vector<uint8_t> h_bg(size);
        checkCudaErrors(cudaMemcpy(h_bg.data(), d_bg, size, cudaMemcpyDeviceToHost));

        std::string outPath1 = framework::GetOutputPath("picoater_tests", "picoater_background.bmp");
        stbi_write_bmp(outPath1.c_str(), W, H, 1, h_bg.data());
        std::cout << "[Save] " << outPath1 << "\n";


        // 下載並存圖 - Mura
        std::vector<uint8_t> h_mura(size);
        checkCudaErrors(cudaMemcpy(h_mura.data(), d_mura, size, cudaMemcpyDeviceToHost));

        std::string outPath2 = framework::GetOutputPath("picoater_tests", "picoater_mura.bmp");
        stbi_write_bmp(outPath2.c_str(), W, H, 1, h_mura.data());
        std::cout << "[Save] " << outPath2 << "\n";


        cudaFree(d_in);
        cudaFree(d_bg);
        cudaFree(d_mura);


        std::cout << Color::GREEN << "Modulese Tests Completed." << Color::RESET << "\n";

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}