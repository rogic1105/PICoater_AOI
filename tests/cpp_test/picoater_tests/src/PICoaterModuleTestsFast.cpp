// PICoater_AOI\tests\cpp_test\picoater_tests\src\PICoaterModuleTestsFast.cpp

#include "framework/test_utils.hpp"
#include "core_cv/base/cuda_utils.hpp"
#include "core_cv/base/cuda_memory.hpp"
#include "core_cv/imgcodecs/core_imgcodecs_fast.hpp"
#include "core_cv/imgproc/core_transform.hpp"

#include "cpp_utils/timer_utils.hpp"
#include "cpp_utils/terminal_colors.hpp"
#include "Module_GetPICoaterBackground.hpp"

#include <future> 
#include <iostream>


void PICoaterModuleTestsFast(const std::string& imgPath) {
    std::cout << Color::CYAN << "\n========= Running PICoater Module Tests (Fast IO) =========" << Color::RESET << "\n";

    // 1. 定義 Host Pinned Memory (輸入與輸出)
    uint8_t* h_pinned_in = nullptr;
    uint8_t* h_pinned_bg = nullptr;
    uint8_t* h_pinned_mura = nullptr;
    uint8_t* h_pinned_ridge = nullptr;

    // 2. 定義 Device Memory (GPU)
    uint8_t* d_in = nullptr;
    uint8_t* d_bg = nullptr;
    uint8_t* d_mura = nullptr;
    uint8_t* d_ridge = nullptr;

    try {
        // 預估最大可能尺寸 (例如 16384 x 10000)，先分配夠大的 Pinned Memory
        // 這樣讀圖時就不用 realloc
        size_t max_size = 16384 * 10000;
        h_pinned_in = (uint8_t*)core::alloc_pinned_memory(max_size);

        int w = 0, h = 0;

        // --- A. 極速讀圖測試 (Direct to Pinned) ---
        {
            TIME_SCOPE_MS("Fast Load BMP (SSD -> Pinned Memory)");
            // 直接從 SSD 讀到 Pinned Memory，不經過 CPU Cache/Vector
            if (!core::fast_read_bmp_8bit(imgPath, w, h, h_pinned_in, max_size)) {
                std::cerr << "Fast load failed! Check file path or format.\n";
                return;
            }
            std::cout << "Loaded: " << w << "x" << h << " (Gray 8-bit)\n";
        }

        size_t img_size = (size_t)w * h;

        // --- B. 分配 GPU 記憶體 ---
        checkCudaErrors(cudaMalloc(&d_in, img_size));
        checkCudaErrors(cudaMalloc(&d_bg, img_size));
        checkCudaErrors(cudaMalloc(&d_mura, img_size));
        checkCudaErrors(cudaMalloc(&d_ridge, img_size));

        // --- C. 分配輸出 Pinned Memory ---
        h_pinned_bg = (uint8_t*)core::alloc_pinned_memory(img_size);
        h_pinned_mura = (uint8_t*)core::alloc_pinned_memory(img_size);
        h_pinned_ridge = (uint8_t*)core::alloc_pinned_memory(img_size);

        // --- D. 上傳圖片 (Pinned -> Device) ---
        {
            TIME_SCOPE_MS("Upload (Pinned -> Device)");
            // 因為是 Pinned Memory，這步會跑全速 DMA
            checkCudaErrors(cudaMemcpy(d_in, h_pinned_in, img_size, cudaMemcpyHostToDevice));
        }

        // --- E. 執行 AOI 模組 ---
        float bgSigma = 2.0f;
        float ridgeSigma = 9.0f;
        const char* ridgeMode = "vertical";

        {
            picoater::PICoaterDetector detector;
            detector.Initialize(w, h);

            TIME_SCOPE_MS_SYNC("Module: PICoater Detector Run (GPU)", cudaDeviceSynchronize());
            detector.Run(d_in, d_bg, d_mura, d_ridge, bgSigma, ridgeSigma, ridgeMode, 0);
        }

        // --- F. 下載結果 (Device -> Pinned) ---
        {
            TIME_SCOPE_MS("Download Result (Device -> Host Pinned)");
            checkCudaErrors(cudaMemcpy(h_pinned_bg, d_bg, img_size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_pinned_mura, d_mura, img_size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_pinned_ridge, d_ridge, img_size, cudaMemcpyDeviceToHost));
        }

        // --- G. 極速存圖測試 (Parallel) ---
        {
            TIME_SCOPE_MS("Fast Save BMP (Parallel + Raw Write)");

            std::string outPath1 = framework::GetOutputPath("picoater_tests", "fast_bg.bmp");
            std::string outPath2 = framework::GetOutputPath("picoater_tests", "fast_mura.bmp");
            std::string outPath3 = framework::GetOutputPath("picoater_tests", "fast_ridge.bmp");

            auto f1 = std::async(std::launch::async, [&] {
                core::fast_write_bmp_8bit(outPath1, w, h, h_pinned_bg);
                });
            auto f2 = std::async(std::launch::async, [&] {
                core::fast_write_bmp_8bit(outPath2, w, h, h_pinned_mura);
                });
            auto f3 = std::async(std::launch::async, [&] {
                core::fast_write_bmp_8bit(outPath3, w, h, h_pinned_ridge);
                });

            f1.get(); f2.get(); f3.get();
        }

        std::cout << Color::GREEN << "AOI Pipeline Test Completed." << Color::RESET << "\n\n";


        // --- H. 額外測試: 單純 IO 極限測試 ---
        // 為了驗證頻寬，我們模擬存 3 張原始圖片 (input) 到硬碟
        std::cout << Color::CYAN << "--- Running Extra IO Stress Test ---" << Color::RESET << "\n";

        {
            TIME_SCOPE_MS("Stress Test: Save 3 Raw Images (Parallel)");
            std::string ioTest1 = framework::GetOutputPath("picoater_tests", "stress_1.bmp");
            std::string ioTest2 = framework::GetOutputPath("picoater_tests", "stress_2.bmp");
            std::string ioTest3 = framework::GetOutputPath("picoater_tests", "stress_3.bmp");

            // 注意：這裡必須用 h_pinned_in (Host Memory)，不能用 d_in (Device Memory)
            // fast_write_bmp_8bit 只能讀取 CPU 記憶體

            auto f1 = std::async(std::launch::async, [&] {
                core::fast_write_bmp_8bit(ioTest1, w, h, h_pinned_in);
                });
            auto f2 = std::async(std::launch::async, [&] {
                core::fast_write_bmp_8bit(ioTest2, w, h, h_pinned_in);
                });
            auto f3 = std::async(std::launch::async, [&] {
                core::fast_write_bmp_8bit(ioTest3, w, h, h_pinned_in);
                });

            f1.get(); f2.get(); f3.get();
        }

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    // 清理資源
    if (d_in) cudaFree(d_in);
    if (d_bg) cudaFree(d_bg);
    if (d_mura) cudaFree(d_mura);
    if (d_ridge) cudaFree(d_ridge);

    core::free_pinned_memory(h_pinned_in); // 別忘了釋放 Input
    core::free_pinned_memory(h_pinned_bg);
    core::free_pinned_memory(h_pinned_mura);
    core::free_pinned_memory(h_pinned_ridge);

    std::cout << Color::GREEN << "All Tests Finished." << Color::RESET << "\n";
}


// [新增] 模擬 7 顆相機的 Context
struct CamContext {
    int id;
    picoater::PICoaterDetector detector;

    // GPU Buffers
    uint8_t* d_in = nullptr;
    uint8_t* d_bg = nullptr;
    uint8_t* d_mura = nullptr;
    uint8_t* d_ridge = nullptr;

    // Pinned Buffers (Input & Output)
    uint8_t* h_pinned_in = nullptr;
    uint8_t* h_pinned_thumb = nullptr;

    // Stream
    cudaStream_t stream = nullptr;

    int w = 0, h = 0;
    size_t img_size = 0;

    void Initialize(int width, int height) {
        w = width; h = height;
        img_size = w * h;

        // 1. 建立 Stream
        checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        // 2. 分配 GPU 記憶體
        checkCudaErrors(cudaMalloc(&d_in, img_size));
        checkCudaErrors(cudaMalloc(&d_bg, img_size));
        checkCudaErrors(cudaMalloc(&d_mura, img_size));
        checkCudaErrors(cudaMalloc(&d_ridge, img_size));

        // 3. 分配 Pinned Memory
        h_pinned_in = (uint8_t*)core::alloc_pinned_memory(img_size);
        h_pinned_thumb = (uint8_t*)core::alloc_pinned_memory(2000 * 2000); // 預留夠大的縮圖空間

        // 4. 初始化 Detector
        detector.Initialize(w, h);
    }

    void Release() {
        if (d_in) cudaFree(d_in);
        if (d_bg) cudaFree(d_bg);
        if (d_mura) cudaFree(d_mura);
        if (d_ridge) cudaFree(d_ridge);

        core::free_pinned_memory(h_pinned_in);
        core::free_pinned_memory(h_pinned_thumb);

        if (stream) cudaStreamDestroy(stream);
    }
};

void PICoaterModuleTestsMultiThread(const std::string& imgPath) {
    const int NUM_CAMS = 7;
    const int THUMB_W = 1000;

    std::cout << Color::CYAN << "Initializing " << NUM_CAMS << " Cameras..." << Color::RESET << "\n";

    // 1. 讀取圖片尺寸 (只讀一次 header 或第一張圖)
    int w = 0, h = 0;
    // 這裡我們作弊一下，先讀一次拿到尺寸，模擬系統已知規格
    // 實際上每顆相機都一樣大
    {
        uint8_t* temp = nullptr;
        // 為了簡單，這裡用 stbi 讀一下 header 或是假定已知
        // 為了嚴謹，我們先用 fast_read 讀到一個臨時 buffer
        size_t max_size = 16384 * 10000;
        uint8_t* temp_pinned = (uint8_t*)core::alloc_pinned_memory(max_size);
        if (!core::fast_read_bmp_8bit(imgPath, w, h, temp_pinned, max_size)) {
            std::cerr << "Init failed.\n"; return;
        }
        core::free_pinned_memory(temp_pinned);
    }
    int thumb_h = (int)((float)h / w * THUMB_W);

    // 2. 準備 7 個 Context (模擬 C# 的 _algoPool)
    std::vector<CamContext> cams(NUM_CAMS);
    for (int i = 0; i < NUM_CAMS; ++i) {
        cams[i].id = i;
        cams[i].Initialize(w, h);
    }

    std::cout << "Image Size: " << w << "x" << h << "\n";
    std::cout << "Starting Parallel Execution...\n";

    // 3. 執行平行測試
    {
        TIME_SCOPE_MS("Total Time (7 Cams Parallel)");

        std::vector<std::future<double>> futures;

        for (int i = 0; i < NUM_CAMS; ++i) {
            futures.push_back(std::async(std::launch::async, [&](int idx) -> double {
                // 每個執行緒內部的計時器
                auto start = std::chrono::high_resolution_clock::now();

                CamContext& ctx = cams[idx];

                // A. 讀圖 (IO)
                // 為了模擬真實情況，我們讓每個 thread 讀同一張圖 (或者你可以準備 7 張不同的圖)
                // 注意：多執行緒讀同一個檔案，OS 的 File Cache 會介入，速度可能會比讀 7 個不同檔案快一點
                // 但如果是 NVMe，差異不大
                if (!core::fast_read_bmp_8bit(imgPath, ctx.w, ctx.h, ctx.h_pinned_in, ctx.img_size)) {
                    return 0.0;
                }

                // B. 上傳 (Async)
                checkCudaErrors(cudaMemcpyAsync(ctx.d_in, ctx.h_pinned_in, ctx.img_size, cudaMemcpyHostToDevice, ctx.stream));

                // C. 運算 (Async)
                ctx.detector.Run(ctx.d_in, ctx.d_bg, ctx.d_mura, ctx.d_ridge,
                    2.0f, 9.0f, "vertical", ctx.stream);

                // D. GPU 縮圖 (Async)
                // 借用 d_bg 放縮圖
                uint8_t* d_thumb_temp = ctx.d_bg;
                core::resize_u8_gpu(ctx.d_mura, ctx.w, ctx.h, d_thumb_temp, THUMB_W, thumb_h, ctx.stream);

                // E. 下載縮圖 (Async)
                size_t thumb_size = THUMB_W * thumb_h;
                checkCudaErrors(cudaMemcpyAsync(ctx.h_pinned_thumb, d_thumb_temp, thumb_size, cudaMemcpyDeviceToHost, ctx.stream));

                // F. 等待完成
                checkCudaErrors(cudaStreamSynchronize(ctx.stream));

                auto end = std::chrono::high_resolution_clock::now();
                return std::chrono::duration<double, std::milli>(end - start).count();
                }, i));
        }

        // 等待所有執行緒完成並收集時間
        for (int i = 0; i < NUM_CAMS; ++i) {
            double ms = futures[i].get();
            std::cout << "Cam " << i + 1 << ": " << ms << " ms\n";
        }
    }

    // 4. 清理
    for (int i = 0; i < NUM_CAMS; ++i) {
        cams[i].Release();
    }
}