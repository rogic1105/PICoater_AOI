// PICoater_AOI\tests\cpp_test\picoater_tests\src\PICoaterModuleTestsFast.cpp

#include "framework/test_utils.hpp"
#include "core_cv/base/cuda_utils.hpp"
#include "core_cv/base/cuda_memory.hpp"
#include "core_cv/imgcodecs/core_imgcodecs_fast.hpp"
#include "core_cv/imgproc/core_transform.hpp"


#include <stb/stb_image_write.h>

#include "cpp_utils/timer_utils.hpp"
#include "cpp_utils/terminal_colors.hpp"
#include "Module_GetPICoaterBackground.hpp"

#include <future> 
#include <iostream>


void PICoaterModuleTestsFast(const std::string& imgPath) {
    std::cout << Color::CYAN << "\n========= Running PICoater Module Tests (Fast IO) =========" << Color::RESET << "\n";

    // 1. 定義 Host Pinned Memory (輸入與輸出)
    uint8_t* h_pinned_in = nullptr;
    uint8_t* h_in = nullptr;
    uint8_t* h_pinned_bg = nullptr;
    uint8_t* h_pinned_mura = nullptr;
    uint8_t* h_pinned_ridge = nullptr;
    uint8_t* h_pinned_heatmap = nullptr; // [新增]
    float* h_pinned_mura_curve = nullptr;

    // 2. 定義 Device Memory (GPU)
    uint8_t* d_in = nullptr;
    uint8_t* d_bg = nullptr;
    uint8_t* d_mura = nullptr;
    uint8_t* d_ridge = nullptr;
    uint8_t* d_heatmap = nullptr; // [新增]
    float* d_mura_curve = nullptr;

    try {
        // 預估最大可能尺寸 (例如 16384 x 10000)
        size_t max_size = 16384 * 10000;
        h_pinned_in = (uint8_t*)core::alloc_pinned_memory(max_size);

        int w = 0, h = 0;

        // --- A. 極速讀圖測試 (Direct to Pinned) ---
        {
            TIME_SCOPE_MS("Fast Load BMP (SSD -> Pinned Memory)");
            if (!core::fast_read_bmp_8bit(imgPath, w, h, h_pinned_in, max_size)) {
                std::cerr << "Fast load failed! Check file path or format.\n";
                return;
            }
            std::cout << "Loaded: " << w << "x" << h << " (Gray 8-bit)\n";
        }

        size_t img_size = (size_t)w * h;
        size_t heatmap_size = img_size * 3; // [新增] RGB 3 channels

        // --- B. 分配 GPU 記憶體 ---
        checkCudaErrors(cudaMalloc(&d_in, img_size));
        checkCudaErrors(cudaMalloc(&d_bg, img_size));
        checkCudaErrors(cudaMalloc(&d_mura, img_size));
        checkCudaErrors(cudaMalloc(&d_ridge, img_size));
        checkCudaErrors(cudaMalloc(&d_mura_curve, w * sizeof(float)));

        // --- C. 分配輸出 Pinned Memory ---
        h_in = (uint8_t*)core::alloc_pinned_memory(img_size);
        h_pinned_bg = (uint8_t*)core::alloc_pinned_memory(img_size);
        h_pinned_mura = (uint8_t*)core::alloc_pinned_memory(img_size);
        h_pinned_ridge = (uint8_t*)core::alloc_pinned_memory(img_size);
        h_pinned_mura_curve = (float*)core::alloc_pinned_memory(w * sizeof(float));

        // --- D. 上傳圖片 (Pinned -> Device) ---
        {
            TIME_SCOPE_MS("Upload (Pinned -> Device)");
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

            detector.Run(
                d_in, 
                d_bg,
                d_mura, 
                d_ridge, 
                d_mura_curve,
                bgSigma, 
                ridgeSigma, 
                ridgeMode,
                0);
        }

        // --- F. 下載結果 (Device -> Pinned) ---
        {
            TIME_SCOPE_MS("Download Result (Device -> Host Pinned)");
            checkCudaErrors(cudaMemcpy(h_in, d_in, img_size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_pinned_bg, d_bg, img_size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_pinned_mura, d_mura, img_size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_pinned_ridge, d_ridge, img_size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_pinned_mura_curve, d_mura_curve, w * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // --- G. 極速存圖測試 (Parallel) ---
        {
            TIME_SCOPE_MS("Fast Save BMP (Parallel + Raw Write)");

            std::string outPath1 = framework::GetOutputPath("picoater_tests", "fast_ori.bmp");
            std::string outPath2 = framework::GetOutputPath("picoater_tests", "fast_bg.bmp");
            std::string outPath3 = framework::GetOutputPath("picoater_tests", "fast_mura.bmp");
            std::string outPath4 = framework::GetOutputPath("picoater_tests", "fast_ridge.bmp");
            std::string outPath5 = framework::GetOutputPath("picoater_tests", "fast_heatmap.bmp");

            auto f1 = std::async(std::launch::async, [&] {
                core::fast_write_bmp_8bit(outPath1, w, h, h_in);
                });
            auto f2 = std::async(std::launch::async, [&] {
                core::fast_write_bmp_8bit(outPath2, w, h, h_pinned_bg);
                });
            auto f3 = std::async(std::launch::async, [&] {
                core::fast_write_bmp_8bit(outPath3, w, h, h_pinned_mura);
                });
            auto f4 = std::async(std::launch::async, [&] {
                core::fast_write_bmp_8bit(outPath4, w, h, h_pinned_ridge);
                });
            auto f5 = std::async(std::launch::async, [&] {
                core::fast_write_bmp_24bit(outPath5, w, h, h_pinned_heatmap);
                });

            f2.get(); f3.get(); f4.get(); f5.get();
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
    if (d_heatmap) cudaFree(d_heatmap); // [新增]
    if (d_mura_curve) cudaFree(d_mura_curve);

    core::free_pinned_memory(h_pinned_in);
    core::free_pinned_memory(h_pinned_bg);
    core::free_pinned_memory(h_pinned_mura);
    core::free_pinned_memory(h_pinned_ridge);
    core::free_pinned_memory(h_pinned_heatmap); // [新增]
    core::free_pinned_memory(h_pinned_mura_curve);

    std::cout << Color::GREEN << "All Tests Finished." << Color::RESET << "\n";
}


struct CamContext {
    int id;
    picoater::PICoaterDetector detector;

    // GPU Buffers
    uint8_t* d_in = nullptr;
    uint8_t* d_bg = nullptr;
    uint8_t* d_mura = nullptr;
    uint8_t* d_ridge = nullptr;
    uint8_t* d_heatmap = nullptr; // [新增]
    float* d_mura_curve = nullptr;

    // Pinned Buffers
    uint8_t* h_pinned_in = nullptr;
    uint8_t* h_pinned_thumb = nullptr;
    float* h_pinned_mura_curve = nullptr;

    cudaStream_t stream = nullptr;

    int w = 0, h = 0;
    size_t img_size = 0;

    void Initialize(int width, int height) {
        w = width; h = height;
        img_size = w * h;

        checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        checkCudaErrors(cudaMalloc(&d_in, img_size));
        checkCudaErrors(cudaMalloc(&d_bg, img_size));
        checkCudaErrors(cudaMalloc(&d_mura, img_size));
        checkCudaErrors(cudaMalloc(&d_ridge, img_size));
        checkCudaErrors(cudaMalloc(&d_mura_curve, w * sizeof(float)));

        h_pinned_in = (uint8_t*)core::alloc_pinned_memory(img_size);
        h_pinned_thumb = (uint8_t*)core::alloc_pinned_memory(2000 * 2000);
        h_pinned_mura_curve = (float*)core::alloc_pinned_memory(w * sizeof(float));

        if (!h_pinned_in || !h_pinned_thumb || !h_pinned_mura_curve) {
            std::cerr << "CamContext " << id << ": Failed to allocate Host Pinned Memory!\n";
            throw std::runtime_error("Pinned Memory Allocation Failed");
        }

        detector.Initialize(w, h);
    }

    void Release() {
        if (d_in) cudaFree(d_in);
        if (d_bg) cudaFree(d_bg);
        if (d_mura) cudaFree(d_mura);
        if (d_ridge) cudaFree(d_ridge);
        if (d_heatmap) cudaFree(d_heatmap); // [新增]
        if (d_mura_curve) cudaFree(d_mura_curve);

        core::free_pinned_memory(h_pinned_in);
        core::free_pinned_memory(h_pinned_thumb);
        core::free_pinned_memory(h_pinned_mura_curve);

        if (stream) cudaStreamDestroy(stream);
    }
};

void PICoaterModuleTestsMultiThread(const std::string& imgPath, const int NUM_CAMS) {


    // 1. [修改] 預先讀取圖片到共用 Buffer (模擬 FrameGrabber 準備好資料)
    int w = 0, h = 0;
    uint8_t* shared_source_img = nullptr;
    size_t img_size = 0;

    {
        size_t max_size = 16384 * 10000;
        shared_source_img = (uint8_t*)core::alloc_pinned_memory(max_size); // 暫存區

        if (!core::fast_read_bmp_8bit(imgPath, w, h, shared_source_img, max_size)) {
            std::cerr << "Init failed: Cannot load image.\n";
            core::free_pinned_memory(shared_source_img);
            return;
        }
        img_size = (size_t)w * h;
        std::cout << "Image Pre-loaded: " << w << "x" << h << " (" << (img_size / 1024 / 1024) << " MB)\n";
    }

    const int THUMB_W = 1000;
    int thumb_h = (int)((float)h / w * THUMB_W);

    // 2. Context
    std::cout << Color::CYAN << "Initializing Shared GPU Resources (1 Context)..." << Color::RESET << "\n";

    CamContext shared_ctx; // 只宣告一個！
    shared_ctx.id = 0;
    shared_ctx.Initialize(w, h); // 這裡佔用約 2.5GB VRAM

    std::cout << "Starting Serial Execution with Resource Reuse...\n";

    // 暖身 (Warmup)
    {
        std::cout << "Warming up...\n";
        // 隨便跑一次讓 GPU 醒來
        std::memcpy(shared_ctx.h_pinned_in, shared_source_img, img_size);
        checkCudaErrors(cudaMemcpyAsync(shared_ctx.d_in, shared_ctx.h_pinned_in, shared_ctx.img_size, cudaMemcpyHostToDevice, shared_ctx.stream));
        shared_ctx.detector.Run(shared_ctx.d_in, shared_ctx.d_bg, shared_ctx.d_mura, shared_ctx.d_ridge, shared_ctx.d_mura_curve, 2.0f, 9.0f, "vertical", shared_ctx.stream);
        checkCudaErrors(cudaStreamSynchronize(shared_ctx.stream));
    }

    {
        TIME_SCOPE_MS("Total Time (Resource Reuse)");

        for (int i = 0; i < NUM_CAMS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            // 這裡直接使用 shared_ctx
            // 模擬：將資料複製進去
            std::memcpy(shared_ctx.h_pinned_in, shared_source_img, img_size);

            // A. Upload
            checkCudaErrors(cudaMemcpyAsync(shared_ctx.d_in, shared_ctx.h_pinned_in, shared_ctx.img_size, cudaMemcpyHostToDevice, shared_ctx.stream));

            // B. Run (GPU 內部的暫存區會被複寫，這在 AOI 是正常的，因為我們只關心當前的結果)
            shared_ctx.detector.Run(
                shared_ctx.d_in, shared_ctx.d_bg, shared_ctx.d_mura, shared_ctx.d_ridge, shared_ctx.d_mura_curve,
                2.0f, 9.0f, "vertical", shared_ctx.stream
            );

            // C. Resize
            uint8_t* d_thumb_temp = shared_ctx.d_bg;
            core::resize_u8_gpu(shared_ctx.d_mura, shared_ctx.w, shared_ctx.h, d_thumb_temp, 1000, thumb_h, shared_ctx.stream);

            // D. Download
            size_t thumb_size = 1000 * thumb_h;
            checkCudaErrors(cudaMemcpyAsync(shared_ctx.h_pinned_thumb, d_thumb_temp, thumb_size, cudaMemcpyDeviceToHost, shared_ctx.stream));
            // ... copy curve ...

            // E. Sync
            checkCudaErrors(cudaStreamSynchronize(shared_ctx.stream));

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << "Cam " << i + 1 << " (Reuse): " << ms << " ms\n";
        }
    }

    // 清理
    shared_ctx.Release();
    core::free_pinned_memory(shared_source_img);


}