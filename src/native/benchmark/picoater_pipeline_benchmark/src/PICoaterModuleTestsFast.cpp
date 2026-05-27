// PICoater_AOI\tests\cpp_test\picoater_pipeline_benchmark\src\PICoaterModuleTestsFast.cpp

#include "bench_framework/test_utils.hpp"
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

    // 1. �w�q Host Pinned Memory (��J�P��X)
    uint8_t* h_pinned_in = nullptr;
    uint8_t* h_in = nullptr;
    uint8_t* h_pinned_bg = nullptr;
    uint8_t* h_pinned_mura = nullptr;
    uint8_t* h_pinned_ridge = nullptr;
    uint8_t* h_pinned_heatmap = nullptr; // [�s�W]
    float* h_pinned_mura_mean = nullptr;
	float* h_pinned_mura_max = nullptr;

    // 2. �w�q Device Memory (GPU)
    uint8_t* d_in = nullptr;
    uint8_t* d_bg = nullptr;
    uint8_t* d_mura = nullptr;
    uint8_t* d_ridge = nullptr;
    uint8_t* d_heatmap = nullptr; // [�s�W]
    float* d_mura_curve_mean = nullptr;
	float* d_mura_curve_max = nullptr;
    float* d_mura_row_curve_mean = nullptr;
    float* d_mura_row_curve_max = nullptr;

    try {
        // �w���̤j�i��ؤo (�Ҧp 16384 x 10000)
        size_t max_size = 16384 * 10000;
        h_pinned_in = (uint8_t*)core::alloc_pinned_memory(max_size);

        int w = 0, h = 0;

        // --- A. ���tŪ�ϴ��� (Direct to Pinned) ---
        {
            TIME_SCOPE_MS("Fast Load BMP (SSD -> Pinned Memory)");
            if (!core::fast_read_bmp_8bit(imgPath, w, h, h_pinned_in, max_size)) {
                std::cerr << "Fast load failed! Check file path or format.\n";
                return;
            }
            std::cout << "Loaded: " << w << "x" << h << " (Gray 8-bit)\n";
        }

        size_t img_size = (size_t)w * h;
        size_t heatmap_size = img_size * 3; // [�s�W] RGB 3 channels

        // --- B. ���t GPU �O���� ---
        checkCudaErrors(cudaMalloc(&d_in, img_size));
        checkCudaErrors(cudaMalloc(&d_bg, img_size));
        checkCudaErrors(cudaMalloc(&d_mura, img_size));
        checkCudaErrors(cudaMalloc(&d_ridge, img_size));
        checkCudaErrors(cudaMalloc(&d_mura_curve_mean, w * sizeof(float)));
		checkCudaErrors(cudaMalloc(&d_mura_curve_max, w * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_mura_row_curve_mean, h * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_mura_row_curve_max, h * sizeof(float)));

        // --- C. ���t��X Pinned Memory ---
        h_in = (uint8_t*)core::alloc_pinned_memory(img_size);
        h_pinned_bg = (uint8_t*)core::alloc_pinned_memory(img_size);
        h_pinned_mura = (uint8_t*)core::alloc_pinned_memory(img_size);
        h_pinned_ridge = (uint8_t*)core::alloc_pinned_memory(img_size);
        h_pinned_mura_mean = (float*)core::alloc_pinned_memory(w * sizeof(float));
        h_pinned_mura_max = (float*)core::alloc_pinned_memory(w * sizeof(float));

        // --- D. �W�ǹϤ� (Pinned -> Device) ---
        {
            TIME_SCOPE_MS("Upload (Pinned -> Device)");
            checkCudaErrors(cudaMemcpy(d_in, h_pinned_in, img_size, cudaMemcpyHostToDevice));
        }

        // --- E. ���� AOI �Ҳ� ---
        float bgSigma = 2.0f;
        float ridgeSigma = 9.0f;
		float hessianMaxFactor = 1.0f;
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
                d_mura_curve_mean,
                d_mura_curve_max,
                d_mura_row_curve_mean,
                d_mura_row_curve_max,
                bgSigma,
                ridgeSigma,
                hessianMaxFactor,
                ridgeMode,
                nullptr, 0);
        }

        // --- F. �U�����G (Device -> Pinned) ---
        {
            TIME_SCOPE_MS("Download Result (Device -> Host Pinned)");
            checkCudaErrors(cudaMemcpy(h_in, d_in, img_size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_pinned_bg, d_bg, img_size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_pinned_mura, d_mura, img_size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_pinned_ridge, d_ridge, img_size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_pinned_mura_mean, d_mura_curve_mean, w * sizeof(float), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(h_pinned_mura_max, d_mura_curve_max, w * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // --- G. ���t�s�ϴ��� (Parallel) ---
        {
            TIME_SCOPE_MS("Fast Save BMP (Parallel + Raw Write)");

            std::string outPath1 = bench_framework::GetOutputPath("picoater_pipeline_benchmark", "fast_ori.bmp");
            std::string outPath2 = bench_framework::GetOutputPath("picoater_pipeline_benchmark", "fast_bg.bmp");
            std::string outPath3 = bench_framework::GetOutputPath("picoater_pipeline_benchmark", "fast_mura.bmp");
            std::string outPath4 = bench_framework::GetOutputPath("picoater_pipeline_benchmark", "fast_ridge.bmp");
            std::string outPath5 = bench_framework::GetOutputPath("picoater_pipeline_benchmark", "fast_heatmap.bmp");

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

        // --- I. CPU SIMD Implementation Test (New) ---
                // ���� CPU ���� (AVX2 + OpenMP) ���B�⵲�G�P�s��
        std::cout << Color::CYAN << "\n--- Running CPU SIMD Implementation Test ---" << Color::RESET << "\n";
        {
            // 1. ���t CPU ��X�w�İ�
            uint8_t* h_cpu_mura = (uint8_t*)core::alloc_pinned_memory(img_size);

            // 2. �إ� Detector ���
            picoater::PICoaterDetector detector_cpu;
            detector_cpu.Initialize(w, h); // ��l�� (�]�w���e)

            // 3. ���� RunCPU (�u�] Column Mean & Background ��B)
            // h_pinned_in �O�w�g���J����l�v��
            {
                TIME_SCOPE_MS("Module: PICoater Detector Run (CPU SIMD)");
                detector_cpu.RunCPU(h_pinned_in, h_cpu_mura, bgSigma);
            }

            // 4. �s������
            {
                TIME_SCOPE_MS("Save CPU Result BMP");
                std::string outPathCPU = bench_framework::GetOutputPath("picoater_pipeline_benchmark", "cpu_simd_mura.bmp");
                core::fast_write_bmp_8bit(outPathCPU, w, h, h_cpu_mura);
                std::cout << "Saved CPU result to: " << outPathCPU << "\n";
            }

            // �M�z���a�귽
            core::free_pinned_memory(h_cpu_mura);
        }

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    // �M�z�귽
    if (d_in) cudaFree(d_in);
    if (d_bg) cudaFree(d_bg);
    if (d_mura) cudaFree(d_mura);
    if (d_ridge) cudaFree(d_ridge);
    if (d_heatmap) cudaFree(d_heatmap); // [�s�W]
    if (d_mura_curve_mean) cudaFree(d_mura_curve_mean);

    core::free_pinned_memory(h_pinned_in);
    core::free_pinned_memory(h_pinned_bg);
    core::free_pinned_memory(h_pinned_mura);
    core::free_pinned_memory(h_pinned_ridge);
    core::free_pinned_memory(h_pinned_heatmap); // [�s�W]
    core::free_pinned_memory(h_pinned_mura_mean);

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
    uint8_t* d_heatmap = nullptr; // [�s�W]
    float* d_mura_curve_mean = nullptr;
	float* d_mura_curve_max = nullptr;
    float* d_mura_row_curve_mean = nullptr;
    float* d_mura_row_curve_max = nullptr;

    // Pinned Buffers
    uint8_t* h_pinned_in = nullptr;
    uint8_t* h_pinned_thumb = nullptr;
    float* h_pinned_mura_mean = nullptr;
	float* h_pinned_mura_max = nullptr;

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
        checkCudaErrors(cudaMalloc(&d_mura_curve_mean, w * sizeof(float)));
		checkCudaErrors(cudaMalloc(&d_mura_curve_max, w * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_mura_row_curve_mean, h * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_mura_row_curve_max, h * sizeof(float)));

        h_pinned_in = (uint8_t*)core::alloc_pinned_memory(img_size);
        h_pinned_thumb = (uint8_t*)core::alloc_pinned_memory(2000 * 2000);
        h_pinned_mura_mean = (float*)core::alloc_pinned_memory(w * sizeof(float));
        h_pinned_mura_max = (float*)core::alloc_pinned_memory(w * sizeof(float));

        if (!h_pinned_in || !h_pinned_thumb || !h_pinned_mura_mean) {
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
        if (d_heatmap) cudaFree(d_heatmap); // [�s�W]
        if (d_mura_curve_mean) cudaFree(d_mura_curve_mean);
        if (d_mura_curve_max) cudaFree(d_mura_curve_max);
        if (d_mura_row_curve_mean) cudaFree(d_mura_row_curve_mean);
        if (d_mura_row_curve_max) cudaFree(d_mura_row_curve_max);

        core::free_pinned_memory(h_pinned_in);
        core::free_pinned_memory(h_pinned_thumb);
        core::free_pinned_memory(h_pinned_mura_mean);

        if (stream) cudaStreamDestroy(stream);
    }
};

void PICoaterModuleTestsMultiThread(const std::string& imgPath, const int NUM_CAMS) {


    // 1. [�ק�] �w��Ū���Ϥ���@�� Buffer (���� FrameGrabber �ǳƦn���)
    int w = 0, h = 0;
    uint8_t* shared_source_img = nullptr;
    size_t img_size = 0;

    {
        size_t max_size = 16384 * 10000;
        shared_source_img = (uint8_t*)core::alloc_pinned_memory(max_size); // �Ȧs��

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

    CamContext shared_ctx; // �u�ŧi�@�ӡI
    shared_ctx.id = 0;
    shared_ctx.Initialize(w, h); // �o�̦��ά� 2.5GB VRAM

    std::cout << "Starting Serial Execution with Resource Reuse...\n";

    // �x�� (Warmup)
    {
        std::cout << "Warming up...\n";
        // �H�K�]�@���� GPU ����
        std::memcpy(shared_ctx.h_pinned_in, shared_source_img, img_size);
        checkCudaErrors(cudaMemcpyAsync(shared_ctx.d_in, shared_ctx.h_pinned_in, shared_ctx.img_size, cudaMemcpyHostToDevice, shared_ctx.stream));
        shared_ctx.detector.Run(
            shared_ctx.d_in, shared_ctx.d_bg, shared_ctx.d_mura, shared_ctx.d_ridge,
            shared_ctx.d_mura_curve_mean, shared_ctx.d_mura_curve_max,
            shared_ctx.d_mura_row_curve_mean, shared_ctx.d_mura_row_curve_max,
            2.0f, 9.0f, 1.0f, "vertical", nullptr, shared_ctx.stream
        );
        checkCudaErrors(cudaStreamSynchronize(shared_ctx.stream));
    }

    {
        TIME_SCOPE_MS("Total Time (Resource Reuse)");

        for (int i = 0; i < NUM_CAMS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            // �o�̪����ϥ� shared_ctx
            // �����G�N��ƽƻs�i�h
            std::memcpy(shared_ctx.h_pinned_in, shared_source_img, img_size);

            // A. Upload
            checkCudaErrors(cudaMemcpyAsync(shared_ctx.d_in, shared_ctx.h_pinned_in, shared_ctx.img_size, cudaMemcpyHostToDevice, shared_ctx.stream));

            // B. Run (GPU �������Ȧs�Ϸ|�Q�Ƽg�A�o�b AOI �O���`���A�]���ڭ̥u���߷��e�����G)
            shared_ctx.detector.Run(
                shared_ctx.d_in, shared_ctx.d_bg, shared_ctx.d_mura, shared_ctx.d_ridge,
                shared_ctx.d_mura_curve_mean, shared_ctx.d_mura_curve_max,
                shared_ctx.d_mura_row_curve_mean, shared_ctx.d_mura_row_curve_max,
                2.0f, 9.0f, 1.0f, "vertical", nullptr, shared_ctx.stream
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

    // �M�z
    shared_ctx.Release();
    core::free_pinned_memory(shared_source_img);


}