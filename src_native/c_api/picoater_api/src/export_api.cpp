// AOI_SDK\src_native\c_api\picoater_api\src\export_api.cpp

#include "export_c/export_api.h"

#include "Module_GetPICoaterBackground.hpp"
#include "core_cv/base/cuda_memory.hpp"
#include "core_cv/base/cuda_utils.hpp" 

#include "core_cv/imgcodecs/core_imgcodecs.hpp"

#include <cuda_runtime.h>
#include <iostream>

// === 內部封裝結構 ===
// 這個 Context 負責持有 Detector 以及 I/O 用的 Device Memory
struct PICoaterContext {
    picoater::PICoaterDetector detector;

    // 這些是 GPU 上的緩衝區，用來接 C# 傳進來的資料
    // Detector Class 只管理中間暫存 (temp)，不管理 I/O，所以這裡要管理
    uint8_t* d_in = nullptr;
    uint8_t* d_bg = nullptr;
    uint8_t* d_mura = nullptr;
    uint8_t* d_ridge = nullptr;

    int current_w = 0;
    int current_h = 0;

    ~PICoaterContext() {
        ReleaseBuffers();
    }

    void ReleaseBuffers() {
        if (d_in) cudaFree(d_in);
        if (d_bg) cudaFree(d_bg);
        if (d_mura) cudaFree(d_mura);
        if (d_ridge) cudaFree(d_ridge);
        d_in = nullptr;
        d_bg = nullptr;
        d_mura = nullptr;
        d_ridge = nullptr;
        current_w = 0;
        current_h = 0;
    }

    // 分配 I/O GPU 記憶體
    bool AllocateBuffers(int w, int h) {
        if (current_w == w && current_h == h) return true;

        ReleaseBuffers();
        current_w = w;
        current_h = h;
        size_t size = w * h * sizeof(uint8_t);

        cudaError_t err;
        err = cudaMalloc(&d_in, size);    if (err) return false;
        err = cudaMalloc(&d_bg, size);    if (err) return false;
        err = cudaMalloc(&d_mura, size);  if (err) return false;
        err = cudaMalloc(&d_ridge, size); if (err) return false;

        // 同時也要初始化內部的 Detector
        detector.Initialize(w, h);
        return true;
    }
};

// === 巨集：安全捕捉例外 ===
#define SAFE_EXEC(stmt) \
    try { \
        stmt; \
        return 0; \
    } catch (const std::exception& e) { \
        std::cerr << "[DLL Error] " << e.what() << std::endl; \
        return -1; \
    } catch (...) { \
        std::cerr << "[DLL Error] Unknown exception." << std::endl; \
        return -2; \
    }

// === API 實作 ===

extern "C" {

    // --- 記憶體管理 (直接轉發給 core_cv) ---
    PICOATER_API unsigned char* PICoater_AllocPinned(unsigned long long size) {
        // 呼叫 core_cv/base/cuda_memory.hpp
        return static_cast<unsigned char*>(core::alloc_pinned_memory(static_cast<size_t>(size)));
    }

    PICOATER_API void PICoater_FreePinned(unsigned char* ptr) {
        core::free_pinned_memory(ptr);
    }

    // --- 物件生命週期 ---
    PICOATER_API PICoaterHandle PICoater_Create() {
        // new 一個 Context (包含 Detector 和 I/O Buffers)
        return new PICoaterContext();
    }

    PICOATER_API void PICoater_Destroy(PICoaterHandle handle) {
        if (handle) {
            auto* ctx = static_cast<PICoaterContext*>(handle);
            delete ctx; // 這會自動呼叫解構子釋放 GPU 記憶體
        }
    }

    PICOATER_API int PICoater_Initialize(PICoaterHandle handle, int width, int height) {
        if (!handle) return -1;
        auto* ctx = static_cast<PICoaterContext*>(handle);

        SAFE_EXEC(
            if (!ctx->AllocateBuffers(width, height)) {
                return -3; // Malloc failed
            }
                );
    }

    // --- 核心執行 ---
    PICOATER_API int PICoater_Run(
        PICoaterHandle handle,
        const unsigned char* h_in,
        unsigned char* h_bg_out,
        unsigned char* h_mura_out,
        unsigned char* h_ridge_out,
        float bgSigmaFactor,
        float ridgeSigma,
        const char* ridgeMode,
        void* stream
    ) {
        if (!handle || !h_in) return -1;
        auto* ctx = static_cast<PICoaterContext*>(handle);

        // 如果還沒初始化，防呆
        if (ctx->current_w == 0 || ctx->current_h == 0) return -4;

        cudaStream_t cuStream = static_cast<cudaStream_t>(stream);
        size_t size = ctx->current_w * ctx->current_h * sizeof(uint8_t);

        SAFE_EXEC(
            // 1. 上傳 (Host Pinned -> Device)
            // 因為 h_in 是 Pinned Memory，這會觸發全速 DMA
            cudaMemcpyAsync(ctx->d_in, h_in, size, cudaMemcpyHostToDevice, cuStream);

        // 2. 計算 (Device -> Device)
        ctx->detector.Run(
            ctx->d_in,
            ctx->d_bg,
            ctx->d_mura,
            ctx->d_ridge,
            bgSigmaFactor,
            ridgeSigma,
            ridgeMode,
            cuStream
        );

        // 3. 下載 (Device -> Host Pinned)
        if (h_bg_out)    cudaMemcpyAsync(h_bg_out, ctx->d_bg, size, cudaMemcpyDeviceToHost, cuStream);
        if (h_mura_out)  cudaMemcpyAsync(h_mura_out, ctx->d_mura, size, cudaMemcpyDeviceToHost, cuStream);
        if (h_ridge_out) cudaMemcpyAsync(h_ridge_out, ctx->d_ridge, size, cudaMemcpyDeviceToHost, cuStream);

        // 4. 同步
        // 由於 C# 呼叫通常是 Blocking 的，這裡等待 GPU 完成是安全的做法
        // 如果希望 C# 端也是 Async，可以把這行拿掉，但 C# 就要自己負責同步
        cudaStreamSynchronize(cuStream);
            );
    }


    PICOATER_API int PICoater_LoadThumbnail(const char* path, int targetWidth, unsigned char* outBuffer, int* outRealW, int* outRealH) {
        try {
            // 呼叫 core_cv 的 CPU 函式
            return core::load_thumbnail_cpu(path, targetWidth, outBuffer, outRealW, outRealH);
        }
        catch (...) {
            return -99;
        }
    }

}