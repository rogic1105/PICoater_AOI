// AOI_SDK\src_native\c_api\picoater_api\src\export_api.cpp

#include "export_c/export_api.h"

#include "Module_GetPICoaterBackground.hpp"
#include "core_cv/base/cuda_memory.hpp"
#include "core_cv/base/cuda_utils.hpp" 

#include "core_cv/imgcodecs/core_imgcodecs.hpp"
#include "core_cv/imgcodecs/core_imgcodecs_fast.hpp"

#include "core_cv/imgproc/core_transform.hpp"

#include <cuda_runtime.h>
#include <iostream>

// === 內部封裝結構 ===
// 這個 Context 負責持有 Detector 以及 I/O 用的 Device Memory
struct PICoaterContext {
    picoater::PICoaterDetector detector;

    // GPU Buffers
    uint8_t* d_in = nullptr;
    uint8_t* d_bg = nullptr;
    uint8_t* d_mura = nullptr;
    uint8_t* d_ridge = nullptr;

    // [新增] 曲線用的 GPU Buffer
    float* d_mura_curve = nullptr;

    cudaStream_t stream = nullptr;
    int current_w = 0;
    int current_h = 0;

    PICoaterContext() {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }

    ~PICoaterContext() {
        ReleaseBuffers();
        if (stream) cudaStreamDestroy(stream);
    }

    void ReleaseBuffers() {
        if (d_in) cudaFree(d_in);
        if (d_bg) cudaFree(d_bg);
        if (d_mura) cudaFree(d_mura);
        if (d_ridge) cudaFree(d_ridge);

        // [新增] 釋放曲線
        if (d_mura_curve) cudaFree(d_mura_curve);

        d_in = nullptr; d_bg = nullptr; d_mura = nullptr; d_ridge = nullptr;
        d_mura_curve = nullptr;
        current_w = 0; current_h = 0;
    }

    bool AllocateBuffers(int w, int h) {
        if (current_w == w && current_h == h) return true;
        ReleaseBuffers();
        current_w = w;
        current_h = h;
        size_t size = w * h * sizeof(uint8_t);
        size_t curve_size = w * sizeof(float); // [新增] 大小為 寬度 * 4 bytes

        if (cudaMalloc(&d_in, size) != cudaSuccess) return false;
        if (cudaMalloc(&d_bg, size) != cudaSuccess) return false;
        if (cudaMalloc(&d_mura, size) != cudaSuccess) return false;
        if (cudaMalloc(&d_ridge, size) != cudaSuccess) return false;

        // [新增] 分配曲線記憶體
        if (cudaMalloc(&d_mura_curve, curve_size) != cudaSuccess) return false;

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
        float* h_mura_curve_out, // [新增] 接收曲線的 Host 指標 (可以是 nullptr)
        float bgSigmaFactor,
        float ridgeSigma,
        const char* ridgeMode,
        void* stream
    ) {
        if (!handle || !h_in) return -1;
        auto* ctx = static_cast<PICoaterContext*>(handle);
        if (ctx->current_w == 0 || ctx->current_h == 0) return -4;

        cudaStream_t cuStream = static_cast<cudaStream_t>(stream);
        size_t size = ctx->current_w * ctx->current_h * sizeof(uint8_t);
        size_t curve_size = ctx->current_w * sizeof(float);

        SAFE_EXEC(
            // 1. 上傳
            cudaMemcpyAsync(ctx->d_in, h_in, size, cudaMemcpyHostToDevice, cuStream);

        // 2. 計算 (傳入 d_mura_curve)
        ctx->detector.Run(
            ctx->d_in, ctx->d_bg, ctx->d_mura, ctx->d_ridge,
            ctx->d_mura_curve, // [新增] 傳入 GPU buffer
            bgSigmaFactor, ridgeSigma, ridgeMode, cuStream
        );

        // 3. 下載影像
        if (h_bg_out)    cudaMemcpyAsync(h_bg_out, ctx->d_bg, size, cudaMemcpyDeviceToHost, cuStream);
        if (h_mura_out)  cudaMemcpyAsync(h_mura_out, ctx->d_mura, size, cudaMemcpyDeviceToHost, cuStream);
        if (h_ridge_out) cudaMemcpyAsync(h_ridge_out, ctx->d_ridge, size, cudaMemcpyDeviceToHost, cuStream);

        // [新增] 4. 下載曲線 (如果使用者有提供 buffer)
        if (h_mura_curve_out) {
            cudaMemcpyAsync(h_mura_curve_out, ctx->d_mura_curve, curve_size, cudaMemcpyDeviceToHost, cuStream);
        }

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

    // [新增] 實作 FastRead
    PICOATER_API bool PICoater_FastReadBMP(const char* filepath, int* w, int* h, unsigned char* outBuffer, int bufferSize) {
        try {
            int width = 0, height = 0;
            // 呼叫 core_cv 的極速讀取
            bool res = core::fast_read_bmp_8bit(filepath, width, height, outBuffer, bufferSize);
            if (res) {
                if (w) *w = width;
                if (h) *h = height;
            }
            return res;
        }
        catch (...) {
            return false;
        }
    }

    // [新增] 實作 FastWrite
    PICOATER_API bool PICoater_FastWriteBMP(const char* filepath, int w, int h, const unsigned char* inBuffer) {
        try {
            // 呼叫 core_cv 的極速寫入
            return core::fast_write_bmp_8bit(filepath, w, h, inBuffer);
        }
        catch (...) {
            return false;
        }
    }


    PICOATER_API int PICoater_Run_And_GetThumbnail(
        PICoaterHandle handle,
        const unsigned char* h_in,
        unsigned char* h_thumb_out,
        float* h_mura_curve_out, // [新增] 同時拿縮圖和曲線
        int thumb_w,
        int thumb_h,
        float bgSigma, float ridgeSigma, const char* rMode
    ) {
        if (!handle || !h_in || !h_thumb_out) return -1;
        auto* ctx = static_cast<PICoaterContext*>(handle);

        size_t size = ctx->current_w * ctx->current_h;
        size_t curve_size = ctx->current_w * sizeof(float);

        SAFE_EXEC(
            // 1. 上傳
            cudaMemcpyAsync(ctx->d_in, h_in, size, cudaMemcpyHostToDevice, ctx->stream);

        // 2. 運算
        ctx->detector.Run(
            ctx->d_in, ctx->d_bg, ctx->d_mura, ctx->d_ridge,
            ctx->d_mura_curve, // [新增]
            bgSigma, ridgeSigma, rMode, ctx->stream
        );

        // 3. 縮圖
        uint8_t * d_thumb_temp = ctx->d_bg; // 借用 buffer
        core::resize_u8_gpu(ctx->d_mura, ctx->current_w, ctx->current_h,
            d_thumb_temp, thumb_w, thumb_h, ctx->stream);

        // 4. 下載縮圖
        size_t thumb_size = thumb_w * thumb_h;
        cudaMemcpyAsync(h_thumb_out, d_thumb_temp, thumb_size, cudaMemcpyDeviceToHost, ctx->stream);

        // [新增] 5. 下載曲線
        if (h_mura_curve_out) {
            cudaMemcpyAsync(h_mura_curve_out, ctx->d_mura_curve, curve_size, cudaMemcpyDeviceToHost, ctx->stream);
        }

        cudaStreamSynchronize(ctx->stream);
            );
        return 0;
    }

}