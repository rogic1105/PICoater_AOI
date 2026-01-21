// PICoater_AOI\src_native\c_api\picoater_api\src\export_api.cpp

#include "export_c/export_api.h"
#include "Module_GetPICoaterBackground.hpp"
#include "core_cv/base/cuda_utils.hpp"
#include "core_cv/base/cuda_memory.hpp"
#include "core_cv/imgcodecs/core_imgcodecs_fast.hpp"
// 為了 LoadThumbnail 需要 stb resize 或類似實作，這裡假設 core_cv 有提供，或使用簡單實作
// 若 core_cv 沒有 resize_cpu，這裡可能需要 include stb_image_resize.h
// 假設專案已有 stb，我們直接用 core::fast_read + 簡單 resize，或是完整讀取
#include <iostream>
#include <vector>

// 內部 Context 結構
struct PICoaterContext {
    picoater::PICoaterDetector detector;
    int width = 0;
    int height = 0;
    size_t img_size = 0;

    // GPU Buffers
    uint8_t* d_in = nullptr;
    uint8_t* d_bg = nullptr;
    uint8_t* d_mura = nullptr;
    uint8_t* d_ridge = nullptr;
    float* d_mura_curve = nullptr;


    void Release() {
        if (d_in) cudaFree(d_in);
        if (d_bg) cudaFree(d_bg);
        if (d_mura) cudaFree(d_mura);
        if (d_ridge) cudaFree(d_ridge);
        if (d_mura_curve) cudaFree(d_mura_curve);


        d_in = nullptr; d_bg = nullptr; d_mura = nullptr;
        d_ridge = nullptr; d_mura_curve = nullptr;
        detector.Release();
    }
};

// --- Context API ---

PICOATER_API PICoaterHandle PICoater_Create() {
    return new PICoaterContext();
}

PICOATER_API void PICoater_Destroy(PICoaterHandle handle) {
    if (handle) {
        PICoaterContext* ctx = (PICoaterContext*)handle;
        ctx->Release();
        delete ctx;
    }
}

PICOATER_API int PICoater_Initialize(PICoaterHandle handle, int width, int height) {
    if (!handle) return -1;
    PICoaterContext* ctx = (PICoaterContext*)handle;

    if (ctx->width == width && ctx->height == height) return 0;

    ctx->Release();
    ctx->width = width;
    ctx->height = height;
    ctx->img_size = (size_t)width * height;

    try {
        cudaMalloc(&ctx->d_in, ctx->img_size);
        cudaMalloc(&ctx->d_bg, ctx->img_size);
        cudaMalloc(&ctx->d_mura, ctx->img_size);
        cudaMalloc(&ctx->d_ridge, ctx->img_size);
        cudaMalloc(&ctx->d_mura_curve, width * sizeof(float));


        ctx->detector.Initialize(width, height);
    }
    catch (...) {
        return -2;
    }
    return 0;
}

PICOATER_API int PICoater_Run(
    PICoaterHandle handle,
    const uint8_t* h_img_in,
    uint8_t* h_bg_out,
    uint8_t* h_mura_out,
    uint8_t* h_ridge_out,
    float* h_mura_curve_out,
    float bgSigma,
    float ridgeSigma,
    int heatmap_lower_thres,
    float heatmap_alpha,
    const char* ridgeMode
) {
    if (!handle) return -1;
    PICoaterContext* ctx = (PICoaterContext*)handle;

    // 1. Upload
    cudaMemcpy(ctx->d_in, h_img_in, ctx->img_size, cudaMemcpyHostToDevice);

    // 2. Run
    ctx->detector.Run(
        ctx->d_in, 
        ctx->d_bg, 
        ctx->d_mura,
        ctx->d_ridge,
        ctx->d_mura_curve,
        bgSigma,
        ridgeSigma, 
        ridgeMode, 
        0
    );

    // 3. Download
    if (h_bg_out) cudaMemcpy(h_bg_out, ctx->d_bg, ctx->img_size, cudaMemcpyDeviceToHost);
    if (h_mura_out) cudaMemcpy(h_mura_out, ctx->d_mura, ctx->img_size, cudaMemcpyDeviceToHost);
    if (h_ridge_out) cudaMemcpy(h_ridge_out, ctx->d_ridge, ctx->img_size, cudaMemcpyDeviceToHost);
    if (h_mura_curve_out) cudaMemcpy(h_mura_curve_out, ctx->d_mura_curve, ctx->width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    return 0;
}

// --- Helper API ---

PICOATER_API void* PICoater_AllocPinned(size_t size) {
    return core::alloc_pinned_memory(size);
}

PICOATER_API void PICoater_FreePinned(void* ptr) {
    core::free_pinned_memory(ptr);
}

PICOATER_API bool PICoater_FastReadBMP(const char* filepath, int* width, int* height, uint8_t* pData, size_t bufferSize) {
    if (!filepath || !width || !height) return false;
    int w = 0, h = 0;
    // 呼叫 core_cv 的 fast read
    bool res = core::fast_read_bmp_8bit(filepath, w, h, pData, bufferSize);
    *width = w;
    *height = h;
    return res;
}

// 簡單的縮圖載入: 讀取全圖 -> CPU Resize (Nearest Neighbor for speed or use STB if available)
// 這裡為了保持範例完整，若 core_cv 沒有 cpu resize，我們做一個簡單的採樣
PICOATER_API int PICoater_LoadThumbnail(const char* filepath, int targetWidth, uint8_t* outBuffer, int* outRealW, int* outRealH) {
    int w = 0, h = 0;
    // 1. 為了讀取方便，這裡需要一個臨時 buffer。
    // 但因為我們不知道圖多大，如果 core::fast_read_bmp_8bit 需要預分配 buffer，我們可能需要先讀 header。
    // 假設我們有夠大的 buffer (e.g. 200MB 暫存)，或者直接 new 一個 (慢一點但安全)。

    // 為了安全起見，這裡先分配一塊大記憶體 (假設最大 16K*10K)
    // 注意：這會比較吃記憶體，正式版建議改用 STB 讀 header 再 alloc
    static size_t MAX_SIZE = 16384ULL * 10000ULL;
    uint8_t* temp = (uint8_t*)core::alloc_pinned_memory(MAX_SIZE);

    if (!core::fast_read_bmp_8bit(filepath, w, h, temp, MAX_SIZE)) {
        core::free_pinned_memory(temp);
        return -1;
    }

    // 2. 計算縮圖尺寸
    float scale = (float)targetWidth / w;
    int thumbH = (int)(h * scale);
    *outRealW = targetWidth;
    *outRealH = thumbH;

    // 3. CPU Resize (Nearest Neighbor) - 簡單實作
    for (int y = 0; y < thumbH; ++y) {
        int srcY = (int)(y / scale);
        if (srcY >= h) srcY = h - 1;

        for (int x = 0; x < targetWidth; ++x) {
            int srcX = (int)(x / scale);
            if (srcX >= w) srcX = w - 1;

            outBuffer[y * targetWidth + x] = temp[srcY * w + srcX];
        }
    }

    core::free_pinned_memory(temp);
    return 0;
}