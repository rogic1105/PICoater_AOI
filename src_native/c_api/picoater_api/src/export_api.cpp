// PICoater_AOI\src_native\c_api\picoater_api\src\export_api.cpp

#include "export_c/export_api.h"
#include "Module_GetPICoaterBackground.hpp"
#include "core_cv/base/cuda_utils.hpp"
#include "core_cv/base/cuda_memory.hpp"
#include "core_cv/imgcodecs/core_imgcodecs_fast.hpp"
#include "core_cv/imgproc/core_transform.hpp" // [必須] 引用 GPU Resize
#include <iostream>
#include <vector>

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
    float* d_mura_curve_mean = nullptr;
	float* d_mura_curve_max = nullptr;

    void Release() {
        if (d_in) cudaFree(d_in);
        if (d_bg) cudaFree(d_bg);
        if (d_mura) cudaFree(d_mura);
        if (d_ridge) cudaFree(d_ridge);
        if (d_mura_curve_mean) cudaFree(d_mura_curve_mean);
		if (d_mura_curve_max) cudaFree(d_mura_curve_max);

        d_in = nullptr; 
        d_bg = nullptr;
        d_mura = nullptr;
        d_ridge = nullptr;
        d_mura_curve_mean = nullptr;
		d_mura_curve_max = nullptr;
        detector.Release();
    }
};

// ... (Create, Destroy, Initialize 保持不變) ...

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
        cudaMalloc(&ctx->d_mura_curve_mean, width * sizeof(float));
        cudaMalloc(&ctx->d_mura_curve_max, width * sizeof(float));

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
    float* h_mura_curve_mean,
	float* h_mura_curve_max,
    float bgSigma,
    float ridgeSigma,
    int heatmap_lower_thres,
    float heatmap_alpha,
    const char* ridgeMode
) {
    if (!handle) return -1;
    PICoaterContext* ctx = (PICoaterContext*)handle;

    cudaMemcpy(ctx->d_in, h_img_in, ctx->img_size, cudaMemcpyHostToDevice);

    ctx->detector.Run(
        ctx->d_in, ctx->d_bg, ctx->d_mura, ctx->d_ridge, ctx->d_mura_curve_mean, ctx->d_mura_curve_max,
        bgSigma, ridgeSigma, ridgeMode, 0
    );

    if (h_bg_out) cudaMemcpy(h_bg_out, ctx->d_bg, ctx->img_size, cudaMemcpyDeviceToHost);
    if (h_mura_out) cudaMemcpy(h_mura_out, ctx->d_mura, ctx->img_size, cudaMemcpyDeviceToHost);
    if (h_ridge_out) cudaMemcpy(h_ridge_out, ctx->d_ridge, ctx->img_size, cudaMemcpyDeviceToHost);
    if (h_mura_curve_mean) cudaMemcpy(h_mura_curve_mean, ctx->d_mura_curve_mean, ctx->width * sizeof(float), cudaMemcpyDeviceToHost);
	if (h_mura_curve_max) cudaMemcpy(h_mura_curve_max, ctx->d_mura_curve_max, ctx->width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    return 0;
}

// [New] GPU 縮圖實作
PICOATER_API int PICoater_RunThumbnail_GPU(
    PICoaterHandle handle,
    const uint8_t* h_img_in,
    int targetW,
    uint8_t* h_thumb_out,
    int* outRealW,
    int* outRealH
) {
    if (!handle) return -1;
    PICoaterContext* ctx = (PICoaterContext*)handle;

    // 1. Upload Full Image
    cudaMemcpy(ctx->d_in, h_img_in, ctx->img_size, cudaMemcpyHostToDevice);

    // 2. Calculate Size
    float scale = (float)targetW / ctx->width;
    int targetH = (int)(ctx->height * scale);
    *outRealW = targetW;
    *outRealH = targetH;

    // 3. GPU Resize
    // 借用 d_bg 作為輸出 buffer (因為是縮圖模式，不需要保留背景圖)
    // core::resize_u8_gpu(src, srcW, srcH, dst, dstW, dstH, stream)
    core::resize_u8_gpu(ctx->d_in, ctx->width, ctx->height, ctx->d_bg, targetW, targetH, 0);

    // 4. Download Thumbnail
    cudaMemcpy(h_thumb_out, ctx->d_bg, targetW * targetH, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    return 0;
}

// Helper API
PICOATER_API void* PICoater_AllocPinned(size_t size) {
    return core::alloc_pinned_memory(size);
}

PICOATER_API void PICoater_FreePinned(void* ptr) {
    core::free_pinned_memory(ptr);
}

PICOATER_API bool PICoater_FastReadBMP(const char* filepath, int* width, int* height, uint8_t* pData, size_t bufferSize) {
    if (!filepath || !width || !height) return false;
    int w = 0, h = 0;
    bool res = core::fast_read_bmp_8bit(filepath, w, h, pData, bufferSize);
    *width = w;
    *height = h;
    return res;
}

PICOATER_API int PICoater_Run_WithThumb(
    PICoaterHandle handle,
    const uint8_t* h_img_in,
    uint8_t* h_ridge_thumb_out,
    int thumbW,
    int thumbH,
    float* h_mura_curve_mean,
    float* h_mura_curve_max,
    float bgSigma,
    float ridgeSigma,
    int heatmap_lower_thres,
    float heatmap_alpha,
    const char* ridgeMode
) {
    if (!handle) return -1;
    PICoaterContext* ctx = (PICoaterContext*)handle;

    // 1. 上傳原圖
    cudaMemcpy(ctx->d_in, h_img_in, ctx->img_size, cudaMemcpyHostToDevice);

    // 2. 執行演算法 (GPU)
    ctx->detector.Run(
        ctx->d_in, ctx->d_bg, ctx->d_mura, ctx->d_ridge, ctx->d_mura_curve_mean, ctx->d_mura_curve_max,
        bgSigma, ridgeSigma, ridgeMode, 0
    );

    // 3. [關鍵] GPU 內直接縮圖 (Full -> Thumb)
    // 借用 d_bg 當臨時 buffer，把 d_ridge 縮小存進去
    uint8_t* d_thumb = ctx->d_bg;
    core::resize_u8_gpu(ctx->d_ridge, ctx->width, ctx->height, d_thumb, thumbW, thumbH, 0);

    // 4. 只下載縮圖 (快!)
    if (h_ridge_thumb_out) {
        cudaMemcpy(h_ridge_thumb_out, d_thumb, thumbW * thumbH, cudaMemcpyDeviceToHost);
    }

    // 5. 下載曲線數據
    if (h_mura_curve_mean) {
        cudaMemcpy(h_mura_curve_mean, ctx->d_mura_curve_mean, ctx->width * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // [獨立檢查] 必須確認 h_mura_curve_max 不為 nullptr 才能寫入
    if (h_mura_curve_max) {
        cudaMemcpy(h_mura_curve_max, ctx->d_mura_curve_max, ctx->width * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
    return 0;
}