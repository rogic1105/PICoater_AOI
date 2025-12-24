// AOI_LIB/ExportCDLL/src/export_api.cpp

#include "export_c/export_api.h"
#include "GeneratePcbSegMask/GeneratePcbSegMask.hpp"
#include "core/core_ops.hpp"   // 引用 CoreLib 測試功能 (Brighten等)
#include "core/cuda_utils.hpp" // 引用 checkCudaErrors
#include <cuda_runtime.h>

// ==========================================
//  Helper: 通用 CUDA 執行器 (給測試功能用)
// ==========================================
typedef void (*KernelFunc)(const uint8_t*, uint8_t*, int, int, int, cudaStream_t);

int RunCudaKernel(const uint8_t* h_in, uint8_t* h_out, int W, int H, int param, KernelFunc func) {
    if (!h_in || !h_out || W <= 0 || H <= 0) return -1;

    uint8_t* d_in = nullptr, * d_out = nullptr;
    size_t size = (size_t)W * H;

    try {
        checkCudaErrors(cudaMalloc(&d_in, size));
        checkCudaErrors(cudaMalloc(&d_out, size));

        // Host -> Device
        checkCudaErrors(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

        // Run Kernel
        func(d_in, d_out, W, H, param, 0);
        checkCudaErrors(cudaGetLastError());

        // Device -> Host
        checkCudaErrors(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));
    }
    catch (...) {
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        return -99;
    }

    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    return 0;
}

// ==========================================
//  DLL 導出介面
// ==========================================
extern "C" {

    // 1. 主要功能：PCB Mask 生成
    int EXP_CC BuildFullMaskFromFFT_C(
        const uint8_t* img_gray, int H, int W,
        float fft_th, int bw_th, int border_t,
        uint8_t* full_mask_out)
    {
        try {
            // [修正點]：這裡不需要再轉 vector 了！
            // 直接把 C# 傳進來的指標 (img_gray) 傳給你的全指標版函式
            GeneratePcbSegMask(img_gray, H, W, fft_th, bw_th, border_t, full_mask_out);
            return 0;
        }
        catch (...) {
            return -99;
        }
    }

    // 2. [測試用] 亮度調整
    int EXP_CC Process_Brighten(const uint8_t* src, uint8_t* dst, int W, int H, int value) {
        return RunCudaKernel(src, dst, W, H, value, core::brighten_u8_gpu);
    }

    // 3. [測試用] 二值化
    int EXP_CC Process_Threshold(const uint8_t* src, uint8_t* dst, int W, int H, int value) {
        // Lambda 轉型適配
        auto lambda = [](const uint8_t* i, uint8_t* o, int w, int h, int p, cudaStream_t s) {
            core::threshold_u8_gpu(i, o, w, h, (uint8_t)p, s);
            };
        return RunCudaKernel(src, dst, W, H, value, lambda);
    }

    // 4. [測試用] 反轉
    int EXP_CC Process_Invert(const uint8_t* src, uint8_t* dst, int W, int H) {
        auto lambda = [](const uint8_t* i, uint8_t* o, int w, int h, int p, cudaStream_t s) {
            core::invert_u8_gpu(i, o, w, h, s);
            };
        return RunCudaKernel(src, dst, W, H, 0, lambda);
    }
}