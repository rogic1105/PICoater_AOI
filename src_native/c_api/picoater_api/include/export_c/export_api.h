// PICoater_AOI\src_native\c_api\picoater_api\include\export_c\export_api.h

#pragma once

#ifdef _WIN32
#define PICOATER_API __declspec(dllexport)
#else
#define PICOATER_API
#endif

#include <cstdint>

extern "C" {
    // 定義 Handle
    typedef void* PICoaterHandle;

    // [Core] Context 管理
    PICOATER_API PICoaterHandle PICoater_Create();
    PICOATER_API void PICoater_Destroy(PICoaterHandle handle);
    PICOATER_API int PICoater_Initialize(PICoaterHandle handle, int width, int height);

    // [Algo] 執行檢測
    // [修正] 移除 h_heatmap_out，保留其他設定參數以符合 detector 需求
    PICOATER_API int PICoater_Run(
        PICoaterHandle handle,
        const uint8_t* h_img_in,
        uint8_t* h_bg_out,
        uint8_t* h_mura_out,
        uint8_t* h_ridge_out,
        float* h_mura_curve_out,
        float bgSigma,
        float ridgeSigma,
        int heatmap_lower_thres, // 仍需保留，傳給內部用
        float heatmap_alpha,     // 仍需保留，傳給內部用
        const char* ridgeMode
    );

    // [Helper] 記憶體管理 (Pinned Memory)
    PICOATER_API void* PICoater_AllocPinned(size_t size);
    PICOATER_API void PICoater_FreePinned(void* ptr);

    // [Helper] 快速 IO
    PICOATER_API bool PICoater_FastReadBMP(
        const char* filepath,
        int* width,
        int* height,
        uint8_t* pData,
        size_t bufferSize
    );

    // [Helper] 讀取縮圖
    PICOATER_API int PICoater_LoadThumbnail(
        const char* filepath,
        int targetWidth,
        uint8_t* outBuffer,
        int* outRealW,
        int* outRealH
    );
}