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

    // [Algo] 執行完整檢測
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
		float hessianMaxFactor,
        const char* ridgeMode
    );

    // [New] 僅執行 GPU 縮圖 (高速模式)
    // 利用已分配的 GPU Memory 進行縮圖，避免 CPU 計算
    PICOATER_API int PICoater_RunThumbnail_GPU(
        PICoaterHandle handle,
        const uint8_t* h_img_in, // Full size 原始圖
        int targetW,             // 目標寬度
        uint8_t* h_thumb_out,    // 輸出 Buffer (大小需 >= targetW * targetH)
        int* outRealW,
        int* outRealH
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
		float hessianMaxFactor,
        const char* ridgeMode
    );
}