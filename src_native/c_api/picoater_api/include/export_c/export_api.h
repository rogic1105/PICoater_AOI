// AOI_SDK\src_native\c_api\picoater_api\include\export_c\export_api.h
#pragma once

#if defined(_WIN32)
#ifdef PICOATER_EXPORTS
#define PICOATER_API __declspec(dllexport)
#else
#define PICOATER_API __declspec(dllimport)
#endif
#else
#define PICOATER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // 定義 Handle，隱藏內部實作
    typedef void* PICoaterHandle;

    // --- 記憶體管理 API (給 C# 用) ---
    // 分配鎖頁記憶體 (Pinned Memory)
    PICOATER_API unsigned char* PICoater_AllocPinned(unsigned long long size);

    // 釋放鎖頁記憶體
    PICOATER_API void PICoater_FreePinned(unsigned char* ptr);


    // --- 演算法 API ---
    PICOATER_API PICoaterHandle PICoater_Create();
    PICOATER_API int PICoater_Initialize(PICoaterHandle handle, int width, int height);

    PICOATER_API int PICoater_Run(
        PICoaterHandle handle,
        const unsigned char* h_in,       // [In]  Host Pinned Pointer
        unsigned char* h_bg_out,         // [Out] Host Pinned Pointer
        unsigned char* h_mura_out,       // [Out] Host Pinned Pointer
        unsigned char* h_ridge_out,      // [Out] Host Pinned Pointer
        float bgSigmaFactor,
        float ridgeSigma,
        const char* ridgeMode,
        void* stream                     // cudaStream_t
    );

    PICOATER_API void PICoater_Destroy(PICoaterHandle handle);

    // --- [新增] 極速 IO API (Fast IO) ---

    // 縮圖預覽 (原本的 CPU 縮放功能)
    PICOATER_API int PICoater_LoadThumbnail(
        const char* path,
        int targetWidth,
        unsigned char* outBuffer,
        int* outRealW,
        int* outRealH
    );

    // 極速讀取 8-bit BMP (直接讀入 Buffer，不縮放)
    // C# 必須先 AllocPinned 足夠大的空間來接
    PICOATER_API bool PICoater_FastReadBMP(
        const char* filepath,
        int* w,
        int* h,
        unsigned char* outBuffer,
        int bufferSize
    );

    // 極速寫入 8-bit BMP
    PICOATER_API bool PICoater_FastWriteBMP(
        const char* filepath,
        int w,
        int h,
        const unsigned char* inBuffer
    );

#ifdef __cplusplus
}
#endif