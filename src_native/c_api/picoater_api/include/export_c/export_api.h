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
    // 1. 建立實例
    PICOATER_API PICoaterHandle PICoater_Create();

    // 2. 初始化 (分配 GPU 資源)
    PICOATER_API int PICoater_Initialize(PICoaterHandle handle, int width, int height);

    // 3. 執行 (接受 Host Pinned Pointers)
    // C# 端傳入 AllocPinned 拿到的指標
    PICOATER_API int PICoater_Run(
        PICoaterHandle handle,
        const unsigned char* h_in,       // [In]  Host Pinned Pointer
        unsigned char* h_bg_out,         // [Out] Host Pinned Pointer
        unsigned char* h_mura_out,       // [Out] Host Pinned Pointer
        unsigned char* h_ridge_out,      // [Out] Host Pinned Pointer
        float bgSigmaFactor,
        float ridgeSigma,
        const char* ridgeMode,
        void* stream                     // cudaStream_t (可傳 IntPtr.Zero)
    );

    // 4. 銷毀
    PICOATER_API void PICoater_Destroy(PICoaterHandle handle);

#ifdef __cplusplus
}
#endif