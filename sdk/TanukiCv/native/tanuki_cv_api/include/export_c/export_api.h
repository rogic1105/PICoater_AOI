
#ifndef TANUKI_CV_API_EXPORT_API_H_
#define TANUKI_CV_API_EXPORT_API_H_

#include <cstdint>
#include <stdbool.h> // for bool in C

#ifdef TANUKI_CV_API_EXPORTS
#define TANUKI_CV_API __declspec(dllexport)
#else
#define TANUKI_CV_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CORE_CV_SUCCESS 0
#define CORE_CV_ERROR_UNKNOWN -1
#define CORE_CV_ERROR_NULL_POINTER -2
#define CORE_CV_ERROR_INVALID_PARAM -3
#define CORE_CV_ERROR_CUDA -4

    // --- [新增] GPU 暖身 (Warm-Up) ---
    // 強迫 CUDA context / driver 提早初始化：在 GPU 內分配小 buffer、跑一個 kernel、釋放。
    // 「怎麼暖身」屬 GPU 內部細節，封裝在 native，不洩漏到 caller。
    // 回傳 0 (CORE_CV_SUCCESS) 表成功。
    TANUKI_CV_API int TanukiCv_WarmUp();

    // --- [新增] 記憶體管理 (Pinned Memory) ---
    TANUKI_CV_API unsigned char* TanukiCv_AllocPinned(unsigned long long size);
    TANUKI_CV_API void TanukiCv_FreePinned(unsigned char* ptr);

    // --- [新增] 極速 IO (Fast IO) ---
    TANUKI_CV_API bool TanukiCv_FastReadBMP(const char* filepath, int* w, int* h, unsigned char* outBuffer, int bufferSize);
    TANUKI_CV_API bool TanukiCv_FastWriteBMP(const char* filepath, int w, int h, const unsigned char* inBuffer);


    // --- 影像處理運算子 ---
    TANUKI_CV_API int TanukiCv_Brighten(const uint8_t* src_ptr, int width, int height, int value, uint8_t* dst_ptr);
    TANUKI_CV_API int TanukiCv_Threshold(const uint8_t* src_ptr, int width, int height, uint8_t threshold, uint8_t* dst_ptr);
    TANUKI_CV_API int TanukiCv_Invert(const uint8_t* src_ptr, int width, int height, uint8_t* dst_ptr);
    TANUKI_CV_API int TanukiCv_Convolution(const uint8_t* src_ptr, int width, int height, const float* mask_ptr, int mask_size, uint8_t* dst_ptr);


    // --- [新增] GPU 記憶體管理 (進階模式用) ---
    TANUKI_CV_API int TanukiCv_MallocGPU(unsigned char** d_ptr, int width, int height);
    TANUKI_CV_API int TanukiCv_FreeGPU(unsigned char* d_ptr);
    TANUKI_CV_API int TanukiCv_Upload(const unsigned char* h_src, unsigned char* d_dst, int width, int height);
    TANUKI_CV_API int TanukiCv_Download(const unsigned char* d_src, unsigned char* h_dst, int width, int height);

    // --- [新增] 純 GPU 運算 API (不含 Memcpy) ---
    // 這些函式的輸入/輸出指標，必須是 GPU 指標 (d_ptr)
    TANUKI_CV_API int TanukiCv_Brighten_GPU(const uint8_t* d_src, int width, int height, int value, uint8_t* d_dst);
    TANUKI_CV_API int TanukiCv_Threshold_GPU(const uint8_t* d_src, int width, int height, uint8_t threshold, uint8_t* d_dst);
    TANUKI_CV_API int TanukiCv_Invert_GPU(const uint8_t* d_src, int width, int height, uint8_t* d_dst);
    TANUKI_CV_API int TanukiCv_Convolution_GPU(const uint8_t* d_src, int width, int height, const float* d_mask, int mask_size, uint8_t* d_dst);

    // --- [新增] Float 資源管理 (給 Mask 用) ---
        // count: 浮點數的數量 (例如 3x3 mask，count = 9)
    TANUKI_CV_API int TanukiCv_MallocGPU_Float(float** d_ptr, int count);
    TANUKI_CV_API int TanukiCv_FreeGPU_Float(float* d_ptr);
    TANUKI_CV_API int TanukiCv_Upload_Float(const float* h_src, float* d_dst, int count);

    // --- [新增] GPU 縮圖 ---
    // 從 Host 讀入全尺寸影像，在 GPU 上縮放後寫回 Host。
    // h_src / h_dst 若為 Pinned Memory，H<->D 傳輸走 DMA 加速。
    TANUKI_CV_API int TanukiCv_Resize_GPU(
        const uint8_t* h_src, int src_w, int src_h,
        uint8_t*       h_dst, int dst_w, int dst_h);

#ifdef __cplusplus
}
#endif

#endif