// AOI_SDK\core_cv_api\include\export_c\export_api.h

#ifndef CORE_CV_API_EXPORT_API_H_
#define CORE_CV_API_EXPORT_API_H_

#include <cstdint>
#include <stdbool.h> // for bool in C

#ifdef CORE_CV_API_EXPORTS
#define CORE_CV_API __declspec(dllexport)
#else
#define CORE_CV_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CORE_CV_SUCCESS 0
#define CORE_CV_ERROR_UNKNOWN -1
#define CORE_CV_ERROR_NULL_POINTER -2
#define CORE_CV_ERROR_INVALID_PARAM -3
#define CORE_CV_ERROR_CUDA -4

    // --- [新增] 記憶體管理 (Pinned Memory) ---
    CORE_CV_API unsigned char* CoreCV_AllocPinned(unsigned long long size);
    CORE_CV_API void CoreCV_FreePinned(unsigned char* ptr);

    // --- [新增] 極速 IO (Fast IO) ---
    CORE_CV_API bool CoreCV_FastReadBMP(const char* filepath, int* w, int* h, unsigned char* outBuffer, int bufferSize);
    CORE_CV_API bool CoreCV_FastWriteBMP(const char* filepath, int w, int h, const unsigned char* inBuffer);


    // --- 影像處理運算子 ---
    CORE_CV_API int CoreCV_Brighten(const uint8_t* src_ptr, int width, int height, int value, uint8_t* dst_ptr);
    CORE_CV_API int CoreCV_Threshold(const uint8_t* src_ptr, int width, int height, uint8_t threshold, uint8_t* dst_ptr);
    CORE_CV_API int CoreCV_Invert(const uint8_t* src_ptr, int width, int height, uint8_t* dst_ptr);
    CORE_CV_API int CoreCV_Convolution(const uint8_t* src_ptr, int width, int height, const float* mask_ptr, int mask_size, uint8_t* dst_ptr);


    // --- [新增] GPU 記憶體管理 (進階模式用) ---
    CORE_CV_API int CoreCV_MallocGPU(unsigned char** d_ptr, int width, int height);
    CORE_CV_API int CoreCV_FreeGPU(unsigned char* d_ptr);
    CORE_CV_API int CoreCV_Upload(const unsigned char* h_src, unsigned char* d_dst, int width, int height);
    CORE_CV_API int CoreCV_Download(const unsigned char* d_src, unsigned char* h_dst, int width, int height);

    // --- [新增] 純 GPU 運算 API (不含 Memcpy) ---
    // 這些函式的輸入/輸出指標，必須是 GPU 指標 (d_ptr)
    CORE_CV_API int CoreCV_Brighten_GPU(const uint8_t* d_src, int width, int height, int value, uint8_t* d_dst);
    CORE_CV_API int CoreCV_Threshold_GPU(const uint8_t* d_src, int width, int height, uint8_t threshold, uint8_t* d_dst);
    CORE_CV_API int CoreCV_Invert_GPU(const uint8_t* d_src, int width, int height, uint8_t* d_dst);
    CORE_CV_API int CoreCV_Convolution_GPU(const uint8_t* d_src, int width, int height, const float* d_mask, int mask_size, uint8_t* d_dst);

    // --- [新增] Float 資源管理 (給 Mask 用) ---
        // count: 浮點數的數量 (例如 3x3 mask，count = 9)
    CORE_CV_API int CoreCV_MallocGPU_Float(float** d_ptr, int count);
    CORE_CV_API int CoreCV_FreeGPU_Float(float* d_ptr);
    CORE_CV_API int CoreCV_Upload_Float(const float* h_src, float* d_dst, int count);


#ifdef __cplusplus
}
#endif

#endif