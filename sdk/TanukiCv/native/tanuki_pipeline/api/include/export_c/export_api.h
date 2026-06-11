#pragma once
// tanuki_pipeline_api 的 C ABI —— 與舊 picoater_api 相同（drop-in：切換只需改 DLL 名 + 重驗 runtime）。
// 內部改用 tanuki::pipeline::find_stream_ridgeline。函式名暫保留 PICoaterAPI_*（4b 再連同 run(name,json) 重設計）。

#ifdef _WIN32
#define TANUKI_PIPELINE_API __declspec(dllexport)
#else
#define TANUKI_PIPELINE_API
#endif

#include <cstdint>

extern "C" {

typedef void* AoiPipelineHandle;

typedef struct AoiInputImageC {
    int width;
    int height;
    const uint8_t* data;
    void* stream;
} AoiInputImageC;

typedef struct AoiOutputBuffersC {
    int width;
    int height;
    uint8_t* background_data;
    uint8_t* mura_data;
    uint8_t* ridge_data;
    float* mura_curve_mean;
    float* mura_curve_max;
    float* mura_row_curve_mean;
    float* mura_row_curve_max;
    void* stream;
} AoiOutputBuffersC;

typedef struct AoiAlgorithmParamsC {
    float bg_sigma_factor;
    float ridge_sigma;
    float hessian_max_factor;
    const char* ridge_mode;
    const float* precomputed_col_mean;  // host pointer, size = width. NULL = per-frame mode.
} AoiAlgorithmParamsC;

TANUKI_PIPELINE_API AoiPipelineHandle PICoaterAPI_CreatePipeline();

TANUKI_PIPELINE_API int PICoaterAPI_ProcessPipeline(
    AoiPipelineHandle handle,
    const AoiInputImageC* input,
    const AoiAlgorithmParamsC* params,
    const AoiOutputBuffersC* output);

TANUKI_PIPELINE_API const char* PICoaterAPI_GetLastError(AoiPipelineHandle handle);

TANUKI_PIPELINE_API void PICoaterAPI_DestroyPipeline(AoiPipelineHandle handle);

TANUKI_PIPELINE_API int PICoaterAPI_ComputeColumnMean(
    AoiPipelineHandle handle,
    const AoiInputImageC* input,
    float bg_sigma_factor,
    float* out_col_mean);  /* host buffer, size = input->width */

}  // extern "C"
