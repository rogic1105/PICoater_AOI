#pragma once
// tanuki_pipeline_api 的 C ABI（4b 定版）：單一簽名服務所有 pipeline。
//   Create(pipeline 名, json 選項)  → 依名字建食譜（目前 "find_stream_ridgeline"；選項如 {"ridge_method":"hessian"}）
//   Process(handle, input, json 參數, precomputed_col_mean, output)
//     演算法參數走 json 字串（各 pipeline parse 自己要的 key）→ 加參數/加 pipeline 不破 ABI。
//     指標類（GPU/host buffer）不進 json：input/output 用 struct、precomputed_col_mean 獨立引數。
// 注意：同步 cudaMemcpy 與 stream kernel 混用，靠 legacy default stream 隱式同步才正確；
//       勿開 --default-stream per-thread（與舊 picoater_api 行為一致）。

#ifdef _WIN32
#define TANUKI_PIPELINE_API __declspec(dllexport)
#else
#define TANUKI_PIPELINE_API
#endif

#include <cstdint>

extern "C" {

typedef void* TanukiPipelineHandle;

typedef struct TanukiPipelineInputC {
    int width;
    int height;
    const uint8_t* data;   // host 灰階 buffer
    void* stream;          // cudaStream_t（可 null）
} TanukiPipelineInputC;

typedef struct TanukiPipelineOutputC {
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
} TanukiPipelineOutputC;

// 建 pipeline。pipeline_name 如 "find_stream_ridgeline"；json_options 可 NULL（如 {"ridge_method":"hessian"}）。
// 未知 pipeline / 方法回 NULL。
TANUKI_PIPELINE_API TanukiPipelineHandle TanukiPipeline_Create(
    const char* pipeline_name,
    const char* json_options);

// 跑一幀。json_params 範例：
//   {"bg_sigma_factor":1.0,"ridge_sigma":1.0,"hessian_max_factor":0.3,"ridge_mode":"vertical+horizontal"}
// precomputed_col_mean：host float*（大小 = width）；NULL = 每幀自算背景。
// 回 0 成功；-1 參數錯；-2 執行錯（GetLastError 查）。
TANUKI_PIPELINE_API int TanukiPipeline_Process(
    TanukiPipelineHandle handle,
    const TanukiPipelineInputC* input,
    const char* json_params,
    const float* precomputed_col_mean,
    const TanukiPipelineOutputC* output);

TANUKI_PIPELINE_API const char* TanukiPipeline_GetLastError(TanukiPipelineHandle handle);

TANUKI_PIPELINE_API void TanukiPipeline_Destroy(TanukiPipelineHandle handle);

// 工具：robust column mean（背景採集用；bg_sigma_factor 為離群門檻）。out_col_mean = host float*，大小 = width。
TANUKI_PIPELINE_API int TanukiPipeline_ComputeColumnMean(
    TanukiPipelineHandle handle,
    const TanukiPipelineInputC* input,
    float bg_sigma_factor,
    float* out_col_mean);

}  // extern "C"
