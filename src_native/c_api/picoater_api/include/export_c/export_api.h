#pragma once

#ifdef _WIN32
#define PICOATER_API __declspec(dllexport)
#else
#define PICOATER_API
#endif

#include <cstdint>

extern "C" {

typedef void* AoiPipelineHandle;
typedef void* PlcAdapterHandle;

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

PICOATER_API AoiPipelineHandle PICoaterAPI_CreatePipeline();

PICOATER_API int PICoaterAPI_ProcessPipeline(
    AoiPipelineHandle handle,
    const AoiInputImageC* input,
    const AoiAlgorithmParamsC* params,
    const AoiOutputBuffersC* output);

PICOATER_API const char* PICoaterAPI_GetLastError(AoiPipelineHandle handle);

PICOATER_API void PICoaterAPI_DestroyPipeline(AoiPipelineHandle handle);

PICOATER_API int PICoaterAPI_ComputeColumnMean(
    AoiPipelineHandle handle,
    const AoiInputImageC* input,
    float bg_sigma_factor,
    float* out_col_mean);  /* host buffer, size = input->width */

PICOATER_API PlcAdapterHandle PICoaterAPI_CreateMockPlc();

PICOATER_API void PICoaterAPI_DestroyPlc(PlcAdapterHandle handle);

PICOATER_API int PICoaterAPI_PlcConnect(PlcAdapterHandle handle);

PICOATER_API int PICoaterAPI_PlcReadBit(
    PlcAdapterHandle handle, int address, bool* value);

PICOATER_API int PICoaterAPI_PlcWriteBit(
    PlcAdapterHandle handle, int address, bool value);

}  // extern "C"
