#pragma once

#ifdef _WIN32
#define PICOATER_API __declspec(dllexport)
#else
#define PICOATER_API
#endif

#include <cstdint>

extern "C" {

typedef void* AoiPipelineHandle;

PICOATER_API AoiPipelineHandle PICoaterAPI_CreatePipeline();

PICOATER_API int PICoaterAPI_ProcessPipeline(
    AoiPipelineHandle handle,
    int width,
    int height,
    const uint8_t* d_input,
    uint8_t* d_background_output,
    uint8_t* d_mura_output,
    uint8_t* d_ridge_output,
    float* d_mura_curve_mean_output,
    float* d_mura_curve_max_output,
    float bg_sigma_factor,
    float ridge_sigma,
    float hessian_max_factor,
    const char* ridge_mode,
    void* stream);

PICOATER_API const char* PICoaterAPI_GetLastError(AoiPipelineHandle handle);

PICOATER_API void PICoaterAPI_DestroyPipeline(AoiPipelineHandle handle);

}  // extern "C"
