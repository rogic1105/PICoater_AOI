//Module_GetPICoaterBackground.hpp


#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace picoater {

    void GetPICoaterBackground_gpu(
        const uint8_t* d_in,
        uint8_t* d_bg_out,
        uint8_t* d_mura_out,
        uint8_t* d_ridge_out, // 新增
        int width, int height,
        float bgSigmaFactor,
        float ridgeSigma,     // 新增
        const char* ridgeMode,// 新增
        cudaStream_t stream = 0
    );

}

