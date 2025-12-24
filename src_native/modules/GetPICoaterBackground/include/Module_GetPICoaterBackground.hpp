//Module_GetPICoaterBackground.hpp


#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace picoater {

    void GetPICoaterBackground_gpu(
        const uint8_t* d_in,
        uint8_t* d_bg_out,    // 為了 debug 用的背景圖
        uint8_t* d_mura_out,  // 最終結果
        int W, int H,
        float sigmaFactor,
        cudaStream_t s = 0
    );

}

