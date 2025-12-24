//Module_GetPICoaterBackground.cu

#include "Module_GetPICoaterBackground.hpp"
#include "core/core_ops.hpp"   // 改引用 ops
#include "core/cuda_utils.hpp" // 為了 CUDA_CHECK

namespace picoater {

    void GetPICoaterBackground_gpu(
        const uint8_t* d_in,
        uint8_t* d_bg_out,      // 輸出的擴展背景圖 (H x W)
        uint8_t* d_mura_out,    // 輸出的 Mura 圖 (H x W)
        int W, int H,
        float sigmaFactor,
        cudaStream_t s
    ) {
        // 1. 準備暫存 (1D Column Background)
        uint8_t* d_col_bg = nullptr;
        CUDA_CHECK(cudaMalloc(&d_col_bg, W * sizeof(uint8_t)));

        // 2. Step 1: 計算 1D 背景
        core::calcColumnBackground_u8_gpu(d_in, d_col_bg, W, H, sigmaFactor, s);

        // 3. Step 2: 擴展背景 (1D -> 2D) 用於存圖驗證
        core::expandBackground_u8_gpu(d_col_bg, d_bg_out, W, H, s);

        // 4. Step 3: 計算 Mura (原圖 - 背景 + 127)
        core::subtractBackground_u8_gpu(d_in, d_col_bg, d_mura_out, W, H, s);

        // 5. 清理暫存
        CUDA_CHECK(cudaFree(d_col_bg));
    }
}