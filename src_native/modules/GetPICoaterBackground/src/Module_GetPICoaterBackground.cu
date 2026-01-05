// PICoater_AOI\src_native\modules\GetPICoaterBackground\src\Module_GetPICoaterBackground.cu

#include "Module_GetPICoaterBackground.hpp"
#include "core_cv/imgproc/core_ops.hpp"
#include "core_cv/base/cuda_utils.hpp"

namespace picoater {

    void GetPICoaterBackground_gpu(
        const uint8_t* d_in,
        uint8_t* d_bg_out,      // [Output] 擴展後的背景圖
        uint8_t* d_mura_out,    // [Output] Mura 圖 (AbsDiff)
        uint8_t* d_ridge_out,   // [Output] Ridge 偵測結果
        int width, int height,
        float bgSigmaFactor,    // 背景計算用的 Sigma
        float ridgeSigma,       // Ridge 偵測用的 Sigma
        const char* ridgeMode,  // Ridge 偵測模式
        cudaStream_t stream
    ) {
        // 1. 準備暫存記憶體
        // d_col_bg: 1D 背景
        // d_blur_tmp: 高斯模糊後的暫存圖 (新增)
        uint8_t* d_col_bg = nullptr;
        uint8_t* d_blur_tmp = nullptr;

        CUDA_CHECK(cudaMalloc(&d_col_bg, width * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_blur_tmp, width * height * sizeof(uint8_t)));


        // Step 0: 預處理 - Gaussian Blur
        core::gaussianBlur_u8_gpu(d_in, d_blur_tmp, width, height, 5.0f, 11, stream);


        // 2. 計算 1D 背景 (Column-wise)
        core::calcColumnBackground_u8_gpu(d_blur_tmp, d_col_bg, width, height, bgSigmaFactor, stream);

        // 3. 擴展背景 (1D -> 2D) 
        core::expandBackground_u8_gpu(d_col_bg, d_bg_out, width, height, stream);

        // 4. 計算 Mura (模糊圖 - 背景)
        core::subtractBackgroundAbs_u8_gpu(d_blur_tmp, d_col_bg, d_mura_out, width, height, stream);

        // 5. 計算 Hessian Ridge 
        core::hessianRidge_u8_gpu(d_mura_out, d_ridge_out, width, height, ridgeSigma, ridgeMode, stream);

        // 6. 清理暫存
        CUDA_CHECK(cudaFree(d_col_bg));
        CUDA_CHECK(cudaFree(d_blur_tmp));
    }

}