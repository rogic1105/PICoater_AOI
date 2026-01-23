// PICoater_AOI\src_native\modules\GetPICoaterBackground\include\Module_GetPICoaterBackground.hpp

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace picoater {

    class PICoaterDetector {
    public:
        PICoaterDetector();
        ~PICoaterDetector();

        void Initialize(int width, int height);

        void Run(
            const uint8_t* d_in,
            uint8_t* d_bg_out,
            uint8_t* d_mura_out,
            uint8_t* d_ridge_out,
            float* d_mura_curve_mean,
            float* d_mura_curve_max,
            float bgSigmaFactor,
            float ridgeSigma,
            float hessianMaxFactor,
            const char* ridgeMode,
            cudaStream_t stream = 0
        );

        void Release();

    private:
        int m_width = 0;
        int m_height = 0;

        // 主要輸出與中間緩衝
        float* d_col_mean = nullptr;
        uint8_t* d_col_bg_ = nullptr;
        uint8_t* d_blur_tmp_ = nullptr; // 給背景運算用的暫存

        // [新增] 這是效能優化的關鍵：單一對齊記憶體池
        // 這是真正被 cudaMalloc 和 cudaFree 的對象
        void* d_workspace_ = nullptr;

        // 下面這些只是 "View" (指向 workspace 內部的指標)，不需要 free
        uint8_t* d_hessian_u8_ = nullptr;
        float* d_hessian_f32_ = nullptr;
        float* d_hessian_resp_ = nullptr;
    };

}