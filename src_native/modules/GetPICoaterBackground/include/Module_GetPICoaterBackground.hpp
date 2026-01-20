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

        /**
         * @brief 執行檢測流程
         * * @param d_in 輸入影像 (Device Pointer, 8-bit)
         * @param d_bg_out 輸出背景 (Device Pointer, 8-bit)
         * @param d_mura_out 輸出 Mura (Device Pointer, 8-bit)
         * @param d_ridge_out 輸出 Ridge (Device Pointer, 8-bit)
         * @param d_mura_curve_out 輸出曲線 (Device Pointer, Float)
         * @param d_heatmap_out [新增] 輸出疊加圖 (Device Pointer, BGR, w*h*3 bytes). 若為 nullptr 則不執行疊圖。
         * @param bgSigmaFactor 背景 sigma 參數
         * @param ridgeSigma Ridge sigma 參數
         * @param heatmap_lower_thres [新增] Heatmap 顯示下限 (低於此值的 Ridge 不顯示顏色)
         * @param heatmap_alpha [新增] 原圖權重 (0.0~1.0)
         * @param ridgeMode Ridge 模式 ("vertical" 等)
         * @param stream CUDA Stream
         */
        void Run(
            const uint8_t* d_in,
            uint8_t* d_bg_out,
            uint8_t* d_mura_out,
            uint8_t* d_ridge_out,
            float* d_mura_curve_out,
            uint8_t* d_heatmap_out,
            float bgSigmaFactor,
            float ridgeSigma,
            int heatmap_lower_thres,
            float heatmap_alpha,
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