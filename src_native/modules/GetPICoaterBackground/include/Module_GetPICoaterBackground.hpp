#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <vector> // 新增

namespace picoater {

    class PICoaterDetector {
    public:
        PICoaterDetector();
        ~PICoaterDetector();

        void Initialize(int width, int height);

        // 原有的 GPU Run
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

        // [新增] CPU Run 介面
        // 先只跑前兩步並存圖
        void RunCPU(
            const uint8_t* h_in,
            uint8_t* h_mura_out, // 這裡對應 GPU 的 d_mura_out
            float bgSigmaFactor
        );

        void Release();

    private:
        int m_width = 0;
        int m_height = 0;

        // GPU Buffers
        float* d_col_mean = nullptr;
        uint8_t* d_col_bg_ = nullptr;
        uint8_t* d_blur_tmp_ = nullptr;
        void* d_workspace_ = nullptr;

        // GPU View Pointers
        uint8_t* d_hessian_u8_ = nullptr;
        float* d_hessian_f32_ = nullptr;
        float* d_hessian_resp_ = nullptr;

        // [新增] CPU Buffers
        // 使用 std::vector 管理較方便，自動釋放
        std::vector<float> h_col_mean;
    };
}