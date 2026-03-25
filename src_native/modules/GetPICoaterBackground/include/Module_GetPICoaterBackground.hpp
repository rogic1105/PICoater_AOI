#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <vector> // �s�W

namespace picoater {

    class PICoaterDetector {
    public:
        PICoaterDetector();
        ~PICoaterDetector();

        void Initialize(int width, int height);

        // �즳�� GPU Run
        void Run(
            const uint8_t* d_in,
            uint8_t* d_bg_out,
            uint8_t* d_mura_out,
            uint8_t* d_ridge_out,
            float* d_mura_curve_mean,
            float* d_mura_curve_max,
            float* d_mura_row_curve_mean,
            float* d_mura_row_curve_max,
            float bgSigmaFactor,
            float ridgeSigma,
            float hessianMaxFactor,
            const char* ridgeMode,
            cudaStream_t stream = 0
        );

        // [�s�W] CPU Run ����
        // ���u�]�e��B�æs��
        void RunCPU(
            const uint8_t* h_in,
            uint8_t* h_mura_out, // �o�̹��� GPU �� d_mura_out
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

        // [�s�W] CPU Buffers
        // �ϥ� std::vector �޲z����K�A�۰�����
        std::vector<float> h_col_mean;
    };
}