// PICoater_AOI\src_native\modules\GetPICoaterBackground\src\Module_GetPICoaterBackground.cu

#include "Module_GetPICoaterBackground.hpp"
#include "core_cv/base/cuda_utils.hpp"

#include "core_cv/imgproc/core_filters.hpp"
#include "core_cv/imgproc/core_background.hpp"
#include "core_cv/imgproc/core_features.hpp"

namespace picoater {

    // [Helper] 記憶體對齊計算工具 (對齊到 256 bytes，符合 CUDA 最佳存取粒度)
    inline size_t alignUp(size_t offset, size_t alignment = 256) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    PICoaterDetector::PICoaterDetector() {}
    PICoaterDetector::~PICoaterDetector() { Release(); }

    void PICoaterDetector::Release() {
        // 只需要釋放 "擁有權" 的記憶體
        if (d_col_bg_) cudaFree(d_col_bg_);
        if (d_blur_tmp_) cudaFree(d_blur_tmp_);

        // [關鍵] 只需要釋放總 workspace，內部的指標只是借用位址，不需要 free
        if (d_workspace_) cudaFree(d_workspace_);

        d_col_bg_ = nullptr;
        d_blur_tmp_ = nullptr;
        d_workspace_ = nullptr;

        // 歸零指標，避免懸空
        d_hessian_u8_ = nullptr;
        d_hessian_f32_ = nullptr;
        d_hessian_resp_ = nullptr;
    }

    void PICoaterDetector::Initialize(int width, int height) {
        if (m_width == width && m_height == height) return;

        Release();
        m_width = width;
        m_height = height;
        size_t num_pixels = width * height;

        CUDA_CHECK(cudaMalloc(&d_col_bg_, num_pixels * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_blur_tmp_, num_pixels * sizeof(uint8_t)));

        // --- 計算 Hessian 需要的偏移 (用於指派指標) ---
        size_t offset = 0;
        auto alignUp = [](size_t off) { return (off + 255) & ~255; };

        size_t off_u8 = alignUp(offset);
        offset = off_u8 + num_pixels * sizeof(uint8_t);

        size_t off_f32 = alignUp(offset);
        offset = off_f32 + num_pixels * sizeof(float);

        size_t off_resp = alignUp(offset);
        offset = off_resp + num_pixels * sizeof(float);

        size_t hessian_req_size = offset; // Hessian 需要這麼多

        // --- 計算 Gaussian 需要的大小 ---
        // Gaussian 需要 3 個 float buffer + mask (假設 mask 1KB)
        size_t gaussian_req_size = (num_pixels * sizeof(float)) * 3 + 1024;

        // [關鍵] 總大小取兩者之最大值
        size_t total_size = (gaussian_req_size > hessian_req_size) ? gaussian_req_size : hessian_req_size;

        // --- 單次分配 ---
        CUDA_CHECK(cudaMalloc(&d_workspace_, total_size));

        // --- 指標指派 (給 Hessian 用) ---
        // 這些指標在 Run Hessian 時會用到
        uint8_t* base = (uint8_t*)d_workspace_;
        d_hessian_u8_ = (uint8_t*)(base + off_u8);
        d_hessian_f32_ = (float*)(base + off_f32);
        d_hessian_resp_ = (float*)(base + off_resp);
    }

    void PICoaterDetector::Run(
        const uint8_t* d_in,
        uint8_t* d_bg_out,
        uint8_t* d_mura_out,
        uint8_t* d_ridge_out,
        float bgSigmaFactor,
        float ridgeSigma,
        const char* ridgeMode,
        cudaStream_t stream
    ) {
        if (m_width == 0) return;

        // 1. 背景計算流程
        // [關鍵修正] 直接傳 d_workspace_ (總記憶體池) 給 Gaussian
        // 因為 Initialize 已經確保 d_workspace_ 足夠容納 Gaussian 所需的 3 個 float buffers

        core::gaussianBlur_u8_gpu(d_in, d_blur_tmp_, m_width, m_height, 5.0f, 11, stream, d_workspace_);

        core::calcColumnBackground_u8_gpu(d_blur_tmp_, d_col_bg_, m_width, m_height, bgSigmaFactor, stream);
        core::expandBackground_u8_gpu(d_col_bg_, d_bg_out, m_width, m_height, stream);
        core::subtractBackgroundAbs_u8_gpu(d_blur_tmp_, d_col_bg_, d_mura_out, m_width, m_height, stream);

        // 2. 脊線計算流程
        // 這裡會複用 d_workspace_ 的前段部分，但這時候 Gaussian 已經做完了，所以安全
        core::hessianRidge_u8_gpu(
            d_mura_out,
            d_ridge_out,
            m_width, m_height,
            ridgeSigma,
            ridgeMode,
            stream,
            d_hessian_u8_,
            d_hessian_f32_,
            d_hessian_resp_
        );
    }
}