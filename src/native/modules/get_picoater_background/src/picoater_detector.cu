#include "picoater_detector.hpp"
#include "core_cv/base/cuda_utils.hpp"

#include "core_cv/imgproc/core_filters.hpp"
#include "core_cv/imgproc/core_background.hpp"
#include "core_cv/imgproc/core_features.hpp"
#include "core_cv/imgproc/core_utils.hpp" // for overlay_heatmap_gpu

#include "cpp_utils/timer_utils.hpp"

namespace picoater {

    // Align an offset up to `alignment` bytes (default 256, CUDA-friendly).
    inline size_t alignUp(size_t offset, size_t alignment = 256) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    // Neutralized curve: multiply a float array in place by scale_factor, no clamp (keep peaks).
    __global__ void k_scale_f32_inplace(float* d, int N, float s) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) d[i] *= s;
    }

    inline void scale_f32_inplace_gpu(float* d_buf, int N, float scale, cudaStream_t stream) {
        if (d_buf == nullptr || N <= 0) return;
        int block = 256;
        int grid  = (N + block - 1) / block;
        k_scale_f32_inplace<<<grid, block, 0, stream>>>(d_buf, N, scale);
    }

    PICoaterDetector::PICoaterDetector() {}
    PICoaterDetector::~PICoaterDetector() { Release(); }

    void PICoaterDetector::Release() {
        // Only free the buffers we actually own.
        if (d_col_mean) cudaFree(d_col_mean);
        if (d_col_bg_) cudaFree(d_col_bg_);
        if (d_blur_tmp_) cudaFree(d_blur_tmp_);

        // Only free the single workspace; the hessian pointers are views into it (no free).
        if (d_workspace_) cudaFree(d_workspace_);

        d_col_mean = nullptr;
        d_col_bg_ = nullptr;
        d_blur_tmp_ = nullptr;
        d_workspace_ = nullptr;

        // Null the view pointers to avoid dangling references.
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

        CUDA_CHECK(cudaMalloc(&d_col_mean, width * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_col_bg_, num_pixels * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_blur_tmp_, num_pixels * sizeof(uint8_t)));

        // --- Workspace size needed by the Hessian path ---
        size_t offset = 0;
        auto alignUp = [](size_t off) { return (off + 255) & ~255; };

        size_t off_u8 = alignUp(offset);
        offset = off_u8 + num_pixels * sizeof(uint8_t);

        size_t off_f32 = alignUp(offset);
        offset = off_f32 + num_pixels * sizeof(float);

        size_t off_resp = alignUp(offset);
        offset = off_resp + num_pixels * sizeof(float);

        size_t hessian_req_size = offset; // bytes the Hessian path needs

        // --- Workspace size needed by the Gaussian path ---
        // Gaussian needs 3 float buffers + mask (assume mask ~1KB).
        size_t gaussian_req_size = (num_pixels * sizeof(float)) * 3 + 1024;

        // Allocate one workspace sized for the larger of the two.
        size_t total_size = (gaussian_req_size > hessian_req_size) ? gaussian_req_size : hessian_req_size;

        // --- Single allocation ---
        CUDA_CHECK(cudaMalloc(&d_workspace_, total_size));

        // --- Sub-pointers (views into workspace, used by the Hessian path in Run) ---
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
        float* d_mura_curve_mean,
        float* d_mura_curve_max,
        float* d_mura_row_curve_mean,
        float* d_mura_row_curve_max,
        float bgSigmaFactor,
        float ridgeSigma,
        float hessianMaxFactor,
        const char* ridgeMode,
        const float* d_precomputed_col_mean,
        cudaStream_t stream
    ) {
        if (m_width == 0) return;

        int sigma_col = 1;

        TIME_SCOPE_MS_SYNC("Total Run Time", cudaStreamSynchronize(stream));

        // Step 1: Column Means (skip if precomputed background provided)
        if (d_precomputed_col_mean != nullptr) {
            cudaMemcpyAsync(d_col_mean, d_precomputed_col_mean,
                            m_width * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        } else {
            tanuki::core::calcColumnMeans_RemoveOutliers_gpu(d_in, d_col_mean, m_width, m_height, sigma_col, stream);
        }

        // Step 2: Background Removal -> d_mura_out (bg-removed image)
        tanuki::core::calcColumnBackground_u8_gpu(d_in, d_col_mean, d_mura_out, m_width, m_height, stream);

        // Step 3: Gaussian blur (once, shared by all ridge directions)
        int ksize = (int)(6.0f * ridgeSigma + 1.0f);
        if (ksize % 2 == 0) ksize++;
        tanuki::core::gaussianBlur_gpu<uint8_t, float>(
            d_mura_out, d_hessian_f32_, m_width, m_height, ridgeSigma, ksize, stream, d_workspace_);

        // Parse ridgeMode: "vertical", "horizontal", "vertical+horizontal"
        bool doVertical   = (strcmp(ridgeMode, "vertical") == 0 || strcmp(ridgeMode, "vertical+horizontal") == 0);
        bool doHorizontal = (strcmp(ridgeMode, "horizontal") == 0 || strcmp(ridgeMode, "vertical+horizontal") == 0);

        float scale_factor = 255.0f / hessianMaxFactor;
        int num_pixels = m_width * m_height;

        // Step 4: Vertical ridge -> d_ridge_out + col curves
        if (doVertical) {
            tanuki::core::computeHessianResponse_gpu(d_hessian_f32_, d_hessian_resp_, m_width, m_height,
                                             tanuki::core::detectionMode::VERTICAL, stream);

            // Neutralized: curves are computed from the raw float Hessian response
            // (before clamp+u8 quantization), then scaled by 255/normValue (no clamp),
            // preserving peak info so thresholds can be re-tuned at review time.
            tanuki::core::calcColumnMeans_gpu<float>(
                d_hessian_resp_, d_mura_curve_mean, m_width, m_height, stream, d_workspace_);
            tanuki::core::calcColumnMax_gpu<float>(
                d_hessian_resp_, d_mura_curve_max, m_width, m_height, stream);
            scale_f32_inplace_gpu(d_mura_curve_mean, m_width, scale_factor, stream);
            scale_f32_inplace_gpu(d_mura_curve_max,  m_width, scale_factor, stream);

            // u8 ridge image (for display; still goes through the scale+clamp path)
            tanuki::core::scale_clamp_f32_to_u8_gpu(d_hessian_resp_, d_ridge_out, num_pixels, scale_factor, stream);
        }

        // Step 5: Horizontal ridge + row curves
        if (doHorizontal) {
            tanuki::core::computeHessianResponse_gpu(d_hessian_f32_, d_hessian_resp_, m_width, m_height,
                                             tanuki::core::detectionMode::HORIZONTAL, stream);

            // Neutralized: row curves from raw float Hessian response, then 255/normValue scale (no clamp).
            if (d_mura_row_curve_mean != nullptr) {
                tanuki::core::calcRowMeans_gpu<float>(d_hessian_resp_, d_mura_row_curve_mean, m_width, m_height, stream);
                scale_f32_inplace_gpu(d_mura_row_curve_mean, m_height, scale_factor, stream);
            }
            if (d_mura_row_curve_max != nullptr) {
                tanuki::core::calcRowMax_gpu<float>(d_hessian_resp_, d_mura_row_curve_max, m_width, m_height, stream);
                scale_f32_inplace_gpu(d_mura_row_curve_max, m_height, scale_factor, stream);
            }

            // u8 ridge image (for display)
            if (doVertical) {
                // vertical+horizontal: horizontal image -> d_mura_out (overwrite bg-removed, already consumed)
                tanuki::core::scale_clamp_f32_to_u8_gpu(d_hessian_resp_, d_mura_out, num_pixels, scale_factor, stream);
            } else {
                // horizontal only: horizontal image -> d_ridge_out (main output)
                tanuki::core::scale_clamp_f32_to_u8_gpu(d_hessian_resp_, d_ridge_out, num_pixels, scale_factor, stream);
            }
        }

    }
}
