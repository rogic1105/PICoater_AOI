
#include "core_cv/base/cuda_utils.hpp"
#include "features_kernels.cuh"
#include "imgproc/utils/utils_kernels.cuh"
#include "core_cv/imgproc/core_filters.hpp"
#include "core_cv/imgproc/core_utils.hpp"

namespace tanuki { namespace core {

    void sobel_u8_gpu(const uint8_t* d_in, uint8_t* d_out, int W, int H, cudaStream_t s) {
        dim3 grid, block;
        get_optimal_launch_2d(k_sobelMagnitude_u8, W, H, grid, block);
        k_sobelMagnitude_u8 << <grid, block, 0, s >> > (d_in, d_out, W, H);
        CUDA_CHECK(cudaGetLastError());
    }

    void computeHessianResponse_gpu(
        const float* d_src,
        float* d_dst,
        int width,
        int height,
        detectionMode mode,
        cudaStream_t stream
    ) {
        int num_pixels = width * height;
        int grid, block;

        // �N���h�� launch �޿�ʸ˦b�o��
        get_optimal_launch_1d(k_hessianResponse, num_pixels, grid, block);
        k_hessianResponse << < grid, block, 0, stream >> > (d_src, d_dst, width, height, mode);

        // �o�̥i�H�� Error Check
        CUDA_CHECK(cudaGetLastError());
    }

    void hessianRidge_u8_gpu(
        const uint8_t* d_in,
        uint8_t* d_out,
        int W, int H,
        float sigma,
        const char* mode_str,
        float hessianMaxFactor,
        float* d_temp_blur_f32,
        float* d_temp_response,
        cudaStream_t s,
        void* d_workspace

    ) {
        int num_pixels = W * H;
        int gridSize, blockSize;


        // �����h���T
        int ksize = (int)(6.0f * sigma + 1.0f);
        if (ksize % 2 == 0) ksize++;
        tanuki::core::gaussianBlur_gpu<uint8_t, float>(d_in, d_temp_blur_f32, W, H, sigma, ksize, s, d_workspace);

        detectionMode mode = detectionMode::VERTICAL;
        if (strcmp(mode_str, "horizontal") == 0) mode = detectionMode::HORIZONTAL;
        else if (strcmp(mode_str, "both") == 0) mode = detectionMode::BOTH;
        tanuki::core::computeHessianResponse_gpu(d_temp_blur_f32, d_temp_response, W, H, mode, s);

        float scale_factor = 255.0f / hessianMaxFactor;
        tanuki::core::scale_clamp_f32_to_u8_gpu(d_temp_response, d_out, num_pixels, scale_factor, s);

    }


}}  // namespace core, tanuki