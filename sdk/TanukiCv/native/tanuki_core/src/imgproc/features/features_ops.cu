
#include "tanuki/core/base/cuda_utils.hpp"
#include "features_kernels.cuh"
#include "imgproc/utils/utils_kernels.cuh"
#include "tanuki/core/imgproc/core_filters.hpp"
#include "tanuki/core/imgproc/core_utils.hpp"

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


        get_optimal_launch_1d(k_hessianResponse, num_pixels, grid, block);
        k_hessianResponse << < grid, block, 0, stream >> > (d_src, d_dst, width, height, mode);


        CUDA_CHECK(cudaGetLastError());
    }

    // 註：原 hessianRidge_u8_gpu（blur+hessian+scale 組合）已移除 →
    //     改由 tanuki_pipeline 的 ridge_hessian module 負責（組合 core primitive）。

}}  // namespace core, tanuki