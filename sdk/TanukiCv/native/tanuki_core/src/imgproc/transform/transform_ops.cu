
#include "tanuki/core/base/cuda_utils.hpp"
#include "tanuki/core/imgproc/core_transform.hpp"


#include "transform_kernels.cuh"
#include <vector>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


namespace tanuki { namespace core {

    void resize_u8_gpu(const uint8_t* d_src, int src_w, int src_h,
        uint8_t* d_dst, int dst_w, int dst_h,
        cudaStream_t stream) {

        dim3 gridDim, blockDim;



        get_optimal_launch_2d(k_resize_nearest_u8, dst_w, dst_h, gridDim, blockDim);


        k_resize_nearest_u8 << <gridDim, blockDim, 0, stream >> > (
            d_src, src_w, src_h, d_dst, dst_w, dst_h
            );

        CUDA_CHECK(cudaGetLastError());
    }

    void downsample_max_f32_to_f16_gpu(const float* d_src, int src_w, int src_h,
        uint16_t* d_dst, int dst_w, int dst_h,
        cudaStream_t stream) {
        dim3 gridDim, blockDim;
        get_optimal_launch_2d(k_downsample_max_f32_to_f16, dst_w, dst_h, gridDim, blockDim);
        k_downsample_max_f32_to_f16<<<gridDim, blockDim, 0, stream>>>(
            d_src, src_w, src_h, d_dst, dst_w, dst_h);
        CUDA_CHECK(cudaGetLastError());
    }

}}  // namespace core, tanuki
