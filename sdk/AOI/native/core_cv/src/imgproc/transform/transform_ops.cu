// AOI_SDK\core_cv\src\imgproc\transform\transform_ops.cu

#include "core_cv/base/cuda_utils.hpp"
#include "core_cv/imgproc/core_transform.hpp"


#include "transform_kernels.cuh"
#include <vector>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>


namespace core {

    void resize_u8_gpu(const uint8_t* d_src, int src_w, int src_h,
        uint8_t* d_dst, int dst_w, int dst_h,
        cudaStream_t stream) {

        dim3 gridDim, blockDim;

        // [關鍵] 使用你提供的 2D Launch Helper
        // 它會自動算出最佳的 blockDim (例如 32x32 或 32x16)
        get_optimal_launch_2d(k_resize_nearest_u8, dst_w, dst_h, gridDim, blockDim);

        // 啟動 Kernel
        k_resize_nearest_u8 << <gridDim, blockDim, 0, stream >> > (
            d_src, src_w, src_h, d_dst, dst_w, dst_h
            );

        CUDA_CHECK(cudaGetLastError());
    }

}