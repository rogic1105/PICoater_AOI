// AOI_SDK\core_cv_api\src\export_api.cpp

#include "export_c/export_api.h"

#include "core_cv/imgproc/core_background.hpp"
#include "core_cv/imgproc/core_enhance.hpp"
#include "core_cv/imgproc/core_features.hpp"
#include "core_cv/imgproc/core_filters.hpp"
#include "core_cv/imgproc/core_utils.hpp"
// [新增] 引用 Fast IO 與 Memory 模組
#include "core_cv/imgcodecs/core_imgcodecs_fast.hpp"
#include "core_cv/base/cuda_memory.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

#define CHECK_CUDA(call)                                                  \
  {                                                                       \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
      std::cerr << "[CoreCV API] CUDA Error: "                            \
                << cudaGetErrorString(err) << "\n";                       \
      return CORE_CV_ERROR_CUDA;                                          \
    }                                                                     \
  }

extern "C" {

    // --- [新增] 記憶體與 IO 實作 ---

    CORE_CV_API unsigned char* CoreCV_AllocPinned(unsigned long long size) {
        return (unsigned char*)core::alloc_pinned_memory((size_t)size);
    }

    CORE_CV_API void CoreCV_FreePinned(unsigned char* ptr) {
        core::free_pinned_memory(ptr);
    }

    CORE_CV_API bool CoreCV_FastReadBMP(const char* filepath, int* w, int* h, unsigned char* outBuffer, int bufferSize) {
        try {
            int width = 0, height = 0;
            bool res = core::fast_read_bmp_8bit(filepath, width, height, outBuffer, bufferSize);
            if (res) {
                if (w) *w = width;
                if (h) *h = height;
            }
            return res;
        }
        catch (...) { return false; }
    }

    CORE_CV_API bool CoreCV_FastWriteBMP(const char* filepath, int w, int h, const unsigned char* inBuffer) {
        try {
            return core::fast_write_bmp_8bit(filepath, w, h, inBuffer);
        }
        catch (...) { return false; }
    }


    // --- 影像處理實作 (保持原本邏輯，但在 Pinned Memory 下會變快) ---

    int CoreCV_Brighten(const uint8_t* src_ptr, int width, int height, int value, uint8_t* dst_ptr) {
        if (!src_ptr || !dst_ptr) return CORE_CV_ERROR_NULL_POINTER;
        if (width <= 0 || height <= 0) return CORE_CV_ERROR_INVALID_PARAM;

        size_t size = static_cast<size_t>(width) * height * sizeof(uint8_t);
        uint8_t* d_in = nullptr;
        uint8_t* d_out = nullptr;

        try {
            CHECK_CUDA(cudaMalloc(&d_in, size));
            CHECK_CUDA(cudaMalloc(&d_out, size));

            // 如果 src_ptr 是透過 CoreCV_AllocPinned 分配的，這行 Memcpy 會自動變成 Async DMA
            CHECK_CUDA(cudaMemcpy(d_in, src_ptr, size, cudaMemcpyHostToDevice));

            core::brighten_u8_gpu(d_in, d_out, width, height, value, 0);

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(dst_ptr, d_out, size, cudaMemcpyDeviceToHost));

            cudaFree(d_in);
            cudaFree(d_out);
            return CORE_CV_SUCCESS;
        }
        catch (const std::exception& e) {
            std::cerr << "[CoreCV API] Exception: " << e.what() << "\n";
            if (d_in) cudaFree(d_in);
            if (d_out) cudaFree(d_out);
            return CORE_CV_ERROR_UNKNOWN;
        }
    }

    int CoreCV_Threshold(const uint8_t* src_ptr, int width, int height, uint8_t threshold, uint8_t* dst_ptr) {
        if (!src_ptr || !dst_ptr) return CORE_CV_ERROR_NULL_POINTER;
        size_t size = static_cast<size_t>(width) * height * sizeof(uint8_t);
        uint8_t* d_in = nullptr;
        uint8_t* d_out = nullptr;

        try {
            CHECK_CUDA(cudaMalloc(&d_in, size));
            CHECK_CUDA(cudaMalloc(&d_out, size));

            CHECK_CUDA(cudaMemcpy(d_in, src_ptr, size, cudaMemcpyHostToDevice));

            core::threshold_u8_gpu(d_in, d_out, width, height, threshold, 0);
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(dst_ptr, d_out, size, cudaMemcpyDeviceToHost));

            cudaFree(d_in);
            cudaFree(d_out);
            return CORE_CV_SUCCESS;
        }
        catch (...) {
            if (d_in) cudaFree(d_in);
            if (d_out) cudaFree(d_out);
            return CORE_CV_ERROR_UNKNOWN;
        }
    }

    int CoreCV_Invert(const uint8_t* src_ptr, int width, int height, uint8_t* dst_ptr) {
        if (!src_ptr || !dst_ptr) return CORE_CV_ERROR_NULL_POINTER;
        size_t size = static_cast<size_t>(width) * height * sizeof(uint8_t);
        uint8_t* d_in = nullptr;
        uint8_t* d_out = nullptr;

        try {
            CHECK_CUDA(cudaMalloc(&d_in, size));
            CHECK_CUDA(cudaMalloc(&d_out, size));
            CHECK_CUDA(cudaMemcpy(d_in, src_ptr, size, cudaMemcpyHostToDevice));

            core::invert_u8_gpu(d_in, d_out, width, height, 0); // 呼叫 core_ops (請確認已實作)
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(dst_ptr, d_out, size, cudaMemcpyDeviceToHost));
            cudaFree(d_in);
            cudaFree(d_out);
            return CORE_CV_SUCCESS;
        }
        catch (...) {
            if (d_in) cudaFree(d_in);
            if (d_out) cudaFree(d_out);
            return CORE_CV_ERROR_UNKNOWN;
        }
    }

    int CoreCV_Convolution(const uint8_t* src_ptr, int width, int height, const float* mask_ptr, int mask_size, uint8_t* dst_ptr) {
        if (!src_ptr || !dst_ptr || !mask_ptr) return CORE_CV_ERROR_NULL_POINTER;
        size_t img_size = static_cast<size_t>(width) * height * sizeof(uint8_t);
        size_t mask_bytes = static_cast<size_t>(mask_size) * mask_size * sizeof(float);
        uint8_t* d_in = nullptr;
        uint8_t* d_out = nullptr;
        float* d_mask = nullptr;

        try {
            CHECK_CUDA(cudaMalloc(&d_in, img_size));
            CHECK_CUDA(cudaMalloc(&d_out, img_size));
            CHECK_CUDA(cudaMalloc(&d_mask, mask_bytes));

            CHECK_CUDA(cudaMemcpy(d_in, src_ptr, img_size, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_mask, mask_ptr, mask_bytes, cudaMemcpyHostToDevice));

            core::convolution_u8_gpu(d_in, d_out, width, height, d_mask, mask_size, 0);
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(dst_ptr, d_out, img_size, cudaMemcpyDeviceToHost));

            cudaFree(d_in);
            cudaFree(d_out);
            cudaFree(d_mask);
            return CORE_CV_SUCCESS;
        }
        catch (...) {
            if (d_in) cudaFree(d_in);
            if (d_out) cudaFree(d_out);
            if (d_mask) cudaFree(d_mask);
            return CORE_CV_ERROR_UNKNOWN;
        }
    }

    // ------------ GPU 記憶體管理 ---------------
    CORE_CV_API int CoreCV_MallocGPU(unsigned char** d_ptr, int width, int height) {
        size_t size = (size_t)width * height;
        CHECK_CUDA(cudaMalloc((void**)d_ptr, size));
        return CORE_CV_SUCCESS;
    }

    CORE_CV_API int CoreCV_FreeGPU(unsigned char* d_ptr) {
        if (d_ptr) CHECK_CUDA(cudaFree(d_ptr));
        return CORE_CV_SUCCESS;
    }

    CORE_CV_API int CoreCV_Upload(const unsigned char* h_src, unsigned char* d_dst, int width, int height) {
        size_t size = (size_t)width * height;
        // 如果 h_src 是 Pinned Memory，這裡會跑 Async DMA
        CHECK_CUDA(cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice));
        return CORE_CV_SUCCESS;
    }

    CORE_CV_API int CoreCV_Download(const unsigned char* d_src, unsigned char* h_dst, int width, int height) {
        size_t size = (size_t)width * height;
        CHECK_CUDA(cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost));
        return CORE_CV_SUCCESS;
    }

    // --- 純 GPU 運算 (極速版) ---
    CORE_CV_API int CoreCV_Brighten_GPU(const uint8_t* d_src, int width, int height, int value, uint8_t* d_dst) {
        // 沒有 Malloc，沒有 Memcpy，只有 Kernel Launch
        core::brighten_u8_gpu(d_src, d_dst, width, height, value, 0);
        // 不做 Sync，讓 CPU 可以馬上量測 Launch 時間 (或做 Sync 量測執行時間)
        CHECK_CUDA(cudaDeviceSynchronize());
        return CORE_CV_SUCCESS;
    }

    CORE_CV_API int CoreCV_Threshold_GPU(const uint8_t* d_src, int width, int height, uint8_t threshold, uint8_t* d_dst) {
        core::threshold_u8_gpu(d_src, d_dst, width, height, threshold, 0);
        CHECK_CUDA(cudaDeviceSynchronize());
        return CORE_CV_SUCCESS;
    }

    CORE_CV_API int CoreCV_Invert_GPU(const uint8_t* d_src, int width, int height, uint8_t* d_dst) {
        core::invert_u8_gpu(d_src, d_dst, width, height, 0);
        CHECK_CUDA(cudaDeviceSynchronize());
        return CORE_CV_SUCCESS;
    }

    // 注意：Mask 也必須已經在 GPU 上
    CORE_CV_API int CoreCV_Convolution_GPU(const uint8_t* d_src, int width, int height, const float* d_mask, int mask_size, uint8_t* d_dst) {
        core::convolution_u8_gpu(d_src, d_dst, width, height, d_mask, mask_size, 0);
        CHECK_CUDA(cudaDeviceSynchronize());
        return CORE_CV_SUCCESS;
    }

    // --- [新增] Float 資源實作 ---
    CORE_CV_API int CoreCV_MallocGPU_Float(float** d_ptr, int count) {
        size_t size = (size_t)count * sizeof(float);
        CHECK_CUDA(cudaMalloc((void**)d_ptr, size));
        return CORE_CV_SUCCESS;
    }

    CORE_CV_API int CoreCV_FreeGPU_Float(float* d_ptr) {
        if (d_ptr) CHECK_CUDA(cudaFree(d_ptr));
        return CORE_CV_SUCCESS;
    }

    CORE_CV_API int CoreCV_Upload_Float(const float* h_src, float* d_dst, int count) {
        size_t size = (size_t)count * sizeof(float);
        CHECK_CUDA(cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice));
        return CORE_CV_SUCCESS;
    }

}


