
#include "export_c/export_api.h"

#include "tanuki/core/imgproc/core_background.hpp"
#include "tanuki/core/imgproc/core_enhance.hpp"
#include "tanuki/core/imgproc/core_features.hpp"
#include "tanuki/core/imgproc/core_filters.hpp"
#include "tanuki/core/imgproc/core_utils.hpp"
// [新增] 引用 Fast IO 與 Memory 模組
#include "tanuki/core/imgcodecs/core_imgcodecs_fast.hpp"
#include "tanuki/core/base/cuda_memory.hpp"
#include "tanuki/core/imgproc/core_transform.hpp"

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

    // --- [新增] GPU 暖身實作 ---
    // 在 GPU 內分配兩塊小 buffer、跑一個 threshold kernel、釋放。
    // 第一次 cudaMalloc / kernel launch 會強迫 CUDA context + driver 載入，
    // 把這成本提前付掉，避免之後第一張正式影像處理變慢。
    // 暖身細節全留在 native，caller 只需呼叫一次。
    TANUKI_CV_API int TanukiCv_WarmUp() {
        const int W = 64;
        const int H = 64;
        const size_t size = (size_t)W * H;

        uint8_t* d_src = nullptr;
        uint8_t* d_dst = nullptr;

        try {
            CHECK_CUDA(cudaMalloc((void**)&d_src, size));
            CHECK_CUDA(cudaMalloc((void**)&d_dst, size));

            // 隨便跑一個 kernel 強迫 context 初始化
            tanuki::core::threshold_u8_gpu(d_src, d_dst, W, H, 128, 0);
            CHECK_CUDA(cudaDeviceSynchronize());

            cudaFree(d_src);
            cudaFree(d_dst);
            return CORE_CV_SUCCESS;
        }
        catch (...) {
            if (d_src) cudaFree(d_src);
            if (d_dst) cudaFree(d_dst);
            return CORE_CV_ERROR_UNKNOWN;
        }
    }

    // --- [新增] 記憶體與 IO 實作 ---

    TANUKI_CV_API unsigned char* TanukiCv_AllocPinned(unsigned long long size) {
        return (unsigned char*)tanuki::core::alloc_pinned_memory((size_t)size);
    }

    TANUKI_CV_API void TanukiCv_FreePinned(unsigned char* ptr) {
        tanuki::core::free_pinned_memory(ptr);
    }

    TANUKI_CV_API bool TanukiCv_FastReadBMP(const char* filepath, int* w, int* h, unsigned char* outBuffer, int bufferSize) {
        try {
            int width = 0, height = 0;
            bool res = tanuki::core::fast_read_bmp_8bit(filepath, width, height, outBuffer, bufferSize);
            if (res) {
                if (w) *w = width;
                if (h) *h = height;
            }
            return res;
        }
        catch (...) { return false; }
    }

    TANUKI_CV_API bool TanukiCv_FastWriteBMP(const char* filepath, int w, int h, const unsigned char* inBuffer) {
        try {
            return tanuki::core::fast_write_bmp_8bit(filepath, w, h, inBuffer);
        }
        catch (...) { return false; }
    }


    // --- 影像處理實作 (保持原本邏輯，但在 Pinned Memory 下會變快) ---

    int TanukiCv_Brighten(const uint8_t* src_ptr, int width, int height, int value, uint8_t* dst_ptr) {
        if (!src_ptr || !dst_ptr) return CORE_CV_ERROR_NULL_POINTER;
        if (width <= 0 || height <= 0) return CORE_CV_ERROR_INVALID_PARAM;

        size_t size = static_cast<size_t>(width) * height * sizeof(uint8_t);
        uint8_t* d_in = nullptr;
        uint8_t* d_out = nullptr;

        try {
            CHECK_CUDA(cudaMalloc(&d_in, size));
            CHECK_CUDA(cudaMalloc(&d_out, size));

            // 如果 src_ptr 是透過 TanukiCv_AllocPinned 分配的，這行 Memcpy 會自動變成 Async DMA
            CHECK_CUDA(cudaMemcpy(d_in, src_ptr, size, cudaMemcpyHostToDevice));

            tanuki::core::brighten_u8_gpu(d_in, d_out, width, height, value, 0);

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

    int TanukiCv_Threshold(const uint8_t* src_ptr, int width, int height, uint8_t threshold, uint8_t* dst_ptr) {
        if (!src_ptr || !dst_ptr) return CORE_CV_ERROR_NULL_POINTER;
        size_t size = static_cast<size_t>(width) * height * sizeof(uint8_t);
        uint8_t* d_in = nullptr;
        uint8_t* d_out = nullptr;

        try {
            CHECK_CUDA(cudaMalloc(&d_in, size));
            CHECK_CUDA(cudaMalloc(&d_out, size));

            CHECK_CUDA(cudaMemcpy(d_in, src_ptr, size, cudaMemcpyHostToDevice));

            tanuki::core::threshold_u8_gpu(d_in, d_out, width, height, threshold, 0);
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

    int TanukiCv_Invert(const uint8_t* src_ptr, int width, int height, uint8_t* dst_ptr) {
        if (!src_ptr || !dst_ptr) return CORE_CV_ERROR_NULL_POINTER;
        size_t size = static_cast<size_t>(width) * height * sizeof(uint8_t);
        uint8_t* d_in = nullptr;
        uint8_t* d_out = nullptr;

        try {
            CHECK_CUDA(cudaMalloc(&d_in, size));
            CHECK_CUDA(cudaMalloc(&d_out, size));
            CHECK_CUDA(cudaMemcpy(d_in, src_ptr, size, cudaMemcpyHostToDevice));

            tanuki::core::invert_u8_gpu(d_in, d_out, width, height, 0); // 呼叫 core_ops (請確認已實作)
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

    int TanukiCv_Convolution(const uint8_t* src_ptr, int width, int height, const float* mask_ptr, int mask_size, uint8_t* dst_ptr) {
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

            tanuki::core::convolution_u8_gpu(d_in, d_out, width, height, d_mask, mask_size, 0);
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
    TANUKI_CV_API int TanukiCv_MallocGPU(unsigned char** d_ptr, int width, int height) {
        size_t size = (size_t)width * height;
        CHECK_CUDA(cudaMalloc((void**)d_ptr, size));
        return CORE_CV_SUCCESS;
    }

    TANUKI_CV_API int TanukiCv_FreeGPU(unsigned char* d_ptr) {
        if (d_ptr) CHECK_CUDA(cudaFree(d_ptr));
        return CORE_CV_SUCCESS;
    }

    TANUKI_CV_API int TanukiCv_Upload(const unsigned char* h_src, unsigned char* d_dst, int width, int height) {
        size_t size = (size_t)width * height;
        // 如果 h_src 是 Pinned Memory，這裡會跑 Async DMA
        CHECK_CUDA(cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice));
        return CORE_CV_SUCCESS;
    }

    TANUKI_CV_API int TanukiCv_Download(const unsigned char* d_src, unsigned char* h_dst, int width, int height) {
        size_t size = (size_t)width * height;
        CHECK_CUDA(cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost));
        return CORE_CV_SUCCESS;
    }

    // --- 純 GPU 運算 (極速版) ---
    TANUKI_CV_API int TanukiCv_Brighten_GPU(const uint8_t* d_src, int width, int height, int value, uint8_t* d_dst) {
        // 沒有 Malloc，沒有 Memcpy，只有 Kernel Launch
        tanuki::core::brighten_u8_gpu(d_src, d_dst, width, height, value, 0);
        // 不做 Sync，讓 CPU 可以馬上量測 Launch 時間 (或做 Sync 量測執行時間)
        CHECK_CUDA(cudaDeviceSynchronize());
        return CORE_CV_SUCCESS;
    }

    TANUKI_CV_API int TanukiCv_Threshold_GPU(const uint8_t* d_src, int width, int height, uint8_t threshold, uint8_t* d_dst) {
        tanuki::core::threshold_u8_gpu(d_src, d_dst, width, height, threshold, 0);
        CHECK_CUDA(cudaDeviceSynchronize());
        return CORE_CV_SUCCESS;
    }

    TANUKI_CV_API int TanukiCv_Invert_GPU(const uint8_t* d_src, int width, int height, uint8_t* d_dst) {
        tanuki::core::invert_u8_gpu(d_src, d_dst, width, height, 0);
        CHECK_CUDA(cudaDeviceSynchronize());
        return CORE_CV_SUCCESS;
    }

    // 注意：Mask 也必須已經在 GPU 上
    TANUKI_CV_API int TanukiCv_Convolution_GPU(const uint8_t* d_src, int width, int height, const float* d_mask, int mask_size, uint8_t* d_dst) {
        tanuki::core::convolution_u8_gpu(d_src, d_dst, width, height, d_mask, mask_size, 0);
        CHECK_CUDA(cudaDeviceSynchronize());
        return CORE_CV_SUCCESS;
    }

    // --- [新增] Float 資源實作 ---
    TANUKI_CV_API int TanukiCv_MallocGPU_Float(float** d_ptr, int count) {
        size_t size = (size_t)count * sizeof(float);
        CHECK_CUDA(cudaMalloc((void**)d_ptr, size));
        return CORE_CV_SUCCESS;
    }

    TANUKI_CV_API int TanukiCv_FreeGPU_Float(float* d_ptr) {
        if (d_ptr) CHECK_CUDA(cudaFree(d_ptr));
        return CORE_CV_SUCCESS;
    }

    TANUKI_CV_API int TanukiCv_Upload_Float(const float* h_src, float* d_dst, int count) {
        size_t size = (size_t)count * sizeof(float);
        CHECK_CUDA(cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice));
        return CORE_CV_SUCCESS;
    }

    // --- [新增] GPU 縮圖實作 ---
    // h_src / h_dst 若為 Pinned Memory，cudaMemcpy 自動走非同步 DMA，速度更快。
    TANUKI_CV_API int TanukiCv_Resize_GPU(
        const uint8_t* h_src, int src_w, int src_h,
        uint8_t*       h_dst, int dst_w, int dst_h)
    {
        if (!h_src || !h_dst)                          return CORE_CV_ERROR_NULL_POINTER;
        if (src_w <= 0 || src_h <= 0 ||
            dst_w <= 0 || dst_h <= 0)                  return CORE_CV_ERROR_INVALID_PARAM;

        uint8_t* d_src = nullptr;
        uint8_t* d_dst = nullptr;

        try {
            CHECK_CUDA(cudaMalloc(&d_src, (size_t)src_w * src_h));
            CHECK_CUDA(cudaMalloc(&d_dst, (size_t)dst_w * dst_h));

            // 若 h_src 是 Pinned Memory，此 memcpy 走 DMA 加速
            CHECK_CUDA(cudaMemcpy(d_src, h_src, (size_t)src_w * src_h, cudaMemcpyHostToDevice));

            tanuki::core::resize_u8_gpu(d_src, src_w, src_h, d_dst, dst_w, dst_h, /*stream=*/nullptr);

            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            // 若 h_dst 是 Pinned Memory，此 memcpy 走 DMA 加速
            CHECK_CUDA(cudaMemcpy(h_dst, d_dst, (size_t)dst_w * dst_h, cudaMemcpyDeviceToHost));

            cudaFree(d_src);
            cudaFree(d_dst);
            return CORE_CV_SUCCESS;
        }
        catch (...) {
            if (d_src) cudaFree(d_src);
            if (d_dst) cudaFree(d_dst);
            return CORE_CV_ERROR_UNKNOWN;
        }
    }

}


