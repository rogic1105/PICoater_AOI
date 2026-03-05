// AOI_SDK\core_cv\include\core_cv\base\cuda_memory.hpp

#pragma once
#include <cuda_runtime.h>
#include <cstdlib> // size_t

namespace core {

    // 分配鎖頁記憶體 (Pinned Memory)
    // 使用 void** 是為了配合 CUDA API 風格，或者直接回傳 void* 也可以
    inline void* alloc_pinned_memory(size_t size) {
        void* ptr = nullptr;
        // cudaHostAllocDefault: 可攜性高
        // cudaHostAllocMapped: 如果需要 Zero-Copy (Device直接存取Host)，但通常比較慢
        cudaError_t err = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            return nullptr;
        }
        return ptr;
    }

    // 釋放鎖頁記憶體
    inline void free_pinned_memory(void* ptr) {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }

}