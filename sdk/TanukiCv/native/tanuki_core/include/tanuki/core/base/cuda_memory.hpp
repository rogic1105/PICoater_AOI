
#pragma once
#include <cuda_runtime.h>
#include <cstdlib> // size_t

namespace tanuki { namespace core {

    // 配置鎖頁記憶體（Pinned Memory，cudaHostAlloc）—— H2D/D2H 傳輸較快、可非同步；失敗回 nullptr。
    inline void* alloc_pinned_memory(size_t size) {
        void* ptr = nullptr;
        // cudaHostAllocDefault：標準 pinned。
        // （需 Zero-Copy／Device 直存 Host 時可改 cudaHostAllocMapped，通常較慢，這裡不用。）
        cudaError_t err = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            return nullptr;
        }
        return ptr;
    }

    // 釋放鎖頁記憶體（cudaFreeHost）。
    inline void free_pinned_memory(void* ptr) {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }

}}  // namespace core, tanuki