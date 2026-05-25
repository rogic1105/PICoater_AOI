// AOI_SDK\core_cv\include\core_cv\base\cuda_utils.hpp

#pragma once
#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


// =========================================================
// [新增] VSCode IntelliSense Fix
// 讓編輯器看得懂 __global__, blockIdx 等關鍵字，不影響實際編譯
// =========================================================
#if defined(__INTELLISENSE__) || defined(__RESHARPER__)

    // 1. 基礎關鍵字欺騙
#ifndef __CUDACC__
#define __CUDACC__
#endif

#define __global__
#define __device__
#define __host__
#define __forceinline__
#define __noinline__
#define __shared__
#define __constant__
#define __managed__
#define __restrict__

// 2. 核心同步函式
inline void __syncthreads() {}
inline void __threadfence() {}
inline void __threadfence_block() {}
inline void __threadfence_system() {}

// 3. 核心索引變數 (模擬 dim3 結構)
struct __cuda_fake_dim3 { unsigned int x, y, z; };
extern __cuda_fake_dim3 gridDim;
extern __cuda_fake_dim3 blockDim;
extern __cuda_fake_dim3 blockIdx;
extern __cuda_fake_dim3 threadIdx;
extern int warpSize;

// 4. 常用數學與原子操作 (Atomic)
// 為了讓 IDE 不會報錯 "atomicAdd undefined"
template<typename T> inline T atomicAdd(T* address, T val) { return *address; }
template<typename T> inline T atomicSub(T* address, T val) { return *address; }
template<typename T> inline T atomicExch(T* address, T val) { return *address; }
template<typename T> inline T atomicMin(T* address, T val) { return *address; }
template<typename T> inline T atomicMax(T* address, T val) { return *address; }
template<typename T> inline T atomicCAS(T* address, T compare, T val) { return *address; }

// 5. 其他 CUDA 內建函式
inline void __sincosf(float x, float* s, float* c) {}
inline float __fdividef(float x, float y) { return x / y; }
// 如果有用到 __launch_bounds__
#define __launch_bounds__(max_threads_per_block, min_blocks_per_multiprocessor)

#endif
// =========================================================


inline void cudaCheck(cudaError_t e, const char* expr, const char* file, int line) {
    if (e != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") + cudaGetErrorString(e) +
            " at " + file + ":" + std::to_string(line) + " (" + expr + ")"
        );
    }
}
#define CUDA_CHECK(x) cudaCheck((x), #x, __FILE__, __LINE__)

// CUDA 錯誤檢查
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << result
            << " (" << cudaGetErrorString(result) << ") in " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// cuFFT 錯誤檢查
#define checkCufftErrors(val) check((val), #val, __FILE__, __LINE__)
inline void check(cufftResult result, const char* func, const char* file, int line) {
    if (result != CUFFT_SUCCESS) {
        std::cerr << "cuFFT error at " << file << ":" << line << " code=" << result
            << " in " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


#define CUDA_OK(x) do{ cudaError_t _e=(x); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s failed @%s:%d : %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(_e)); \
  std::exit(1);} }while(0)

inline void dbg_sync(const char* tag) {
    CUDA_OK(cudaGetLastError());        // 先撈 launch error
    CUDA_OK(cudaDeviceSynchronize());   // 等 GPU 完成
    fprintf(stderr, "[SYNC OK] %s\n", tag);
}

// 下載 uint8 並列印前幾個、計數非零、min/max
inline void dbg_dump_u8(const char* tag, const uint8_t* d_ptr, size_t N, size_t head = 32) {
    std::vector<uint8_t> h(N);
    CUDA_OK(cudaMemcpy(h.data(), d_ptr, N, cudaMemcpyDeviceToHost));
    size_t nz = 0; uint8_t mn = 255, mx = 0;
    for (size_t i = 0; i < N; i++) { nz += (h[i] != 0); if (h[i] < mn) mn = h[i]; if (h[i] > mx) mx = h[i]; }
    fprintf(stderr, "[%s] u8 N=%zu  min=%u max=%u  nonzero=%zu (%.2f%%)\n",
        tag, N, (unsigned)mn, (unsigned)mx, nz, 100.0 * double(nz) / double(N));
    size_t k = std::min(head, N);
    fprintf(stderr, "[%s] head:", tag);
    for (size_t i = 0; i < k; i++) fprintf(stderr, " %u", (unsigned)h[i]);
    fprintf(stderr, "\n");
}

// 下載 float 並列印 min/max
inline void dbg_dump_f32_minmax(const char* tag, const float* d_ptr, size_t N) {
    std::vector<float> h(N);
    CUDA_OK(cudaMemcpy(h.data(), d_ptr, N * sizeof(float), cudaMemcpyDeviceToHost));
    auto mm = std::minmax_element(h.begin(), h.end());
    float mn = (mm.first != h.end()) ? *mm.first : 0.0f;
    float mx = (mm.second != h.end()) ? *mm.second : 0.0f;
    std::fprintf(stderr, "[%s] f32 N=%zu  min=%g max=%g\n", tag, N, mn, mx);
}

template <typename KernelFunc>
inline void get_optimal_launch_1d(KernelFunc kernel, int n, int& gridSize, int& blockSize, int dynamicSMemSize = 0) {
    int minGridSize; // 最小 Grid 數 (API 需要這個參數但我們這邊暫時用不到)

    // 魔法函式：詢問驅動程式最佳 Block 大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, dynamicSMemSize, n);

    // 根據算出來的最佳 Block 計算 Grid
    gridSize = (n + blockSize - 1) / blockSize;
}

// [新增] 自動計算 2D 最佳 Block/Grid
template <typename KernelFunc>
inline void get_optimal_launch_2d(KernelFunc kernel, int W, int H, dim3& gridDim, dim3& blockDim, int dynamicSMemSize = 0) {
    int minGridSize;
    int maxBlockSize;

    // 1. 取得最佳的總 Thread 數 (例如 1024 或 512)
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, kernel, dynamicSMemSize, W * H);

    // 2. 將 1D 的總數拆成 2D (X, Y)
    // 通常設定 block.x 為 32 (因為 Warp Size 是 32)，剩下的給 block.y
    // 例如 maxBlockSize = 1024 -> 32 x 32
    // 例如 maxBlockSize = 512  -> 32 x 16

    blockDim.x = 32;
    blockDim.y = maxBlockSize / blockDim.x;
    blockDim.z = 1;

    // 3. 計算 Grid
    gridDim.x = (W + blockDim.x - 1) / blockDim.x;
    gridDim.y = (H + blockDim.y - 1) / blockDim.y;
    gridDim.z = 1;
}