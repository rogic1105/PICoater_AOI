
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
// [�s�W] VSCode IntelliSense Fix
// ���s�边�ݱo�� __global__, blockIdx ������r�A���v�T��ڽsĶ
// =========================================================
#if defined(__INTELLISENSE__) || defined(__RESHARPER__)

    // 1. ��¦����r���F
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

// 2. �֤ߦP�B�禡
inline void __syncthreads() {}
inline void __threadfence() {}
inline void __threadfence_block() {}
inline void __threadfence_system() {}

// 3. �֤߯����ܼ� (���� dim3 ���c)
struct __cuda_fake_dim3 { unsigned int x, y, z; };
extern __cuda_fake_dim3 gridDim;
extern __cuda_fake_dim3 blockDim;
extern __cuda_fake_dim3 blockIdx;
extern __cuda_fake_dim3 threadIdx;
extern int warpSize;

// 4. �`�μƾǻP��l�ާ@ (Atomic)
// ���F�� IDE ���|���� "atomicAdd undefined"
template<typename T> inline T atomicAdd(T* address, T val) { return *address; }
template<typename T> inline T atomicSub(T* address, T val) { return *address; }
template<typename T> inline T atomicExch(T* address, T val) { return *address; }
template<typename T> inline T atomicMin(T* address, T val) { return *address; }
template<typename T> inline T atomicMax(T* address, T val) { return *address; }
template<typename T> inline T atomicCAS(T* address, T compare, T val) { return *address; }

// 5. ��L CUDA ���ب禡
inline void __sincosf(float x, float* s, float* c) {}
inline float __fdividef(float x, float y) { return x / y; }
// �p�G���Ψ� __launch_bounds__
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

// CUDA ���~�ˬd
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << result
            << " (" << cudaGetErrorString(result) << ") in " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// cuFFT ���~�ˬd
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
    CUDA_OK(cudaGetLastError());        // ���� launch error
    CUDA_OK(cudaDeviceSynchronize());   // �� GPU ����
    fprintf(stderr, "[SYNC OK] %s\n", tag);
}

// �U�� uint8 �æC�L�e�X�ӡB�p�ƫD�s�Bmin/max
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

// �U�� float �æC�L min/max
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
    int minGridSize; // �̤p Grid �� (API �ݭn�o�ӰѼƦ��ڭ̳o��ȮɥΤ���)

    // �]�k�禡�G�߰��X�ʵ{���̨� Block �j�p
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, dynamicSMemSize, n);

    // �ھں�X�Ӫ��̨� Block �p�� Grid
    gridSize = (n + blockSize - 1) / blockSize;
}

// [�s�W] �۰ʭp�� 2D �̨� Block/Grid
template <typename KernelFunc>
inline void get_optimal_launch_2d(KernelFunc kernel, int W, int H, dim3& gridDim, dim3& blockDim, int dynamicSMemSize = 0) {
    int minGridSize;
    int maxBlockSize;

    // 1. ���o�̨Ϊ��` Thread �� (�Ҧp 1024 �� 512)
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, kernel, dynamicSMemSize, W * H);

    // 2. �N 1D ���`�Ʃ 2D (X, Y)
    // �q�`�]�w block.x �� 32 (�]�� Warp Size �O 32)�A�ѤU���� block.y
    // �Ҧp maxBlockSize = 1024 -> 32 x 32
    // �Ҧp maxBlockSize = 512  -> 32 x 16

    blockDim.x = 32;
    blockDim.y = maxBlockSize / blockDim.x;
    blockDim.z = 1;

    // 3. �p�� Grid
    gridDim.x = (W + blockDim.x - 1) / blockDim.x;
    gridDim.y = (H + blockDim.y - 1) / blockDim.y;
    gridDim.z = 1;
}