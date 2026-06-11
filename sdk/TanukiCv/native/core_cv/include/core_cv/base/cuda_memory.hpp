
#pragma once
#include <cuda_runtime.h>
#include <cstdlib> // size_t

namespace tanuki { namespace core {

    // ���t�ꭶ�O���� (Pinned Memory)
    // �ϥ� void** �O���F�t�X CUDA API ����A�Ϊ̪����^�� void* �]�i�H
    inline void* alloc_pinned_memory(size_t size) {
        void* ptr = nullptr;
        // cudaHostAllocDefault: �i��ʰ�
        // cudaHostAllocMapped: �p�G�ݭn Zero-Copy (Device�����s��Host)�A���q�`����C
        cudaError_t err = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            return nullptr;
        }
        return ptr;
    }

    // �����ꭶ�O����
    inline void free_pinned_memory(void* ptr) {
        if (ptr) {
            cudaFreeHost(ptr);
        }
    }

}}  // namespace core, tanuki