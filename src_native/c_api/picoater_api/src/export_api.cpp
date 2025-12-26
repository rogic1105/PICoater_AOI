// PiCoater_api\src\export_api.cpp

#include "export_c/export_api.h"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// 引用你的演算法模組
#include "Module_GetPICoaterBackground.hpp" 
// 假設這個標頭檔在 include path 中可以被找到

// 內部使用的錯誤檢查輔助巨集 (不拋出例外，改為回傳錯誤碼)
#define CHECK_CUDA(call)                                          \
  {                                                               \
    cudaError_t err = call;                                       \
    if (err != cudaSuccess) {                                     \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err)      \
                << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
      return -1; /* CUDA 錯誤通用代碼 */                          \
    }                                                             \
  }

int Picoater_GetBackground(
    const uint8_t* input_img,
    int width,
    int height,
    float sigma,
    uint8_t* output_bg,
    uint8_t* output_mura) {

    // 1. 參數基本檢查
    if (input_img == nullptr || output_bg == nullptr || output_mura == nullptr) {
        std::cerr << "Error: Input or Output pointers are null.\n";
        return -2; // 指標錯誤
    }
    if (width <= 0 || height <= 0) {
        std::cerr << "Error: Invalid image dimensions.\n";
        return -3; // 維度錯誤
    }

    // 2. 準備 GPU 資源
    size_t img_size = static_cast<size_t>(width) * height * sizeof(uint8_t);
    uint8_t* d_in = nullptr;
    uint8_t* d_bg = nullptr;
    uint8_t* d_mura = nullptr;

    // 使用 try-catch 區塊來確保發生 C++ 例外時也能釋放 GPU 記憶體 (RAII 更好，但這是 C API 實作層)
    try {
        CHECK_CUDA(cudaMalloc(&d_in, img_size));
        CHECK_CUDA(cudaMalloc(&d_bg, img_size));
        CHECK_CUDA(cudaMalloc(&d_mura, img_size));

        // 3. 資料上傳 (Host -> Device)
        CHECK_CUDA(cudaMemcpy(d_in, input_img, img_size, cudaMemcpyHostToDevice));

        // 4. 執行核心演算法
        // 注意：假設這個函數是同步的，或是我們需要手動同步
        picoater::GetPICoaterBackground_gpu(d_in, d_bg, d_mura, width, height, sigma, 0);

        // 檢查 Kernel 啟動是否有錯
        CHECK_CUDA(cudaGetLastError());
        // 等待 GPU 完成 (對於 C API 呼叫通常建議同步，除非有非同步介面設計)
        CHECK_CUDA(cudaDeviceSynchronize());

        // 5. 資料下載 (Device -> Host)
        CHECK_CUDA(cudaMemcpy(output_bg, d_bg, img_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(output_mura, d_mura, img_size, cudaMemcpyDeviceToHost));

        // 6. 釋放資源
        cudaFree(d_in);
        cudaFree(d_bg);
        cudaFree(d_mura);

        return 0; // 成功

    }
    catch (const std::exception& e) {
        std::cerr << "Exception in Picoater_GetBackground: " << e.what() << "\n";

        // 發生例外時的清理
        if (d_in) cudaFree(d_in);
        if (d_bg) cudaFree(d_bg);
        if (d_mura) cudaFree(d_mura);

        return -99; // 未知例外
    }
}