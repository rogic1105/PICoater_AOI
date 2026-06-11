#include "background_sub.hpp"
#include "tanuki/pipeline/registry.hpp"
#include "tanuki/core/imgproc/core_background.hpp"
#include <cuda_runtime.h>

namespace tanuki { namespace pipeline {

BackgroundSubModule::~BackgroundSubModule() { Release(); }

void BackgroundSubModule::Release() {
    if (d_col_mean_) cudaFree(d_col_mean_);
    d_col_mean_ = nullptr; w_ = 0; h_ = 0;
}

void BackgroundSubModule::EnsureBuffers(int w, int h) {
    if (w == w_ && h == h_) return;
    Release();
    w_ = w; h_ = h;
    cudaMalloc(&d_col_mean_, (size_t)w * sizeof(float));
}

bool BackgroundSubModule::Initialize() { return true; } // lazy 配置在 Process

bool BackgroundSubModule::Process(const InputImage& input, const Params& params, OutputBuffers* output) {
    if (!input.data || !output || !output->mura_data) { err_ = "background_sub: null input/output buffer"; return false; }
    EnsureBuffers(input.width, input.height);
    cudaStream_t s = (cudaStream_t)input.stream;

    // Step 1：column 背景估計（有 precomputed 直接用，否則 robust column mean）
    if (params.precomputed_col_mean != nullptr) {
        cudaMemcpyAsync(d_col_mean_, params.precomputed_col_mean,
                        (size_t)w_ * sizeof(float), cudaMemcpyDeviceToDevice, s);
    } else {
        tanuki::core::calcColumnMeans_RemoveOutliers_gpu<uint8_t>(
            input.data, d_col_mean_, w_, h_, params.bg_sigma_factor, s);
    }

    // Step 2：column 背景相減 → 去背影像（寫 output->mura_data，供下個 module 接手）
    tanuki::core::calcColumnBackground_u8_gpu(input.data, d_col_mean_, output->mura_data, w_, h_, s);
    return true;
}

std::string BackgroundSubModule::GetLastError() const { return err_; }

TANUKI_REGISTER_MODULE("background_sub", BackgroundSubModule);

}} // namespace tanuki::pipeline
