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

bool BackgroundSubModule::EnsureBuffers(int w, int h) {
    if (w == w_ && h == h_ && d_col_mean_ != nullptr) return true;
    Release();
    if (cudaMalloc(&d_col_mean_, (size_t)w * sizeof(float)) != cudaSuccess) {
        err_ = "background_sub: cudaMalloc col_mean failed";
        d_col_mean_ = nullptr;
        return false;
    }
    w_ = w; h_ = h;
    return true;
}

bool BackgroundSubModule::Initialize() { return true; } // lazy 配置在 Process

bool BackgroundSubModule::Process(const InputImage& input, const Params& params, OutputBuffers* output) {
    if (!input.data || !output || !output->mura_data) { err_ = "background_sub: null input/output buffer"; return false; }
    if (!EnsureBuffers(input.width, input.height)) return false;
    cudaStream_t s = (cudaStream_t)input.stream;

    // Step 1：column 背景估計（有 precomputed 直接用，否則 robust column mean）
    if (params.precomputed_col_mean != nullptr) {
        if (cudaMemcpyAsync(d_col_mean_, params.precomputed_col_mean,
                            (size_t)w_ * sizeof(float), cudaMemcpyDeviceToDevice, s) != cudaSuccess) {
            err_ = "background_sub: copy precomputed col_mean failed";
            return false;
        }
    } else {
        // ⚠ parity 保留：舊 PICoaterDetector::Run 在這裡「硬編 sigma_col=1、無視 bgSigmaFactor 參數」。
        //   為了與舊版輸出位元級一致（drop-in 驗證可比對），這裡同樣固定 1。
        //   app 其實傳 bg_sigma_factor=2.0（InspectionEngineConfig.DefaultBgSigma）——若要「啟用」
        //   這個參數（可能是修舊 bug），改回 params.bg_sigma_factor 並在 migration doc 記 known diff。
        const float sigma_col = 1.0f;
        tanuki::core::calcColumnMeans_RemoveOutliers_gpu<uint8_t>(
            input.data, d_col_mean_, w_, h_, sigma_col, s);
    }

    // Step 2：column 背景相減 → 去背影像（寫 output->mura_data，供下個 module 接手）
    tanuki::core::calcColumnBackground_u8_gpu(input.data, d_col_mean_, output->mura_data, w_, h_, s);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err_ = std::string("background_sub: CUDA error: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

std::string BackgroundSubModule::GetLastError() const { return err_; }

TANUKI_REGISTER_MODULE("background_sub", BackgroundSubModule);

}} // namespace tanuki::pipeline
