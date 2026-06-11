#include "ridge_hessian.hpp"
#include "tanuki/pipeline/registry.hpp"
#include "tanuki/core/imgproc/core_filters.hpp"     // gaussianBlur_gpu
#include "tanuki/core/imgproc/core_features.hpp"    // computeHessianResponse_gpu, detectionMode
#include "tanuki/core/imgproc/core_background.hpp"  // calcColumn/Row Means/Max
#include "tanuki/core/imgproc/core_utils.hpp"       // scale_clamp_f32_to_u8_gpu
#include <cuda_runtime.h>
#include <cstring>

namespace tanuki { namespace pipeline {

// 曲線中性化：float 陣列原地 ×scale（不 clamp、保峰值，供 review 時重調閾值）。沿用原 PICoaterDetector。
__global__ static void k_scale_f32_inplace(float* d, int N, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) d[i] *= s;
}
static void scale_f32_inplace_gpu(float* d_buf, int N, float scale, cudaStream_t stream) {
    if (d_buf == nullptr || N <= 0) return;
    int block = 256, grid = (N + block - 1) / block;
    k_scale_f32_inplace<<<grid, block, 0, stream>>>(d_buf, N, scale);
}

RidgeHessianModule::~RidgeHessianModule() { Release(); }

void RidgeHessianModule::Release() {
    if (d_workspace_) cudaFree(d_workspace_);
    d_workspace_ = nullptr; d_blur_f32_ = nullptr; d_resp_ = nullptr; w_ = 0; h_ = 0;
}

void RidgeHessianModule::EnsureBuffers(int w, int h) {
    if (w == w_ && h == h_) return;
    Release();
    w_ = w; h_ = h;
    size_t num_pixels = (size_t)w * h;
    auto alignUp = [](size_t off) { return (off + 255) & ~255; };

    // hessian 路徑 workspace 佈局（u8 + f32 + resp views），沿用原 PICoaterDetector::Initialize
    size_t offset = 0;
    size_t off_u8  = alignUp(offset); offset = off_u8  + num_pixels * sizeof(uint8_t);
    size_t off_f32 = alignUp(offset); offset = off_f32 + num_pixels * sizeof(float);
    size_t off_resp= alignUp(offset); offset = off_resp+ num_pixels * sizeof(float);
    size_t hessian_req = offset;
    size_t gaussian_req = num_pixels * sizeof(float) * 3 + 1024; // gaussian 內部 3 float + mask
    size_t total = (gaussian_req > hessian_req) ? gaussian_req : hessian_req;

    cudaMalloc(&d_workspace_, total);
    uint8_t* base = (uint8_t*)d_workspace_;
    d_blur_f32_ = (float*)(base + off_f32);
    d_resp_     = (float*)(base + off_resp);
}

bool RidgeHessianModule::Initialize() { return true; } // lazy

bool RidgeHessianModule::Process(const InputImage& input, const Params& params, OutputBuffers* output) {
    if (!output || !output->mura_data || !output->ridge_data) { err_ = "ridge_hessian: null output buffer"; return false; }
    EnsureBuffers(input.width, input.height);
    cudaStream_t s = (cudaStream_t)input.stream;
    const uint8_t* d_src = output->mura_data; // 去背影像（前一 module background_sub 寫的）
    int W = w_, H = h_, num_pixels = W * H;

    // Step 3：gaussian blur（一次，供各方向共用）
    int ksize = (int)(6.0f * params.ridge_sigma + 1.0f);
    if (ksize % 2 == 0) ksize++;
    tanuki::core::gaussianBlur_gpu<uint8_t, float>(d_src, d_blur_f32_, W, H, params.ridge_sigma, ksize, s, d_workspace_);

    const char* mode = params.ridge_mode ? params.ridge_mode : "";
    bool doVertical   = (strcmp(mode, "vertical") == 0 || strcmp(mode, "vertical+horizontal") == 0);
    bool doHorizontal = (strcmp(mode, "horizontal") == 0 || strcmp(mode, "vertical+horizontal") == 0);
    float scale_factor = 255.0f / params.hessian_max_factor;

    // Step 4：切向脊線 → ridge_data + 切向曲線
    if (doVertical) {
        tanuki::core::computeHessianResponse_gpu(d_blur_f32_, d_resp_, W, H, tanuki::core::detectionMode::VERTICAL, s);
        tanuki::core::calcColumnMeans_gpu<float>(d_resp_, output->mura_curve_mean, W, H, s, d_workspace_);
        tanuki::core::calcColumnMax_gpu<float>(d_resp_, output->mura_curve_max, W, H, s);
        scale_f32_inplace_gpu(output->mura_curve_mean, W, scale_factor, s);
        scale_f32_inplace_gpu(output->mura_curve_max,  W, scale_factor, s);
        tanuki::core::scale_clamp_f32_to_u8_gpu(d_resp_, output->ridge_data, num_pixels, scale_factor, s);
    }

    // Step 5：法向脊線 + 法向曲線
    if (doHorizontal) {
        tanuki::core::computeHessianResponse_gpu(d_blur_f32_, d_resp_, W, H, tanuki::core::detectionMode::HORIZONTAL, s);
        if (output->mura_row_curve_mean) {
            tanuki::core::calcRowMeans_gpu<float>(d_resp_, output->mura_row_curve_mean, W, H, s);
            scale_f32_inplace_gpu(output->mura_row_curve_mean, H, scale_factor, s);
        }
        if (output->mura_row_curve_max) {
            tanuki::core::calcRowMax_gpu<float>(d_resp_, output->mura_row_curve_max, W, H, s);
            scale_f32_inplace_gpu(output->mura_row_curve_max, H, scale_factor, s);
        }
        // 兩向都做：法向影像覆寫 mura_data（去背已被消費）；只法向：寫 ridge_data
        uint8_t* dst = doVertical ? output->mura_data : output->ridge_data;
        tanuki::core::scale_clamp_f32_to_u8_gpu(d_resp_, dst, num_pixels, scale_factor, s);
    }
    return true;
}

std::string RidgeHessianModule::GetLastError() const { return err_; }

TANUKI_REGISTER_MODULE("ridge_hessian", RidgeHessianModule);

}} // namespace tanuki::pipeline
