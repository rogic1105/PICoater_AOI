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

bool RidgeHessianModule::EnsureBuffers(int w, int h) {
    if (w == w_ && h == h_ && d_workspace_ != nullptr) return true;
    Release();
    size_t num_pixels = (size_t)w * h;
    auto alignUp = [](size_t off) { return (off + 255) & ~(size_t)255; };

    // Workspace 佈局（race 修正版，勿回退舊排法）：
    //   gaussianBlur_gpu<u8,float> 會把同一塊 workspace「從 offset 0」bump 配內部 scratch
    //   （f32_src 4N + f32_temp 4N + mask），輸出則直接寫 caller 給的 d_blur_f32_。
    //   舊 PICoaterDetector 把 blur view 放在 offset≈N → 與 scratch 的 f32_temp 重疊 →
    //   col-pass kernel 一邊讀 temp 一邊寫 dst（重疊區 = 真 GPU data race，靠 block 排程矇對）。
    //   修正：blur / resp view 排在 gaussian scratch 區之後，零重疊（代價 +8N bytes）。
    //   （原版還有個沒人用的 u8 槽 = 舊 d_hessian_u8_ 遺跡，已刪。）
    size_t scratch_bytes = alignUp(num_pixels * sizeof(float) * 2 + 4096); // gaussian 內部 src+temp+mask
    size_t off_blur = scratch_bytes;
    size_t off_resp = alignUp(off_blur + num_pixels * sizeof(float));
    size_t total    = off_resp + num_pixels * sizeof(float);

    if (cudaMalloc(&d_workspace_, total) != cudaSuccess) {
        err_ = "ridge_hessian: cudaMalloc workspace failed";
        d_workspace_ = nullptr;
        return false;
    }
    w_ = w; h_ = h;
    uint8_t* base = (uint8_t*)d_workspace_;
    d_blur_f32_ = (float*)(base + off_blur);
    d_resp_     = (float*)(base + off_resp);
    return true;
}

bool RidgeHessianModule::Process(const InputImage& input, const Params& params, OutputBuffers* output) {
    if (!output || !output->mura_data || !output->ridge_data) { err_ = "ridge_hessian: null output buffer"; return false; }
    if (!EnsureBuffers(input.width, input.height)) return false;
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
        tanuki::core::calcColumnMeans_gpu<float>(d_resp_, output->mura_curve_mean, W, H, s);
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

    // launch 錯誤回收（async 執行錯誤由 caller 的 sync 點承接；這裡至少抓 launch 失敗）
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err_ = std::string("ridge_hessian: CUDA error: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

std::string RidgeHessianModule::GetLastError() const { return err_; }

TANUKI_REGISTER_MODULE("ridge_hessian", RidgeHessianModule);

}} // namespace tanuki::pipeline
