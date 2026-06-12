#include "export_c/export_api.h"

#include <cuda_runtime.h>
#include <memory>
#include <string>

#include "tanuki/core/imgproc/core_background.hpp"   // calcColumnMeans_RemoveOutliers（ComputeColumnMean 用）
#include "find_stream_ridgeline.hpp"                  // CreateFindStreamRidgeline

namespace {

struct PipelineContext {
    std::unique_ptr<tanuki::pipeline::Pipeline> pipeline;
    std::string last_error;

    int width = 0, height = 0;
    size_t image_size = 0;

    uint8_t* d_input = nullptr;
    uint8_t* d_background = nullptr;
    uint8_t* d_mura = nullptr;
    uint8_t* d_ridge = nullptr;
    float* d_curve_mean = nullptr;
    float* d_curve_max = nullptr;
    float* d_row_curve_mean = nullptr;
    float* d_row_curve_max = nullptr;

    ~PipelineContext() { ReleaseBuffers(); }

    void ReleaseBuffers() {
        if (d_input) cudaFree(d_input);
        if (d_background) cudaFree(d_background);
        if (d_mura) cudaFree(d_mura);
        if (d_ridge) cudaFree(d_ridge);
        if (d_curve_mean) cudaFree(d_curve_mean);
        if (d_curve_max) cudaFree(d_curve_max);
        if (d_row_curve_mean) cudaFree(d_row_curve_mean);
        if (d_row_curve_max) cudaFree(d_row_curve_max);
        d_input = d_background = d_mura = d_ridge = nullptr;
        d_curve_mean = d_curve_max = d_row_curve_mean = d_row_curve_max = nullptr;
        width = height = 0; image_size = 0;
    }

    bool EnsureBuffers(int new_width, int new_height, std::string* error) {
        if (new_width <= 0 || new_height <= 0) { *error = "width and height must be positive."; return false; }
        if (width == new_width && height == new_height && d_input != nullptr) return true;
        ReleaseBuffers();
        width = new_width; height = new_height;
        image_size = (size_t)width * (size_t)height;
        if (cudaMalloc(&d_input, image_size) != cudaSuccess ||
            cudaMalloc(&d_background, image_size) != cudaSuccess ||
            cudaMalloc(&d_mura, image_size) != cudaSuccess ||
            cudaMalloc(&d_ridge, image_size) != cudaSuccess ||
            cudaMalloc(&d_curve_mean, width * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_curve_max, width * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_row_curve_mean, height * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_row_curve_max, height * sizeof(float)) != cudaSuccess) {
            *error = "Failed to allocate internal CUDA buffers for pipeline processing.";
            ReleaseBuffers();
            return false;
        }
        return true;
    }
};

}  // namespace

extern "C" {

AoiPipelineHandle PICoaterAPI_CreatePipeline() {
    auto* ctx = new PipelineContext();
    ctx->pipeline = tanuki::pipeline::CreateFindStreamRidgeline("hessian");
    if (!ctx->pipeline) { delete ctx; return nullptr; } // 未知方法食譜回 nullptr → 對外明確失敗
    return reinterpret_cast<AoiPipelineHandle>(ctx);
}

int PICoaterAPI_ProcessPipeline(AoiPipelineHandle handle,
                                const AoiInputImageC* input,
                                const AoiAlgorithmParamsC* params,
                                const AoiOutputBuffersC* output) {
    if (handle == nullptr) return -1;
    if (input == nullptr || params == nullptr || output == nullptr || input->data == nullptr) return -1;
    auto* ctx = reinterpret_cast<PipelineContext*>(handle);

    if (!ctx->EnsureBuffers(input->width, input->height, &ctx->last_error)) return -2;

    if (cudaMemcpy(ctx->d_input, input->data, ctx->image_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        ctx->last_error = "Failed to copy input image from host to CUDA memory."; return -2;
    }

    tanuki::pipeline::InputImage in;
    in.width = input->width; in.height = input->height; in.data = ctx->d_input; in.stream = input->stream;

    // precomputed column mean：host → GPU（用 d_curve_mean 暫存）
    if (params->precomputed_col_mean != nullptr) {
        if (cudaMemcpy(ctx->d_curve_mean, params->precomputed_col_mean,
                       input->width * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            ctx->last_error = "Failed to copy precomputed column mean to GPU."; return -2;
        }
    }

    tanuki::pipeline::Params p;
    p.bg_sigma_factor = params->bg_sigma_factor;
    p.ridge_sigma = params->ridge_sigma;
    p.hessian_max_factor = params->hessian_max_factor;
    p.ridge_mode = params->ridge_mode;
    p.precomputed_col_mean = params->precomputed_col_mean != nullptr ? ctx->d_curve_mean : nullptr;

    tanuki::pipeline::OutputBuffers out;
    out.width = output->width > 0 ? output->width : input->width;
    out.height = output->height > 0 ? output->height : input->height;
    out.background_data = ctx->d_background;
    out.mura_data = ctx->d_mura;
    out.ridge_data = ctx->d_ridge;
    out.mura_curve_mean = ctx->d_curve_mean;
    out.mura_curve_max = ctx->d_curve_max;
    out.mura_row_curve_mean = ctx->d_row_curve_mean;
    out.mura_row_curve_max = ctx->d_row_curve_max;
    out.stream = output->stream != nullptr ? output->stream : input->stream;

    if (!ctx->pipeline->Process(in, p, &out)) { ctx->last_error = ctx->pipeline->GetLastError(); return -2; }

    auto d2h = [&](void* dst, const void* src, size_t bytes, const char* what) -> bool {
        if (dst == nullptr) return true;
        if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            ctx->last_error = std::string("Failed to copy ") + what + " output to host."; return false;
        }
        return true;
    };
    if (!d2h(output->background_data, ctx->d_background, ctx->image_size, "background")) return -2;
    if (!d2h(output->mura_data, ctx->d_mura, ctx->image_size, "mura")) return -2;
    if (!d2h(output->ridge_data, ctx->d_ridge, ctx->image_size, "ridge")) return -2;
    if (!d2h(output->mura_curve_mean, ctx->d_curve_mean, input->width * sizeof(float), "mura mean curve")) return -2;
    if (!d2h(output->mura_curve_max, ctx->d_curve_max, input->width * sizeof(float), "mura max curve")) return -2;
    if (!d2h(output->mura_row_curve_mean, ctx->d_row_curve_mean, input->height * sizeof(float), "row mean curve")) return -2;
    if (!d2h(output->mura_row_curve_max, ctx->d_row_curve_max, input->height * sizeof(float), "row max curve")) return -2;

    ctx->last_error.clear();
    return 0;
}

const char* PICoaterAPI_GetLastError(AoiPipelineHandle handle) {
    if (handle == nullptr) return "Invalid pipeline handle.";
    return reinterpret_cast<PipelineContext*>(handle)->last_error.c_str();
}

void PICoaterAPI_DestroyPipeline(AoiPipelineHandle handle) {
    if (handle == nullptr) return;
    delete reinterpret_cast<PipelineContext*>(handle);
}

int PICoaterAPI_ComputeColumnMean(AoiPipelineHandle handle,
                                  const AoiInputImageC* input,
                                  float bg_sigma_factor,
                                  float* out_col_mean) {
    if (handle == nullptr || input == nullptr || out_col_mean == nullptr || input->data == nullptr) return -1;
    auto* ctx = reinterpret_cast<PipelineContext*>(handle);
    if (!ctx->EnsureBuffers(input->width, input->height, &ctx->last_error)) return -2;
    if (cudaMemcpy(ctx->d_input, input->data, ctx->image_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        ctx->last_error = "Failed to copy input image for column mean."; return -2;
    }
    int sigma_col = (int)bg_sigma_factor; if (sigma_col < 1) sigma_col = 1;
    tanuki::core::calcColumnMeans_RemoveOutliers_gpu<uint8_t>(
        ctx->d_input, ctx->d_curve_mean, input->width, input->height, (float)sigma_col, nullptr);
    cudaDeviceSynchronize();
    if (cudaMemcpy(out_col_mean, ctx->d_curve_mean, input->width * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        ctx->last_error = "Failed to copy column mean to host."; return -2;
    }
    ctx->last_error.clear();
    return 0;
}

}  // extern "C"
