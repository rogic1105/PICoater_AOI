#include "export_c/export_api.h"
#include "json_lite.hpp"

#include <cuda_runtime.h>
#include <memory>
#include <string>

#include "tanuki/core/imgproc/core_background.hpp"   // calcColumnMeans_RemoveOutliers（ComputeColumnMean 用）
#include "tanuki/core/imgproc/core_transform.hpp"     // resize_u8_gpu（存檔縮圖 fused）
#include "find_stream_ridgeline.hpp"                  // CreateFindStreamRidgeline

namespace {

using tanuki::pipeline::jsonlite::get_number;
using tanuki::pipeline::jsonlite::get_string;

struct PipelineContext {
    std::unique_ptr<tanuki::pipeline::Pipeline> pipeline;
    std::string last_error;
    std::string ridge_mode_buf;   // 持有 json 取出的字串（Params.ridge_mode 是 const char*）

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

    // 存檔縮圖用可重用 device workspace（避免每張 resize 各 cudaMalloc/cudaFree）。
    uint8_t* d_resize = nullptr;
    size_t   resize_cap = 0;
    uint16_t* d_standard_half = nullptr;
    size_t standard_half_cap = 0;

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
        if (d_resize) cudaFree(d_resize);
        if (d_standard_half) cudaFree(d_standard_half);
        d_input = d_background = d_mura = d_ridge = nullptr;
        d_curve_mean = d_curve_max = d_row_curve_mean = d_row_curve_max = nullptr;
        d_resize = nullptr; resize_cap = 0;
        d_standard_half = nullptr; standard_half_cap = 0;
        width = height = 0; image_size = 0;
    }

    // 確保 d_resize 至少 bytes 大（on-demand 成長、重用）。
    bool EnsureResize(size_t bytes) {
        if (d_resize != nullptr && resize_cap >= bytes) return true;
        if (d_resize) { cudaFree(d_resize); d_resize = nullptr; resize_cap = 0; }
        if (cudaMalloc(&d_resize, bytes) != cudaSuccess) { d_resize = nullptr; return false; }
        resize_cap = bytes;
        return true;
    }

    bool EnsureStandardHalf(size_t sample_count) {
        size_t bytes = sample_count * 2 * sizeof(uint16_t);
        if (d_standard_half != nullptr && standard_half_cap >= bytes) return true;
        if (d_standard_half) { cudaFree(d_standard_half); d_standard_half = nullptr; standard_half_cap = 0; }
        if (cudaMalloc(&d_standard_half, bytes) != cudaSuccess) return false;
        standard_half_cap = bytes;
        return true;
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

TanukiPipelineHandle TanukiPipeline_Create(const char* pipeline_name, const char* json_options) {
    std::string name = pipeline_name ? pipeline_name : "";
    std::unique_ptr<tanuki::pipeline::Pipeline> pipe;

    if (name == "find_stream_ridgeline") {
        std::string method = get_string(json_options, "ridge_method", "hessian");
        pipe = tanuki::pipeline::CreateFindStreamRidgeline(method);
    }
    // 未來新 pipeline 在此加分支（單一 API 簽名不變）

    if (!pipe) return nullptr;   // 未知 pipeline / 未知方法 → 明確失敗

    auto* ctx = new PipelineContext();
    ctx->pipeline = std::move(pipe);
    return reinterpret_cast<TanukiPipelineHandle>(ctx);
}

int TanukiPipeline_Process(TanukiPipelineHandle handle,
                           const TanukiPipelineInputC* input,
                           const char* json_params,
                           const float* precomputed_col_mean,
                           const TanukiPipelineOutputC* output) {
    if (handle == nullptr) return -1;
    if (input == nullptr || output == nullptr || input->data == nullptr) return -1;
    auto* ctx = reinterpret_cast<PipelineContext*>(handle);

    if (!ctx->EnsureBuffers(input->width, input->height, &ctx->last_error)) return -2;

    if (cudaMemcpy(ctx->d_input, input->data, ctx->image_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        ctx->last_error = "Failed to copy input image from host to CUDA memory."; return -2;
    }

    tanuki::pipeline::InputImage in;
    in.width = input->width; in.height = input->height; in.data = ctx->d_input; in.stream = input->stream;

    // precomputed column mean：host → GPU（用 d_curve_mean 暫存）
    if (precomputed_col_mean != nullptr) {
        if (cudaMemcpy(ctx->d_curve_mean, precomputed_col_mean,
                       input->width * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            ctx->last_error = "Failed to copy precomputed column mean to GPU."; return -2;
        }
    }

    // 演算法參數：json → Params（各 module 取自己要的 key；缺 key 用 Params 預設）
    tanuki::pipeline::Params p;
    p.bg_sigma_factor = get_number(json_params, "bg_sigma_factor", p.bg_sigma_factor);
    p.ridge_sigma = get_number(json_params, "ridge_sigma", p.ridge_sigma);
    p.hessian_max_factor = get_number(json_params, "hessian_max_factor", p.hessian_max_factor);
    ctx->ridge_mode_buf = get_string(json_params, "ridge_mode", p.ridge_mode);
    p.ridge_mode = ctx->ridge_mode_buf.c_str();
    p.precomputed_col_mean = precomputed_col_mean != nullptr ? ctx->d_curve_mean : nullptr;

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

    bool want_standard = output->standard_width > 0 && output->standard_height > 0 &&
        (output->resized_hessian_column_half != nullptr || output->resized_hessian_row_half != nullptr);
    size_t standard_samples = (size_t)output->standard_width * (size_t)output->standard_height;
    if (want_standard) {
        if (!ctx->EnsureStandardHalf(standard_samples)) {
            ctx->last_error = "Failed to allocate standard Hessian workspace.";
            return -2;
        }
        out.standard_width = output->standard_width;
        out.standard_height = output->standard_height;
        out.hessian_column_half = output->resized_hessian_column_half != nullptr
            ? ctx->d_standard_half : nullptr;
        out.hessian_row_half = output->resized_hessian_row_half != nullptr
            ? ctx->d_standard_half + standard_samples : nullptr;
    }

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

    // 存檔縮圖（fused，一進多出）：用檢測後仍 resident 的 device buffer 就地縮，免二次 H2D。
    //   raw←d_input、V←d_ridge、H←d_mura（"vertical+horizontal" 下 d_ridge=V、d_mura=H）。
    //   個別 dst 為 NULL 則跳過；resize_width/height<=0 則整段跳過（純 live 幀不縮）。
    if (output->resize_width > 0 && output->resize_height > 0) {
        int rw = output->resize_width, rh = output->resize_height;
        size_t rbytes = (size_t)rw * rh;
        if (!ctx->EnsureResize(rbytes)) { ctx->last_error = "Failed to allocate resize workspace."; return -2; }
        cudaStream_t rs = (cudaStream_t)out.stream;
        auto resize_d2h = [&](const uint8_t* d_src, uint8_t* h_dst, const char* what) -> bool {
            if (h_dst == nullptr) return true;
            tanuki::core::resize_u8_gpu(d_src, ctx->width, ctx->height, ctx->d_resize, rw, rh, rs);
            if (cudaMemcpy(h_dst, ctx->d_resize, rbytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
                ctx->last_error = std::string("Failed to copy resized ") + what + " to host."; return false;
            }
            return true;
        };
        if (!resize_d2h(ctx->d_input, output->resized_raw,   "raw"))   return -2;
        if (!resize_d2h(ctx->d_ridge, output->resized_ridge, "ridge")) return -2;
        if (!resize_d2h(ctx->d_mura,  output->resized_mura,  "mura"))  return -2;
    }

    if (want_standard) {
        size_t half_bytes = standard_samples * sizeof(uint16_t);
        if (output->resized_hessian_column_half != nullptr &&
            !d2h(output->resized_hessian_column_half, ctx->d_standard_half,
                 half_bytes, "standard column Hessian")) return -2;
        if (output->resized_hessian_row_half != nullptr &&
            !d2h(output->resized_hessian_row_half, ctx->d_standard_half + standard_samples,
                 half_bytes, "standard row Hessian")) return -2;
    }

    ctx->last_error.clear();
    return 0;
}

const char* TanukiPipeline_GetLastError(TanukiPipelineHandle handle) {
    if (handle == nullptr) return "Invalid pipeline handle.";
    return reinterpret_cast<PipelineContext*>(handle)->last_error.c_str();
}

void TanukiPipeline_Destroy(TanukiPipelineHandle handle) {
    if (handle == nullptr) return;
    delete reinterpret_cast<PipelineContext*>(handle);
}

int TanukiPipeline_ComputeColumnMean(TanukiPipelineHandle handle,
                                     const TanukiPipelineInputC* input,
                                     float bg_sigma_factor,
                                     float* out_col_mean) {
    if (handle == nullptr || input == nullptr || out_col_mean == nullptr || input->data == nullptr) return -1;
    auto* ctx = reinterpret_cast<PipelineContext*>(handle);
    if (!ctx->EnsureBuffers(input->width, input->height, &ctx->last_error)) return -2;
    if (cudaMemcpy(ctx->d_input, input->data, ctx->image_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        ctx->last_error = "Failed to copy input image for column mean."; return -2;
    }
    int sigma_col = (int)bg_sigma_factor; if (sigma_col < 1) sigma_col = 1;  // 沿用舊版整數截斷行為
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
