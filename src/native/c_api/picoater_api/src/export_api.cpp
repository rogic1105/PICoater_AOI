#include "export_c/export_api.h"

#include <cuda_runtime.h>

#include <memory>
#include <string>

#include "core_cv/imgproc/core_background.hpp"

#include "../../../modules/GetPICoaterBackground/include/module_get_picoater_background.hpp"
#include "../../../pipeline/aoi_pipeline.hpp"

namespace {

struct AoiPipelineContext {
  picoater::aoi::AoiPipeline pipeline;
  std::string last_error;

  int width = 0;
  int height = 0;
  size_t image_size = 0;

  uint8_t* d_input = nullptr;
  uint8_t* d_background = nullptr;
  uint8_t* d_mura = nullptr;
  uint8_t* d_ridge = nullptr;
  float* d_curve_mean = nullptr;
  float* d_curve_max = nullptr;
  float* d_row_curve_mean = nullptr;
  float* d_row_curve_max = nullptr;

  ~AoiPipelineContext() { ReleaseBuffers(); }

  void ReleaseBuffers() {
    if (d_input != nullptr) cudaFree(d_input);
    if (d_background != nullptr) cudaFree(d_background);
    if (d_mura != nullptr) cudaFree(d_mura);
    if (d_ridge != nullptr) cudaFree(d_ridge);
    if (d_curve_mean != nullptr) cudaFree(d_curve_mean);
    if (d_curve_max != nullptr) cudaFree(d_curve_max);
    if (d_row_curve_mean != nullptr) cudaFree(d_row_curve_mean);
    if (d_row_curve_max != nullptr) cudaFree(d_row_curve_max);

    d_input = nullptr;
    d_background = nullptr;
    d_mura = nullptr;
    d_ridge = nullptr;
    d_curve_mean = nullptr;
    d_curve_max = nullptr;
    d_row_curve_mean = nullptr;
    d_row_curve_max = nullptr;
    width = 0;
    height = 0;
    image_size = 0;
  }

  bool EnsureBuffers(int new_width, int new_height, std::string* error) {
    if (new_width <= 0 || new_height <= 0) {
      *error = "width and height must be positive.";
      return false;
    }

    if (width == new_width && height == new_height && d_input != nullptr) {
      return true;
    }

    ReleaseBuffers();

    width = new_width;
    height = new_height;
    image_size = static_cast<size_t>(width) * static_cast<size_t>(height);

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
  auto* context = new AoiPipelineContext();
  context->pipeline.AddModule(
      std::make_unique<picoater::aoi::GetPICoaterBackgroundModule>());
  return reinterpret_cast<AoiPipelineHandle>(context);
}

int PICoaterAPI_ProcessPipeline(AoiPipelineHandle handle,
                                const AoiInputImageC* input,
                                const AoiAlgorithmParamsC* params,
                                const AoiOutputBuffersC* output) {
  if (handle == nullptr) {
    return -1;
  }

  if (input == nullptr || params == nullptr || output == nullptr ||
      input->data == nullptr) {
    return -1;
  }

  auto* context = reinterpret_cast<AoiPipelineContext*>(handle);

  if (!context->EnsureBuffers(input->width, input->height, &context->last_error)) {
    return -2;
  }

  if (cudaMemcpy(context->d_input,
                 input->data,
                 context->image_size,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    context->last_error = "Failed to copy input image from host to CUDA memory.";
    return -2;
  }

  picoater::aoi::AoiInputImage input_image;
  input_image.width = input->width;
  input_image.height = input->height;
  input_image.data = context->d_input;
  input_image.stream = input->stream;

  // Pre-computed column mean: copy from host to GPU if provided
  if (params->precomputed_col_mean != nullptr) {
    if (cudaMemcpy(context->d_curve_mean,
                   params->precomputed_col_mean,
                   input->width * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      context->last_error = "Failed to copy precomputed column mean to GPU.";
      return -2;
    }
  }

  picoater::aoi::AoiAlgorithmParams algo_params;
  algo_params.bg_sigma_factor = params->bg_sigma_factor;
  algo_params.ridge_sigma = params->ridge_sigma;
  algo_params.hessian_max_factor = params->hessian_max_factor;
  algo_params.ridge_mode = params->ridge_mode;
  algo_params.precomputed_col_mean =
      params->precomputed_col_mean != nullptr ? context->d_curve_mean : nullptr;

  picoater::aoi::AoiOutputBuffers output_image;
  output_image.width = output->width > 0 ? output->width : input->width;
  output_image.height = output->height > 0 ? output->height : input->height;
  output_image.background_data = context->d_background;
  output_image.mura_data = context->d_mura;
  output_image.ridge_data = context->d_ridge;
  output_image.mura_curve_mean = context->d_curve_mean;
  output_image.mura_curve_max = context->d_curve_max;
  output_image.mura_row_curve_mean = context->d_row_curve_mean;
  output_image.mura_row_curve_max = context->d_row_curve_max;
  output_image.stream = output->stream != nullptr ? output->stream : input->stream;

  if (!context->pipeline.Process(input_image, algo_params, &output_image)) {
    context->last_error = context->pipeline.GetLastError();
    return -2;
  }

  if (output->background_data != nullptr &&
      cudaMemcpy(output->background_data,
                 context->d_background,
                 context->image_size,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    context->last_error = "Failed to copy background output to host.";
    return -2;
  }

  if (output->mura_data != nullptr &&
      cudaMemcpy(
          output->mura_data, context->d_mura, context->image_size, cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
    context->last_error = "Failed to copy mura output to host.";
    return -2;
  }

  if (output->ridge_data != nullptr &&
      cudaMemcpy(output->ridge_data,
                 context->d_ridge,
                 context->image_size,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    context->last_error = "Failed to copy ridge output to host.";
    return -2;
  }

  if (output->mura_curve_mean != nullptr &&
      cudaMemcpy(output->mura_curve_mean,
                 context->d_curve_mean,
                 input->width * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    context->last_error = "Failed to copy mura mean curve output to host.";
    return -2;
  }

  if (output->mura_curve_max != nullptr &&
      cudaMemcpy(output->mura_curve_max,
                 context->d_curve_max,
                 input->width * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    context->last_error = "Failed to copy mura max curve output to host.";
    return -2;
  }

  if (output->mura_row_curve_mean != nullptr &&
      cudaMemcpy(output->mura_row_curve_mean,
                 context->d_row_curve_mean,
                 input->height * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    context->last_error = "Failed to copy row mean curve output to host.";
    return -2;
  }

  if (output->mura_row_curve_max != nullptr &&
      cudaMemcpy(output->mura_row_curve_max,
                 context->d_row_curve_max,
                 input->height * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    context->last_error = "Failed to copy row max curve output to host.";
    return -2;
  }

  context->last_error.clear();
  return 0;
}

const char* PICoaterAPI_GetLastError(AoiPipelineHandle handle) {
  if (handle == nullptr) {
    return "Invalid pipeline handle.";
  }

  auto* context = reinterpret_cast<AoiPipelineContext*>(handle);
  return context->last_error.c_str();
}

void PICoaterAPI_DestroyPipeline(AoiPipelineHandle handle) {
  if (handle == nullptr) {
    return;
  }

  auto* context = reinterpret_cast<AoiPipelineContext*>(handle);
  delete context;
}

int PICoaterAPI_ComputeColumnMean(AoiPipelineHandle handle,
                                  const AoiInputImageC* input,
                                  float bg_sigma_factor,
                                  float* out_col_mean) {
  if (handle == nullptr || input == nullptr || out_col_mean == nullptr ||
      input->data == nullptr) {
    return -1;
  }

  auto* context = reinterpret_cast<AoiPipelineContext*>(handle);

  if (!context->EnsureBuffers(input->width, input->height,
                              &context->last_error)) {
    return -2;
  }

  if (cudaMemcpy(context->d_input, input->data, context->image_size,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    context->last_error = "Failed to copy input image for column mean.";
    return -2;
  }

  int sigma_col = static_cast<int>(bg_sigma_factor);
  if (sigma_col < 1) sigma_col = 1;

  // Only Step 1: compute column means with outlier removal
  core::calcColumnMeans_RemoveOutliers_gpu<uint8_t>(
      context->d_input, context->d_curve_mean, input->width, input->height,
      sigma_col, nullptr);
  cudaDeviceSynchronize();

  if (cudaMemcpy(out_col_mean, context->d_curve_mean,
                 input->width * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    context->last_error = "Failed to copy column mean to host.";
    return -2;
  }

  context->last_error.clear();
  return 0;
}

}  // extern "C"
