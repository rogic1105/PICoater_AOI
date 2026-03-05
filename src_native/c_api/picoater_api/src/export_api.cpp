#include "export_c/export_api.h"

#include <cuda_runtime.h>

#include <memory>
#include <string>

#include "../../../modules/GetPICoaterBackground/include/module_get_picoater_background.hpp"
#include "../../../pipeline/aoi_pipeline.hpp"
#include "../../../plc/i_plc_adapter.hpp"

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

  ~AoiPipelineContext() { ReleaseBuffers(); }

  void ReleaseBuffers() {
    if (d_input != nullptr) cudaFree(d_input);
    if (d_background != nullptr) cudaFree(d_background);
    if (d_mura != nullptr) cudaFree(d_mura);
    if (d_ridge != nullptr) cudaFree(d_ridge);
    if (d_curve_mean != nullptr) cudaFree(d_curve_mean);
    if (d_curve_max != nullptr) cudaFree(d_curve_max);

    d_input = nullptr;
    d_background = nullptr;
    d_mura = nullptr;
    d_ridge = nullptr;
    d_curve_mean = nullptr;
    d_curve_max = nullptr;
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
        cudaMalloc(&d_curve_max, width * sizeof(float)) != cudaSuccess) {
      *error = "Failed to allocate internal CUDA buffers for pipeline processing.";
      ReleaseBuffers();
      return false;
    }

    return true;
  }
};

struct PlcAdapterContext {
  std::unique_ptr<picoater::plc::IPlcAdapter> adapter;
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
                                int width,
                                int height,
                                const uint8_t* d_input,
                                uint8_t* d_background_output,
                                uint8_t* d_mura_output,
                                uint8_t* d_ridge_output,
                                float* d_mura_curve_mean_output,
                                float* d_mura_curve_max_output,
                                float bg_sigma_factor,
                                float ridge_sigma,
                                float hessian_max_factor,
                                const char* ridge_mode,
                                void* stream) {
  if (handle == nullptr) {
    return -1;
  }

  if (d_input == nullptr) {
    return -1;
  }

  auto* context = reinterpret_cast<AoiPipelineContext*>(handle);

  if (!context->EnsureBuffers(width, height, &context->last_error)) {
    return -2;
  }

  if (cudaMemcpy(context->d_input,
                 d_input,
                 context->image_size,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    context->last_error = "Failed to copy input image from host to CUDA memory.";
    return -2;
  }

  picoater::aoi::AoiImage input_image;
  input_image.width = width;
  input_image.height = height;
  input_image.data = context->d_input;
  input_image.bg_sigma_factor = bg_sigma_factor;
  input_image.ridge_sigma = ridge_sigma;
  input_image.hessian_max_factor = hessian_max_factor;
  input_image.ridge_mode = ridge_mode;
  input_image.stream = stream;

  picoater::aoi::AoiImage output_image;
  output_image.width = width;
  output_image.height = height;
  output_image.background_data = context->d_background;
  output_image.mura_data = context->d_mura;
  output_image.ridge_data = context->d_ridge;
  output_image.mura_curve_mean = context->d_curve_mean;
  output_image.mura_curve_max = context->d_curve_max;
  output_image.stream = stream;

  if (!context->pipeline.Process(input_image, &output_image)) {
    context->last_error = context->pipeline.GetLastError();
    return -2;
  }

  if (d_background_output != nullptr &&
      cudaMemcpy(d_background_output,
                 context->d_background,
                 context->image_size,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    context->last_error = "Failed to copy background output to host.";
    return -2;
  }

  if (d_mura_output != nullptr &&
      cudaMemcpy(
          d_mura_output, context->d_mura, context->image_size, cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
    context->last_error = "Failed to copy mura output to host.";
    return -2;
  }

  if (d_ridge_output != nullptr &&
      cudaMemcpy(d_ridge_output,
                 context->d_ridge,
                 context->image_size,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    context->last_error = "Failed to copy ridge output to host.";
    return -2;
  }

  if (d_mura_curve_mean_output != nullptr &&
      cudaMemcpy(d_mura_curve_mean_output,
                 context->d_curve_mean,
                 width * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    context->last_error = "Failed to copy mura mean curve output to host.";
    return -2;
  }

  if (d_mura_curve_max_output != nullptr &&
      cudaMemcpy(d_mura_curve_max_output,
                 context->d_curve_max,
                 width * sizeof(float),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    context->last_error = "Failed to copy mura max curve output to host.";
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

PlcAdapterHandle PICoaterAPI_CreateMockPlc() {
  auto* context = new PlcAdapterContext();
  context->adapter = std::make_unique<picoater::plc::MockPlcAdapter>();
  return reinterpret_cast<PlcAdapterHandle>(context);
}

void PICoaterAPI_DestroyPlc(PlcAdapterHandle handle) {
  if (handle == nullptr) {
    return;
  }

  auto* context = reinterpret_cast<PlcAdapterContext*>(handle);
  delete context;
}

int PICoaterAPI_PlcConnect(PlcAdapterHandle handle) {
  if (handle == nullptr) {
    return -1;
  }

  auto* context = reinterpret_cast<PlcAdapterContext*>(handle);
  return context->adapter->Connect() ? 0 : -2;
}

int PICoaterAPI_PlcReadBit(PlcAdapterHandle handle, int address, bool* value) {
  if (handle == nullptr || value == nullptr) {
    return -1;
  }

  auto* context = reinterpret_cast<PlcAdapterContext*>(handle);
  return context->adapter->ReadBit(address, value) ? 0 : -2;
}

int PICoaterAPI_PlcWriteBit(PlcAdapterHandle handle, int address, bool value) {
  if (handle == nullptr) {
    return -1;
  }

  auto* context = reinterpret_cast<PlcAdapterContext*>(handle);
  return context->adapter->WriteBit(address, value) ? 0 : -2;
}

}  // extern "C"
