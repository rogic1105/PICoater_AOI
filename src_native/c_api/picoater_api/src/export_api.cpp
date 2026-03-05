#include "export_c/export_api.h"

#include <memory>
#include <string>

#include "../../../pipeline/aoi_pipeline.hpp"
#include "../../../modules/GetPICoaterBackground/include/module_get_picoater_background.hpp"

namespace {

struct AoiPipelineContext {
  picoater::aoi::AoiPipeline pipeline;
  std::string last_error;
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

  auto* context = reinterpret_cast<AoiPipelineContext*>(handle);

  picoater::aoi::AoiImage input_image;
  input_image.width = width;
  input_image.height = height;
  input_image.data = const_cast<uint8_t*>(d_input);
  input_image.bg_sigma_factor = bg_sigma_factor;
  input_image.ridge_sigma = ridge_sigma;
  input_image.hessian_max_factor = hessian_max_factor;
  input_image.ridge_mode = ridge_mode;
  input_image.stream = stream;

  picoater::aoi::AoiImage output_image;
  output_image.width = width;
  output_image.height = height;
  output_image.background_data = d_background_output;
  output_image.mura_data = d_mura_output;
  output_image.ridge_data = d_ridge_output;
  output_image.mura_curve_mean = d_mura_curve_mean_output;
  output_image.mura_curve_max = d_mura_curve_max_output;
  output_image.stream = stream;

  if (!context->pipeline.Process(input_image, &output_image)) {
    context->last_error = context->pipeline.GetLastError();
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

}  // extern "C"
