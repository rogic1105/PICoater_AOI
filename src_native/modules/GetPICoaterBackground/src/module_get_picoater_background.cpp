#include "module_get_picoater_background.hpp"

namespace picoater {
namespace aoi {

namespace {
constexpr const char* kDefaultRidgeMode = "dark";
}  // namespace

GetPICoaterBackgroundModule::GetPICoaterBackgroundModule() = default;

GetPICoaterBackgroundModule::~GetPICoaterBackgroundModule() {
  detector_.Release();
}

bool GetPICoaterBackgroundModule::Initialize() {
  last_error_.clear();
  initialized_ = true;
  return true;
}

bool GetPICoaterBackgroundModule::Process(const AoiInputImage& input_image,
                                          const AoiAlgorithmParams& params,
                                          AoiOutputBuffers* output_image) {
  if (!initialized_ && !Initialize()) {
    if (last_error_.empty()) {
      last_error_ = "Failed to initialize GetPICoaterBackgroundModule.";
    }
    return false;
  }

  if (!ValidateInputImage(input_image)) {
    return false;
  }

  if (output_image == nullptr) {
    last_error_ = "output_image must not be null.";
    return false;
  }

  if (!ValidateOutputImage(*output_image)) {
    return false;
  }

  output_image->width = input_image.width;
  output_image->height = input_image.height;

  detector_.Initialize(input_image.width, input_image.height);
  detector_.Run(
      input_image.data,
      output_image->background_data,
      output_image->mura_data,
      output_image->ridge_data,
      output_image->mura_curve_mean,
      output_image->mura_curve_max,
      output_image->mura_row_curve_mean,
      output_image->mura_row_curve_max,
      params.bg_sigma_factor,
      params.ridge_sigma,
      params.hessian_max_factor,
      params.ridge_mode == nullptr ? kDefaultRidgeMode : params.ridge_mode,
      reinterpret_cast<cudaStream_t>(input_image.stream));

  last_error_.clear();
  return true;
}

std::string GetPICoaterBackgroundModule::GetLastError() const {
  return last_error_;
}

bool GetPICoaterBackgroundModule::ValidateInputImage(
    const AoiInputImage& input_image) {
  if (input_image.width <= 0 || input_image.height <= 0) {
    last_error_ = "input_image width/height must be positive.";
    return false;
  }

  if (input_image.data == nullptr) {
    last_error_ = "input_image.data must not be null.";
    return false;
  }

  return true;
}

bool GetPICoaterBackgroundModule::ValidateOutputImage(
    const AoiOutputBuffers& output_image) {
  if (output_image.mura_data == nullptr) {
    last_error_ = "output_image.mura_data must not be null.";
    return false;
  }

  if (output_image.ridge_data == nullptr) {
    last_error_ = "output_image.ridge_data must not be null.";
    return false;
  }

  if (output_image.mura_curve_mean == nullptr ||
      output_image.mura_curve_max == nullptr) {
    last_error_ =
        "output_image.mura_curve_mean and output_image.mura_curve_max must not be null.";
    return false;
  }

  return true;
}

}  // namespace aoi
}  // namespace picoater
