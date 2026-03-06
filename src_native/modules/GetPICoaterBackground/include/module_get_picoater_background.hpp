#ifndef PICOATER_AOI_SRC_NATIVE_MODULES_GET_PICOATER_BACKGROUND_MODULE_HPP_
#define PICOATER_AOI_SRC_NATIVE_MODULES_GET_PICOATER_BACKGROUND_MODULE_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../../i_aoi_module.hpp"

namespace picoater {

class PICoaterDetector {
 public:
  PICoaterDetector();
  ~PICoaterDetector();

  void Initialize(int width, int height);

  void Run(const uint8_t* d_in,
           uint8_t* d_bg_out,
           uint8_t* d_mura_out,
           uint8_t* d_ridge_out,
           float* d_mura_curve_mean,
           float* d_mura_curve_max,
           float bg_sigma_factor,
           float ridge_sigma,
           float hessian_max_factor,
           const char* ridge_mode,
           cudaStream_t stream = 0);

  void RunCPU(const uint8_t* h_in,
              uint8_t* h_mura_out,
              float bg_sigma_factor);

  void Release();

 private:
  int m_width = 0;
  int m_height = 0;

  float* d_col_mean = nullptr;
  uint8_t* d_col_bg_ = nullptr;
  uint8_t* d_blur_tmp_ = nullptr;
  void* d_workspace_ = nullptr;

  uint8_t* d_hessian_u8_ = nullptr;
  float* d_hessian_f32_ = nullptr;
  float* d_hessian_resp_ = nullptr;

  std::vector<float> h_col_mean;
};

}  // namespace picoater

namespace picoater::aoi {

class GetPICoaterBackgroundModule : public IAoiModule {
 public:
  GetPICoaterBackgroundModule();
  ~GetPICoaterBackgroundModule() override;

  bool Initialize() override;
  bool Process(const AoiImage& input_image, AoiImage* output_image) override;
  std::string GetLastError() const override;

 private:
  bool ValidateInputImage(const AoiImage& input_image);
  bool ValidateOutputImage(const AoiImage& output_image);

  ::picoater::PICoaterDetector detector_;
  bool initialized_ = false;
  std::string last_error_;
};

}  // namespace picoater::aoi

#endif  // PICOATER_AOI_SRC_NATIVE_MODULES_GET_PICOATER_BACKGROUND_MODULE_HPP_
