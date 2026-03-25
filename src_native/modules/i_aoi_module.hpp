#ifndef PICOATER_AOI_SRC_NATIVE_MODULES_I_AOI_MODULE_HPP_
#define PICOATER_AOI_SRC_NATIVE_MODULES_I_AOI_MODULE_HPP_

#include <cstdint>
#include <string>

namespace picoater {
namespace aoi {

struct AoiInputImage {
  int width = 0;
  int height = 0;

  // Primary image buffer pointer on GPU.
  uint8_t* data = nullptr;

  void* stream = nullptr;
};

struct AoiOutputBuffers {
  int width = 0;
  int height = 0;

  // Optional output buffers on GPU.
  uint8_t* background_data = nullptr;
  uint8_t* mura_data = nullptr;
  uint8_t* ridge_data = nullptr;
  float* mura_curve_mean = nullptr;
  float* mura_curve_max = nullptr;
  float* mura_row_curve_mean = nullptr;
  float* mura_row_curve_max = nullptr;

  void* stream = nullptr;
};

struct AoiAlgorithmParams {
  // Processing parameters.
  float bg_sigma_factor = 1.0f;
  float ridge_sigma = 1.0f;
  float hessian_max_factor = 1.0f;
  const char* ridge_mode = "dark";
};

class IAoiModule {
 public:
  virtual ~IAoiModule() = default;

  virtual bool Initialize() = 0;
  virtual bool Process(const AoiInputImage& input,
                       const AoiAlgorithmParams& params,
                       AoiOutputBuffers* output) = 0;
  virtual std::string GetLastError() const = 0;
};

}  // namespace aoi
}  // namespace picoater

#endif  // PICOATER_AOI_SRC_NATIVE_MODULES_I_AOI_MODULE_HPP_
