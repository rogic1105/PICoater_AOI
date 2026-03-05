#ifndef PICOATER_AOI_SRC_NATIVE_MODULES_GET_PICOATER_BACKGROUND_MODULE_HPP_
#define PICOATER_AOI_SRC_NATIVE_MODULES_GET_PICOATER_BACKGROUND_MODULE_HPP_

#include <string>

#include "../../i_aoi_module.hpp"
#include "Module_GetPICoaterBackground.hpp"

namespace picoater {
namespace aoi {

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

}  // namespace aoi
}  // namespace picoater

#endif  // PICOATER_AOI_SRC_NATIVE_MODULES_GET_PICOATER_BACKGROUND_MODULE_HPP_
