#ifndef PICOATER_AOI_SRC_NATIVE_MODULES_GET_PICOATER_BACKGROUND_MODULE_HPP_
#define PICOATER_AOI_SRC_NATIVE_MODULES_GET_PICOATER_BACKGROUND_MODULE_HPP_

#include <string>

#include "../../i_aoi_module.hpp"

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
  std::string last_error_;
};

}  // namespace aoi
}  // namespace picoater

#endif  // PICOATER_AOI_SRC_NATIVE_MODULES_GET_PICOATER_BACKGROUND_MODULE_HPP_
