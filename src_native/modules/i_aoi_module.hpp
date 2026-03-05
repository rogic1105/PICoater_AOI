#ifndef PICOATER_AOI_SRC_NATIVE_MODULES_I_AOI_MODULE_HPP_
#define PICOATER_AOI_SRC_NATIVE_MODULES_I_AOI_MODULE_HPP_

#include <string>

namespace picoater {
namespace aoi {

struct AoiImage;

class IAoiModule {
 public:
  virtual ~IAoiModule() = default;

  virtual bool Initialize() = 0;
  virtual bool Process(const AoiImage& input_image, AoiImage* output_image) = 0;
  virtual std::string GetLastError() const = 0;
};

}  // namespace aoi
}  // namespace picoater

#endif  // PICOATER_AOI_SRC_NATIVE_MODULES_I_AOI_MODULE_HPP_
