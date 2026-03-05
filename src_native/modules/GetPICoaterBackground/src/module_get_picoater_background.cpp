#include "module_get_picoater_background.hpp"

namespace picoater {
namespace aoi {

GetPICoaterBackgroundModule::GetPICoaterBackgroundModule() = default;

GetPICoaterBackgroundModule::~GetPICoaterBackgroundModule() = default;

bool GetPICoaterBackgroundModule::Initialize() {
  last_error_.clear();
  return true;
}

bool GetPICoaterBackgroundModule::Process(const AoiImage& input_image,
                                          AoiImage* output_image) {
  (void)input_image;
  if (output_image == nullptr) {
    last_error_ = "output_image must not be null.";
    return false;
  }

  last_error_.clear();
  return true;
}

std::string GetPICoaterBackgroundModule::GetLastError() const {
  return last_error_;
}

}  // namespace aoi
}  // namespace picoater
