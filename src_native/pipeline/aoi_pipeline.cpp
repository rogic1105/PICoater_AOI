#include "aoi_pipeline.hpp"

#include <utility>

namespace picoater {
namespace aoi {

void AoiPipeline::AddModule(std::unique_ptr<IAoiModule> module) {
  if (module == nullptr) {
    return;
  }

  modules_.push_back(std::move(module));
}

bool AoiPipeline::Process(const AoiImage& input_image, AoiImage* output_image) {
  if (output_image == nullptr) {
    last_error_ = "output_image must not be null.";
    return false;
  }

  if (modules_.empty()) {
    last_error_ = "No AOI module has been registered to the pipeline.";
    return false;
  }

  AoiImage current_input = input_image;
  AoiImage current_output = *output_image;

  for (const auto& module : modules_) {
    if (!module->Process(current_input, &current_output)) {
      last_error_ = module->GetLastError();
      return false;
    }
    current_input = current_output;
  }

  *output_image = current_output;
  last_error_.clear();
  return true;
}

std::string AoiPipeline::GetLastError() const {
  return last_error_;
}

}  // namespace aoi
}  // namespace picoater
