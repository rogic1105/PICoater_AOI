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

bool AoiPipeline::Process(const AoiInputImage& input_image,
                          const AoiAlgorithmParams& params,
                          AoiOutputBuffers* output_buffers) {
  if (output_buffers == nullptr) {
    last_error_ = "output_buffers must not be null.";
    return false;
  }

  if (modules_.empty()) {
    last_error_ = "No AOI module has been registered to the pipeline.";
    return false;
  }

  AoiInputImage current_input = input_image;
  AoiOutputBuffers current_output = *output_buffers;

  for (const auto& module : modules_) {
    if (!module->Process(current_input, params, &current_output)) {
      last_error_ = module->GetLastError();
      return false;
    }
    current_input.width = current_output.width;
    current_input.height = current_output.height;
    current_input.data = current_output.background_data;
    current_input.stream = current_output.stream;
  }

  *output_buffers = current_output;
  last_error_.clear();
  return true;
}

std::string AoiPipeline::GetLastError() const {
  return last_error_;
}

}  // namespace aoi
}  // namespace picoater
