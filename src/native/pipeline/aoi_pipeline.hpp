#ifndef PICOATER_AOI_SRC_NATIVE_PIPELINE_AOI_PIPELINE_HPP_
#define PICOATER_AOI_SRC_NATIVE_PIPELINE_AOI_PIPELINE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "modules/i_aoi_module.hpp"

namespace picoater {
namespace aoi {

class AoiPipeline {
 public:
  void AddModule(std::unique_ptr<IAoiModule> module);
  bool Process(const AoiInputImage& input_image,
               const AoiAlgorithmParams& params,
               AoiOutputBuffers* output_buffers);
  std::string GetLastError() const;

 private:
  std::vector<std::unique_ptr<IAoiModule>> modules_;
  std::string last_error_;
};

}  // namespace aoi
}  // namespace picoater

#endif  // PICOATER_AOI_SRC_NATIVE_PIPELINE_AOI_PIPELINE_HPP_
