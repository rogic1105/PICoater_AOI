#include "tanuki/pipeline/pipeline.hpp"

namespace tanuki { namespace pipeline {

void Pipeline::AddModule(std::unique_ptr<IModule> module) {
    if (module) modules_.push_back(std::move(module));
}

bool Pipeline::Process(const InputImage& input, const Params& params, OutputBuffers* output) {
    last_error_.clear();
    for (auto& m : modules_) {
        if (!m->Process(input, params, output)) {
            last_error_ = std::string("module '") + m->Name() + "' failed: " + m->GetLastError();
            return false;
        }
    }
    return true;
}

std::string Pipeline::GetLastError() const { return last_error_; }

}} // namespace tanuki::pipeline
