#pragma once
// Pipeline 工頭：持有一串 module，依序跑（食譜 = 加哪些 module、什麼順序）。
// 從 src/native picoater::aoi::AoiPipeline 帶過來、改 namespace。
#include <memory>
#include <string>
#include <vector>
#include "tanuki/pipeline/i_module.hpp"

namespace tanuki { namespace pipeline {

class Pipeline {
public:
    void AddModule(std::unique_ptr<IModule> module);

    // 依序跑每個 module；任一 module 回 false 即中止，錯誤記在 last_error_。
    bool Process(const InputImage& input, const Params& params, OutputBuffers* output);

    std::string GetLastError() const;
    size_t ModuleCount() const { return modules_.size(); }

private:
    std::vector<std::unique_ptr<IModule>> modules_;
    std::string last_error_;
};

}} // namespace tanuki::pipeline
