#pragma once
// find_stream_ridgeline pipeline 食譜：找「流水圖」的脊線（mura 檢測）。
// 流程：background_sub（去背）→ ridge_*（脊線偵測，方法可選）。
// 換脊線方法 = 改 ridge_method 參數（目前 "hessian"，未來 "gabor"…），不改 pipeline 結構。
#include "tanuki/pipeline/pipeline.hpp"
#include <memory>
#include <string>

namespace tanuki { namespace pipeline {

// 建一條 find_stream_ridgeline pipeline。ridge_method 選脊線法（預設 hessian）。
std::unique_ptr<Pipeline> CreateFindStreamRidgeline(const std::string& ridge_method = "hessian");

}} // namespace tanuki::pipeline
