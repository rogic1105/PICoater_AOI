#pragma once
// 去背 module：robust column 背景估計（或用 precomputed）+ column 背景相減 → 去背影像。
// 組合 tanuki_core 的 calcColumnMeans_RemoveOutliers + calcColumnBackground（module 擁有食譜、不擁有 kernel）。
#include "tanuki/pipeline/i_module.hpp"
#include <string>

namespace tanuki { namespace pipeline {

class BackgroundSubModule : public IModule {
public:
    ~BackgroundSubModule() override;
    bool Initialize() override;
    bool Process(const InputImage& input, const Params& params, OutputBuffers* output) override;
    std::string GetLastError() const override;
    const char* Name() const override { return "background_sub"; }

private:
    bool EnsureBuffers(int w, int h);   // 依輸入尺寸 lazy 配置（失敗設 err_ 回 false）
    void Release();

    int w_ = 0, h_ = 0;
    float* d_col_mean_ = nullptr;       // 本 module 擁有的 column 背景 buffer
    std::string err_;
};

}} // namespace tanuki::pipeline
