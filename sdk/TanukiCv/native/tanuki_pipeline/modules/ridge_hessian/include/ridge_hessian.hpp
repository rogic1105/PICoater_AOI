#pragma once
// 脊線 module（hessian 法）：blur → hessian 響應 → 曲線(切/法向 mean/max) → scale_clamp u8。
// 組合 tanuki_core 的 gaussianBlur + computeHessianResponse + calcColumn/Row Means/Max + scale_clamp。
// 讀「去背影像」（前一個 module 寫的 output->mura_data），寫 output->ridge_data + 曲線。
// 換脊線方法 = 換成另一個實作此角色的 module（如 ridge_gabor），pipeline 食譜選 name。
#include "tanuki/pipeline/i_module.hpp"
#include <string>

namespace tanuki { namespace pipeline {

class RidgeHessianModule : public IModule {
public:
    ~RidgeHessianModule() override;
    bool Initialize() override;
    bool Process(const InputImage& input, const Params& params, OutputBuffers* output) override;
    std::string GetLastError() const override;
    const char* Name() const override { return "ridge_hessian"; }

private:
    void EnsureBuffers(int w, int h);   // 依尺寸 lazy 配置 workspace + views
    void Release();

    int w_ = 0, h_ = 0;
    void* d_workspace_ = nullptr;   // 單一 workspace（gaussian 內部 scratch + hessian views，沿用原設計）
    float* d_blur_f32_ = nullptr;   // workspace 內 view：blur 輸出
    float* d_resp_ = nullptr;       // workspace 內 view：hessian 響應
    std::string err_;
};

}} // namespace tanuki::pipeline
