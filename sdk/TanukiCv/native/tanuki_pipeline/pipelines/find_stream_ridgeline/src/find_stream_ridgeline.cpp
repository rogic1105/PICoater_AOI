#include "find_stream_ridgeline.hpp"
#include "background_sub.hpp"
#include "ridge_hessian.hpp"

namespace tanuki { namespace pipeline {

std::unique_ptr<Pipeline> CreateFindStreamRidgeline(const std::string& ridge_method) {
    auto p = std::make_unique<Pipeline>();

    // Step 1：去背（流水圖 column 背景相減）
    p->AddModule(std::make_unique<BackgroundSubModule>());

    // Step 2：脊線偵測（方法由 ridge_method 選；直接 new 以強制連結模組——
    //          registry 的 static 自註冊在 static lib 會被 linker 丟 TU，故食譜不走 Create(name)）
    if (ridge_method == "hessian" || ridge_method.empty()) {
        p->AddModule(std::make_unique<RidgeHessianModule>());
    }
    // else if (ridge_method == "gabor") p->AddModule(std::make_unique<RidgeGaborModule>());
    else {
        return nullptr; // 未知方法 → 明確失敗（不回「只有去背」的半成品讓 caller 猜）
    }

    return p;
}

}} // namespace tanuki::pipeline
