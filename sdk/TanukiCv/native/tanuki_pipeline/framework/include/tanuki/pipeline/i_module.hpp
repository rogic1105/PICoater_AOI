#pragma once
// 模組介面（一個可換的「步驟/方法」）。組合 tanuki_core primitive 完成一件事，可被 pipeline 串接。
// 範例：background_sub（去背）、ridge_hessian（脊線，hessian 法）。換方法 = 換實作此介面的 module。
#include <string>
#include "tanuki/pipeline/types.hpp"

namespace tanuki { namespace pipeline {

class IModule {
public:
    virtual ~IModule() = default;

    // 一次性初始化（配置常駐資源等）。
    virtual bool Initialize() = 0;

    // 處理一幀：吃 input + params，填 output buffer。
    virtual bool Process(const InputImage& input, const Params& params, OutputBuffers* output) = 0;

    // 最近一次錯誤訊息（Process 回 false 時查）。
    virtual std::string GetLastError() const = 0;

    // 模組名（registry 註冊 / log 用），如 "background_sub" / "ridge_hessian"。
    virtual const char* Name() const = 0;
};

}} // namespace tanuki::pipeline
