#pragma once
// 模組註冊表（name → factory）：讓 pipeline 食譜 / API 依「名字」runtime 建 module，
// 達成「換方法」（ridge_hessian ↔ ridge_gabor）不用重編、不破 ABI。
// module 用 TANUKI_REGISTER_MODULE 自我註冊。
#include <functional>
#include <map>
#include <memory>
#include <string>
#include "tanuki/pipeline/i_module.hpp"

namespace tanuki { namespace pipeline {

class ModuleRegistry {
public:
    using Factory = std::function<std::unique_ptr<IModule>()>;

    static ModuleRegistry& Instance();

    void Register(const std::string& name, Factory factory);
    std::unique_ptr<IModule> Create(const std::string& name) const; // 查無回 nullptr
    bool Has(const std::string& name) const;

private:
    ModuleRegistry() = default;
    std::map<std::string, Factory> factories_;
};

// 自我註冊輔助：在 module .cu/.cpp 檔尾放
//   TANUKI_REGISTER_MODULE("ridge_hessian", RidgeHessianModule);
struct ModuleRegistrar {
    ModuleRegistrar(const std::string& name, ModuleRegistry::Factory f) {
        ModuleRegistry::Instance().Register(name, std::move(f));
    }
};

#define TANUKI_REGISTER_MODULE(NAME, TYPE)                                   \
    static ::tanuki::pipeline::ModuleRegistrar _tanuki_reg_##TYPE(           \
        NAME, [] { return std::unique_ptr<::tanuki::pipeline::IModule>(new TYPE()); })

}} // namespace tanuki::pipeline
