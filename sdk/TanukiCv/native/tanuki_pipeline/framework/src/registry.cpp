#include "tanuki/pipeline/registry.hpp"

namespace tanuki { namespace pipeline {

ModuleRegistry& ModuleRegistry::Instance() {
    static ModuleRegistry inst;   // Meyers singleton（單一註冊表）
    return inst;
}

void ModuleRegistry::Register(const std::string& name, Factory factory) {
    factories_[name] = std::move(factory);
}

std::unique_ptr<IModule> ModuleRegistry::Create(const std::string& name) const {
    auto it = factories_.find(name);
    return it == factories_.end() ? nullptr : it->second();
}

bool ModuleRegistry::Has(const std::string& name) const {
    return factories_.find(name) != factories_.end();
}

}} // namespace tanuki::pipeline
