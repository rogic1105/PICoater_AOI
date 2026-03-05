// AOI_SDK\cpp_utils\include\cpp_utils\timer_utils.hpp

#pragma once
#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <utility>

// 舊 API：start()/stop()，輸出格式與你原版相同
class Timer {
public:
    explicit Timer(const std::string& name) : name_(name) {}
    void start() { start_ = clock::now(); }
    void stop() {
        auto end = clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() / 1000.0;
        std::cout << std::fixed << std::setprecision(3)
            << "\nElapsed time = " << ms << " ms on " << name_ << "\n";
    }
private:
    using clock = std::chrono::high_resolution_clock;
    std::string name_;
    std::chrono::time_point<clock> start_;
};

// RAII 版：進入區塊即開始，離開自動印時間（同樣用 high_resolution_clock 與舊版同格式）
class ScopeTimerHR {
public:
    explicit ScopeTimerHR(const char* name) : name_(name), start_(clock::now()) {}
    explicit ScopeTimerHR(const std::string& name) : ScopeTimerHR(name.c_str()) {}
    ~ScopeTimerHR() {
        auto end = clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() / 1000.0;
        std::cout << std::fixed << std::setprecision(3)
            << "\nElapsed time = " << ms << " ms on " << name_ << "\n";
    }
private:
    using clock = std::chrono::high_resolution_clock;
    const char* name_;
    std::chrono::time_point<clock> start_;
};

// RAII + 自訂「結束前同步」：例如先 cudaDeviceSynchronize() 再取時間
template<class Sync>
class ScopeTimerSyncHR {
public:
    ScopeTimerSyncHR(const char* name, Sync&& sync)
        : name_(name), sync_(std::forward<Sync>(sync)), start_(clock::now()) {
    }
    ~ScopeTimerSyncHR() {
        sync_(); // 先做同步，再取 end
        auto end = clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() / 1000.0;
        std::cout << std::fixed << std::setprecision(3)
            << "\nElapsed time = " << ms << " ms on " << name_ << "\n";
    }
private:
    using clock = std::chrono::high_resolution_clock;
    const char* name_;
    Sync sync_;
    std::chrono::time_point<clock> start_;
};

template<class Sync>
inline ScopeTimerSyncHR<Sync> MakeScopeTimerSyncHR(const char* name, Sync&& sync) {
    return ScopeTimerSyncHR<Sync>(name, std::forward<Sync>(sync));
}

// 巨集（好用一點）
#define AOI_CONCAT_INNER(a,b) a##b
#define AOI_CONCAT(a,b) AOI_CONCAT_INNER(a,b)

#define TIME_SCOPE_MS(label) ::ScopeTimerHR AOI_CONCAT(_scope_timer_hr_, __COUNTER__){label}
// 會在印時間前先執行 sync_expr（例如 cudaDeviceSynchronize()）
#define TIME_SCOPE_MS_SYNC(label, sync_expr) \
    auto AOI_CONCAT(_scope_timer_sync_hr_, __COUNTER__) = ::MakeScopeTimerSyncHR(label, [&](){ sync_expr; })

#define TIME_FUNC_MS() ::ScopeTimerHR AOI_CONCAT(_func_timer_hr_, __COUNTER__){__func__}
