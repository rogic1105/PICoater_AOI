
#pragma once
#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <utility>

// �� API�Gstart()/stop()�A��X�榡�P�A�쪩�ۦP
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

// RAII ���G�i�J�϶�Y�}�l�A���}�۰ʦL�ɶ��]�P�˥� high_resolution_clock �P�ª��P�榡�^
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

// RAII + �ۭq�u����e�P�B�v�G�Ҧp�� cudaDeviceSynchronize() �A���ɶ�
template<class Sync>
class ScopeTimerSyncHR {
public:
    ScopeTimerSyncHR(const char* name, Sync&& sync)
        : name_(name), sync_(std::forward<Sync>(sync)), start_(clock::now()) {
    }
    ~ScopeTimerSyncHR() {
        sync_(); // �����P�B�A�A�� end
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

// �����]�n�Τ@�I�^
#define AOI_CONCAT_INNER(a,b) a##b
#define AOI_CONCAT(a,b) AOI_CONCAT_INNER(a,b)

#define TIME_SCOPE_MS(label) ::ScopeTimerHR AOI_CONCAT(_scope_timer_hr_, __COUNTER__){label}
// �|�b�L�ɶ��e������ sync_expr�]�Ҧp cudaDeviceSynchronize()�^
#define TIME_SCOPE_MS_SYNC(label, sync_expr) \
    auto AOI_CONCAT(_scope_timer_sync_hr_, __COUNTER__) = ::MakeScopeTimerSyncHR(label, [&](){ sync_expr; })

#define TIME_FUNC_MS() ::ScopeTimerHR AOI_CONCAT(_func_timer_hr_, __COUNTER__){__func__}
