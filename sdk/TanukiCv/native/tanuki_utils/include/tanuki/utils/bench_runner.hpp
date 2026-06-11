#pragma once
// 微基準跑測器（純統計，0 CUDA 依賴）—— tanuki_utils 共用唯一來源。
// 計時器不可知：呼叫端給「跑一次 → 回傳該次毫秒」的 lambda（GPU 用 GpuTimer、CPU 用 chrono），
// 本檔只負責 warmup + 跑 N 次 + 算 mean/median/min/max/stddev + 吞吐 + 印表格/CSV。
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <numeric>

namespace tanuki { namespace utils {

struct BenchStats {
    std::string name;
    double mean_ms = 0, median_ms = 0, min_ms = 0, max_ms = 0, stddev_ms = 0;
    int    iters = 0;
    double megapixels = 0;       // 該案資料量（W*H/1e6），算吞吐用
    double mpix_per_s = 0;       // = megapixels / mean_ms * 1000
};

/// 跑測：warmup 丟掉（含暖機）→ 跑 iters 次（timed_once 每次回傳該次毫秒）→ 統計。
template <class TimedOnce>
inline BenchStats bench_run(const std::string& name, int warmup, int iters,
                            double megapixels, TimedOnce&& timed_once) {
    for (int i = 0; i < warmup; ++i) (void)timed_once();   // 暖機（JIT/cache）丟棄
    std::vector<double> t;
    t.reserve(iters);
    for (int i = 0; i < iters; ++i) t.push_back((double)timed_once());

    BenchStats s;
    s.name = name; s.iters = iters; s.megapixels = megapixels;
    if (t.empty()) return s;
    std::sort(t.begin(), t.end());
    s.min_ms = t.front();
    s.max_ms = t.back();
    s.median_ms = t[t.size() / 2];
    s.mean_ms = std::accumulate(t.begin(), t.end(), 0.0) / t.size();
    double var = 0;
    for (double x : t) var += (x - s.mean_ms) * (x - s.mean_ms);
    s.stddev_ms = std::sqrt(var / t.size());
    s.mpix_per_s = s.mean_ms > 0 ? megapixels / s.mean_ms * 1000.0 : 0;
    return s;
}

// ── 輸出 ────────────────────────────────────────────────────────────────
inline void bench_print_header() {
    std::printf("%-38s %9s %9s %9s %8s %10s\n",
                "benchmark", "mean(ms)", "median", "min", "stddev", "MPix/s");
    std::printf("%s\n", std::string(86, '-').c_str());
}
inline void bench_print_row(const BenchStats& s) {
    std::printf("%-38s %9.3f %9.3f %9.3f %8.3f %10.1f\n",
                s.name.c_str(), s.mean_ms, s.median_ms, s.min_ms, s.stddev_ms, s.mpix_per_s);
}
/// CSV 一列（含硬體 tag 由呼叫端在表頭/欄位補；此處輸出純數據列）。
inline void bench_csv_header(std::FILE* f) {
    std::fprintf(f, "name,mean_ms,median_ms,min_ms,max_ms,stddev_ms,iters,megapixels,mpix_per_s\n");
}
inline void bench_csv_row(std::FILE* f, const BenchStats& s) {
    std::fprintf(f, "%s,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%.3f,%.1f\n",
                 s.name.c_str(), s.mean_ms, s.median_ms, s.min_ms, s.max_ms,
                 s.stddev_ms, s.iters, s.megapixels, s.mpix_per_s);
}

}} // namespace tanuki::utils
