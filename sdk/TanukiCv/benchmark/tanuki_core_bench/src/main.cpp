#define _CRT_SECURE_NO_WARNINGS   // 允許 std::fopen（benchmark 寫 CSV）
// tanuki_core_bench —— tanuki::core 速度/資源 micro-benchmark（自寫，0 外部依賴）。
// 計時/統計/系統資訊全用 tanuki_utils 共用工具。示範對象：threshold（二值化）。
//
// 用法：tanuki_core_bench.exe [--csv out.csv]
//   印表格到 stdout；給 --csv 則另寫 CSV（含硬體 tag）供 Python 報告管線後製。
//
// 新增 op / 變體：照 bench_op_threshold 樣板加一個函式，在 main 裡呼叫即可。
#include <tanuki/utils/gpu_timer.hpp>
#include <tanuki/utils/bench_runner.hpp>
#include <tanuki/utils/sys_info.hpp>
#include <tanuki/core/imgproc/core_enhance.hpp>   // threshold_u8_gpu

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <chrono>

using namespace tanuki::utils;

// CPU 基準（單執行緒，沒優化）——當對照組，凸顯 GPU/未來 SIMD 的差距。
static void cpu_threshold(const uint8_t* in, uint8_t* out, size_t n, uint8_t thr) {
    for (size_t i = 0; i < n; ++i) out[i] = in[i] >= thr ? 255 : 0;
}

// threshold 一個尺寸：GPU(resident) / GPU(含 H2D+D2H) / CPU naive 三個變體。
static void bench_op_threshold(int W, int H, uint8_t THR, int warm, int iters,
                               std::FILE* csv) {
    size_t N = (size_t)W * H;
    double mp = N / 1e6;
    char tag[64];

    std::vector<uint8_t> hin(N), hout(N);
    for (size_t i = 0; i < N; ++i) hin[i] = (uint8_t)((i * 131) % 256);

    uint8_t *din = nullptr, *dout = nullptr;
    cudaMalloc(&din, N); cudaMalloc(&dout, N);
    cudaMemcpy(din, hin.data(), N, cudaMemcpyHostToDevice);

    auto emit = [&](const BenchStats& s) { bench_print_row(s); if (csv) bench_csv_row(csv, s); };

    // 1) GPU kernel only（資料已在 GPU）——pipeline 中段的真實成本
    std::snprintf(tag, sizeof(tag), "GPU_kernel/%dx%d", W, H);
    emit(bench_run(tag, warm, iters, mp, [&] {
        GpuTimer t; t.start();
        tanuki::core::threshold_u8_gpu(din, dout, W, H, THR);
        return t.stop_ms();
    }));

    // 2) GPU + H2D/D2H（含 PCIe 來回）——單獨呼叫的端到端成本
    std::snprintf(tag, sizeof(tag), "GPU_h2d2h/%dx%d", W, H);
    emit(bench_run(tag, warm, iters, mp, [&] {
        GpuTimer t; t.start();
        cudaMemcpy(din, hin.data(), N, cudaMemcpyHostToDevice);
        tanuki::core::threshold_u8_gpu(din, dout, W, H, THR);
        cudaMemcpy(hout.data(), dout, N, cudaMemcpyDeviceToHost);
        return t.stop_ms();
    }));

    // 3) CPU naive（host chrono）
    std::snprintf(tag, sizeof(tag), "CPU_naive/%dx%d", W, H);
    emit(bench_run(tag, warm, iters, mp, [&] {
        auto a = std::chrono::high_resolution_clock::now();
        cpu_threshold(hin.data(), hout.data(), N, THR);
        auto b = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(b - a).count();
    }));

    std::printf("\n");
    cudaFree(din); cudaFree(dout);
}

int main(int argc, char** argv) {
    const char* csvPath = nullptr;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) csvPath = argv[++i];

    SysInfo si = query_sys_info();
    std::printf("Hardware: %s\n", sys_info_tag(si).c_str());
    std::printf("Op: threshold (binarization), 8-bit gray, thr=128\n\n");
    bench_print_header();

    std::FILE* csv = csvPath ? std::fopen(csvPath, "w") : nullptr;
    if (csv) { std::fprintf(csv, "# %s\n", sys_info_tag(si).c_str()); bench_csv_header(csv); }

    const uint8_t THR = 128;
    const int WARM = 5, ITERS = 50;
    const int sizes[][2] = { {1024, 768}, {2048, 1536}, {4096, 3000} };
    for (auto& s : sizes) bench_op_threshold(s[0], s[1], THR, WARM, ITERS, csv);

    if (csv) std::fclose(csv);
    std::printf("decision hint: threshold 是 memory-bound；GPU_kernel 通常遠快，但 GPU_h2d2h 含傳輸\n"
                "若 > CPU 表示『單獨做不值得上 GPU、只在資料已駐留 GPU(pipeline) 才划算』。\n");
    return 0;
}
