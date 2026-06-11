#define _CRT_SECURE_NO_WARNINGS
// find_stream_ridgeline pipeline benchmark —— 端到端 GPU 計時（去背→脊線整條），用 tanuki_utils harness。
// 用法：find_stream_ridgeline_bench.exe [--csv out.csv]
#include <tanuki/utils/gpu_timer.hpp>
#include <tanuki/utils/bench_runner.hpp>
#include <tanuki/utils/sys_info.hpp>
#include "find_stream_ridgeline.hpp"

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdio>

using namespace tanuki::utils;
using namespace tanuki::pipeline;

static void bench_size(Pipeline* pipe, int W, int H, int warm, int iters, std::FILE* csv) {
    size_t N = (size_t)W * H;
    double mp = N / 1e6;
    char tag[64];

    std::vector<uint8_t> hin(N);
    for (size_t i = 0; i < N; ++i) hin[i] = (uint8_t)((i * 131) % 256);

    uint8_t *d_in = nullptr, *d_bg = nullptr, *d_mura = nullptr, *d_ridge = nullptr;
    float *d_cmean = nullptr, *d_cmax = nullptr, *d_rmean = nullptr, *d_rmax = nullptr;
    cudaMalloc(&d_in, N); cudaMalloc(&d_bg, N); cudaMalloc(&d_mura, N); cudaMalloc(&d_ridge, N);
    cudaMalloc(&d_cmean, W * sizeof(float)); cudaMalloc(&d_cmax, W * sizeof(float));
    cudaMalloc(&d_rmean, H * sizeof(float)); cudaMalloc(&d_rmax, H * sizeof(float));
    cudaMemcpy(d_in, hin.data(), N, cudaMemcpyHostToDevice);

    InputImage in; in.width = W; in.height = H; in.data = d_in; in.stream = nullptr;
    Params p; p.bg_sigma_factor = 1.0f; p.ridge_sigma = 1.0f; p.hessian_max_factor = 1.0f;
    p.ridge_mode = "vertical+horizontal"; p.precomputed_col_mean = nullptr;
    OutputBuffers out; out.width = W; out.height = H;
    out.background_data = d_bg; out.mura_data = d_mura; out.ridge_data = d_ridge;
    out.mura_curve_mean = d_cmean; out.mura_curve_max = d_cmax;
    out.mura_row_curve_mean = d_rmean; out.mura_row_curve_max = d_rmax; out.stream = nullptr;

    std::snprintf(tag, sizeof(tag), "find_stream_ridgeline/%dx%d", W, H);
    auto s = bench_run(tag, warm, iters, mp, [&] {
        GpuTimer t; t.start();
        pipe->Process(in, p, &out);
        return t.stop_ms();
    });
    bench_print_row(s);
    if (csv) bench_csv_row(csv, s);

    cudaFree(d_in); cudaFree(d_bg); cudaFree(d_mura); cudaFree(d_ridge);
    cudaFree(d_cmean); cudaFree(d_cmax); cudaFree(d_rmean); cudaFree(d_rmax);
}

int main(int argc, char** argv) {
    const char* csvPath = nullptr;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) csvPath = argv[++i];

    SysInfo si = query_sys_info();
    std::printf("Hardware: %s\n", sys_info_tag(si).c_str());
    std::printf("Pipeline: find_stream_ridgeline (background_sub -> ridge_hessian), ridge_mode=vertical+horizontal\n\n");
    bench_print_header();

    std::FILE* csv = csvPath ? std::fopen(csvPath, "w") : nullptr;
    if (csv) { std::fprintf(csv, "# %s\n", sys_info_tag(si).c_str()); bench_csv_header(csv); }

    auto pipe = CreateFindStreamRidgeline("hessian");
    const int WARM = 5, ITERS = 30;
    const int sizes[][2] = { {1024, 768}, {2048, 1536}, {4096, 3000} };
    for (auto& sz : sizes) bench_size(pipe.get(), sz[0], sz[1], WARM, ITERS, csv);

    if (csv) std::fclose(csv);
    return 0;
}
