#pragma once
// AOI 影像 pipeline 的 I/O 契約（GPU buffer + 參數）。從 src/native picoater::aoi 帶過來、改 namespace。
#include <cstdint>

namespace tanuki { namespace pipeline {

// 輸入影像（GPU 上的灰階 buffer）。
struct InputImage {
    int width = 0;
    int height = 0;
    uint8_t* data = nullptr;   // device 指標（GPU 灰階）
    void* stream = nullptr;    // cudaStream_t（void* 避免 framework 拉 CUDA header）
};

// 輸出 buffer 組（GPU；各步驟視需要填）。
struct OutputBuffers {
    int width = 0;
    int height = 0;
    uint8_t* background_data = nullptr;   // 去背結果
    uint8_t* mura_data = nullptr;         // mura 圖
    uint8_t* ridge_data = nullptr;        // 脊線圖
    float* mura_curve_mean = nullptr;     // 切向曲線 mean/max
    float* mura_curve_max = nullptr;
    float* mura_row_curve_mean = nullptr; // 法向曲線 mean/max
    float* mura_row_curve_max = nullptr;
    int standard_width = 0;
    int standard_height = 0;
    uint16_t* hessian_column_half = nullptr;
    uint16_t* hessian_row_half = nullptr;
    void* stream = nullptr;
};

// 演算法參數（各 module 取自己需要的；換方法用 registry by name，不放這）。
struct Params {
    float bg_sigma_factor = 1.0f;
    float ridge_sigma = 1.0f;
    float hessian_max_factor = 1.0f;
    // "vertical" / "horizontal" / "vertical+horizontal"。
    // （舊 code 預設 "dark"＝不被任何路徑解析 → 默默零輸出的 footgun，已改 sane default。）
    const char* ridge_mode = "vertical+horizontal";

    // 預先算好的 column mean（GPU 指標）。非 null 時跳過每幀 calcColumnMeans，直接拿來去背。
    const float* precomputed_col_mean = nullptr;
};

}} // namespace tanuki::pipeline
