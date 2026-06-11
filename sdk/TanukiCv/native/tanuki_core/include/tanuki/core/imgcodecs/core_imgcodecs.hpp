
#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace tanuki { namespace core {

    // 通用影像結構（RAII 風格，vector 自動管理記憶體）。
    struct Image {
        int w = 0;
        int h = 0;
        int c = 0;
        std::vector<uint8_t> data; // 像素 buffer（vector 自動釋放）

        bool empty() const { return data.empty(); }
    };

    /**
     * @brief 讀取圖片
     * @param filepath 路徑
     * @param desired_channels
     *   0 = 自動（原圖幾通道就讀幾通道）
     *   1 = 強制灰階（最快、最省記憶體）
     *   3 = 強制 RGB
     */
    Image imread(const std::string& filepath, int desired_channels = 0);

    /**
     * @brief 儲存圖片（BMP）
     * @param filepath 路徑
     * @param img 影像物件
     * @return true 成功
     */
    bool imwrite(const std::string& filepath, const Image& img);

    // 相容用多載：直接給 raw buffer 存檔（Pinned Memory 存檔需求）。
    bool imwrite(const std::string& filepath, int w, int h, int c, const void* data);

    // CPU 縮圖讀取。
    int load_thumbnail_cpu(const char* filepath, int target_width, uint8_t* out_buffer, int* out_real_w, int* out_real_h);
}}  // namespace core, tanuki
