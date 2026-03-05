// AOI_SDK\core_cv\include\core_cv\imgcodecs\core_imgcodecs.hpp

#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace core {

    // 定義一個通用的影像結構 (RAII 風格，自動管理記憶體)
    struct Image {
        int w = 0;
        int h = 0;
        int c = 0;
        std::vector<uint8_t> data; // 使用 vector 自動管理記憶體

        bool empty() const { return data.empty(); }
    };

    /**
     * @brief 讀取圖片
     * @param filepath 路徑
     * @param desired_channels
     * 0 = 自動 (原圖是幾通道就讀幾通道)
     * 1 = 強制轉灰階 (最快，省記憶體)
     * 3 = 強制轉 RGB
     */
    Image imread(const std::string& filepath, int desired_channels = 0);

    /**
     * @brief 儲存圖片 (BMP)
     * @param filepath 路徑
     * @param img 影像物件
     * @return true 成功
     */
    bool imwrite(const std::string& filepath, const Image& img);

    // 原本的寫法 (為了相容你的 Pinned Memory 存檔需求)
    bool imwrite(const std::string& filepath, int w, int h, int c, const void* data);

    // CPU 縮圖讀取 (你原本寫好的)
    int load_thumbnail_cpu(const char* filepath, int target_width, uint8_t* out_buffer, int* out_real_w, int* out_real_h);
}