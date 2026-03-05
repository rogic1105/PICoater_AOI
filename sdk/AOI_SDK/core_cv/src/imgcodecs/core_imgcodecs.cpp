// AOI_SDK\core_cv\src\imgcodecs\core_imgcodecs.cpp

#include "core_cv/imgcodecs/core_imgcodecs.hpp"

#include "stb/stb_image.h" 
#include "stb/stb_image_write.h"

#include <iostream>
#include <stdexcept>

namespace core {

    // 1. 讀取圖片
    Image imread(const std::string& filepath, int desired_channels) {
        Image res;

        // stbi_load 最後一個參數：
        // 0: 保持原圖通道數
        // 1: 強制轉灰階 (Gray)
        // 3: 強制轉 RGB
        unsigned char* ptr = stbi_load(filepath.c_str(), &res.w, &res.h, &res.c, desired_channels);

        if (!ptr) {
            // 讀取失敗回傳空圖，或者你可以選擇 throw exception
            std::cerr << "[Error] Failed to load image: " << filepath << "\n";
            return res;
        }

        // 如果指定了通道，需更新結構中的通道數
        if (desired_channels != 0) {
            res.c = desired_channels;
        }

        // 複製數據到 vector (這樣 stbi 的記憶體就可以釋放了，避免 memory leak)
        size_t size = res.w * res.h * res.c;
        res.data.assign(ptr, ptr + size);

        // 釋放 STB 原始記憶體
        stbi_image_free(ptr);

        return res;
    }

    // 2. 儲存圖片 (Wrapper around stbi_write_bmp)
    bool imwrite(const std::string& filepath, int w, int h, int c, const void* data) {
        // stbi_write_bmp 回傳非 0 代表成功
        return stbi_write_bmp(filepath.c_str(), w, h, c, data) != 0;
    }

    bool imwrite(const std::string& filepath, const Image& img) {
        if (img.empty()) return false;
        return imwrite(filepath, img.w, img.h, img.c, img.data.data());
    }

    // --- 以下是你原本的縮圖代碼 (保持不變) ---
    static void resize_gray_nearest(const uint8_t* src, int w, int h, uint8_t* dst, int new_w, int new_h) {
        float scale_x = (float)w / new_w;
        float scale_y = (float)h / new_h;

        for (int y = 0; y < new_h; ++y) {
            const uint8_t* src_row = src + (int)(y * scale_y) * w;
            uint8_t* dst_row = dst + y * new_w;
            for (int x = 0; x < new_w; ++x) {
                dst_row[x] = src_row[(int)(x * scale_x)];
            }
        }
    }

    int load_thumbnail_cpu(const char* filepath, int target_width, uint8_t* out_buffer, int* out_real_w, int* out_real_h) {
        if (!filepath || !out_buffer) return -1;
        int w, h, channels;
        uint8_t* img = stbi_load(filepath, &w, &h, &channels, 1); // 強制灰階
        if (!img) return -2;

        float ratio = (float)h / w;
        int new_h = (int)(target_width * ratio);

        resize_gray_nearest(img, w, h, out_buffer, target_width, new_h);

        if (out_real_w) *out_real_w = target_width;
        if (out_real_h) *out_real_h = new_h;

        stbi_image_free(img);
        return 0;
    }
}