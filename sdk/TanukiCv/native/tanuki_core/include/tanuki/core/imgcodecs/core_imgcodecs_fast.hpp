
#pragma once
#include <string>
#include <cstdint>

namespace tanuki { namespace core {

    // 快速讀取 BMP（僅支援 8-bit 灰階）：直接讀入指定 buffer（建議傳 Pinned Memory）。成功回 true。
    bool fast_read_bmp_8bit(const std::string& filepath, int& w, int& h, uint8_t* out_buffer, int buffer_size);

    // 快速寫入 BMP（僅支援 8-bit 灰階）：直接把 buffer 寫進檔案。
    bool fast_write_bmp_8bit(const std::string& filepath, int w, int h, const uint8_t* data);

    void fast_write_bmp_24bit(const std::string& filepath, int width, int height, const uint8_t* pData);

}}  // namespace core, tanuki
