
#pragma once
#include <string>
#include <cstdint>

namespace tanuki { namespace core {

    // ���tŪ�� BMP (�Ȥ䴩 8-bit �Ƕ�)
    // ����Ū�J���w�� buffer (��ĳ�ǤJ Pinned Memory)
    // ���\�^�� true
    bool fast_read_bmp_8bit(const std::string& filepath, int& w, int& h, uint8_t* out_buffer, int buffer_size);

    // ���t�g�J BMP (�Ȥ䴩 8-bit �Ƕ�)
    // ������ buffer �g�J�w��
    bool fast_write_bmp_8bit(const std::string& filepath, int w, int h, const uint8_t* data);

    void fast_write_bmp_24bit(const std::string& filepath, int width, int height, const uint8_t* pData);

}}  // namespace core, tanuki