
// [����ץ�] ������b��� #include ���e
#define _CRT_SECURE_NO_WARNINGS 

#include "core_cv/imgcodecs/core_imgcodecs_fast.hpp"
#include <cstdio>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath> // std::abs

namespace tanuki { namespace core {

#pragma pack(push, 1)
    struct BmpHeader {
        uint16_t signature; // 'BM'
        uint32_t fileSize;
        uint16_t reserved1;
        uint16_t reserved2;
        uint32_t dataOffset;
    };

    struct BmpInfoHeader {
        uint32_t headerSize; // 40
        int32_t width;
        int32_t height;
        uint16_t planes;     // 1
        uint16_t bpp;        // 8
        uint32_t compression;// 0
        uint32_t imageSize;  // 0 or w*h
        int32_t xRes;
        int32_t yRes;
        uint32_t colorsUsed; // 256
        uint32_t colorsImportant;
    };
#pragma pack(pop)

    bool fast_write_bmp_8bit(const std::string& filepath, int w, int h, const uint8_t* data) {
        FILE* f = fopen(filepath.c_str(), "wb");
        if (!f) return false;

        // 1. �ǳ� Header
        const int paletteSize = 1024;
        const int headerSize = sizeof(BmpHeader) + sizeof(BmpInfoHeader);
        const int offset = headerSize + paletteSize;

        // Padding �p��
        int stride = (w + 3) & (~3);
        int imageSize = stride * h;
        int fileSize = offset + imageSize;

        BmpHeader fileHeader = { 0x4D42, (uint32_t)fileSize, 0, 0, (uint32_t)offset };
        // �`�N height �s���t�� (-h) �N�� Top-Down�A��K�˵�
        BmpInfoHeader infoHeader = { 40, w, -h, 1, 8, 0, (uint32_t)imageSize, 2835, 2835, 256, 0 };

        // 2. �g�J Header
        fwrite(&fileHeader, sizeof(fileHeader), 1, f);
        fwrite(&infoHeader, sizeof(infoHeader), 1, f);

        // 3. �g�J�զ�L
        uint8_t palette[1024];
        for (int i = 0; i < 256; ++i) {
            palette[i * 4 + 0] = i; // B
            palette[i * 4 + 1] = i; // G
            palette[i * 4 + 2] = i; // R
            palette[i * 4 + 3] = 0; // A
        }
        fwrite(palette, 1, 1024, f);

        // 4. �g�J����
        if (stride == w) {
            fwrite(data, 1, (size_t)w * h, f);
        }
        else {
            std::vector<uint8_t> padding(stride - w, 0);
            for (int y = 0; y < h; ++y) {
                fwrite(data + y * w, 1, w, f);
                fwrite(padding.data(), 1, padding.size(), f);
            }
        }

        fclose(f);
        return true;
    }

    bool fast_read_bmp_8bit(const std::string& filepath, int& w, int& h, uint8_t* out_buffer, int buffer_size) {
        FILE* f = fopen(filepath.c_str(), "rb");
        if (!f) return false;

        BmpHeader fileHeader;
        BmpInfoHeader infoHeader;

        if (fread(&fileHeader, sizeof(fileHeader), 1, f) != 1) { fclose(f); return false; }
        if (fileHeader.signature != 0x4D42) { fclose(f); return false; }

        if (fread(&infoHeader, sizeof(infoHeader), 1, f) != 1) { fclose(f); return false; }

        w = infoHeader.width;
        h = std::abs(infoHeader.height);

        if (infoHeader.bpp != 8) {
            fclose(f);
            std::cerr << "[FastRead] Error: Not an 8-bit BMP\n";
            return false;
        }

        int stride = (w + 3) & (~3);
        size_t dataSize = (size_t)stride * h;

        if (buffer_size < dataSize) {
            fclose(f);
            std::cerr << "[FastRead] Error: Buffer too small\n";
            return false;
        }

        fseek(f, fileHeader.dataOffset, SEEK_SET);
        size_t readCount = fread(out_buffer, 1, dataSize, f);
        fclose(f);

        return readCount == dataSize;
    }

#pragma pack(push, 1)
    struct BMPHeader {
        uint16_t bfType{ 0x4D42 };       // 'BM'
        uint32_t bfSize{ 0 };            // File size
        uint16_t bfReserved1{ 0 };
        uint16_t bfReserved2{ 0 };
        uint32_t bfOffBits{ 54 };        // Offset to image data
        uint32_t biSize{ 40 };           // Info header size
        int32_t  biWidth{ 0 };
        int32_t  biHeight{ 0 };
        uint16_t biPlanes{ 1 };
        uint16_t biBitCount{ 24 };       // 24-bit
        uint32_t biCompression{ 0 };
        uint32_t biSizeImage{ 0 };
        int32_t  biXPelsPerMeter{ 0 };
        int32_t  biYPelsPerMeter{ 0 };
        uint32_t biClrUsed{ 0 };
        uint32_t biClrImportant{ 0 };
    };
#pragma pack(pop)

    void fast_write_bmp_24bit(const std::string& filepath, int width, int height, const uint8_t* pData) {
        if (!pData) return;

        // �p�� Row Padding (BMP �C�@�楲���O 4 bytes ������)
        int row_stride = width * 3;
        int padding_size = (4 - (row_stride % 4)) % 4;
        int padded_stride = row_stride + padding_size;

        size_t data_size = (size_t)padded_stride * height;
        size_t file_size = sizeof(BMPHeader) + data_size;

        BMPHeader header;
        header.bfSize = (uint32_t)file_size;
        header.biWidth = width;
        header.biHeight = -height; // �t�ȥN�� Top-down (�ѤW���U)�A���Ȭ� Bottom-up
        // �`�N�G�Y��ϬO���V�A�o��] -height �i�����g�J�F�Y�����˵������䴩�A�i�]���Ȧ��n�˵ۼg
        // �j�h�Ʋ{�N�˵������䴩 Top-down BMP�C
        header.biSizeImage = (uint32_t)data_size;

        FILE* fp = nullptr;
#ifdef _WIN32
        fopen_s(&fp, filepath.c_str(), "wb");
#else
        fp = fopen(filepath.c_str(), "wb");
#endif

        if (!fp) {
            std::cerr << "Failed to open file for writing: " << filepath << "\n";
            return;
        }

        // 1. �g�J Header
        fwrite(&header, sizeof(BMPHeader), 1, fp);

        // 2. �g�J Data (�]�t Padding)
        if (padding_size == 0) {
            // �p�G��n���A�������g�J (�̧�)
            fwrite(pData, 1, (size_t)width * height * 3, fp);
        }
        else {
            // �ݭn Padding�A�v��g�J
            std::vector<uint8_t> padBytes(padding_size, 0);
            for (int y = 0; y < height; ++y) {
                const uint8_t* row_ptr = pData + (size_t)y * row_stride;
                fwrite(row_ptr, 1, row_stride, fp);
                fwrite(padBytes.data(), 1, padding_size, fp);
            }
        }

        fclose(fp);
    }

}}  // namespace core, tanuki