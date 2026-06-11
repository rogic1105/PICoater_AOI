
#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace tanuki { namespace core {

    // �w�q�@�ӳq�Ϊ��v�����c (RAII ����A�۰ʺ޲z�O����)
    struct Image {
        int w = 0;
        int h = 0;
        int c = 0;
        std::vector<uint8_t> data; // �ϥ� vector �۰ʺ޲z�O����

        bool empty() const { return data.empty(); }
    };

    /**
     * @brief Ū���Ϥ�
     * @param filepath ��|
     * @param desired_channels
     * 0 = �۰� (��ϬO�X�q�D�NŪ�X�q�D)
     * 1 = �j����Ƕ� (�̧֡A�ٰO����)
     * 3 = �j���� RGB
     */
    Image imread(const std::string& filepath, int desired_channels = 0);

    /**
     * @brief �x�s�Ϥ� (BMP)
     * @param filepath ��|
     * @param img �v������
     * @return true ���\
     */
    bool imwrite(const std::string& filepath, const Image& img);

    // �쥻���g�k (���F�ۮe�A�� Pinned Memory �s�ɻݨD)
    bool imwrite(const std::string& filepath, int w, int h, int c, const void* data);

    // CPU �Y��Ū�� (�A�쥻�g�n��)
    int load_thumbnail_cpu(const char* filepath, int target_width, uint8_t* out_buffer, int* out_real_w, int* out_real_h);
}}  // namespace core, tanuki