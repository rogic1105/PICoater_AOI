
#pragma once

#include <functional>
#include <string>

namespace bench_framework {

    // �w�q���ը禡���з�ñ�W
    using TestEntryFunc = std::function<void(const std::string&)>;

    // ���o�Τ@�����տ�X��|
    // �|�۰ʫإ߸�Ƨ�: artifacts/<suiteName>/
    // �^��: ���㪺�ɮ׵����|
    std::string GetOutputPath(const std::string& suiteName, const std::string& fileName);

    // �q�Ϊ����ձҰʾ�
    int RunAOITestBootstrap(const std::string& suiteName, TestEntryFunc testFunc);

}  // namespace bench_framework