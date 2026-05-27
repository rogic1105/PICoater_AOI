
#include "bench_framework/test_utils.hpp"

#include <filesystem>
#include <iostream>
#include <string>

#include "cpp_utils/terminal_colors.hpp"

#ifndef REPO_ROOT
#error "REPO_ROOT is not defined! Please check your Directory.Build.props."
#endif

namespace fs = std::filesystem;

namespace bench_framework {

    // ��@�G���o��X��|�æ۰ʫإ߸�Ƨ�
    std::string GetOutputPath(const std::string& suiteName, const std::string& fileName) {
        fs::path root = REPO_ROOT;

        // �զX��|: <Root>/artifacts/<REPO_ROOT>/<FileName>
        fs::path outputDir = root / "artifacts" / suiteName;
        fs::path fullPath = outputDir / fileName;

        // �۰ʫإߥؿ� (�p�G���s�b)
        if (!fs::exists(outputDir)) {
            try {
                fs::create_directories(outputDir);
                std::cout << Color::CYAN << "[Info] Created output directory: "
                    << outputDir.string() << Color::RESET << "\n";
            }
            catch (const std::exception& e) {
                std::cerr << Color::RED << "[Error] Failed to create output dir: "
                    << e.what() << Color::RESET << "\n";
            }
        }

        // �^�ǼзǤƪ��r�� (�� STB Image ��)
        return fullPath.string();
    }

    int RunAOITestBootstrap(const std::string& suiteName, TestEntryFunc testFunc) {

        // ���ո�Ƹ�|���props
        fs::path projectRoot = PROJECT_ROOT;
        fs::path srcFolder = DATA_FOLDER_PATH;
        fs::path imagePath = TARGET_IMAGE_PATH;
        fs::path fullPath = projectRoot / srcFolder / imagePath;

        fullPath.make_preferred();

        // 2. �w���ˬd
        if (!fs::exists(fullPath)) {
            std::cerr << Color::RED << "[Error] Cannot find test image!" << Color::RESET << "\n";
            std::cerr << Color::RED << "Looking at: " << fullPath << Color::RESET << "\n";
            std::cerr << "Root: " << projectRoot << "\n";
            // �o�̥i�H��ܤ��Ȱ����� return�A��K�۰ʤƴ���
            std::cout << "Press Enter to exit.\n";
            std::cin.get();
            return -1;
        }

        std::string testFullPath = fullPath.string();

        // 3. ����w��T��
        std::cout << Color::YELLOW << "Starting " << suiteName << "..." << Color::RESET << "\n";
        std::cout << Color::YELLOW << "Target Image: " << imagePath.filename().string() << Color::RESET << "\n";
        std::cout << "--------------------------------------------------\n";

        // 4. ����ǤJ�����ը禡
        try {
            testFunc(testFullPath);
        }
        catch (const std::exception& e) {
            std::cerr << Color::RED << "\n[Fatal Error] Test crashed: " << e.what() << Color::RESET << "\n";
            return -1;
        }

        // 5. ���� (�b CI ��ҩΧ妸����ɡA�o�̥i�H�Ҽ{�����Ȱ�)
        std::cout << "\n" << Color::GREEN << "Test Finished. Press Enter to exit." << Color::RESET << "\n";
        std::cin.get();
        return 0;
    }

}  // namespace bench_framework