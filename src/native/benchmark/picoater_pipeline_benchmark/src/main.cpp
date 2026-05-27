// PICoater_AOI\tests\cpp_test\picoater_pipeline_benchmark\src\main.cpp

#include "bench_framework/test_utils.hpp"
#include <iostream>

// �ŧi�~�����ը禡
void PICoaterModuleTestsFast(const std::string& imgPath);  // �s��k (�ϥ� Fast IO)
void PICoaterModuleTestsMultiThread(const std::string& imgPath, const int NUM_CAMS);

int main() {
    // �w�q�@�� Wrapper �禡�A�̧ǩI�s��Ӵ���
    auto run_all_tests = [](const std::string& imgPath) {
        std::cout << "imgPath: " << imgPath << "\n";
        std::cout << "\n\n";
        std::cout << "\n==================================================";
        std::cout << "\n[Test 1] Fast IO Method (Raw Read/Write / Optimized)";
        std::cout << "\n==================================================\n";
        PICoaterModuleTestsFast(imgPath);

        std::cout << "\n\n";
        std::cout << "\n==================================================";
        std::cout << "\n[Test 2] Multi-Thread Simulation (1 Cameras)";
        std::cout << "\n==================================================\n";
        PICoaterModuleTestsMultiThread(imgPath, 1);

        std::cout << "\n\n";
        std::cout << "\n==================================================";
        std::cout << "\n[Test 3] Multi-Thread Simulation (7 Cameras)";
        std::cout << "\n==================================================\n";
        PICoaterModuleTestsMultiThread(imgPath, 10);
        };

    // �N Wrapper �ǵ� Bootstrap
    return bench_framework::RunAOITestBootstrap("PICoater Full Comparison Tests", run_all_tests);
}