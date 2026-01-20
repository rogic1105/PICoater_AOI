// PICoater_AOI\tests\cpp_test\picoater_tests\src\main.cpp

#include "framework/test_utils.hpp"
#include <iostream>

// 宣告外部測試函式
void PICoaterModuleTestsFast(const std::string& imgPath);  // 新方法 (使用 Fast IO)
void PICoaterModuleTestsMultiThread(const std::string& imgPath);

int main() {
    // 定義一個 Wrapper 函式，依序呼叫兩個測試
    auto run_all_tests = [](const std::string& imgPath) {

        std::cout << "\n\n";
        std::cout << "\n==================================================";
        std::cout << "\n[Test 2] Fast IO Method (Raw Read/Write / Optimized)";
        std::cout << "\n==================================================\n";
        PICoaterModuleTestsFast(imgPath);

        std::cout << "\n\n";
        std::cout << "\n==================================================";
        std::cout << "\n[Test 3] Multi-Thread Simulation (7 Cameras)";
        std::cout << "\n==================================================\n";
        PICoaterModuleTestsMultiThread(imgPath);
        };

    // 將 Wrapper 傳給 Bootstrap
    return framework::RunAOITestBootstrap("PICoater Full Comparison Tests", run_all_tests);
}