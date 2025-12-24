// picoater_tests/main.cpp

#include "framework/test_utils.hpp"
// 宣告你的測試進入點
void RunModuleTests(const std::string& imgPath);

int main() {
    // 一行搞定，把 "RunModuleTests" 函式傳進去
    return framework::RunTestBootstrap("PICoater Module Tests", RunModuleTests);
}