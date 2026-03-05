// AOI_SDK\core_cv_tests\src\main.cpp

#include "framework/test_utils.hpp" 

void RunCoreTests(const std::string& imgPath);

int main() {
    // 一行搞定，把 "RunCoreTests" 函式傳進去
    return framework::RunAOITestBootstrap("AOI Core SDK Tests", RunCoreTests);
}