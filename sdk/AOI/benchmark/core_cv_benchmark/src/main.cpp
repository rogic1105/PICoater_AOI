// AOI_SDK\core_cv_benchmark\src\main.cpp

#include "framework/test_utils.hpp" 

void RunCoreTests(const std::string& imgPath);

int main() {
    // �@��d�w�A�� "RunCoreTests" �禡�Ƕi�h
    return framework::RunAOITestBootstrap("AOI Core SDK Tests", RunCoreTests);
}