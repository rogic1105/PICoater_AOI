
#include "bench_framework/test_utils.hpp"

void RunCoreTests(const std::string& imgPath);

int main() {
    // �@��d�w�A�� "RunCoreTests" �禡�Ƕi�h
    return bench_framework::RunAOITestBootstrap("AOI Core SDK Tests", RunCoreTests);
}