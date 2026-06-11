#pragma once
// 系統/硬體資訊查詢（benchmark 報告標機器用）—— tanuki_utils 共用唯一來源。
// GPU：cudaGetDeviceProperties + cudaMemGetInfo；CPU：__cpuid brand；RAM：GlobalMemoryStatusEx。
#include <cuda_runtime.h>
#include <string>
#include <cstdio>
#include <cstring>
#include <array>
#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#endif

namespace tanuki { namespace utils {

struct SysInfo {
    std::string gpu_name;
    int   sm_major = 0, sm_minor = 0;
    double gpu_total_mb = 0, gpu_free_mb = 0;
    std::string cpu_brand;
    double ram_total_mb = 0;
};

inline std::string cpu_brand_string() {
#ifdef _WIN32
    std::array<int, 4> r;
    char b[0x40] = {0};
    __cpuid(r.data(), 0x80000000);
    if ((unsigned)r[0] < 0x80000004) return "unknown-cpu";
    for (unsigned i = 0x80000002, off = 0; i <= 0x80000004; ++i, off += 16) {
        __cpuid(r.data(), i);
        for (int k = 0; k < 4; ++k) std::memcpy(b + off + k * 4, &r[k], 4);
    }
    std::string s(b);
    size_t p = s.find_first_not_of(' '); // trim 前導空白
    return p == std::string::npos ? s : s.substr(p);
#else
    return "unknown-cpu";
#endif
}

inline SysInfo query_sys_info(int device = 0) {
    SysInfo s;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, device) == cudaSuccess) {
        s.gpu_name = p.name; s.sm_major = p.major; s.sm_minor = p.minor;
    }
    size_t freeB = 0, totalB = 0;
    if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) {
        s.gpu_free_mb  = freeB  / 1048576.0;
        s.gpu_total_mb = totalB / 1048576.0;
    }
    s.cpu_brand = cpu_brand_string();
#ifdef _WIN32
    MEMORYSTATUSEX m{}; m.dwLength = sizeof(m);
    if (GlobalMemoryStatusEx(&m)) s.ram_total_mb = m.ullTotalPhys / 1048576.0;
#endif
    return s;
}

/// 一行硬體標籤（印在報告/CSV 表頭）。
inline std::string sys_info_tag(const SysInfo& s) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "GPU=%s(sm_%d%d, %.0fGB) | CPU=%s | RAM=%.0fGB",
                  s.gpu_name.c_str(), s.sm_major, s.sm_minor, s.gpu_total_mb / 1024.0,
                  s.cpu_brand.c_str(), s.ram_total_mb / 1024.0);
    return std::string(buf);
}

}} // namespace tanuki::utils
