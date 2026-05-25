// AOI_SDK\cpp_utils\include\cpp_utils\terminal_colors.hpp
#pragma once

namespace Color {
    // === 控制碼 ===
    constexpr auto RESET = "\033[0m";

    // === 標準色 (較暗) ===
    constexpr auto RED = "\033[31m";
    constexpr auto GREEN = "\033[32m";
    constexpr auto YELLOW = "\033[33m";
    constexpr auto BLUE = "\033[34m";
    constexpr auto MAGENTA = "\033[35m";
    constexpr auto CYAN = "\033[36m";
    constexpr auto WHITE = "\033[37m";

    // === 豔色/亮色 (較亮，推薦用這些) ===
    constexpr auto RED_B = "\033[91m";
    constexpr auto GREEN_B = "\033[92m";
    constexpr auto YELLOW_B = "\033[93m";
    constexpr auto BLUE_B = "\033[94m";
    constexpr auto MAGENTA_B = "\033[95m";
    constexpr auto CYAN_B = "\033[96m";
    constexpr auto WHITE_B = "\033[97m";
}