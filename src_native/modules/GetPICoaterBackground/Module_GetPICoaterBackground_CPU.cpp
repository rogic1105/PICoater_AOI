#include "Module_GetPICoaterBackground.hpp"
#include "core_cv/imgproc/core_background.hpp"
#include "core_cv/imgproc/core_utils.hpp"
#include "cpp_utils/timer_utils.hpp" // 用於計時

// 如果需要 Debug 存圖，請解開以下註解
// #define DEBUG_SAVE_IMAGE
#ifdef DEBUG_SAVE_IMAGE
#include "stb/stb_image_write.h"
#include <string>
#endif

namespace picoater {

    void PICoaterDetector::RunCPU(
        const uint8_t* h_in,
        uint8_t* h_mura_out,
        float bgSigmaFactor
    ) {
        if (m_width == 0 || m_height == 0) return;

        // 1. 確保 CPU 記憶體已分配
        // 如果 Initialize 沒被呼叫或只分配了 GPU，這裡做 lazy initialization
        if (h_col_mean.size() != m_width) {
            h_col_mean.resize(m_width);
        }

        // 使用計時器來觀察 CPU 版效能
        {
            TIME_SCOPE_MS("Total Run Time (CPU)");

            // 步驟 1: 計算列平均 (Column Means)
            {
                // TIME_SCOPE_MS("   1. Calc Column Means (CPU)");
                // 這裡 stride 傳入 m_width，假設 h_in 是緊密排列的 (無 padding)
                core::calcColumnMeans_RemoveOutliers_cpu(
                    h_in,
                    h_col_mean.data(),
                    m_width,
                    m_height,
                    m_width, // stride
                    bgSigmaFactor // sigma_threshold (雖然名字是 bgSigmaFactor，但在此上下文中常被共用)
                );
            }

            // 步驟 2: 計算背景與 Mura
            {
                // TIME_SCOPE_MS("   2. Calc Background & Mura (CPU)");
                core::calcColumnBackground_u8_cpu(
                    h_in,
                    h_col_mean.data(),
                    h_mura_out,
                    m_width,
                    m_height,
                    m_width // stride
                );
            }
        }

    }

}