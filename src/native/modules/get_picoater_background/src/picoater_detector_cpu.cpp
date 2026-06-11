#include "picoater_detector.hpp"
#include "core_cv/imgproc/core_background.hpp"
#include "core_cv/imgproc/core_utils.hpp"
#include "tanuki/utils/timer_utils.hpp"  // timing

// Define DEBUG_SAVE_IMAGE to enable debug image dump.
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

        // Lazy CPU buffer allocation (Initialize only set up the GPU buffers).
        if (h_col_mean.size() != static_cast<size_t>(m_width)) {
            h_col_mean.resize(m_width);
        }

        {
            TIME_SCOPE_MS("Total Run Time (CPU)");

            // Step 1: column means with outlier removal.
            // stride = m_width: h_in is tightly packed by column (no padding).
            {
                tanuki::core::calcColumnMeans_RemoveOutliers_cpu(
                    h_in,
                    h_col_mean.data(),
                    m_width,
                    m_height,
                    m_width,        // stride
                    bgSigmaFactor   // sigma_threshold
                );
            }

            // Step 2: column background + mura.
            {
                tanuki::core::calcColumnBackground_u8_cpu(
                    h_in,
                    h_col_mean.data(),
                    h_mura_out,
                    m_width,
                    m_height,
                    m_width         // stride
                );
            }
        }
    }

}
