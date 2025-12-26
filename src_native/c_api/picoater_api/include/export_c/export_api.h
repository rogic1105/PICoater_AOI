//PiCoater_api\include\export_c\export_api.h

#ifndef PICOATER_API_EXPORT_API_H_
#define PICOATER_API_EXPORT_API_H_

#include <cstdint>

// 定義 DLL 匯出/匯入 巨集
// 在編譯 picoater_api 專案時，需要在 Preprocessor Definitions 加入 PICOATER_API_EXPORTS
#ifdef PICOATER_API_EXPORTS
#define PICOATER_API __declspec(dllexport)
#else
#define PICOATER_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief 執行 PICoater 背景與 Mura 分離演算法 (GPU 版本)
     *
     * @param input_img  [In]  輸入影像指標 (灰階, uint8_t 陣列), 大小為 width * height
     * @param width      [In]  影像寬度
     * @param height     [In]  影像高度
     * @param sigma      [In]  高斯模糊係數 (例如 2.0f)
     * @param output_bg  [Out] 輸出背景影像指標 (Caller 需預先分配 width * height 大小)
     * @param output_mura [Out] 輸出 Mura 影像指標 (Caller 需預先分配 width * height 大小)
     * @return int       0: 成功, 非 0: 失敗代碼
     */
    PICOATER_API int Picoater_GetBackground(
        const uint8_t* input_img,
        int width,
        int height,
        float sigma,
        uint8_t* output_bg,
        uint8_t* output_mura);

#ifdef __cplusplus
}
#endif

#endif  // PICOATER_API_EXPORT_API_H_