#pragma once

#include <cstdint>

// --------------------------------------------------------------------------
//  DLL 導出/匯入 巨集定義
// --------------------------------------------------------------------------
// 當我們在編譯 ExportCDLL 專案時，VS 會自動定義 ExportCDLL_EXPORTS (或類似名稱)
// 這時我們要用 dllexport (導出)。
// 當其他 C++ 專案(如 TestApp) 引用這個標頭檔時，我們要用 dllimport (匯入)。
// --------------------------------------------------------------------------

#ifdef EXPORTCDLL_EXPORTS
#define EXP_API __declspec(dllexport)
#else
#define EXP_API __declspec(dllimport)
#endif

// 定義呼叫慣例 (Calling Convention)
// __cdecl 是 C/C++ 的預設標準，也是 C# CallingConvention.Cdecl 的對應
#define EXP_CC __cdecl


// --------------------------------------------------------------------------
//  C 語言介面 (extern "C")
//  避免 C++ Name Mangling，確保 C# 可以找到正確的函式名稱
// --------------------------------------------------------------------------
extern "C" {

    // 1. PCB Mask 生成 (你的主要功能)
    EXP_API int EXP_CC BuildFullMaskFromFFT_C(
        const uint8_t* img_gray,
        int H, int W,
        float fft_th, int bw_th, int border_t,
        uint8_t* full_mask_out
    );

    // 2. 亮度調整 (測試用)
    EXP_API int EXP_CC Process_Brighten(
        const uint8_t* src,
        uint8_t* dst,
        int W, int H,
        int value
    );

    // 3. 二值化 (測試用)
    EXP_API int EXP_CC Process_Threshold(
        const uint8_t* src,
        uint8_t* dst,
        int W, int H,
        int value
    );

    // 4. 反轉 (測試用)
    EXP_API int EXP_CC Process_Invert(
        const uint8_t* src,
        uint8_t* dst,
        int W, int H
    );

}