# PICoater AOI Solution

PI Coater 線上自動光學檢測系統（Automated Optical Inspection），用於 Anilox Roll 塗佈品質即時監控與離線分析。

## 1. 專案架構概覽 (Repository Structure)

```text
PICoater_AOI/
├── bin/                 # 編譯輸出目錄 (所有的 .exe, .dll, .lib 都會產生於此)
├── build/               # 編譯中間檔案 (Intermediate files, obj)
├── sdk/                 # [核心] AOI_SDK 影像處理引擎
│   ├── core_cv          # CUDA 底層運算庫
│   ├── framework        # 測試框架
│   ├── cpp_utils        # C++ 工具庫
│   └── src_dotnet/
│       ├── AOI.SDK/     # .NET SDK（SmartCanvas 等）
│       └── MilGrabSample/ # MIL 相機擷取參考實作
├── src_native/          # [演算法] 專案特定的 C++ 模組
│   ├── modules/         # 各式檢測功能模組 (如 GetPICoaterBackground)
│   └── c_api/           # 導出給 C# 使用的 DLL 介面層
├── src_dotnet/          # [介面] C# 使用者介面
│   └── AniloxRoll.Monitor/ # 主程式 (WinForms)
├── tests/               # [測試] C++ 單元測試與整合測試
│   └── cpp_test         # (picoater_tests)
├── docs/                # 架構與模式文件
├── third_party/         # 第三方函式庫 (如 stb_image)
└── Directory.Build.props # 全域 MSBuild 設定檔
```

## 2. 功能概述

- **即時監控**（Live View）：7 台線掃相機同步取像，MIL 顯示 + GPU Mura 曲線即時更新
- **影像回顧**（Review）：離線瀏覽歷史影像，支援原圖/V強化圖/H強化圖切換，多張拼接模式
- **檢測數據**（Data）：良率統計（序號/時間模式）、逐序號明細、年月日趨勢圖

## 3. 技術棧

| 層 | 技術 |
|------|------|
| UI | C# WinForms (.NET Framework 4.8) |
| 相機 | Matrox MIL 10.x（Camera Link + CLProtocol） |
| GPU | CUDA（Hessian Ridge Detection、Background Removal、StandardBgSub） |
| 存檔 | JPEG + .bin 曲線（GPU resize 1/5x），可選 BMP 原圖 |

## 4. 系統需求 (Prerequisites)

* **IDE**: Visual Studio 2022 (Community/Pro/Enterprise)
* **Workloads**:
  * 使用 C++ 的桌面開發 (Desktop development with C++)
  * .NET 桌面開發 (.NET desktop development)
* **Framework**: .NET Framework 4.8
* **GPU Toolkit**: NVIDIA CUDA Toolkit 12.8 (必須啟用 Visual Studio 整合)
* **Platform**: Windows x64

## 5. 建置指南 (Build Instructions)

本專案使用 `Directory.Build.props` 統一管理路徑，不需要手動設定 Include/Library 路徑。

1. 使用 Visual Studio 2022 開啟根目錄下的 **`PICoater_AOI.sln`**。
2. 組態設定：**Configuration**: `Release` / **Platform**: `x64`
3. 在方案總管中的「方案」上點擊右鍵 -> **建置方案**。
4. 建置成功後，執行檔位於 `bin/x64/Release/`。

## 6. 執行與測試 (Running & Testing)

### C++ 模組測試 (底層驗證)
* **專案**: `picoater_tests`
* 驗證 CUDA 演算法是否正確，不涉及 GUI。

### C# GUI (主程式)
* **專案**: `AniloxRoll.Monitor`
* 程式啟動時自動載入 `picoater_api.dll` 及 `core_cv_api.dll`。

## 7. 開發文件

詳見 `CLAUDE.md`（Claude Code 規則 + 文件路由）和 `docs/` 目錄：

| 文件 | 內容 |
|------|------|
| `docs/architecture-ui.md` | UI 架構、控制項觸發關係 |
| `docs/architecture-image-pipeline.md` | GPU pipeline、存檔格式 |
| `docs/architecture-acquisition.md` | MIL 取像模組 |
| `docs/architecture-data-stats.md` | 統計與 CSV |
| `docs/MIL_API_Reference.md` | MIL API 參考 |
| `docs/patterns-csharp.md` | C#/WinForms 開發模式 |
| `docs/patterns-performance.md` | 效能優化模式 |
| `docs/patterns-mil.md` | MIL 開發模式 |

## 8. 開發規範 (Development Guide)

### 加入新的 C++ 演算法
1. 在 `src_native/modules` 下建立新的專案 (Static Library)。
2. 實作 `.hpp` 與 `.cu`（參考 `Module_GetPICoaterBackground`）。
3. 在 `Directory.Build.props` 定義新模組的路徑變數。
4. **重要**: 透過**專案參考**加入模組，切勿手動加入 `.lib`。

### 解決連結錯誤 (Troubleshooting)
* **LNK2001 / LNK1181**: 檢查 References 是否勾選相依專案。若依賴 CUDA Code，將「使用程式庫相依性輸入」設為 True。

### Coding Style
* **C++**: 遵循 Google C++ Style Guide。
* **路徑**: 所有專案均繼承 `Directory.Build.props`，請勿寫死絕對路徑。

---

### 聯絡資訊

* **Maintainer**: Chunkuan
* **Department**: AUO / PICoater Project Team
