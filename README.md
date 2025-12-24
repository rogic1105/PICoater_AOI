# PICoater AOI Solution

## 1. 專案架構概覽 (Repository Structure)

本儲存庫包含完整的開發與測試環境，目錄結構如下：

```text
PICoater_AOI/
├── bin/                 # 編譯輸出目錄 (所有的 .exe, .dll, .lib 都會產生於此)
├── build/               # 編譯中間檔案 (Intermediate files, obj)
├── sdk/                 # [核心] AOI_SDK 影像處理引擎
│   ├── core_cv          # CUDA 底層運算庫
│   ├── framework        # 測試框架
│   └── cpp_utils        # C++ 工具庫
├── src_native/          # [演算法] 專案特定的 C++ 模組
│   ├── modules/         # 各式檢測功能模組 (如 GetPICoaterBackground)
│   └── c_api/           # 導出給 C# 使用的 DLL 介面層
├── src_dotnet/          # [介面] C# 使用者介面與測試程式
│   ├── aoi_gui          # 主程式 (WinForms)
│   └── aoi_gui_test     # C# 端的整合測試工具
├── tests/               # [測試] C++ 單元測試與整合測試
│   └── cpp_test         # (picoater_tests)
├── third_party/         # 第三方函式庫 (如 stb_image)
└── Directory.Build.props # 全域 MSBuild 設定檔 (路徑與編譯參數)

```

## 2. 系統需求 (Prerequisites)

在建置此解決方案之前，請確保開發環境已安裝以下組件：

* **IDE**: Visual Studio 2022 (Community/Pro/Enterprise)
* **Workloads**:
* 使用 C++ 的桌面開發 (Desktop development with C++)
* .NET 桌面開發 (.NET desktop development)


* **Framework**: .NET Framework 4.7.2
* **GPU Toolkit**: NVIDIA CUDA Toolkit 12.8 (必須啟用 Visual Studio 整合)
* **Platform**: Windows x64

## 3. 建置指南 (Build Instructions)

本專案使用 `Directory.Build.props` 統一管理路徑，因此不需要手動設定 Include/Library 路徑。

1. 使用 Visual Studio 2022 開啟根目錄下的 **`PICoater_AOI.sln`**。
2. 將上方工具列的組態設定為：
* **Configuration**: `Release` (推薦) 或 `Debug`
* **Platform**: `x64` (必須是 x64，因為 CUDA 不支援 x86)


3. 在方案總管 (Solution Explorer) 中的「方案 'PICoater_AOI'」上點擊右鍵 -> **建置方案 (Build Solution)**。
4. 建置成功後，執行檔將位於：
* `bin/x64/Release/`



## 4. 執行與測試 (Running & Testing)

### C++ 模組測試 (底層驗證)

用於驗證 CUDA 演算法是否正確，不涉及 GUI。

* **專案**: `picoater_tests`
* **執行方式**: 設定為起始專案並執行 (F5)。
* **功能**: 會讀取預設測試圖片，執行 GPU 背景運算，並輸出結果圖。

### C# GUI 測試 (整合驗證)

用於驗證 C# 與 C++ DLL 的串接 (Interop) 以及操作介面。

* **專案**: `aoi_gui` (主程式) 或 `aoi_gui_test` (測試工具)
* **執行方式**: 設定為起始專案並執行。
* **注意**: 程式啟動時會自動載入 `picoater_api.dll` 及其相依的 `core_cv.dll`。

## 5. 開發規範 (Development Guide)

### 加入新的 C++ 演算法

1. 在 `src_native/modules` 下建立新的專案 (Static Library)。
2. 實作 `.hpp` 與 `.cu` (請參考 `Module_GetPICoaterBackground` 的範例)。
3. 在 `Directory.Build.props` 定義新模組的路徑變數 (選擇性)。
4. **重要**: 在 `picoater_api` (DLL) 或 `picoater_tests` (EXE) 中，透過 **專案參考 (Add Reference)** 加入該模組，**切勿**手動在 Linker 設定加入 `.lib`。

### 解決連結錯誤 (Troubleshooting)

* **LNK2001 / LNK1181**:
* 請檢查該專案的 **參考 (References)** 是否已勾選相依的專案 (`framework`, `core_cv` 等)。
* 若依賴的專案包含 CUDA Code (如 `core_cv`)，請在該參考項目上右鍵 -> 屬性 -> **「使用程式庫相依性輸入 (Use Library Dependency Inputs)」** 設為 **True**。



### Coding Style

* **C++**: 遵循 **Google C++ Style Guide**。
* **路徑**: 所有專案均繼承 `Directory.Build.props`，請勿在個別專案中寫死絕對路徑。

---

### 聯絡資訊

* **Maintainer**: Chunkuan
* **Department**: AUO / PICoater Project Team