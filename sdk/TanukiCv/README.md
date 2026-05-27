# TanukiCv — 以 core_cv 為引擎的 .NET 影像 SDK

TanukiCv 是一個高效能的電腦視覺開發框架，底層採用 CUDA 引擎 `core_cv` 進行 GPU 加速，
對上提供 .NET 封裝（`TanukiCv.Core` 純 library + `TanukiCv.Controls` WinForms 控制項），
並附統一的測試框架 (Framework) 與工具庫 (Cpp Utils)。

> **命名**：`TanukiCv` 是對外品牌；引擎層 `core_cv` / `core_cv_api` / `cpp_utils` 名稱保留不動。
>
> **dotnet 分層**：
> - `dotnet/TanukiCv.Core`（ns `TanukiCv.Core` / `TanukiCv.Core.Models` / `TanukiCv.Utils`）— 純 library，封裝 `core_cv_api.dll` 的 P/Invoke、GPU helper、檔案/影像工具。
> - `dotnet/TanukiCv.Controls`（ns `TanukiCv.Controls`）— 獨立 WinForms assembly，含 `SmartCanvas`（zoom/pan PictureBox 子類）；不依賴 `TanukiCv.Core`。

## 1. 系統需求 (Prerequisites)

在開始開發或建置之前，請確保環境已安裝以下組件：

* **IDE**: Visual Studio 2022 (v143 toolset)
* **Language Standard**: C++17
* **GPU Computing**: NVIDIA CUDA Toolkit 12.8 (或相容版本)
* **Architecture**: x64

## 2. 專案架構 (Project Structure)

SDK 的核心檔案位於 `sdk/TanukiCv` 目錄下，主要模組如下：

* **core_cv**: 核心影像處理演算法庫。
* 包含所有 CUDA Kernel 實作 (`.cu`) 與 Host Wrapper (`.cpp`/`.cu`)。
* 負責底層 GPU 記憶體管理與影像運算 (Filter, NCC, Subtraction 等)。


* **bench_framework**: 測試與驗證框架。
* 提供 `RunTestBootstrap` 等標準化測試流程。
* 負責圖片讀取 (STB Image)、結果驗證與錯誤處理。


* **cpp_utils**: 通用 C++ 工具庫。
* 提供計時器 (Timer)、終端機顏色輸出 (Terminal Colors) 等輔助功能。


* **core_cv_api**: C 語言導出介面 (Export C API)。
* 用於編譯 DLL，供 C# (WPF/WinForms) 或 Python 調用。



## 3. 開發規範 (Coding Standards)

本專案嚴格遵循以下規範，請在貢獻程式碼時務必遵守：

* **Coding Style**: 符合 **Google C++ Style Guide**。
* **命名規則**: 變數與函式命名需具備描述性，避免不明縮寫。
* **CUDA 語法**: Kernel 啟動語法必須寫為 `<<<grid, block>>>`，**嚴禁** 寫成 `<< < > >>`。
* **檔案分離**:
* 介面宣告請放在 `.hpp` 或 `.cuh` (Header files)。
* 實作細節請放在 `.cpp` 或 `.cu` (Source files)。
* 禁止在 Header 檔中實作複雜邏輯，以確保靜態庫 (.lib) 能正確生成與連結。



## 4. 如何整合 SDK (Usage Guide)

若要在方案中新增一個使用 AOI SDK 的 C++ 執行檔專案 (例如新的測試工具)，請依照以下步驟設定：

### 步驟 A：繼承全域設定 (Props)

確保你的 `.vcxproj` 檔案有匯入根目錄的 `Directory.Build.props`。這通常是自動的，但如果沒有，請在專案檔開頭加入：

```xml
<Import Project="..\..\..\..\Directory.Build.props" />

```

*(路徑視你的專案深度而定)*

這會自動設定好：

* Output Directory (`bin/x64/Release`)
* Intermediate Directory (`build/obj/...`)
* Include Directories (包含 SDK headers)

### 步驟 B：設定專案參考 (Project References) - **關鍵步驟**

為了避免連結錯誤 (LNK2001/LNK1181) 並確保正確的建置順序，**請勿手動加入 .lib 檔案**。請使用 Visual Studio 的專案參考功能：

1. 在方案總管 (Solution Explorer) 右鍵點擊你的專案。
2. 選擇 **加入 (Add)** -> **參考 (Reference)**。
3. 勾選以下核心專案：
* `core_cv`
* `bench_framework`
* `cpp_utils`
* (若有使用特定模組) `Module_GetPICoaterBackground`


4. 按下確定。

**注意**：針對包含 CUDA 程式碼的靜態庫 (如 `Module_...` 或 `core_cv`)，若發生 `LNK2001` 錯誤，請在該「參考」節點上右鍵 -> 屬性，將 **「使用程式庫相依性輸入 (Use Library Dependency Inputs)」** 設為 **True**。

### 步驟 C：開啟 RDC (Relocatable Device Code)

由於 `core_cv` 使用了 CUDA 動態並行或跨編譯單元連結，相依的執行檔專案必須開啟 RDC：

1. 專案屬性 -> **CUDA C/C++** -> **Common**。
2. 設定 **Generate Relocatable Device Code** 為 **Yes (-rdc=true)**。

### 步驟 D：程式碼範例

以下是一個標準的測試程式進入點範例：

```cpp
#include <iostream>
#include "bench_framework/test_utils.hpp"
#include "core_cv/core_ops.hpp" // 引用核心演算法
#include "Module_GetPICoaterBackground.hpp" // 引用特定模組

// 測試邏輯實作
void RunMyTest(const std::string& imagePath) {
    std::cout << "Processing: " << imagePath << std::endl;
    
    // 這裡通常會進行:
    // 1. 讀取圖片 (使用 bench_framework 或 stb)
    // 2. 配置 GPU 記憶體 (cudaMalloc)
    // 3. 呼叫核心演算法 (core::... 或 picoater::...)
    // 4. 下載結果並儲存
}

int main() {
    // 使用 bench_framework 的啟動器，它會自動處理路徑與例外
    return bench_framework::RunTestBootstrap("My Custom Test Suite", RunMyTest);
}

```

## 5. 新增功能模組 (Adding New Modules)

如果你需要開發新的演算法模組 (例如 `GetPICoaterMura`)，請參照 `src_native/modules/GetPICoaterBackground` 的結構：

1. 建立 `.vcxproj` 專案。
2. 設定輸出為 **Static Library (.lib)**。
3. 確保有 `.hpp` (介面) 與 `.cu` (實作) 檔案。
4. 將實作包在 `namespace picoater` 中。
5. 在 `.cu` 檔中包含 `Host Wrapper` 函式來呼叫 `__global__` Kernel。

## 6. 常見問題排除 (Troubleshooting)

* **LNK1181: 無法開啟輸入檔 'xxx.lib'**
* 原因：專案參考未設定，或者依賴的專案編譯失敗。
* 解法：檢查「步驟 B」，並嘗試「重建方案」。


* **LNK2001: 無法解析的外部符號 (Unresolved External Symbol)**
* 原因：函式宣告與定義不符、漏寫 Namespace、或是靜態庫連結問題。
* 解法：確認 Namespace 正確；在測試專案的「參考」屬性中啟用 `Use Library Dependency Inputs`。


* **nvlink error: Multiple definition...**
* 原因：重複連結。
* 解法：檢查 `Directory.Build.props` 是否有多餘的 `.lib` 依賴設定，請移除手動依賴，改用專案參考。