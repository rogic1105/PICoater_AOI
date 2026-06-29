# TanukiCv — 以 tanuki_core 為引擎的 .NET 影像 SDK

TanukiCv 是一個高效能的電腦視覺開發框架，底層採用 CUDA 引擎 `tanuki_core` 進行 GPU 加速，
對上提供 .NET 封裝（`TanukiCv.Core` 純 library + `TanukiCv.Controls` WinForms 控制項），
並含演算法流程層（`tanuki_pipeline`）與共用工具庫（`tanuki_utils`）。

> **命名**：品牌 `tanuki` 貫穿全層 —— C++ namespace `tanuki::core` / `tanuki::utils` / `tanuki::pipeline`，
> C API `TanukiCv_*`（DLL `tanuki_cv_api.dll`）。源碼一律 UTF-8，vcxproj 帶 `/utf-8`。
>
> **dotnet 分層**：
> - `dotnet/TanukiCv.Core`（ns `TanukiCv.Core` / `TanukiCv.Core.Models` / `TanukiCv.Utils`）— 純 library，封裝 `tanuki_cv_api.dll` 的 P/Invoke、GPU helper、檔案/影像工具、合圖佈局（MergeLayout）與曲線合併（CurveOverviewMerger）演算法。
> - `dotnet/TanukiCv.Controls`（ns `TanukiCv.Controls`）— 獨立 WinForms assembly，含 `ImageCanvas`（zoom/pan）、`LiveDisplayView`（多相機監控）等顯示元件；參考 `TanukiCv.Core`。

## 1. 系統需求 (Prerequisites)

* **IDE**: Visual Studio 2022 (v143 toolset)
* **Language Standard**: C++17
* **GPU Computing**: NVIDIA CUDA Toolkit 12.8 (或相容版本)
* **Architecture**: x64（一律 Release|x64）

## 2. 專案架構 (Project Structure)

SDK 的核心檔案位於 `sdk/TanukiCv` 目錄下，native 主要模組：

* **native/tanuki_core**：核心影像處理演算法庫（primitive 層）。
  * 包含所有 CUDA Kernel 實作（`X_kernels.cu`）與 Host Wrapper（`X_ops.cu`）。
  * 一個 wrapper 包一顆 kernel = 一個「動作」（threshold / gaussianBlur / hessian 響應 / column 統計…）。

* **native/tanuki_utils**：通用 C++ 工具庫。
  * 計時器（`timer_utils`）、GPU 計時（`gpu_timer`，cudaEvent）、benchmark harness（`bench_runner`：warmup+統計+CSV）、硬體資訊（`sys_info`）、終端機顏色輸出。

* **native/tanuki_pipeline**：演算法流程層（core→module→pipeline 分層）。
  * `framework/`：`IModule` 介面 + `Pipeline` 工頭 + `ModuleRegistry`。
  * `modules/`：可抽換的「步驟/方法」（`background_sub` 去背、`ridge_hessian` 脊線）——組合 tanuki_core primitive。
  * `pipelines/find_stream_ridgeline/`：食譜＝串 module 成完整解決方案（含 README + benchmark）。
  * `api/`：`tanuki_pipeline_api.dll` —— 應用程式 P/Invoke 的唯一出口。

* **native/tanuki_cv_api**：通用 CV 的 C 語言導出介面（`TanukiCv_*`）。
  * 編譯 `tanuki_cv_api.dll`，供 C# pinned memory / fast BMP / GPU resize 等呼叫。

* **benchmark/**：`tanuki_core_bench`（C++ 自寫輕量 harness，量 primitive 速度）+ `TanukiCv.BenchUi`。

## 3. 開發規範 (Coding Standards)

* **Coding Style**：符合 **Google C++ Style Guide**；變數與函式命名需具描述性。
* **namespace**：傳統巢狀 `namespace tanuki { namespace core { ... }}`（nvcc 不吃 C++17 `namespace a::b{}`）。
* **CUDA 語法**：Kernel 啟動寫 `<<<grid, block>>>`，**嚴禁** `<< < > >>`。
* **檔案分離**：介面 `.hpp`/`.cuh`、實作 `.cpp`/`.cu`；Header 不放複雜實作（確保 .lib 正確生成連結）。
* **分層判準**：包一顆 kernel = core primitive；組幾個 primitive 成可換步驟 = module；串 module 成完整流程 = pipeline。
* **型別後綴命名（定案）**：core primitive 用顯式型別後綴（`scale_clamp_f32_to_u8_gpu`、`threshold_u8_gpu`），**不**參數化成 `xxx_gpu('u8')`——C 風格 + 顯式實例化下這是業界正解（NPP/OpenCV 同款），型別組合有限且編譯期檢查。「同名不同型別組合」才用 template（如 `gaussianBlur_gpu<T_in,T_out>`）。

## 4. 如何整合 SDK (Usage Guide)

新增一個使用 SDK 的 C++ 執行檔專案（例如新的測試工具）：

### 步驟 A：繼承全域設定 (Props)

確保 `.vcxproj` 在 repo 內（自動繼承根 `Directory.Build.props`：OutDir `bin/x64/Release`、IntDir、include 路徑 `$(CoreCVPath)include` / `$(CppUtilsPath)include`）。

### 步驟 B：設定專案參考 (Project References) — **關鍵步驟**

**請勿手動加入 .lib**，用 VS 專案參考（避免 LNK2001/LNK1181 並確保建置順序）：
* `tanuki_core`（+ 若用 pipeline：`tanuki_pipeline_framework` / `tanuki_pipeline_modules` / `find_stream_ridgeline`）

含 CUDA 的靜態庫若報 `LNK2001`：該「參考」節點右鍵→屬性→**Use Library Dependency Inputs = True**。

### 步驟 C：開啟 RDC (Relocatable Device Code)

專案屬性 → CUDA C/C++ → Common → **Generate Relocatable Device Code = Yes (-rdc=true)**。

### 步驟 D：程式碼範例

實際可跑的完整範例見：
* `benchmark/tanuki_core_bench/src/main.cpp` —— primitive 速度量測（gpu_timer + bench_runner + sys_info 全套用法）。
* `native/tanuki_pipeline/pipelines/find_stream_ridgeline/benchmark/src/main.cpp` —— pipeline 端到端用法（`CreateFindStreamRidgeline` + `Process`）。

```cpp
#include "find_stream_ridgeline.hpp"

auto pipe = tanuki::pipeline::CreateFindStreamRidgeline("hessian");
tanuki::pipeline::InputImage in;   /* GPU 灰階 buffer + 尺寸 */
tanuki::pipeline::Params p;        /* sigma / ridge_mode 等 */
tanuki::pipeline::OutputBuffers out; /* ridge 圖 + 曲線 buffer */
pipe->Process(in, p, &out);
```

## 5. 新增功能模組 (Adding New Modules)

開發新的演算法「步驟/方法」（例如新的脊線法 `ridge_gabor`），參照 `native/tanuki_pipeline/modules/ridge_hessian`：

1. 在 `modules/<名稱>/` 建 `include/` + `src/`，實作 `tanuki::pipeline::IModule`（`Process` 組合 tanuki_core primitive）。
2. 加進 `tanuki_pipeline_modules.vcxproj`（CudaCompile + include 路徑）。
3. 在 pipeline 食譜（如 `find_stream_ridgeline.cpp`）的方法選擇加分支（直接 `new`，registry 自註冊在 static lib 會被 linker 丟）。
4. 同目標不同方法＝抽換 module，**不要**另開 `pipeline_2`。

## 6. 常見問題排除 (Troubleshooting)

* **LNK1181: 無法開啟輸入檔 'xxx.lib'**
  * 原因：專案參考未設定，或依賴專案編譯失敗。解法：檢查步驟 B、重建方案。
* **LNK2001: 無法解析的外部符號**
  * 原因：宣告/定義不符、漏 namespace、或 CUDA 靜態庫連結問題。解法：確認 namespace；參考屬性啟用 `Use Library Dependency Inputs`。
* **nvlink error: Multiple definition...**
  * 原因：重複連結。解法：移除手動 .lib 依賴，改用專案參考。
* **`namespace tanuki::core {` 編譯錯誤（nvcc）**
  * nvcc 不支援 C++17 巢狀 namespace 定義，改傳統 `namespace tanuki { namespace core {`。
