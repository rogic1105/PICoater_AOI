# 遷移 Plan：src/native（產品 pipeline）→ sdk/tanuki_pipeline

> 狀態：**規劃中**（基準 = main @ tanuki 改名完成）。動工前逐階段確認。
> 原則沿用 tanuki 改名那次：**分階段 + 每階段 build 驗證 + checkpoint commit + 隨時可回滾**。

## 1. 目標與動機
把 `src/native`（C++/CUDA 演算法 pipeline）整個搬進 `sdk/`，讓：
- **sdk = 所有 library/演算法（無 GUI）** ── 解決現況「library 程式碼住在 app 層 src/native」的矛盾
- **src = 只剩 C# UI app**（產品的臉）
- 中間用單一 C API DLL（P/Invoke）隔開，依賴單向 `src → sdk`

同時把演算法重新分層成可維護、可抽換、可比較的結構。

## 2. 分層定義（判準）
```
kernel       __global__ GPU code                （core 內最底，X_kernels.cu）
  ↑ 包一顆 kernel + 算 grid/block + launch
primitive    threshold_u8_gpu / gaussianBlur…   （🟢 core 對外 API＝「一個動作」）
  ↑ 組多個 primitive + 實作介面
module       ridge_hessian / background_sub     （🔵 一個可換的「步驟/方法」）
  ↑ 串起來
pipeline     find_stream_ridgeline              （🟣 完整「解決方案/食譜」）
  ↑ 單一出口
api          tanuki_pipeline_api.dll            （🟠 run(pipeline名, 參數)）
  ↑ P/Invoke
src (C# UI)  選 pipeline + 給參數（runtime）    （⚪ 產品 UI）
```
**速記**：包一顆 kernel=primitive；組幾個 primitive 成可換步驟=module；串幾個 module 成完整流程=pipeline。
**判準**：「一個動作」→ core；「一串步驟 / 一種方法」→ module/pipeline。

## 3. 目標結構
```
sdk/
├── tanuki_core/        ← primitive（CUDA wrapper：threshold/gaussianBlur/computeHessianResponse/calcColumnMeans…）
├── tanuki_utils/       ← 共用工具（timer/bench_runner/sys_info）
├── tanuki_pipeline/    ← ★ 新
│   ├── framework/      ← IModule 介面 + Pipeline 工頭 + 模組註冊/選擇（namespace tanuki::pipeline）
│   ├── modules/
│   │   ├── background_sub/   ← 去背步驟（robust column 背景 + 相減）
│   │   ├── ridge_hessian/    ← 脊線步驟法1（blur+hessian+scale，實作 IRidge）
│   │   └── （ridge_gabor… 未來，同 IRidge 可抽換）
│   ├── pipelines/
│   │   └── find_stream_ridgeline/   ← 食譜＝串 background_sub → ridge_* → threshold
│   │       ├── include/ + src/
│   │       ├── benchmark/           ← pipeline benchmark（接 tanuki_utils harness）
│   │       └── README.md + docs/images/   ← 演算法說明 + 範例圖（原圖→去背→脊線）
│   └── api/  tanuki_pipeline_api（→ tanuki_pipeline_api.dll，單一出口 run(name, params)）
└── （bench_framework 刪除）
src/
└── dotnet/AniloxRoll.Monitor/   ← 只剩 C# UI
```

## 4. 搬遷對應表
| 現在 | 搬去 | 備註 |
|---|---|---|
| `tanuki_core` 的 `hessianRidge_u8_gpu` | `modules/ridge_hessian` | 組合（blur+hessian+scale），非 primitive |
| `tanuki_core` 的 `calcColumnBackground_u8_gpu/cpu` | `modules/background_sub` | 背景相減步驟 |
| `tanuki_core` 的 `calcColumnMeans_RemoveOutliers_gpu/cpu` | `modules/background_sub` | 穩健背景估計法 |
| `src/native/modules/i_aoi_module.hpp` + `pipeline/aoi_pipeline.*` | `framework/`（IModule + 工頭） | 改名 namespace `tanuki::pipeline` |
| `src/native/modules/get_picoater_background/`（肥 module） | 拆成 `modules/background_sub` + `modules/ridge_hessian` | 名字消失，PICoaterDetector 拆進 module |
| `src/native/c_api/picoater_api` | `api/`（tanuki_pipeline_api.dll） | 單一出口 |
| `src/native/benchmark/picoater_pipeline_benchmark` | `pipelines/find_stream_ridgeline/benchmark` | 接新 harness |
| 留 `tanuki_core`（primitive） | — | threshold/brighten/invert/sobel/resize/convolution/gaussianBlur/convert*/calcColumn(Row)Means(Max)/computeHessianResponse/overlay_heatmap/imgcodecs |

## 5. 分階段執行（每階段 build 驗證 + commit）
| 階段 | 內容 | 驗證 |
|---|---|---|
| **0 骨架** | 建 `sdk/tanuki_pipeline/{framework,modules,pipelines,api}` 空殼 + 定 IModule/IRidge 介面 | 編得過 |
| **1 framework** | 搬 i_aoi_module + aoi_pipeline → framework，改名 `tanuki::pipeline::IModule`/`Pipeline` | framework 編譯 |
| **2 modules** | 把 🔴 三個 core 函式 + PICoaterDetector 拆成 `background_sub` / `ridge_hessian` module（實作介面），core 移除那三個 | 各 module lib 綠、tanuki_core 仍綠 |
| **3 pipeline** | 建 `find_stream_ridgeline` 食譜（串 module）+ README + 範例圖 | pipeline lib 綠 |
| **4a api 搬移** | picoater_api → `api/`（**先保留現有函式**，只搬位置 + 改 DLL 名 tanuki_pipeline_api.dll）+ 改 C# P/Invoke DLL 名 | .NET app 綠、runtime 對齊（上機驗）|
| **4b api 重設計**（可後續） | 改成單一 `run(pipeline_name, params{method})` runtime 選 module | .NET 改呼叫方式 |
| **5 benchmark** | picoater_pipeline_benchmark → pipeline/benchmark，接 tanuki_utils harness | benchmark 跑出數字 |
| **6 收尾** | 刪 bench_framework + 空的 src/native；更新 .sln/props/docs | 全方案綠 |

## 6. 風險 / 注意
- **階段 4 是最大新工**（不只搬，是 C API 重設計 + P/Invoke 改）。**先 4a 純搬（保行為、上機驗 runtime），4b 重設計另開**，de-risk。
- **nvcc namespace**：新 namespace 用傳統巢狀 `namespace tanuki { namespace pipeline {`（C++17 `A::B{}` nvcc 不吃，沿用 tanuki_core 教訓）。
- **byte-safe**：bulk 改用 Python bytes-level（GNU sed 在 heredoc 吃 backslash、Big5 風險，沿用教訓）。
- **MSBuild 牽連**：dir/vcxproj/.sln/Directory.Build.props 的 `$(...)Path` 屬性都要一起改（沿用 tanuki 改名經驗）。
- **依賴方向**：pipeline → core（單向）；module 之間不互相依賴（都靠 framework 介面）。

## 7. 設計定案（2026-06-11 確認）
- **framework = .lib**（工頭 Pipeline + 模組註冊表 ModuleRegistry，編一次）+ 介面（IModule/IRidge + I/O structs）放 header。
- **module 顆粒度** = 去背一個（background_sub）+ 脊線一個（ridge_hessian），足夠。
- **pipeline 食譜 = 先硬編 C++**（型別安全、現在一個 pipeline）；「換方法」用**參數選**（registry by name），不上 json 食譜（拿 90% 彈性、0 額外基建）。未來多食譜/現場可調再上 json。
- **API = 單一 `run(pipeline_name, json_params)`**（一簽名服務所有 pipeline，C# settings 序列化 json 傳入，native 各自 parse；擴充不破 ABI）。階段 4a 先純搬保行為、4b 才換成此形狀。
