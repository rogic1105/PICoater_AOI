# 遷移 Plan：src/native（產品 pipeline）→ sdk/tanuki_pipeline

> 狀態：✅ **全部完成（含 phase-6 刪舊，2026-06-12）**。本文件轉為歷史紀錄：分層定義（§2）與判準仍為現行規範。
> 原則沿用 tanuki 改名那次：**分階段 + 每階段 build 驗證 + checkpoint commit + 隨時可回滾**。

## ⚡ 目前進度（2026-06-11 自主跑完 0–5）
新 sdk/tanuki_pipeline 全套**已建好 + build 綠 + 實機 RTX5080 跑通**（12MP 整條 3.26ms）：
| 元件 | 產出 | 狀態 |
|---|---|---|
| framework | tanuki_pipeline_framework.lib（IModule/Pipeline/Registry） | ✅ build |
| modules | tanuki_pipeline_modules.lib（background_sub + ridge_hessian） | ✅ build |
| pipeline | find_stream_ridgeline.lib（食譜 + README） | ✅ build + **runtime 跑通** |
| api | tanuki_pipeline_api.dll（**C ABI 與 picoater_api 相同，drop-in**） | ✅ build |
| benchmark | find_stream_ridgeline_bench.exe | ✅ build + 跑出數字 |

**平行建新、未拆舊**：`src/native`（picoater_api.dll）+ bench_framework 全保留，**app 仍走舊路徑可用**。

### 🔎 架構審查 + 修正（2026-06-12，Fable 5 agent 審查、已驗證屬實並修掉）
1. **（高）ridge_hessian workspace data race 已修**：舊 PICoaterDetector 把 blur view 放 offset≈N，
   與 `gaussianBlur_gpu` 從 offset 0 bump 配的內部 scratch（f32_temp）重疊 → col-pass kernel
   讀 temp 同時寫 dst = 真 GPU race（靠 block 排程順序矇對；換卡/尺寸可能浮現）。忠實 port 曾把坑搬來。
   **修正**：blur/resp view 排到 gaussian scratch 區之後、零重疊（+8N bytes）；並刪掉沒人用的 u8 遺跡槽。
2. **（高）bg_sigma parity 已還原**：舊 code 在 column mean 這步「硬編 sigma=1、無視 bgSigmaFactor 參數」，
   新 module 一度改用參數值（app 傳 2.0）→ 輸出必然與舊版不同、drop-in 不成立。
   **暫還原硬編 1.0（求 parity，可乾淨比對）**；要「啟用參數」（可能是修舊 bug）再單獨決定 + 記 known diff。
3. **（中）module 補錯誤檢查**：cudaMalloc / memcpy / launch 失敗現在會設 err_ 回 false（GetLastError 契約兌現）。
4. `Params.ridge_mode` 預設 `"dark"`（不被解析=默默零輸出）→ 改 `"vertical+horizontal"`；
   未知 ridge_method 食譜回 nullptr（api CreatePipeline 有防護）。
5. 審查留待後續：registry static 自註冊 link-drop 保險（4b 前補 `RegisterBuiltinModules()` + 啟動 assert）、
   api 同步 cudaMemcpy 依賴 legacy default stream 隱式同步（與舊版一致，開 per-thread stream 才有事）、
   `calcColumnMeans_gpu` 死參數 d_workspace 可刪、`IModule::Initialize` 無人呼叫（YAGNI 候選）。

### 🔲 交接進度
1. ~~**切換 app**~~ ✅ **已做**：`NativeMethods.cs` DllName 已改 `tanuki_pipeline_api.dll`，即時監控正常。
2. **驗數值**（修正後重驗）：上面第 1、2 修正後，新舊輸出理論上 parity；同輸入比 ridge/curve 確認。
3. ~~**接 .sln + 依賴**~~ ✅ **已做**：tanuki_pipeline 5 個 vcxproj 收進 PICoater_AOI.sln（方案資料夾 sdk/TanukiCv/tanuki_pipeline），
   app ProjectDependencies 加 tanuki_pipeline_api（**picoater_api 依賴暫留＝回滾保險，phase-6 刪舊時一併移除**）。
   全方案 build 0 錯誤；實測刪 DLL 後 sln build 自動重生 → **清 bin 陷阱解除**。
   附帶：舊 picoater_pipeline_benchmark（既有編碼壞、已被 find_stream_ridgeline_bench 取代）關 Build.0 排除出 sln build（目錄留 phase-6 刪）；api 連結修 LNK4098（移 cudart_static）。
4. ~~**刪舊**~~ ✅ **已做（phase-6，2026-06-12）**：`src/native` 整目錄（picoater_api／get_picoater_background／aoi_pipeline／picoater_pipeline_benchmark）+ sdk `bench_framework` + `core_cv_benchmark` 殘檔刪除；.sln 4 專案條目 + app 的 picoater_api 回滾依賴移除；props（BenchFrameworkPath/NativeRoot/LocalModulesPath）清掉；CLAUDE.md×2 / README×2 / skills×3 同步。全方案 build 0 錯誤。**src/ 只剩 dotnet（UI），單一真相在 sdk/tanuki_pipeline。**
5. ~~**可選 4b**~~ ✅ **已做（2026-06-12）**：API 定版 `TanukiPipeline_Create(name, json_options)` + `Process(handle, input, json_params, precomputed_col_mean, output)`（演算法參數 json 化＝加參數/加 pipeline 不破 ABI；json_lite 零依賴 flat parser；指標走 struct/獨立引數）。C# NativeMethods/AoiService 同步切換。
   一併收尾：bg_sigma 參數 honest 化（module 用參數；app 新常數 PerFrameBgSigma=1.0 走 Process＝行為不變、DefaultBgSigma=2.0 留背景採集——兩條路徑本來就不同 sigma）；IModule 移除無人呼叫的 Initialize（YAGNI）；calcColumnMeans_gpu 死參數 d_workspace 刪除；AlgorithmParams 預設 ridge_mode "dark"→"vertical+horizontal"。
   registry 保險（RegisterBuiltinModules）未做＝registry 仍是未用的未來基建（食譜 direct-new 已避開 link-drop），等真用 registry 選 module 再補。
   ⚠ **app 需上機重驗**（P/Invoke 簽名換了：監控 + 取得背景 + 回顧各跑一輪）。

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
| `tanuki_core` 的 `hessianRidge_u8_gpu` | **已刪**（由 `modules/ridge_hessian` 取代） | 組合（blur+hessian+scale），非 primitive，且無人呼叫 |
| `tanuki_core` 的 `calcColumnBackground_u8_gpu/cpu` | **留 core**（`modules/background_sub` 組合它） | 實作時改判：單顆 kernel=primitive；module 擁有食譜、不擁有 kernel |
| `tanuki_core` 的 `calcColumnMeans_RemoveOutliers_gpu/cpu` | **留 core**（`modules/background_sub` 組合它） | 同上（原計畫「搬去 module」已修正，勿誤刪 core 函式） |
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
