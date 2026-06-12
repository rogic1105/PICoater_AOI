# sdk/ — PICoater AOI 可獨立 split 的 library 區

> 此檔在編輯 `sdk/` 下任何檔時載入（巢狀 CLAUDE.md）。專案總規則見 repo 根 `CLAUDE.md`；
> 本檔只談 sdk 區的分層原則與元件地圖。**目標：sdk/ 未來可整包 split 成獨立 repo**，故規則以「能不能帶著走」為準。

## 鐵則（違反即停下重組）

1. **只放 library，無 GUI / 無 exe** — 帶 WinForms 的可執行檔放各元件的 `samples/` 或 repo 根 `tools/`，不放 `*.Core`。引用 sdk 的專案不該被迫拉 UI 依賴。
2. **依賴單向 `src/ → sdk/`，sdk 絕不反向依賴 src/** — 包括不 `using AniloxRoll.Monitor.*`、不接 application-level 設定物件 / 業務 callback。CI lint：`grep -r "using AniloxRoll" sdk/` 應為 0。
3. **每個元件 self-contained 可 split** — 自帶 `Directory.Build.props` / `.gitignore` / `vendor/`（廠商規格書放這，不放 repo 根 `third_party/`）。
4. **vendored third-party 隨元件走** — 如 `TanukiCv/third_party/stb`，split 時一起帶。
5. **拋棄層 vs durable 層要分清** — 「換硬體就整包丟」的程式碼（MIL grabber 封裝）留在 `MIL/`，可重用的演算法（合圖佈局、像素↔mm）放 `TanukiCv/`（durable，跨產品共用）。可重用 IP 不該困在拋棄層。

## 架構原則：演算法分層（kernel → primitive → module → pipeline → api → src）

任何影像/演算法功能，按這個堆疊放（依賴單向往下；違反即停下重組）：

```
kernel       __global__ GPU code               （tanuki_core 內最底，X_kernels.cu）
  ↑ 包一顆 kernel + 算 grid/block + launch
primitive    threshold_u8_gpu / gaussianBlur…  （🟢 tanuki_core 對外 API＝「一個動作」）
  ↑ 組多個 primitive + 實作 IModule 介面
module       ridge_hessian / background_sub    （🔵 tanuki_pipeline/modules＝可換的「步驟/方法」）
  ↑ 串起來（食譜）
pipeline     find_stream_ridgeline             （🟣 tanuki_pipeline/pipelines＝完整「解決方案」）
  ↑ 單一出口
api          tanuki_pipeline_api.dll           （🟠 TanukiPipeline_Create(name,json)/Process(...,json,...)）
  ↑ P/Invoke
src (C# UI)  選 pipeline 名 + 給 json 參數     （⚪ 產品 UI；src 不放演算法）
```

**判準（新功能放哪一層）：**
- 「**一個動作**」（包一顆 kernel）→ core primitive（如 `calcColumnBackground`、`computeHessianResponse`）
- 「**一串步驟 / 一種方法**」（組幾個 primitive、可被換掉）→ module（如 hessian 法找脊線）
- 「**端到端解決方案**」（串 module）→ pipeline

**擴充模式（不破既有結構）：**
- 同目標**換方法**（hessian→gabor）＝新 module 實作 IModule + 食譜加分支，**不開 `pipeline_2`**
- 加**新 pipeline** ＝ `TanukiPipeline_Create` 加分支 + C# 換 name 字串（API 單一簽名不變）
- 加**演算法參數** ＝ json 加 key（各 pipeline parse 自己的；**不破 C ABI、不改 P/Invoke**）
- 指標類（GPU/host buffer）**不進 json**：input/output struct + 獨立引數

**反模式：**
- ❌ 把多步驟組合塞進 core 當 primitive（如已刪除的 `hessianRidge_u8_gpu`＝blur+hessian+scale）
- ❌ module 之間直接互呼（只透過 OutputBuffers 傳遞、由 pipeline 排順序）
- ❌ 演算法寫在 src/（src 只剩 UI）
- ❌ 用 registry 名字選 module 卻不補 static-lib link-drop 保險（目前食譜 direct-new 刻意避開；真用 registry 再補 `RegisterBuiltinModules`）

歷史與細節：`docs/dev/migration-native-to-sdk-pipeline.md`（已完成的遷移紀錄）。

## 元件地圖

```
sdk/
├── TanukiCv/        ← 以 tanuki_core(CUDA) 為引擎的 .NET 影像 SDK（durable，跨產品共用）
│   ├── native/{tanuki_core, tanuki_utils, tanuki_cv_api, tanuki_pipeline}   ← C++/CUDA 引擎 + C API + pipeline 層
│   │     C++ namespace = `tanuki::core`（傳統巢狀 `namespace tanuki { namespace core {`；nvcc 不吃 C++17
│   │     `namespace tanuki::core{}` 形式）。避開超常見 `core` 撞名，供其他 C++ 專案 source/header 重用。
│   │     對外 C API = `extern "C" TanukiCv_*`（DLL `tanuki_cv_api.dll`；.NET P/Invoke 不碰 namespace）。
│   │     tanuki_utils namespace = `tanuki::utils`（原 `Color` 已收編）。源碼一律 UTF-8 + vcxproj 帶 /utf-8
│   │     ★ tanuki_pipeline（namespace `tanuki::pipeline`）= 演算法流程層（分層見上方「架構原則：演算法分層」）：
│   │       framework/（IModule + Pipeline 工頭 + ModuleRegistry）；modules/（background_sub、ridge_hessian）；
│   │       pipelines/find_stream_ridgeline/（找流水圖脊線=mura 檢測，含 README+benchmark）；
│   │       api/（tanuki_pipeline_api.dll＝app P/Invoke 出口；json_lite 零依賴 parser）
│   ├── dotnet/
│   │   ├── TanukiCv.Core         ← 純 library（無 WinForms）：PixelMmMapper 像素↔mm、SystemInfo、PerfTimer、
│   │   │                             MergeLayout（合圖佈局唯一來源）、CurveOverviewMerger（切向全覽曲線合併唯一來源）
│   │   └── TanukiCv.Controls     ← WinForms（→Core）：SmartCanvas / LiveDisplayView / ThumbStrip /
│   │                                 曲線圖 helper（Base/Column/Row）/ GrayBitmap / GrayResizeCpu
│   ├── benchmark/{tanuki_core_bench, TanukiCv.BenchUi}
│   ├── samples/TanukiCv.SysInfoTool
│   └── third_party/stb
├── Bridges/         ← 對外設備 / 系統橋接層（純函式庫 + 介面供 mock 注入）
│   ├── IoBridge/IoBridge.Core/        ← ICP DAS ET-7044（Modbus TCP）+ IModbusTcpClient + IoModuleFactory
│   ├── LightBridge/LightBridge.Core/  ← LTS-3DPA24 RS-232 光源
│   └── StorageBridge/StorageBridge.Core/ ← SMB + 檔案複製 + 循環儲存
│   （各 Bridge 有 samples/ 可執行範例 + vendor/ 廠商規格書）
├── MIL/             ← MIL 集中區（拋棄層：換 grabber 整區換）
│   ├── MilGrabber.Core/   ← MIL 取像/顯示封裝（MilCamera=一台相機 + MultiCameraMerger 即時合圖工頭）
│   ├── samples/MilGrabber.Monitor/  ← 多相機監控範例（MilGrabberPbForm）；三種合圖方式選用
│   └── docs/   ← Matrox CLProtocol / Grabber 規格書
└── docs/            ← 跨專案工程經驗（repo-style / testing pyramid / FSM）
```

## 單一來源（sdk 內已收斂的，勿再抄）

- **合圖佈局** = `TanukiCv.Core.MergeLayout.Compute`（純算術；xOffset + 重疊 boundary，3 策略 `MergeOverlap.Midline/RightOverLeft/LeftOverRight`）。影像合圖（GrabImageStitcher / LiveDisplayView）+ 曲線合圖都呼這份 → 曲線與影像 pixel 對齊。
- **切向全覽曲線合併** = `TanukiCv.Core.CurveOverviewMerger.Merge`（純算術；reuse MergeLayout boundary 唯一歸屬、間空參與分界(黑占位)留 0＝在線相機曲線在與黑布的中線被切、與影像對齊；回傳 mean/max/globalMin/gridMm 純資料，「秀」交呼叫端）。app `CurveMergeHelper.UpdateOverviewChart` 是薄 wrapper（委派 Merge + 接 ColumnCurveChartHelper + StitchMode 視野）；範例可直接呼 Merge 接自己的曲線圖。
  - 例外：`MIL/MultiCameraMerger` 刻意保 MIL-only 自含中線、不引用 TanukiCv（拋棄層隨硬體換）。
  - 註：MergeLayout / CurveOverviewMerger 在 **Core**（純算術），非 Controls —— 純 IP 不困在 WinForms assembly，headless/benchmark/範例 皆可用。
- **像素↔mm** = `TanukiCv.Core.PixelMmMapper`。
- **曝光上限公式** = `MilCameraParams.CalcExposureMaxUs`（`MIL/MilGrabber.Core/MilCamera.Params.cs`）。
- **曲線圖** = `BaseCurveChartHelper`（Template Method）+ `Column`/`Row` 子類；app 與 sample 共用。

## samples/ vs repo 根 tools/

- `sdk/<元件>/samples/` — 只服務單一 sdk 元件的可執行範例（拿掉該元件就沒用），跟元件一起 split。
- repo 根 `tools/` — 跨元件 / 應用層通用工具，不專屬單一元件。
- 判準：「這工具拿掉某個 sdk 元件還有用嗎？」沒用 → samples/；還有用 → tools/。

## Build

- 一律 `Release|x64`。sdk 工具方案 `sdk/Tools.sln`；各 Bridge `sdk/Bridges/*/*.sln`；單一 csproj msbuild 直接 build（依賴自動拉）。
- **`.Core` 輸出位置不可在 monorepo 內改**（VS 方案 P2P 參考寫死共用 bin → CS0006；共用 OutDir 是刻意設計）。
