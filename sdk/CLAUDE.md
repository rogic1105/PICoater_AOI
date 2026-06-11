# sdk/ — PICoater AOI 可獨立 split 的 library 區

> 此檔在編輯 `sdk/` 下任何檔時載入（巢狀 CLAUDE.md）。專案總規則見 repo 根 `CLAUDE.md`；
> 本檔只談 sdk 區的分層原則與元件地圖。**目標：sdk/ 未來可整包 split 成獨立 repo**，故規則以「能不能帶著走」為準。

## 鐵則（違反即停下重組）

1. **只放 library，無 GUI / 無 exe** — 帶 WinForms 的可執行檔放各元件的 `samples/` 或 repo 根 `tools/`，不放 `*.Core`。引用 sdk 的專案不該被迫拉 UI 依賴。
2. **依賴單向 `src/ → sdk/`，sdk 絕不反向依賴 src/** — 包括不 `using AniloxRoll.Monitor.*`、不接 application-level 設定物件 / 業務 callback。CI lint：`grep -r "using AniloxRoll" sdk/` 應為 0。
3. **每個元件 self-contained 可 split** — 自帶 `Directory.Build.props` / `.gitignore` / `vendor/`（廠商規格書放這，不放 repo 根 `third_party/`）。
4. **vendored third-party 隨元件走** — 如 `TanukiCv/third_party/stb`，split 時一起帶。
5. **拋棄層 vs durable 層要分清** — 「換硬體就整包丟」的程式碼（MIL grabber 封裝）留在 `MIL/`，可重用的演算法（合圖佈局、像素↔mm）放 `TanukiCv/`（durable，跨產品共用）。可重用 IP 不該困在拋棄層。

## 元件地圖

```
sdk/
├── TanukiCv/        ← 以 core_cv(CUDA) 為引擎的 .NET 影像 SDK（durable，跨產品共用）
│   ├── native/{core_cv, cpp_utils, core_cv_api}   ← C++/CUDA 引擎 + C API
│   │     C++ namespace = `tanuki::core`（傳統巢狀 `namespace tanuki { namespace core {`；nvcc 不吃 C++17
│   │     `namespace tanuki::core{}` 形式）。避開超常見 `core` 撞名，供其他 C++ 專案 source/header 重用。
│   │     對外 C API 仍是 `extern "C" CoreCV_*`（.NET P/Invoke 不碰 namespace）。cpp_utils 仍 `Color`（待議）
│   ├── dotnet/
│   │   ├── TanukiCv.Core         ← 純 library（無 WinForms）：PixelMmMapper 像素↔mm、SystemInfo、PerfTimer、
│   │   │                             MergeLayout（合圖佈局唯一來源）、CurveOverviewMerger（切向全覽曲線合併唯一來源）
│   │   └── TanukiCv.Controls     ← WinForms（→Core）：SmartCanvas / LiveDisplayView / ThumbStrip /
│   │                                 曲線圖 helper（Base/Column/Row）/ GrayBitmap / GrayResizeCpu
│   ├── benchmark/{bench_framework, core_cv_benchmark, TanukiCv.BenchUi}
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
- **切向全覽曲線合併** = `TanukiCv.Core.CurveOverviewMerger.Merge`（純算術；reuse MergeLayout boundary 唯一歸屬、間空留 0；回傳 mean/max/globalMin/gridMm 純資料，「秀」交呼叫端）。app `CurveMergeHelper.UpdateOverviewChart` 是薄 wrapper（委派 Merge + 接 ColumnCurveChartHelper + StitchMode 視野）；範例可直接呼 Merge 接自己的曲線圖。
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
