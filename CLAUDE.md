# PICoater AOI — Claude Code Rules

## 架構原則：repo 分層（src / sdk / tools / tests / docs）

```
PICoater_AOI/
├── src/                  ← 應用程式（產品交付；只剩 UI）
│   └── dotnet/AniloxRoll.Monitor/  ← C# WinForms 主應用（C++ 演算法已全數搬入 sdk/TanukiCv/native/tanuki_pipeline）
├── sdk/                  ← 可獨立 split 的 library（純函式庫，無 GUI、無 exe）。**有自己的 `sdk/CLAUDE.md`（巢狀，編 sdk 檔時載入；放分層鐵則+元件地圖，隨 split 帶走）**
│   ├── TanukiCv/         ← 以 tanuki_core 為引擎的 .NET 影像 SDK（native/{tanuki_core,tanuki_utils,tanuki_cv_api,tanuki_pipeline〔★演算法流程層 core→module→pipeline：framework(IModule+工頭+registry)+modules(background_sub/ridge_hessian 可換步驟)+pipelines/find_stream_ridgeline(食譜+README+benchmark)+api(tanuki_pipeline_api.dll=app P/Invoke 出口，原 src/native picoater_api 已退場刪除)〕} + dotnet/{TanukiCv.Core 純 library〔含 PixelMmMapper 像素↔mm 公式、SystemInfo CPU/GPU/RAM/螢幕查詢、PerfTimer 通用計時器（量段+視窗 worst-case，計時唯一來源）、MergeLayout 合圖佈局演算法單一來源（純算術；xOffset+重疊分界，3 策略：中線/右覆蓋左/左覆蓋右）、CurveOverviewMerger 欄全覽曲線合併唯一來源（reuse MergeLayout boundary 唯一歸屬、間空參與分界(黑占位)留 0＝在線相機曲線在與黑布的中線被切、與影像對齊；範例同源共用）〕, TanukiCv.Controls WinForms〔→Core；含 ImageCanvas + **顯示 pipeline 共用元件：ImageDisplayView（絞殺榕重寫版多相機監控：主畫面 ImageCanvas+ThumbStrip縮圖+CPU合圖+合圖全部+flip+**LOD 單張&合圖**，統一介面 PushFrame/SetLayout/EnableLod(GrayResize)/FlipVertical；**合圖 LOD＝虛擬圖=完整合圖佈局、provider 逐欄找相機合成可見區+stride 壓緩衝→GrayResize，顯示成本從 ~180ms 降到 ~1ms**；**app（LiveCameraManager）+ 範例（MilGrabberPbForm）都已採用＝兩產品同源唯一來源；舊 MultiCamLiveView 已退場刪除**）/ ThumbStrip（多相機縮圖條：批量 CPU 建圖不閃，唯一來源）/ ThumbView（雙緩衝自繪縮圖葉子）/ GrayBitmap（灰階 bytes→bitmap 唯一來源）/ GrayResizeCpu（純 CPU 雙線性縮放＝LOD 的 CPU provider，無 GPU 機器用）/ GrayResize 委派（LOD resize 插槽：GPU 呼叫端給、CPU 用 GrayResizeCpu）**〕} + benchmark/{tanuki_core_bench,TanukiCv.BenchUi} + samples/TanukiCv.SysInfoTool〔系統資訊 GUI 工具〕 + third_party/stb；self-contained 可 split）
│   ├── Bridges/          ← 對外設備 / 系統橋接層
│   │   ├── IoBridge/                         ← ICP DAS ET-7044 IO module（Modbus TCP）
│   │   │   ├── IoBridge.Core/                ← library（IModbusTcpClient 介面 + ET-7044 實作）
│   │   │   └── samples/                      ← 可執行範例（ManualControl / Automation GUI）
│   │   ├── LightBridge/LightBridge.Core/     ← RS-232 LTS-3DPA24 光源
│   │   └── StorageBridge/StorageBridge.Core/ ← SMB + 檔案複製 + 循環儲存
│   ├── MIL/              ← MIL 集中區（MilGrabber.Core MIL 封裝 library〔MilCamera=一台相機〕 + samples/MilGrabber.Monitor 多相機監控範例（**繪圖模式 MIL 直繪 panel vs PictureBox**〔「設定」tab PropertyGrid 選、初始化後軟鎖、釋放解鎖〕；PictureBox 模式：MilCamera FrameReady → GetFrameBytes → tanuki_core GPU resize 縮圖 → ImageCanvas 繪 + 合圖，測 GDI 即時取像卡不卡。class=MilGrabberPbForm（1127→535 行，按職責拆 partial：`.cs` 核心生命週期/ctor + `.PictureBox.cs` 顯示路徑 + `.Params.cs` 參數接線 + `.Telemetry.cs` ListView/timer + `.Config.cs` json 載入）；含 `[PbTiming]` 計時 log（縮圖/顯示 max）+ chkLod 動態LOD（停住裁可見區+GPU 縮到 panel→縮小看全圖便宜、放大看真細節）+ 滾輪相對 fit + 雙擊 fit / 三擊實體 1:1〔FOV 輸入框算 mm/px〕。**原獨立 MilGrabber.PictureBox 範例已併入此**。**tabParams 含「設定」tab**（單一 PropertyGrid `propertyGridMerge` 綁 `PbSettings`，仿 app propertyGridSettings 的 SSoT；散落 chk/num/cmb/radio 全收進來：繪圖模式/合圖方式/動態LOD/重疊策略 互斥用 enum〔`EnumDescConverter` 顯示中文 [Description]〕+ 上下翻轉/FOV/縮圖倍率 + OPS/Start CamRow8〔Cam1~8 可展開〕；`PropertyValueChanged`→`ApplyPbSettings` 一次套用全部，view 不擁有邏輯。合圖全部=無畫面相機黑占位；佈局用 sdk `TanukiCv.Controls.MergeLayout`，巨圖 `MergeMaxW=30000` cap 防 GDI 16-bit 座標 wrap 內容錯位）） + docs）；隔離 MIL，換 grabber 整區換。**MIL 合圖（MultiCameraMerger）走 sdk 單一來源 `TanukiCv.Core.MergeLayout`（2026-06-26 收斂，原 MIL-only 自含一份已移除）—— MergeLayout 零依賴純算術、從拋棄層引用它依賴方向正確(throwaway→durable)；live/回顧/瀑布/曲線全同一份、黑槽（沒影像相機）一致參與中線分界**。**取像同步**：相機需 CLProtocol 套線掃才同頻（btnInit 後自動跑 btnFetchInfo 等 CLProtocol+套線掃，否則 free-run 偶發不同步；硬體無 encoder/外部觸發）
│   └── docs/            ← 跨專案工程經驗（repo-style / testing pyramid / FSM）
├── tools/                ← 跨元件 / 應用層通用工具（不專屬單一 sdk 元件）
│   ├── ps/                   ← PowerShell 腳本
│   └── python/               ← Python 工具
├── tests/                ← 純自動化測試（.NET NUnit 三層 + TestRunner.bat/.ps1，量「對不對」）
├── benchmark（量「多快」）→ 不設頂層，跟被測對象住：
│       sdk/TanukiCv/benchmark/tanuki_core_bench（通用 CV）、sdk/TanukiCv/native/tanuki_pipeline/pipelines/find_stream_ridgeline/benchmark（pipeline 端到端）
├── algtest/              ← Python 演算法原型 / 可行性研究（讀 repo 外 05_QA_Validation；非自動化測試）
├── docs/                 ← 文件
│   ├── config/           ← 設定 JSON 範例
│   ├── dev/              ← 開發者參考（MIL API / 廠商規格書）
│   └── user-manual/      ← 操作員說明（hardware-specs / io_diagrams / storage-flow）
├── deploy/               ← 現場部署腳本（PowerShell + JSON）
└── .claude/skills/       ← Claude Code skills（按修改範圍觸發）

**samples/ vs tools/ 區分：**
- `sdk/<元件>/samples/` — **只服務單一 sdk 元件**的可執行範例（拿掉該元件就沒用）。展示「怎麼用這個 library」，self-contained 跟元件一起 split。如 IoBridge 的 ManualControl / Automation GUI。
- `tools/` — **跨元件 / 應用層**通用工具（log analyzer、部署 helper 等），不專屬單一元件。
- 判準：「這工具拿掉某個 sdk 元件還有用嗎？」沒用 → samples/；還有用 → tools/。
```

**核心分層原則（業界 monorepo + Codex/Gemini 共識）：**

1. **library 跟 executable 實體分離** — sdk/ 只放 library（無 GUI、無 exe），exe/工具放 tools/。引用 sdk 的專案不會被迫拉 UI 依賴
2. **sdk/ = 可獨立 split** — 每個元件 self-contained（有自己 Directory.Build.props / .gitignore 更好），未來可 split 為獨立 repo
3. **依賴方向單向** — `src/ → sdk/`；vendored third-party（如 stb）放各 sdk 元件的 `third_party/`（隨元件 split）；**sdk/ 絕對不能反向依賴 src/**
4. **新硬體 bridge 走 sdk/ 模板** — 見 [.claude/skills/add-hardware-bridge.md](.claude/skills/add-hardware-bridge.md)

**業界對照：**
- `src/` ↔ Nx `apps/` / .NET `src/`（dotnet/runtime）
- `sdk/` ↔ Nx `packages/` / .NET `lib/`（Microsoft）
- `tools/` ↔ `tools/` / `bin/`（標準）

## 架構原則：測試分層

`tests/` 按「執行速度 / 副作用 / 失敗影響範圍」分三層 csproj：

| 類型 | csproj | 內容 | 速度 | CI 跑時機 |
|---|---|---|---|---|
| **Unit** | `tests/AniloxRoll.Monitor.Tests/` | 純邏輯、無 IO、無外部依賴（Mock 對外） | < 5ms / case | 每次 commit |
| **Integration** | `tests/AniloxRoll.Monitor.Integration.Tests/` | 含檔案 IO、JSON 讀寫、Mock 硬體 | < 1s / case | PR / nightly |
| **Stress** | `tests/AniloxRoll.Monitor.Stress.Tests/` | 長時間循環、Soak、Load | 數十秒～小時 | 週期跑（隔夜 / 週末 soak 24h） |

**分類規則（測試該歸哪一類）：**
- 用 `Mock<I*>` 注入 + 純函式驗證 → **Unit**
- 用 `Path.GetTempFileName()` / `File.WriteAllText` / 讀寫 JSON / CSV → **Integration**
- `for (i = 0; i < N_BIG; i++)` 或 `Task.Delay(minutes)` → **Stress**

**新測試該寫哪一層？提問順序：**
「這條測試需要設定外部資源（檔案 / mock 硬體）嗎？」→ 是 → Integration
「這條會跑很久（> 1s）嗎？」→ 是 → Stress
「都不是」→ Unit

**InternalsVisibleTo**：`src/dotnet/AniloxRoll.Monitor/Properties/AssemblyInfo.cs` 同時 `InternalsVisibleTo` 三個 test assembly。新增第四個測試 csproj 時要加進去。

**Benchmark（量「多快」，跟被測對象住，非 `tests/`）**：
- `sdk/TanukiCv/benchmark/tanuki_core_bench/` — 通用 CV micro-benchmark（C++）；跟 tanuki_core 同住，隨 sdk split 帶走
- `sdk/TanukiCv/native/tanuki_pipeline/pipelines/find_stream_ridgeline/benchmark/` — pipeline 端到端速度（C++/CUDA；tanuki_utils harness：gpu_timer+bench_runner+sys_info）；跟被測 pipeline 同住，供 agent loop 優化

**Python 演算法**：
- `algtest/` — 演算法原型 / 可行性研究（讀 repo 外 05_QA_Validation 資料；非自動化測試）

**未來擴充**：
- UI 自動化 → `tests/AniloxRoll.Monitor.UITests/`（拆 csproj）
- .NET micro-benchmark → 跟被測 .NET 元件同住（BenchmarkDotNet）

## 架構原則：前端 UI 架構 → `src/dotnet/AniloxRoll.Monitor/CLAUDE.md`

app 的 UI 架構（SSoT 原子結構 / 四層 View-協調-State-Service / 協調層五角色 / 機制政策邊界）
已移至**巢狀 `src/dotnet/AniloxRoll.Monitor/CLAUDE.md`**（編 app 檔時載入，與 sdk/CLAUDE.md 對稱）。

## 架構原則：唯一來源（同一邏輯只寫一份）

> 此條 **OVERRIDE 預設的「避免過早抽象」傾向**：本專案真正重複（同一真相多份）一律收斂，不留分歧空間。

任何「公式、演算法、順序敏感的步驟、常數」出現**第二份**時，**主動提取成唯一來源**（一個函式 / 方法 / 類別），呼叫端共用，不抄多份。**發現即提取並執行，不要只提議等同意** —— 小範圍 private 重構直接做。

**判準（決定該不該抽）：**
- 「這份改了，另一份是不是**一定**要跟著改？」→ 是（同一個真相）→ **抽**
- 否（只是現在像、未來各自演化）→ 不抽（三行相似 < 過早抽象）

**最高危險：順序敏感的重複** —— 多個方法都做「算 A → 設 B → 用 B」且順序重複，順序錯一處就出 bug。**必抽成單一方法把順序鎖死。**

**已知教訓（都是抄多份釀的坑）：**
- 曝光上限公式 `900000/線掃` 曾抄 4 份（主程式 3 + 範例 1）→ 收進 `MilCameraParams.CalcExposureMaxUs`
- 合圖佈局「設座標 + 算位置」在 `EnableMerge`/`RefreshLayout` 各一份且順序不一致 → 切換 StitchMode 後 `xOffset` 用到 `RefOpsMm=0`（除以 0）變垃圾值 → 合圖全黑 → 收進 `MultiCameraMerger.ApplyLayout`（順序鎖死）
- 座標換算 `pixel↔mm`：回顧已收斂進 sdk `ImageDisplayView.OnCanvasStatus`（唯一來源，CanvasInteractionHelper 已刪）；Live MIL 直繪路徑 `LiveCameraManager` 仍一份（公式都走 `PixelMmMapper`，待 Wave3 一併檢視）

**討論 / 設計時的提問順序：**
「這段邏輯有沒有第二份？」 → 「改一處另一處是否一定要跟著改？」 → 是則「抽哪裡、誰呼叫」 — 而不是「複製過來改一改」。

## 術語標準：軸命名（col/row ↔ 欄/列，唯一一組，無例外）

> 收斂歷史包袱：曾有多組詞講同 2 個軸（現行 欄/列、中文舊稱、Vertical/Horizontal、row/col），
> 且 `chartLiveVertical` 配的是 col（欄）= 視覺詞天生反向。**一律收斂成下表這一組，舊詞完全刪除（非註解標「已棄用」）。**

| 軸 | **code 識別字** | **中文（註解/UI/標準名稱）** | 影像意義 | 圖表 | 物理 |
|---|---|---|---|---|---|
| X | **col**（Column） | **欄** | 每「欄」一個值 | 沿 X 畫 | 沿輥圓周 |
| Y | **row**（Row） | **列** | 每「列」一個值 | 沿 Y 畫 | 橫向捲動 |

**鐵則：**
- **欄 = col = X**；**列 = row = Y**。繁中慣例：欄=直行(column)、列=橫列(row)，**勿記反**（此表為唯一來源）。
- **完全退場（刪除，非註解）**：中文軸舊詞、`切線`、以及**我們自己的** `Vertical`/`Horizontal` 識別字（控制項名、欄位、方法）。
- **唯一不碰**：WinForms 框架的 `Vertical/Horizontal`（`Orientation`/`DockStyle`/`ScrollBars`/`TextAlign`/`Anchor`…）— 那不是我們的術語，rename 時手術式排除。
- PropertyGrid「檢出方向」顯示值收斂為 `欄/列/全部`（操作員看 欄/列 比中文軸舊詞直觀、又跟 code 1:1）。
- sdk 既有 `ColumnCurveChartHelper`/`RowCurveChartHelper` 已是 col/row，為對齊基準，不改。
- 控制項對齊：`chart{Live,Review,Data}Vertical`→`...Column`、`chart{Live,Review}Horizontal`→`...Row`（修「Vertical 配 Column」反向）。

## 專案結構

```
PICoater_AOI/
├── src/dotnet/AniloxRoll.Monitor/                 ← C# WinForms 應用程式（src 只剩 UI）
├── sdk/TanukiCv/                                  ← 以 tanuki_core 為引擎的 .NET 影像 SDK（native/{tanuki_core,tanuki_utils,tanuki_cv_api,tanuki_pipeline} + dotnet/{TanukiCv.Core 純 library, TanukiCv.Controls WinForms} + benchmark/{tanuki_core_bench,TanukiCv.BenchUi}）
├── sdk/Bridges/IoBridge/IoBridge.Core/          ← Modbus TCP Client + IModbusTcpClient 介面
├── sdk/Bridges/LightBridge/LightBridge.Core/      ← LTS-3DPA24 RS-232 光源
├── sdk/Bridges/StorageBridge/StorageBridge.Core/  ← SMB 檔案複製 + 循環儲存
├── sdk/docs/                                      ← 跨專案工程經驗（atomic html）
├── tools/io-manual-control/IoBridge.ManualControl/  ← 手動 DI/DO GUI
├── tools/io-automation/IoBridge.Automation/         ← FSM 模擬 GUI
├── tests/AniloxRoll.Monitor.{Tests,Integration.Tests,Stress.Tests}/  ← NUnit 三層 + TestRunner.bat/.ps1
├── benchmark：sdk/TanukiCv/benchmark/tanuki_core_bench/（通用 CV）+ sdk/.../tanuki_pipeline/pipelines/find_stream_ridgeline/benchmark/（pipeline）  ← 速度測試，跟被測對象住
├── algtest/                         ← Python 演算法原型 / 可行性研究
└── deploy/                          ← 現場部署腳本（PowerShell + JSON 參數）
    ├── storage-pc/                  ← 儲存機：固定 IP + SMB 共用 + 防火牆 + Guest 匿名（secedit）
    └── inspection-pc/               ← 檢測機：單 NIC 雙 IP 別名 + Client 端匿名 Guest SMB
```

## Native API

兩組 DLL，均宣告於 `src/dotnet/AniloxRoll.Monitor/Interop/NativeMethods.cs`：

| DLL | 函式 | 用途 |
|-----|------|------|
| `tanuki_pipeline_api.dll` | `TanukiPipeline_Create(name,jsonOpts)` / `Process(h,input,jsonParams,precomputed,output)` / `GetLastError` / `Destroy` / `ComputeColumnMean` | GPU 檢測 pipeline（4b 定版：單一 API run(name,json)，演算法參數走 json 字串、加參數/加 pipeline 不破 ABI；指標走 struct/獨立引數）。**output struct 尾端含可選「存檔縮圖」欄位**（`resize_width/height` + `resized_raw/ridge/mura`）：非 0/非 null 時 pipeline 在檢測同一次 device 停留就地縮（raw←input、V←ridge、H←mura），**免存檔再二次 H2D**；全 0 則跳過（純 live 幀）。決策以 grab 為單位（存圖 grab 才傳目標尺寸） |
| `tanuki_cv_api.dll` | `TanukiCv_AllocPinned` / `TanukiCv_FreePinned` | CUDA pinned memory 管理 |
| `tanuki_cv_api.dll` | `TanukiCv_FastReadBMP` | 快速讀取 BMP（繞過 GDI+） |
| `tanuki_cv_api.dll` | `TanukiCv_Resize_GPU` | GPU 縮圖（**LOD 顯示用**；存檔縮圖已改走 pipeline fused 輸出，不再走這支） |

## P/Invoke 架構規則

**所有 P/Invoke 宣告只能在 `AniloxRoll.Monitor/Interop/NativeMethods.cs`**，不得跨層使用 SDK 的 `TanukiCv.Core.TanukiCvWrapper`。

## 關鍵檔案速查

| 路徑 | 職責 |
|------|------|
| `Interop/NativeMethods.cs` | 唯一 P/Invoke 宣告點 |
| `ImageProcessing/NativeBufferPool.cs` | CUDA pinned buffer 管理 |
| `ImageProcessing/InspectionEngine.ImageProcessing.cs` | 縮圖/全解析度影像處理 |
| `ImageProcessing/InspectionEngineConfig.cs` | MaxWidth=16384, MaxHeight=10000, DefaultSaveResizeScale=5 |
| `ImageProcessing/BatchInspectionService.cs` | Parallel.For 批次縮圖 |
| `UI/Form/AniloxRollForm.cs` | **主檔（核心 bootstrap，~940 行）**：欄位宣告、ctor、`OnFormClosing`、`InitializeSystem`、`InitServiceLayer`/`InitUiLayer`/`InitCameraLayer`、`OnSettingChanged`（SSoT dispatcher，勿拆）、PG events、`UpdateCamCountLabel`。原 3969 行 God Object 已按職責拆 9 個 partial（↓ 同 `partial class AniloxRollForm`） |
| `UI/Form/AniloxRollForm.Live.cs` | Live 監控：grab 流程（`btnLiveGrab_Click`）、即時曲線（`OnLiveCurveData`/`OnLiveRowCurveData`）、Mura 判定（`CheckLiveMura`）、強化/合圖切換（`ApplyMuraEnhance`/`SwitchStitchModeWithEnhanceSequence`） |
| `UI/Form/AniloxRollForm.Review.cs` | 回顧：資料夾/時段載入（`btnReviewSelectFolder_Click`/period 導航/`ApplyReviewEnhance`/`LoadImagesWithReviewConfig`） |
| `UI/Form/AniloxRollForm.Background.cs` | 背景取得/載入/預覽 + 背景判斷（`IsBgBinReady`/`IsStandardBgSubEnabled`） |
| `UI/Form/AniloxRollForm.SettingsTabs.cs` | 右側設定面板 tab 建構（`SetupCameraTab`/`SetupSystemTab`/`Bind*Sync`）+ 相機參數硬體同步（`SyncCameraParamsFromHardware`） |
| `UI/Form/AniloxRollForm.HardwareStatus.cs` | IO/光源/儲存狀態：init（`InitIoController` 設 `ReconnectIntervalMs=3000`/`ReadWriteTimeoutMs=500`、`InitLightController` AutoDetect 背景化）、連線標籤、LED、儲存管理（`TriggerRetentionAndFlagAsync`）。**斷線重連倒數**（`RefreshIoConnLabel`/`UpdateLightConnLabel`/`UpdateStorageConnLabel` 顯示「重連中 Ns…」，秒數源自 `ReconnectIntervalMs`/`*ProbeIntervalTicks`×`TelemetryTickMs` 單一來源；尊重 `_isIoSuspended` 不覆蓋）。**儲存探測** `ProbeStorageReachable`＝解析 UNC host 後 TCP 連 445（繞過 SMB session 快取，重插即恢復；非 `Directory.Exists`）。`OnNetworkAddressChanged`（NetworkChange 事件）拔/插本機網路線即時重探 |
| `UI/Form/AniloxRollForm.DirectionStitch.cs` | V/H 方向/ridge/合圖模式切換（`SwitchRidgeDirection`/`OnStitchModeChangedAsync`） |
| `UI/Form/AniloxRollForm.Data.cs` | 檢測數據 Tab（`SetupDataTab`/grabId 選擇） |
| `UI/Form/AniloxRollForm.Telemetry.cs` | Telemetry/資源監控 timer（`TelemetryTimer_Tick`/`UpdateResourceMonitor`）。`TelemetryTimer_Tick` MIL 查詢經 `LiveTelemetryPresenter.Capture` 背景化（`_telemetryCaptureInFlight` 防堆積）+ `AreCamerasHwReady` gate（CLProtocol 初始化期間不碰 MIL）。注意：timer new+Start 仍在 SettingsTabs.SetupSystemTab |
| `UI/Form/AniloxRollForm.Helpers.cs` | PG refresh（`RefreshGridItem`）/Review 座標（`ViewRangeProvider`）/通用（`FindCameraById`/`IsCanvasFitToScreen`） |
| `UI/Form/AniloxRollForm.Designer.cs` | Form 控制項佈局（VS Designer）。狀態列順序：相機→儲存→光源→IO連線→IO狀態→DIO |
| `Program.cs` | 進入點 + **全域例外攔截**（`ThreadException`/`AppDomain.UnhandledException`/`UnobservedTaskException` → 寫 `AniloxRoll-crash.log` 到 bin 與 %TEMP%；背景執行緒未處理例外不再直接 0xffffffff 終止）+ 損毀 user.config 自刪 + **檔案 trace listener**（Main 開頭加 `TextWriterTraceListener` → `{AniloxRoot或D:\Anilox}\Logs\trace-*.log`，AutoFlush；所有 `Trace.WriteLine`〔含 `[HtRealloc]` 改高度診斷〕落地檔案供離線判讀，stall/hang 也已 flush） |
| `UI/Widgets/FormInteractionHelper.cs` | 回顧資料夾載入、忙碌鎖、設定套用、螢幕校正、ReviewConfig（Wave2 後已去 canvas 代理 + gallery；Wave3 待拆 facade） |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/Interaction/EventGuard.cs` | 可重入 bool 旗標（EventGuard + EventGuardScope），using 自動還原（**Wave1 已搬 sdk**） |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/BaseCurveChartHelper.cs` | **曲線圖抽象基底（已搬 sdk 共用唯一來源）**：Template Method（Build 骨架 + Mean/Max 線 + 閾值線工廠 + PostPaint），子類填方向專屬洞。自包含 0 依賴 app；app+sample 共用。`ShowThresholds`（預設 true）可關紅閾值線（純剖面用）；`RowCurveChartHelper.SetRowPitch` 直接設 mm/列。**範例 chartProfileX/Y 重用此 + ImageDisplayView.CursorProfileChanged 游標剖面（單張＝選定相機全幀；**合圖剖面已做**＝游標列橫跨整張合圖、用 BuildMerge 同份 placements 拼故與畫面 pixel 對齊、游標行取所屬相機；sdk ImageDisplayView 內、app+sample 同享）** |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/ColumnCurveChartHelper.cs` | 欄（X 軸）曲線圖子類：X=位置 mm/Y=值、右側 Y2 刻度、水平 InnerPlotPosition 對齊補償、zoom 同步（chartLiveVertical/Patch/Review*/DataPatch 用）|
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/RowCurveChartHelper.cs` | 列（Y 軸）曲線圖子類：Y=位置、軸旋轉、垂直 InnerPlot 補償（chartLiveHorizontal/ReviewHorizontal 用）|
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/Input/TrackBarWheelInterceptor.cs` | TrackBar 滑鼠滾輪攔截器（**Wave1 已搬 sdk**） |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/Input/ComboBoxWheelReverser.cs` | ComboBox 滑鼠滾輪方向反轉（**Wave1 已搬 sdk**） |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/Interaction/MultiClickDetector.cs` | **多擊偵測器（已搬 sdk 共用唯一來源）**：ImageCanvas 雙擊 fit/三擊 1:1 與 app `camLiveMain` panel route 共用；呼叫端可設定 interval/距離模式保留原手勢語意。 |
| `UI/Widgets/CurveMergeHelper.cs` | **薄 wrapper**：全覽合併數學已抽到 sdk `TanukiCv.Core.CurveOverviewMerger.Merge`（唯一來源，範例同源可共用），本類別只 `.bin 曲線讀取（MergeCurves/MergeRowCurves/GetCurveBasePath，含 CaptureFileNaming 檔名故留 app）+ UpdateOverviewChart「秀」（委派 Merge → 接 ColumnCurveChartHelper + StitchMode 視野）`。欄全覽重疊區依合圖方式 `MergeOverlap`（app 預設 Midline，對齊影像 MultiCameraMerger 中線）唯一歸屬、不再 avg/max、間空留 0（邏輯在 Core） |
| `UI/Presenters/DataStatisticsPresenter.cs` | Data tab 統計邏輯：統計計算、combo 串聯、Period Charts、Mura 空間分布圖（chartDataPatch）、跨 Tab 同步事件。**`listViewGrabDetail` = VirtualMode**（預估常駐 ~1 萬筆：只存 `_visibleDetails`+`RetrieveVirtualItem` 按需建列；點選用 `SelectedIndices`；欄寬 `FitGrabDetailColumnsToContent` 取樣 `_visibleDetails`；符號 unicode —/○/×）|
| `UI/Presenters/ReviewStitchCoordinator.cs` | Review tab 拼接管理：LoadGrabStitchedViewAsync、合圖、ClearStitchedMode、overview chart 聯動。**掉偵補黑（tick 對位）**：跨相機對齊軸交 `FrameTickIndex.BuildAlignedByTick`（硬體 tick）→ fallback `BuildAlignedByStitchKey`（檔名）；每 camId 拿對齊清單（缺幀=null），影像 `StitchCamera`+列曲線 `MergeRowCurves` 共用同份 → 掉幀那格影像黑布+曲線 0 對齊。**換 ID 載入效能**：`LoadGrabStitchedViewAsync` 的 `Task.Run` 內 7 台相機 `Parallel.For` 平行解碼/拼接（imgs[i]/curve[i] 各寫各 index、BitmapPool 有 lock、CurveMergeHelper 無共用 static → 安全；GDI+ 併發為灰色地帶，留意偶發黑塊）+ **`MergeHorizontal` 也移進背景**（原在 UI 執行緒＝swap 卡頓主因）；計時 log `CSV/Stitch/Merge(bg)/UIapply/Total`。**`CurveFlipVertical`**（旗標，未來可做 tool 選項）+ `FlipRowCurveIfNeeded`：線掃相機由下往上拍→回顧影像上下翻轉（StitchCamera），故 row 曲線兩條路徑（逐相機 + Global）都反向才對齊影像；live 不翻轉故不動 |
| `UI/Presenters/LiveTelemetryPresenter.cs` | 16 欄即時 Telemetry。**MIL 查詢背景化**：`Capture(cameras)`（背景執行緒做 16 欄 MdigInquire/MsysInquire ≈195ms，回傳純字串 `CamSnapshot`）+ `Apply(snapshots)`（UI 執行緒只套字串，不碰 MIL）→ 避免 `TelemetryTimer_Tick` 每 500ms 卡 UI 執行緒。`Update()`=同步版（背景用） |
| `Acquisition/AniloxCamera.cs` | 單台相機 composition：持有 `MilCamera _mil`（`sdk/MIL/MilGrabber.Core`）委派 MIL 資源/grab/display/參數/telemetry；自己做檢測/存檔/合圖/曲線（訂閱 `_mil.FrameReady`，hook 內檢測→顯示`PutDisplayBytes`/`CopyToDisplay`→合圖→存檔）。Global merge child-buffer 來源。**存檔縮圖 fused（一進多出）**：`TryApplyPicoaterRidge` 依 `wantResize`（grab-level：EnableAutoCapture && !SuppressCapture && CaptureRootPath && scale>1）把 3 塊 pinned dst（`_rawResizeBuf`/`_procResizeBuf`/`_muraResizeBuf`=raw/V/H）填進 `ProcessImage` 的 output resize 欄位 → pipeline 就地縮、免二次 H2D；`TrySaveCapture` 直接讀預縮 buffer（不再呼 `TanukiCv_Resize_GPU`）。`_lastFrameResized` 守門：detection 失敗幀不存舊縮圖。`HessianSigma`（細線濾除 ridge_sigma）預設＝`InspectionEngineConfig.DefaultRidgeSigma`（勿寫死值） |
| `sdk/MIL/MilGrabber.Core/MilCamera.cs` | MIL 取像/顯示封裝 library（一台相機=一個 MilCamera）：alloc/grab/display/參數/系統資訊/CLProtocol/在線/mouse hook/buffer helper(`GetFrameBytes`/`PutDisplayBytes`/`CopyToDisplay`/`ClearDisplay`)/線掃最大速率(`GetLineRateMaxHz` via CLProtocol M_FEATURE_MAX，grab 後 ~3s 可得)；`FrameReady`/`OnMouseDataChanged`/`OnCameraClicked` 事件。純 MIL 範圍，檢測等非 MIL 由訂閱者做。**ctor `devNum` = 板內固定絕對 device 位置（0-based，對應 M_DEVx）唯一轉換點**：caller（主程式 LiveCameraManager + sample）一律傳 json 固定值、不加 M_DEV0 偏移（相機實體配線固定，少槽卡只列實際 channel）；本機型 M_DEV0=0 為 identity，未來 M_DEV0≠0 只改 ctor 這一行。**原 876 行 God object 已按職責拆 4 partial（同 `partial class MilCamera`，純分檔零邏輯變更）：核心 `MilCamera.cs`（欄位/ctor/Initialize/Grab/Hook/Merge target/Dispose，~342）+ `MilCamera.Params.cs`（Exposure/LineRate/GrabHeight + `MilCameraParams` 公式類）+ `MilCamera.Display.cs`（buffer I/O + 主/副顯示 + mouse hooks）+ `MilCamera.Telemetry.cs`（唯讀遙測 getter）+ `MilCamera.CLProtocol.cs`（CLProtocol 啟用/套參）。Merge target 刻意留核心（hook 內呼叫，耦合緊）**。`MilCameraParams`（純函式參數公式單一真相：`CalcExposureMaxUs(lrHz,expMin,expMaxCap)`=曝光上限=900000/線掃 clamp；主程式+範例共用，勿再各自抄公式）移至 `MilCamera.Params.cs`。**`SetGrabHeight`：①開頭同值守門（高度未變+buffer已配→直接 return 不 realloc）＝防套設定時多餘 realloc 撞背景 CLProtocol enable→CAM1 stall（改高度 stall 主因，2026-06-24 dropdiag 定案）；②改尺寸前 `M_STOP+M_WAIT`+`MdigControl(M_GRAB_ABORT)` drain；③熱路徑禁 MsysInquire/MdigInquire（會 cam1 stall）；高度硬上限 `AcquisitionDefaults.MaxGrabHeightPx=12000`（固定、不分台數；12062 是 grab 中拉單台真硬限）；勿寫相機 Height feature；詳見 `/modify-acquisition` skill** |
| `sdk/MIL/MilGrabber.Core/MultiCameraMerger.cs` | 多相機即時合圖「工頭」library（純 MIL、無 WinForms）：接收一組 MilCamera，算佈局（全域範圍/xOffset/重疊中點分界）+ 分配合併 buffer + 每台 SetMergeTarget。`EnableMerge`/`RefreshLayout` 共用唯一來源 `ApplyLayout`（先設 RefOpsMm 再算 xOffset，**順序鎖死**避免除以 0 變垃圾值）。回傳 `MergedBuffer` + 座標(MinStartMm/RefOpsMm/TotalW/H) 供上層「秀」。貼圖在 MilCamera grab hook（`SetMergeTarget`/`CopyDisplayToMergeTarget`） |
| `Acquisition/CameraFrameSaver.cs` | 存檔 I/O：SaveCapture（背景執行緒）、SaveJpegFromBytes、SaveCurveBinFromArray、Resource Log（CSV: CPU%/RAM/VRAM/GPU ms/Live/Review/StitchMode；啟動時 MergeOldResourceLogs 把「昨天以前」的小檔按日合併為 resource-monitor-yyyyMMdd.csv）。**tick 側車**：`AppendTickSidecar` 把每幀硬體 frame-start tick（`CaptureContext.FrameStartTicks`，源 `MilCamera.LastFrameStartTicks`）寫進當日資料夾 `_ticks.csv`（`baseName,ticks`，static lock 多相機共寫）→ 供回顧 tick 對位精準補黑 |
| `Services/FlowTrace.cs` | **[Flow] 顯示資料流跡唯一出口**：咽喉點各一行（AllocateCameras/StartGrab/StopGrab/FreeCameras/ApplyMainDisplayMode/Enable-Teardown view/firstFrame/SwitchMainDisplay/EnableGlobalMerge），每行帶時間戳+執行緒 ID → 落 `Logs\trace-*.log`。驗證＝與 `/verify-flows` 的 flow 契約（EVT）比對：執行緒內驗全序、跨執行緒驗因果+完整性（不驗非決定性交錯）。首幀追蹤每次 StartGrab 重置＝每輪 grab 都證明「幀有流到 view」 |
| `Services/FrameTickIndex.cs` | **跨相機幀對齊唯一來源**（回顧合圖補黑）：各台「各自獨立掉不同幀」→ seq 會歪、檔名軟體協調戳不知道實際掉哪幀 → 用硬體 frame-start tick（同板 125MHz 同 epoch，物理同時兩幀差 <0.5ms）就近聚類成「時間槽」。`LoadTickMap`（讀 `_ticks.csv` 側車）+ `BuildAlignedByTick`（聚類→各 camId 對齊清單，缺槽=null=補黑）+ `BuildAlignedByStitchKey`（舊資料無側車的檔名 fallback）+ `ComputeThreshold(period)=period/2`（**同槽容差規則單一來源**，回顧批次 + 監控瀑布串流共用）。回傳餵 `GrabImageStitcher.StitchCamera`/`CurveMergeHelper.MergeRowCurves`（吃 null=黑布占位）。**⚠ 跨板 tick 不可相減**（cam1-4 板0/cam5-7 板1 epoch 不同）：現裸全域排序，**7 台跨板會錯配，現在沒事只因測試都同板**（待補 board offset 正規化，連瀑布一起） |
| `UI/Widgets/WaterfallView.cs` | 監控主畫面「瀑布圖」（`he_MainDisplay`==Waterfall）：全幅 7 相機合圖每幀往下接成捲動長圖。**全解析分塊儲存**（`ChunkRows=512` chunk，避開 2GB byte[] 上限）+ **LOD 只在顯示時降採樣**（provider 邊讀邊 nearest 到 dest，不配巨圖暫存）；固定總高 `WaterfallTotalHeight`（預設 30000）。**Ring 循環**＝繞回頂端覆蓋最舊+寫頭畫亮掃描線接縫；**Restart 重來**＝滿了清黑幕重畫。**跨相機對齊與回顧同源**：吃 `AniloxCamera.OnDisplayFrame` 的硬體 tick → 串流半週期聚類成時間槽（pending 槽緩衝 + hold-back grace flush，避免同瞬間晚到幀偽掉幀；週期＝運行最小 delta 抗掉幀；thr 共用 `FrameTickIndex.ComputeThreshold`），某台缺幀那欄補黑。背景 queue+worker 寫 memcpy 不卡 UI。佈局走 `LiveCameraManager.FeedWaterfallLayout`（合圖開用 merger 7 槽、沒開退回設定 start/ops）。**槽數＝配置相機數（7，非線上台數）** |
| `Services/CaptureFileNaming.cs` | 擷取檔名規則單一真相：suffix 常數（`_raw.jpg`/`_proc_v\|h.jpg`/`_mean_v\|max_v\|mean_h\|max_h.bin` + 5 個 legacy fallback）+ `IsRawJpg`/`StripRawJpg`/`BaseFromImagePath`（影像路徑→base 反推）/`ResolveProcJpg`（v/h+legacy 解析）。寫端（CameraFrameSaver/InspectionEngine 存檔）+ 讀端（CurveMergeHelper/ReviewStitchCoordinator/GrabImageStitcher/ImageRepository/InspectionStatisticsService）共用 —— 改命名格式只改這。**只統一檔名字串，fallback 載入行為留各 caller**（避免改 edge case）。另含背景檔名 `BgBin(w,camId)`/`BgGlob`/`BgGlobForCam(camId)`（AniloxRollForm 載/掃背景共用） |
| `Services/CaptureStoragePaths.cs` | 擷取資料日期階層儲存路徑單一真相：`DailyCsv(root,d)`=`{root}\{yyyy}\{yyyyMM}\{yyyyMMdd}.csv`、`DateImageDir(root,d\|yyyymmdd)`=該日影像資料夾。寫端 InspectionLogService + 讀端 InspectionStatisticsService 共用 —— 改目錄結構只改這 |
| `Acquisition/CaptureTimestampCoordinator.cs` | 多相機存檔時間戳同步 |
| `UI/Managers/GlobalMergeCoordinator.cs` | **即時全域合圖「秀」協調者**（2026-06-26 從 LiveCameraManager 提取，internal sealed）：擁有 `_merger`(MultiCameraMerger 工頭)+`_mergedDisplay`+座標鏡像+33ms 防閃刷新 timer+滑鼠 hook，負責 MIL 合圖 display 整個生命週期（alloc/select window/free）+ 視野範圍查詢（TryGetViewRange[Y]）+ merged 分支 zoom/pan/1x/reset/pan-to-center。穩定依賴走 ctor、變動值走 Func 委派（screenMmPerPx/speed/lineRate/IsReleasing）、視野中心選中相機走 `Action<int>` callback（不反向參考整個 LCM）。`showMilDisplay` 現恆 false（顯示鐵則：合圖「秀」一律 CPU——即時=ImageDisplayView、瀑布=WaterfallView；工頭只負責佈局+merge buffer。此協調者的 display 生命週期 code 待 Wave3 清除）。**巨圖 display 必須「先關 M_UPDATE 再 MdispSelectWindow」**（select+SCALE+CENTER 各觸發一次 8.9 萬寬巨圖重繪 → lag + 半貼殘影）。**合圖永遠 Global**（`hb_StitchMode` 寫死 Global，2026-06-13 退場 Vertical） |
| `UI/Managers/LiveDisplayCoordinator.cs` | **即時監控顯示協調者**（2026-06-29 從 LiveCameraManager 提取，internal sealed）：擁有 ImageCanvas/Waterfall/縮圖 panel/狀態 label/選中相機/screen mm-per-px/GPU LOD provider/WheelZoomFilter。負責 `ApplyMainDisplayMode`、`SetLodMode`、`SwitchMainDisplay`、單台 MIL mouse status 文字、ImageCanvas cursor/view-range 轉發、Waterfall layout feeding、非合圖 MIL 直繪 zoom/1x。穩定 UI 依賴走 ctor；變動資料（cameras/settings/lineRate/live 狀態）走 Func，不反向參考整個 LCM；GPU LOD pinned buffer 生命週期委派 `TanukiCv.Core.GpuGrayResizeProvider`（app 注入 `NativeMethods`）。 |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/WaterfallView.cs` | **瀑布顯示元件（已搬 sdk 共用）**：ImageCanvas LOD + chunked gray buffer + MergeLayout placement；`WaterfallFullMode` 也在 sdk。app 只負責餵 frame/layout/settings。 |
| `UI/Managers/LiveCameraManager.cs` | 多台相機生命週期管理、連線數監控（OnCameraCountChanged）、即時全域合圖**編排 + forwarder**（EnableGlobalMerge 建 MilCamera 清單+清各台 secondary；「拼」委派 `MultiCameraMerger`、「秀」委派 `GlobalMergeCoordinator`、ImageCanvas/Waterfall/狀態/滾輪顯示職責委派 `LiveDisplayCoordinator`；`LiveCameraManager.Display.cs` 只留對外 API forwarder）。**改參數協調套用**：`ApplyParamCoordinated(camId,write)`＝只停/寫/開**被改的那一台**（相位已證明不重要 free-run 2-3 條線 → 不再全部一起停/開連累 cam2 stall）；`ApplyParamCoordinated(write)`＝全部（All 滑桿用）。**不同步等出幀**（Thread.Sleep 會凍 UI→Not Responding→拉滑桿排隊 replay 跳值＝暴力漏洞）。**stall 偵測**（`LiveCameraManager.Telemetry.cs` status timer 500ms）：判據＝**`M_PROCESS_FRAME_COUNT`（`GetFrameCount`）有沒有前進**，**非 FPS 門檻**（低線掃合法 FPS 極低如 0.0083，固定門檻會誤判）；幀數凍住超過「2s + 預期幀週期(高度/線掃)×1.5」自動窗才判 stall（縮圖紅「STALL」）；停/開救不回（硬體 CL 失鎖、只有重開程式）故不自動 thrash。詳見 `sdk/MIL/docs/grab-height-param-stall.md`。**參數鎖**（`AniloxRollForm.SettingsTabs.cs`）：`SnapshotLiveFrameCounts`/`AllAdvancedSince` 供 UI 套用期間 disable 控制項、恢復出幀/3s 逾時才解鎖 |
| `Settings/InspectionSettings.cs` | 根設定物件 |
| `Settings/Models/ChartSettings.cs` | 圖表 Y 軸範圍設定（ChartScaleMode + YMax）；StitchMode enum（Vertical / Global） |
| `Settings/Models/ImageViewSettings.cs` | 合圖方式設定（StitchMode） |
| `Settings/Models/MuraChartConfig.cs` | Mura 圖表閾值 PropertyGrid 展開代理 |
| `Settings/Models/CameraParamSettings.cs` | DCF 設定檔路徑 |
| `Settings/Models/LightSettings.cs` | 光源控制器設定（COM Port、Channel、Brightness） |
| `Settings/Models/Defaults/InspectionDefaults.cs` | Inspection 所有預設常數集中（CamOps/IoEnabled/AniloxRootPath/DcfPath/LightChannel…）— 預設值唯一來源 |
| `Settings/Models/Defaults/AcquisitionDefaults.cs` | Acquisition 預設值集中（GrabHeight/ExposureTimeUs/LineRateHz × 7 cam）；NewArray 工廠避免散落 |
| `Settings/Models/Defaults/AppModeDefaults.cs` | AppMode 預設值（Role/StorageMachineConfigFolder/StorageMachineDataPath） |
| `Settings/Models/Defaults/SystemDefaults.cs` | SystemSettings 預設值（7 cam 拓樸 NewCameraDevices；dcf 共用 InspectionDefaults.DcfPath） |
| `Settings/Utilities/DcfPathHelper.cs` | DcfPath 路徑解析：相對 `Config\Radient_Config.dcf` → 絕對 `BaseDir\Config\…`（進 MIL 前呼） |
| `Settings/Stores/SettingsStoreHelper.cs` | Settings Load/Save 共用 helper：JSON 檔案 I/O、regex 解析工具方法 |
| `Settings/Stores/AcquisitionSettingsStore.cs` | 讀寫 acquisition-settings.json |
| `UI/State/UserSessionState.cs` | UI session 持久化 → session-state.json |
| `ImageCatalog/ImageRepository.cs` | 掃描目錄建立索引 |
| `Services/AoiService.cs` | C# ↔ Native P/Invoke wrapper（ProcessImage + ComputeColumnMean） |
| `Services/InspectionLogService.cs` | 每日 CSV 寫入；GrabId = `yyMMdd-HHmmss` 時間戳格式 |
| `Services/InspectionStatisticsService.cs` | CSV 統計服務；LoadConfigForDate（按日期載入 #CFG）；LoadConfigForGrabId / LoadImagePathsForGrabId（單 grab 取 #CFG 與 .bin 路徑，供 chartDataPatch 對齊 chartReviewPatch） |
| `Services/IoState.cs` | IoState enum（FSM 狀態）+ IoSnapshot struct（IO 快照） |
| `Services/IoGrabController.cs` | IO-Grab 連動：IoState FSM、IO 追蹤、Watchdog keepalive；支援 IModbusTcpClient 注入測試。`ReadWriteTimeoutMs`（可設）= 斷線偵測下限（拔線 OS 即報錯近 0ms；斷電靠逾時 ~500ms）；`NextReconnectAtUtc` 供 UI 重連倒數 |
| `Services/CsvConfigSnapshot.cs` | 不可變設定快照（CamOps/CamPos/CamGrabHeight/CamExposureUs/CamLineRateHz/Hessian/ErrorValue/TrimHead/TrimTail） |
| `Services/HessianRescaleHelper.cs` | View-time HM rescale 共用：Ratio / IsNoOp / RescaleInPlace1D\|2D / CloneAndRescale1D\|2D — 5 個公式單一來源 |
| `Services/StorageRetentionService.cs` | 循環儲存：事件驅動（grab 結束/每 10 grab/watchdog），磁碟可用空間低於門檻時刪最舊日期資料夾影像，保留 CSV |
| `Services/CleanupFlagWatcher.cs` | Storage PC 專用：每 10 秒自主查空間 + 清理；同時輪詢 cleanup-request.flag（Inspection PC 寫入）立即觸發 |
| `Settings/Models/AppModeConfig.cs` | 機台角色設定：Role（Inspection/Storage）、StorageMachineConfigFolder、StorageMachineDataPath；Load/Save → Config\app-mode.json |
| `Services/RemoteCopyService.cs` | 背景遠端複製：ConcurrentQueue + 背景執行緒，File.Copy 含重試（3 次） |
| `Services/LightController.cs` | LTS-3DPA24 光源控制器 RS-232 通訊：AutoDetect（先試設定 COM 再掃描）、嚴格 probe（PDF §4.1.4 表-4 驗證：8-byte、cmd/ch echo、XOR checksum）、TurnOn/Off/SetBrightness，跟隨 IO Grab 開關 |
| `UI/Widgets/GrabImageStitcher.cs` | 多張影像垂直拼接 + MergeHorizontal 全域合圖（佈局 xOffset + 重疊中點分界委派 sdk `TanukiCv.Controls.MergeLayout.Compute(Midline)` 單一來源，totalW 保留自家 ALL-slots 含空缺占位版）；LoadCameraImage（internal） |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/Layout/ProportionalScaler.cs` | Form 等比例縮放（重設 Bounds + 重建 Font，非點陣縮放）。`RescaleActiveTabs`（開窗最大化後補縮作用中 tab，解 TabControl lazy-layout）。DPI 感知（`app.manifest` dpiAware=true）+ `WindowState=Maximized` 下文字原生清晰（**Wave1 已搬 sdk**） |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/Layout/RoundedLabel.cs` | 圓角晶片 Label（`Label` 子類）：反鋸齒繪圓角底 + 文字交 `base.OnPaint` 原生繪製；強制無 BorderStyle 方框。用於 IO 運作區與連線燈視覺分組（**已搬 sdk**） |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/UI/ImageCanvas.cs` | PictureBox 子類（`TanukiCv.Controls` 獨立 WinForms assembly）：zoom/pan/edge/ClampPan；自訂白底黑邊十字游標。**畫布資訊 overlay**（`ShowOverlay` 右鍵開關）：游標座標/亮度跟滑鼠、四邊 mm 範圍（X 左右 90° 旋轉、Y 上下）、右下實體倍率；座標/亮度 canvas 自繪，範圍/倍率由 `CanvasInteractionHelper.SetRangeOverlay` 推（mm 換算單一來源仍在 helper）。**效能**：游標 overlay 用 `Region` 區域失效（兩塊分離小矩形，非外接框，快速移動才不變整張）+ 重狀態同步限流 ~30fps + **整圖快取 `_viewCache`**（存「整張圖在當前 zoom 下」的點陣，**不含 pan**；pan 只改 `DrawImageUnscaled` 偏移、不重建 → FitToScreen 拖曳不再每幀重縮整張大圖；只 zoom/Image 變才重建。放大致整圖 > ~6× 控制項面積 → `_viewCache=null` 退回 per-frame 只畫可見區，放大時便宜。**cache 建構內插：縮小(zoom<1，如 fit/overview)用 `HighQualityBilinear` 平滑、放大(zoom>1)用 `NearestNeighbor` 保像素邊界清晰 —— min/mag 標準正解，避免寬圖(如合圖)縮小時 NearestNeighbor 丟像素變馬賽克**）。**滾輪 zoom 防抖**：滾動中不重建 cache、改拉伸舊 cache(`_zoom/_cacheZoom`)，停下 150ms(`_zoomSettleTimer`)才重建一次 → zoom 不每格頓。**動態 LOD（opt-in，預設關）**：`EnableLod(virtualW,virtualH,provider)` 把 zoom/pan 當「導覽虛擬全解析度圖」，停住(150ms)才請 provider 裁可見區+縮到~panel 產 tile（互動用舊 tile 拉伸）；**tile 的 GPU 重算丟背景執行緒**(in-flight guard + pending 用最新視角重算；caller 的 pinned 釋放需與 provider 互斥防 use-after-free)→ 停下不凍 UI；`LodMargin`(1.0=3×3 overscan)+ 拖出範圍節流 120ms 即時補不破圖；`RefreshLod`/`DisableLod`/`UpdateLodVirtualSize`。**`FitRelativeZoom`(opt-in)**：滾輪相對 fit(fit=1×)，上限=bitmap 1:1×`MaxZoomOverBitmap`(8)；滾輪 `×1.1^(e.Delta/120)` 正比轉動量（修卡頓時事件合併漏算）。`ZoomRelativeToFit`=螢幕縮放，**非實體倍率**(兩者互不可取代)；`PhysicalMagnification`=實體倍率(需 SetPhysicalCalibration，唯一來源，1.0x=螢幕1mm=實物1mm)。**多擊手勢(opt-in，單一來源)**：`DoubleClickFitToScreen`(雙擊 fit；只在「非 fit」才動作、已 fit 不歸零讓三擊接手)、`TripleClickPhysical1x`(三擊實體 1:1，需先 `SetPhysicalCalibration(mmPerImagePx,screenMmPerPx)`→`ZoomToOneToOne` 將點選點移到畫布中央，用 `PixelMmMapper.OneToOneZoom`)；偵測委派 `TanukiCv.Controls.MultiClickDetector` + `IsAtFitView()`。事件 `FitPerformed`/`Physical1xPerformed`/`DragStarted` 供上層做 app 專屬記錄(如 `UiActionLogger`)。**主程式回顧畫布 `camReviewMain` 已改用此**(`CanvasInteractionHelper.UpdateCanvasInfo` 餵 `SetPhysicalCalibration`；原 app 的 MultiClickDetector handler/SetPhysicalMagnification1x/IsCanvasFitToScreen 已移除)。Controls→Core 參考。詳見 `sdk/MIL/samples/MilGrabber.Monitor`（先在此驗證，未來搬回顧畫布） |

> 路徑前綴 `src/dotnet/AniloxRoll.Monitor/` 省略以節省空間。

### 測試專案

| 路徑 | 職責 |
|------|------|
| `sdk/Bridges/IoBridge/IoBridge.Core/IModbusTcpClient.cs` | Modbus TCP 介面（供 IoGrabController mock 注入） |
| `sdk/Bridges/IoBridge/IoBridge.Core/IoModuleFactory.cs` | 型號→client 單一決策點：`Create(model)`；新增型號加 case。`Modules/` 按廠商分實作 |
| `sdk/Bridges/IoBridge/IoBridge.Core/Modules/IcpDasModbusTcpClient.cs` | ICP DAS 標準 Modbus（ET 系列通用）；ET-7044 實作 |
| `tests/AniloxRoll.Monitor.Tests/` | NUnit 3.x + Moq 4.x 測試專案 |
| `CsvConfigSnapshotTests.cs` | #CFG round-trip、ContentKey |
| `AcquisitionSettingsTests.cs` | Validate fallback、JSON Save/Load |
| `InspectionLogServiceTests.cs` | CSV 寫入、#CFG 插入、Pass/Fail 判定 |
| `InspectionStatisticsServiceTests.cs` | 時間/序號統計、veto 邏輯、Period 分組 |
| `IoGrabControllerTests.cs` | FSM 狀態機：連線、邊緣偵測、故障恢復、CommLost |
| `StressTests.cs` | 長時間壓力：PLC 100 萬循環、CSV 50 萬筆、Settings 14.5 萬讀寫；STRESS_MINUTES 環境變數控制時長 |

---

## Harness 架構

兩方同步機制確保控制項命名一致：

| 層 | 檔案 | 內容 |
|----|------|------|
| Form | `AniloxRollForm.Designer.cs` | `.Text` 畫面文字 + 控制項 Name |
| 速查表 | `CLAUDE.md` §控制項速查 | 標準名稱 + Name |

**規則：**
1. 改名時兩層同時改，不可只改一層
2. 同功能控制項共用標準名稱（如 Review/Data 的【讀取資料】）
3. Commit 前驗證速查表的 Name 全部存在於 Designer.cs（改過控制項時 grep 對一輪）

---

## 檢測參數速查（PropertyGrid 屬性）

使用者在【檢測設定】看到的參數。溝通格式：「屬性名-值」（例如「欄正規值-0.2」「存檔-T」）。

**範圍**：此表只列「**what** — 參數是什麼 / 預設值 / 屬性名映射」。
**互動行為與 chart 聯動**＝以 code 為真相，配合對應 skill（`/modify-ui` 等）查閱；不在表格內重複描述。

### 0. 機台設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| 機台角色 | `AppRole` | Inspection | Inspection / Storage；變更後寫 app-mode.json，重開程式生效 |

### 1. 機台佈局

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| ── OPS (um) ── | （分隔列，唯讀） | — | — |
| Cam 1~7 | `ab_OpsCam1~ah_OpsCam7` | 24.4140625 | 各相機像素尺寸 |
| A輪速度 (m/min) | `ai_OpsSpeed` → `AniloxRollSpeedMPerMin` | 40.0 | Anilox 輪速 |
| ── Start (mm) ── | （分隔列，唯讀） | — | — |
| Cam 1~7 | `bb_StartCam1~bh_StartCam7` | 0/345/690/1035/1380/1725/2070 | 各相機起始位置 |
| ── Crop (mm) ── | （分隔列，唯讀） | — | — |
| 去頭 | `cb_CropHead` → `Crop.TrimHeadMm` | 0.0 | CAM1 左側裁切 |
| 去尾 | `cc_CropTail` → `Crop.TrimTailMm` | 0.0 | CAM7 右側裁切 |

### 2. 檢測設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| ── 演算法 ── | （分隔列，唯讀） | — | — |
| 去背演算法 | `db_Algorithm` → `Algorithm` | SingleFrameBgSub | None / SingleFrameBgSub / StandardBgSub |
| 欄正規值 | `dc_HessianMaxFactorV` → `HessianMaxFactorV` | 0.3 | V Hessian 正規化係數（capture-time baked-in） |
| 列正規值 | `dd_HessianMaxFactorH` → `HessianMaxFactorH` | 0.3 | H Hessian 正規化係數（view-time only） |
| 細線濾除 | `de_RidgeSigma` → `RidgeSigma` | 9.0 | Ridge 前 Gaussian blur sigma；越大→濾掉越多細線/雜訊（較不敏感），越小→越敏感。走每幀 json 送 native；改設定下次 grab 生效。唯一預設 `InspectionEngineConfig.DefaultRidgeSigma` |
| ── 檢出標準 ── | （分隔列，唯讀） | — | — |
| 檢出方向 | `eb_RidgeDir` → `RidgeDir` | Both | 欄 / 列 / 全部 |
| 欄平均閾值 | `ec_ErrorValueMeanV` → `ErrorValueMeanV` | 0.2 | V chart Mean 閾值線 |
| 欄最大閾值 | `ed_ErrorValueMaxV` → `ErrorValueMaxV` | 0.6 | V chart Max 閾值線 |
| 列平均閾值 | `ee_ErrorValueMeanH` → `ErrorValueMeanH` | 0.2 | H chart Mean 閾值線 |
| 列最大閾值 | `ef_ErrorValueMaxH` → `ErrorValueMaxH` | 0.6 | H chart Max 閾值線 |
| ── 背景校正 ── | （分隔列，唯讀） | — | — |
| 取時間 (sec) | `fb_BackgroundSampleSeconds` → `BackgroundSampleSeconds` | 3 | StandardBgSub 採集時間 |

### 3. 圖表設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| ── 檢測報表 ── | （分隔列，唯讀） | — | — |
| y座標 | `gb_ChartScaleMode` → `ChartScaleMode` | Auto | Auto / Fixed |
| 月產量 | `gc_YearlyYMax` → `ChartDataYieldYearlyYMax` | 50000 | 良率年圖 Y 軸上限 |
| 日產量 | `gd_MonthlyYMax` → `ChartDataYieldMonthlyYMax` | 2000 | 良率月圖 Y 軸上限 |
| 時產量 | `ge_DailyYMax` → `ChartDataYieldDailyYMax` | 300 | 良率日圖 Y 軸上限 |
| ── 主畫面 ── | （分隔列，唯讀） | — | — |
| 合圖方式 | `hb_StitchMode` → `StitchMode` | Global | Vertical / Global |
| 監控強化 | `hc_EnableMuraEnhance` → `EnableMuraEnhance` | false | 即時影像強化 Mura |
| 回顧強化 | `hd_EnableReviewEnhance` → `EnableReviewEnhance` | false | 回顧影像強化 Mura |
| 主畫面顯示 | `he_MainDisplay` → `ImageView.MainDisplay` | ImageCanvas | ImageCanvas（即時合圖）/ Waterfall（瀑布合圖）——主畫面永遠合圖、縮圖永遠即時（顯示鐵則見 app 巢狀 CLAUDE.md）。ImageCanvas：共用 `ImageDisplayView`〔sdk TanukiCv.Controls；與範例同源唯一來源〕在 camLiveMain 疊 ImageCanvas+各 cam 疊 ThumbStrip 縮圖，吃 `AniloxCamera.OnDisplayFrame` bytes→bitmap，CPU合圖+mm overlay+zoom+雙三擊+LOD。滾輪縮放：`WheelZoomFilter` 在 ImageCanvas 模式讓路（`return false`，否則全域 filter 吃掉滾輪→縮不動）。**bin↔主畫面連動**：`ImageDisplayView.ViewRangeMmChanged(left,right,top,bot)`→`LiveCameraManager.OnImageViewRange`→`OnLiveViewRange`事件→form `ApplyLiveViewRange`：欄/overview用X、列用Y zoom 同步（列需 `LiveCameraManager.RowPitchMm` 餵真 row pitch；overview 用 `LiveViewRangeProvider` 沿用同範圍→500ms 重畫不閃）。**TODO：縮圖多相機同步刷 / 關底層 MIL / 回顧側 CanvasInteractionHelper 視野計算收斂進 sdk / sample 重用曲線圖+閾值線選用 / app live 實體化 LOD/flip UI**）|
| 動態LOD | `hf_LiveLod` → `ImageView.LiveLod` | CPU | Off / GPU（TanukiCv GPU 縮）/ CPU（GrayResizeCpu 純 CPU 縮）。ImageCanvas 模式放大巨圖看細節用（顯示成本 ~180ms→~1ms），即時生效。預設 CPU＝無 GPU 機也能跑。`LiveCameraManager.SetLodMode` 套到 `ImageDisplayView.EnableLod`/`DisableLod` |

### 4. 儲存設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| 存檔 | `EnableAutoCapture` | true | 取像時自動存檔 |
| 存原圖 | `SaveOriginalBmp` | false | 額外存原始 BMP |
| Anilox 根目錄 | `AniloxRootPath` | D:\Anilox | 資料根目錄；磁碟不存在時自動 fallback 到 C:\Anilox + MessageBox + 寫回 settings |
| 存檔目錄 | （computed）| `{AniloxRoot}\Captures` | 影像 + 統計 CSV；不顯示於 PropertyGrid |
| 存背景目錄 | （computed）| `{AniloxRoot}\Bg` | StandardBgSub 背景影像；不顯示 |
| Logs 目錄 | （computed）| `{AniloxRoot}\Logs` | Resource Log；不顯示 |
| Dcf 檔 | （跟 exe 走）| `{ExeDir}\Config\Radient_Config.dcf` | MIL DCF；build 自動複製，PG 隱藏 |
| 預留空間 (GB) | `LocalMinFreeGB` | 100 | 磁碟可用空間低於此值觸發循環儲存，刪最舊日期影像（CSV 保留） |
| 遠端路徑 | `RemotePath` | \\192.168.10.20\Anilox\Captures | 遠端複製目標路徑（空=不複製）。單一 SMB share `Anilox` 子目錄 |
| 遠端設定路徑 | `RemoteConfigPath` | \\192.168.10.20\Anilox\Config | [Browsable(false)] 開發者設定；cleanup-request.flag 寫入位置 |

### 5. 光源設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| 啟用光源 | `LightEnabled` | true | 啟用 LTS-3DPA24 光源控制器 |
| COM Port | `LightComPort` | COM17 | RS-232 連接埠；啟動時先試此 port，失敗則自動掃描所有 port（找到後更新此欄位） |
| 通道 | `LightChannel` | 1 | 使用通道（單通道機型固定 1） |
| 亮度 | `LightBrightness` | 255 | 亮度（0~255） |
| 暖機延遲 (ms) | `LightWarmupMs` | 300 | 開燈後等待光源穩定的延遲；Grab 啟動前插入此延遲 |

### 6. IO 設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| IO 型號 | `IoModel` | ET-7044 | 對應 `IoModuleFactory.Create(model)`；換型號改此值。目前支援 ET-7044 |
| 啟用 IO | `IoEnabled` | true | 啟用 IO Modbus TCP |
| IO IP | `IoIp` | 192.168.255.1 | ET-7044 IP |
| IO Port | `IoPort` | 502 | Modbus TCP port |

---

## 控制項速查（標準名稱 → 程式名稱）

每個控制項有一個**標準名稱**（中文），用於對話溝通。

### 監控（tabPageLiveView）

| 標準名稱 | Name | 類型 | 畫面文字 |
|---------|------|------|---------|
| 開始抓取 | `btnLiveGrab` | Button | 開始抓取 / 停止抓取 |
| 取得背景 | `btnLiveGetBackground` | Button | 取得背景 |
| 預覽背景 | `btnLiveViewBackground` | Button | 預覽背景 |
| 監控主畫面 | `camLiveMain` | Panel | — |
| 監控縮圖1~7 | `camLive1~7` | Panel | — |
| 監控欄曲線圖（全覽） | `chartLiveColumn` | Chart | —（原 chartLivePatch 接位改名；舊單台欄 chart 已刪，曲線走全覽合併路徑） |
| 監控列曲線圖 | `chartLiveRow` | Chart | — |
| 暫停 Mura 檢測 | `lblIoDoMura`（點擊切換） | Label | DO1 MURA_DET / DO1 MURA ⏸（黃底=暫停中） |

### 回顧（tabPageReview）

| 標準名稱 | Name | 類型 | 畫面文字 |
|---------|------|------|---------|
| 讀取資料 | `btnReviewSelectFolder`（Review）/ `btnDataSelectFolder`（Data） | Button | 讀取資料 |
| 回顧縮圖1~7 | `camReview1~7` | Panel | —（Wave2：原 PictureBox→Panel，當 ImageDisplayView ThumbStrip 宿主） |
| 回顧主畫面 | `camReviewMain` | Panel | —（Wave2：原 ImageCanvas→Panel，當 ImageDisplayView 宿主；顯示/互動全由 sdk 承接） |
| 回顧欄曲線圖（全覽） | `chartReviewColumn` | Chart | —（原 chartReviewPatch 接位改名；舊單台欄 chart 已刪） |
| 回顧列曲線圖 | `chartReviewRow` | Chart | — |
| 時段群組 | `grpReviewTimePeriod` | GroupBox | 時序 |
| 時段日期（時序cb） | `cbReviewDate` | ComboBox | — |
| 時段時間（時序cb） | `cbReviewTime` | ComboBox | — |
| 單片群組 | `grpReviewGrabNav` | GroupBox | 單片 |
| 單片序號（序號cb） | `cbReviewId` | ComboBox | — |

### 報表（tabPageData）

| 標準名稱 | Name | 類型 | 畫面文字 |
|---------|------|------|---------|
| 篩選異常 | `btnDataShowFail` | Button | 篩選異常 / 顯示全部 |
| 良率卡片1~7 | `camData1~7` | Panel | — |
| Mura 空間分布圖 | `chartDataColumn` | Chart | —（原 chartDataPatch 改名） |
| 明細列表 | `listViewGrabDetail` | ListView | — |
| 序號範圍群組 | `groupBoxGrabIdRange` | GroupBox | 序號範圍 |
| 起始序號 | `cbDataIdStart` | ComboBox | — |
| 結束序號 | `cbDataIdEnd` | ComboBox | — |
| 序號選擇群組 | `grpDataSingleSheet` | GroupBox | 序號選擇 |
| 報表序號 | `cbDataId` | ComboBox | — |
| 良率年圖 | `chartDataYieldYearly` | Chart | — |
| 良率月圖 | `chartDataYieldMonthly` | Chart | — |
| 良率日圖 | `chartDataYieldDaily` | Chart | — |
| 年圖導航 | `cbDataYieldYear` | ComboBox | — |
| 月圖導航 | `cbDataYieldMonth` | ComboBox | — |
| 日圖導航 | `cbDataYieldDay` | ComboBox | — |

### 設定面板

| 標準名稱 | Name | 類型 | 畫面文字 |
|---------|------|------|---------|
| 檢測設定 | `propertyGridSettings` | PropertyGrid | — |
| 說明文字 | `helpRichText` | RichTextBox | — |
| 曝光滑桿1~7 | `trackBarExpCam1~7` + `numExpCam1~7` | TrackBar+NUD | — |
| 線掃滑桿1~7 | `trackBarLrCam1~7` + `numLrCam1~7` | TrackBar+NUD | — |
| 高度滑桿1~7 | `trackBarHtCam1~7` + `numHtCam1~7` | TrackBar+NUD | — |
| Telemetry 列表 | `listViewCameras` | ListView | — |
| 引擎常數列表 | `listViewEngine` | ListView | — |
| 圖表常數列表 | `listViewChartConst` | ListView | — |
| 硬體參數列表 | `listViewHardware` | ListView | — |
| 座標狀態列 | `lblPixelInfo` | Label | 位置:... |
| 相機數狀態 | `lblCamCount` | Label | 相機: N/7 |
| IO 狀態 | `lblIoState` | RoundedLabel | -- → 待機/取像/停止/故障/斷線/關閉/未連線（FSM 運作狀態，非連線燈；已移入 `panelIo` IO 運作區當開頭，與 DIO 燈號同組，皆圓角晶片） |
| IO 連線狀態 | `lblIoConn` | Label | ● IO: -- |
| 光源連線狀態 | `lblLightConn` | Label | ● 光源: -- |
| 儲存電腦連線狀態 | `lblStorageConn` | Label | ● 儲存電腦: -- |
| IO 燈號 | `lblIoDiAlive~lblIoDoPcBusy` | Label×5 | DI0~DO2 |

---

## Skills 路由（取代 docs/dev）

開發知識已整合至 `.claude/skills/`，按修改範圍觸發對應 skill：

| 修改範圍 | Skill | 涵蓋內容 |
|---------|-------|---------|
| UI 控制項、事件、Chart、Canvas | `/modify-ui` | Guard flags、V/H 決策矩陣、StitchMode、Chart 對齊、跨倍率 View、ProportionalScaler、**單一權威閘門**（UI 刷新時序/chart 啟動/timer 更新） |
| Data tab 統計、CSV、Period Charts | `/modify-data-stats` | 統計三模式、CSV 格式、Period Charts、跨 Tab 同步 |
| GPU pipeline、Buffer、存檔格式 | `/modify-pipeline` | CUDA pinned memory、V/H ridge、.bin 格式、ImageRepository、StandardBgSub |
| MIL 取像、相機、CLProtocol、PLC | `/modify-acquisition` | 初始化順序、CLProtocol 延遲啟動、資源釋放、SetGrabHeight、IO FSM |
| C# / WinForms 通用開發 | `/csharp-patterns` | 命名規則、Settings 持久化、WinForms 陷阱、Designer 規則 |
| Native C API 新增/修改 | `/add-native-api` | P/Invoke 宣告、C++ 實作範本 |
| 新增測試 | `/add-test` | Unit/Integration/Stress 三層分類判準 + 模板 |
| 顯示/接線改動後驗證、控制項流程追蹤 | `/verify-flows` | UI 動作流程契約（EVT）：F1~F8 flow 契約 + 偏序驗證規則 + `[Flow]` log 比對（改 LiveCameraManager/LiveDisplayCoordinator/ImageDisplayView/WaterfallView 後必跑）；含任意控制項 call chain 追蹤法 |
| 現場部署 / 網路 / SMB | `/deploy-network` | 雙網段架構、單 NIC 雙 IP、匿名 Guest SMB、編碼陷阱（bat ASCII / ps1 UTF-8 BOM / JSON UTF-8 讀法）、secedit SeDenyNetworkLogonRight |

> Build / commit / 文件同步 不是 skill：規則直接在本檔（§Build 驗證、§Git Workflow 規則、§Harness 架構），每次照做。

### 參考文件（僅供查閱，不自動載入）

| 文件 | 用途 |
|------|------|
| [`docs/dev/architecture-map.md`](docs/dev/architecture-map.md) | **Repo 架構地圖**：分層/依賴方向 + 執行期資料流 + 顯示 pipeline 鳥瞰（高層、穩定，找細節的入口）|
| [`docs/user-manual/io_diagrams.html`](docs/user-manual/io_diagrams.html) | IO FSM 視覺化（ET-7044 ↔ 設備 Nakan）|
| [`docs/user-manual/storage-flow.html`](docs/user-manual/storage-flow.html) | Storage PC 雙寫架構流程圖 |
| [`docs/user-manual/hardware-specs.html`](docs/user-manual/hardware-specs.html) | 7 相機 + Grabber + 光源 + PLC 硬體規格 |
| [`docs/dev/MIL_API_Reference.md`](docs/dev/MIL_API_Reference.md) | MIL .NET API 完整參考（常數、方法、範例）|
| [`docs/dev/system-resources.md`](docs/dev/system-resources.md) | 系統資源用量（GPU/CPU/RAM 評估）|
| [`docs/dev/stress-test-plan.md`](docs/dev/stress-test-plan.md) | 壓力測試規劃（Phase 0~6 + 監控腳本）|
| `docs/dev/CLProtocol/` / `docs/dev/Grabber/` / `docs/dev/LTS_3DPA24/` | 廠商規格書與示範程式 |
| `docs/dev/archive/` | 歷史 review 紀錄（code-review-2026-05-15 等）|

### docs/ 目錄定位

```
docs/
├── config/         ← inspection-settings.json / acquisition-settings.json / dcf 範例
├── dev/            ← 開發者/部署參考（MIL/CLProtocol API、廠商 Grabber/LTS_3DPA24 規格書、code review 紀錄）
└── user-manual/    ← 操作員說明書（UI 流程、IO 圖、硬體規格、lib/ 渲染資源）
```

開發知識統一放 `.claude/skills/`，`docs/dev/` 放大型參考文件。

**撰寫/更新 docs/ 規則**：必須比對實際程式碼確認功能狀態，不可只參考 skills（skills 本身可能滯後）。

### Skills ↔ docs 同步規則

修改功能後，若同時影響開發注意事項和使用者操作流程，**兩邊都要更新**：

| 改動 | 更新 `.claude/skills/` | 更新 `docs/` |
|------|:---:|:---:|
| 新增/修改 UI 控制項行為 | 對應 modify-* skill 的注意事項 | user-manual 操作說明 |
| 新增/修改設定參數 | csharp-patterns（持久化）+ 對應 skill | user-manual 參數說明 |
| 修改 pipeline/存檔格式 | modify-pipeline | — |
| 修改 MIL/PLC 行為 | modify-acquisition | user-manual PLC 操作 |

---

## 實作指引

### 狀態機邏輯

實作 click counting、mode transition、status flag 等狀態機時：
1. 先列出完整的**狀態轉移表**（State + Event → Next State + Action）
2. 與使用者確認後再寫 code
3. 避免用 AND 條件做安全檢查（容易漏邊界情況），優先用比值/閾值比較

### 設定檔機制（Config 跟著 exe，缺檔 → Defaults 重生）

- **所有設定 json 存 `{ExeDir}\Config\`**（`AppDomain.CurrentDomain.BaseDirectory`），不在 `%AppData%`/`%ProgramData%`。各 Store（`InspectionSettingsStore` / `AcquisitionSettingsStore` / `SystemSettings` / `AppModeConfig` / `UserSessionState`）`Load` 時 **檔案不存在 → 用 `*Defaults.cs` 產生並寫回**。
- **設定 json 不進版控、不隨原始碼複製**（`.gitignore: src/.../Config/*.json`；csproj 不放 `<None CopyToOutputDirectory>`）。理由：若 commit 並複製進 bin，開機永遠以那份舊 json 為輸入，**蓋掉 `*Defaults.cs` 這個唯一真相** → 改 Defaults 不生效、刪 bin 也回不到預設。**唯一例外是 DCF**（MIL 二進位、非 Defaults 產生，保留 csproj 複製）。
- **要回到預設值**：刪 `bin\...\Config\*.json`（或整個 bin）後啟動即重生。**改預設值只改 `*Defaults.cs`**。
- **⚠ 序列化型別陷阱**：`JavaScriptSerializer` 無法序列化 `MIL_INT` struct（寫成 `{}`）。持久化的 config model **欄位一律用 `int`/`double`/`string`**，不可用 `MIL_INT`（`CameraHardwareConfig.DevNum` 曾用 MIL_INT → 重生的 `system-settings.json` DevNum 全 `{}` → 讀回全 0 → 多 board 撞號只認到單 board）。消費端要 MIL_INT 時自行 `(MIL_INT)` 轉。

### Build 驗證

- **一律 Release|x64**（主程式 + sdk + tools 全部）。**不要 build Debug** — 開發用 agent + `Trace.WriteLine` / `Console.WriteLine` 檢查（`Debug.WriteLine` 在 Release 是 no-op，不要用）。csproj 殘留的 Debug 配置請忽略，不要選用。
- 修改 `.cs`、`.csproj`、`.sln` 後**立即 build** 確認零錯誤
- 不得在 VS 的 reserved ImportGroup 放自訂 Import
- build 入口：產品 `PICoater_AOI.sln` / sdk 工具 `sdk/Tools.sln` + 各 Bridge `sdk/Bridges/*/{Io,Light,Storage}Bridge.sln` / 單一 `xxx.csproj`（msbuild 直接 build，依賴自動拉）
- **所有 sln 只保留 `Release|x64` 單一組態**（同主方案 `PICoater_AOI.sln`，避免誤選 Debug/AnyCPU；bridge 方案亦同）
- **範例工具輸出位置**：各 sample 的 `samples\Directory.Build.props` 導向 `bin\x64\Release\tools\{io|light|storage|mil}\`；`samples\Directory.Build.targets` 把 `OutputPath` 對齊 `OutDir`（否則 VS F5「遺漏偵錯目標」）。主程式 `AniloxRoll.Monitor` Release build 經 `BuildBridgeTools` target 連帶把四個 Bridge sample 編到同位置（現場部署整包帶走）。**MIL 範例 `MilGrabber.Monitor` → `tools\mil\`**（自有 icon `sdk\MIL\samples\assets\MilGrabber.ico`，csproj `<ApplicationIcon>`；輸出資料夾含自身一份依賴 DLL，與 Bridge 工具同模式＝各工具 self-contained 可單獨跑）。**`.Core` 輸出位置不可在 monorepo 內改**（見記憶 project-core-output-shared-bin：VS 方案 P2P 參考寫死共用 bin → CS0006）
- **主方案 `PICoater_AOI.sln` 已收四個 Bridge sample 工具**（IoBridge.ManualControl/Automation、LightBridge.Control、StorageBridge.Control，掛 Bridges 方案資料夾），可直接「設定為啟動專案 → F5」run。註：`IoBridge.ManualControl` 的 ProjectGuid 與 `AniloxRoll.Monitor.Tests` 原撞 GUID，主 sln 已把 Tests 改 GUID `{C481DC6D-…}` 解衝突
- Build 命令（**必須帶 Platform=x64**，本專案依賴 AMD64 MIL SDK）：
  ```
  cat > /tmp/build.bat << 'EOFBAT'
  @echo off
  "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" %1 /p:Configuration=Release /p:Platform=x64 /v:minimal
  EOFBAT
  cmd //c "$(cygpath -w /tmp/build.bat)" "專案路徑"
  ```

---

## Git Workflow 規則

**未經使用者明確說「commit/push」，不得主動執行任何 git commit 或 git push。**

**每次 commit / push 前，必須先更新相關文件：**

1. `CLAUDE.md` — 更新關鍵檔案速查、Skills 路由
2. `.claude/skills/*.md` — 更新對應的 skill 注意事項（根據改動內容選擇）
3. `README.md` — 更新對外專案說明（若有重大功能變更）

確保文件反映最新的程式碼狀態，讓下次對話能快速上手。
