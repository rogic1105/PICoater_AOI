# PICoater AOI — Claude Code Rules

## 專案結構

```
PICoater_AOI/
├── src_dotnet/AniloxRoll.Monitor/   ← C# WinForms 應用程式
├── src_dotnet/PlcBridge/            ← PLC Modbus TCP 通訊模組
│   ├── PlcBridge.Core/              ← 共用 Modbus TCP Client + Logger（IModbusTcpClient 介面）
│   ├── PlcBridge.ManualControl/     ← 手動 DI/DO 控制工具
│   └── PlcBridge.Automation/        ← FSM 狀態機自動控制工具
├── tests/dotnet_test/AniloxRoll.Monitor.Tests/ ← NUnit 單元 + 壓力測試
├── tests/python_test/               ← Python 測試/工具腳本
├── TestRunner/                      ← 測試啟動器（雙擊 TestRunner.bat）
├── src_native/                      ← C++ pipeline 實作
└── sdk/AOI_SDK/                     ← 共用 SDK (core_cv_api / AOI.SDK)
```

## Native API

兩組 DLL，均宣告於 `src_dotnet/AniloxRoll.Monitor/Interop/NativeMethods.cs`：

| DLL | 函式 | 用途 |
|-----|------|------|
| `picoater_api.dll` | `PICoaterAPI_CreatePipeline` / `ProcessPipeline` / `DestroyPipeline` / `ComputeColumnMean` | GPU 檢測 pipeline |
| `core_cv_api.dll` | `CoreCV_AllocPinned` / `CoreCV_FreePinned` | CUDA pinned memory 管理 |
| `core_cv_api.dll` | `CoreCV_FastReadBMP` | 快速讀取 BMP（繞過 GDI+） |
| `core_cv_api.dll` | `CoreCV_Resize_GPU` | GPU 縮圖 |

## P/Invoke 架構規則

**所有 P/Invoke 宣告只能在 `AniloxRoll.Monitor/Interop/NativeMethods.cs`**，不得跨層使用 SDK 的 `AOI.SDK.Core.CoreCVWrapper`。

## 關鍵檔案速查

| 路徑 | 職責 |
|------|------|
| `Interop/NativeMethods.cs` | 唯一 P/Invoke 宣告點 |
| `ImageProcessing/NativeBufferPool.cs` | CUDA pinned buffer 管理 |
| `ImageProcessing/InspectionEngine.ImageProcessing.cs` | 縮圖/全解析度影像處理 |
| `ImageProcessing/InspectionEngineConfig.cs` | MaxWidth=16384, MaxHeight=10000, DefaultSaveResizeScale=5 |
| `ImageProcessing/BatchInspectionService.cs` | Parallel.For 批次縮圖 |
| `UI/Form/AniloxRollForm.cs` | Form 邏輯：事件、InitializeSystem、Period Charts；內含 helpers: `BindBidirectionalSync`、`GetCurveBasePath`、`PopulateAllGrabIdCombos`、`SetChartYRange`、`FindCameraById`、`MultiClickDetector`、`CheckLiveMura`（Live 即時閾值→DO_MURA）、`ApplyGlobalMergeIfNeeded`（Period 全域合圖） |
| `UI/Form/AniloxRollForm.Designer.cs` | Form 控制項佈局（VS Designer） |
| `UI/Widgets/FormInteractionHelper.cs` | UI 互動、gallery 選擇、計時；ReviewConfig 代理 |
| `UI/Widgets/CanvasInteractionHelper.cs` | Canvas zoom/pan 事件、mm 座標換算；ReviewConfig → GetEffectiveOps/Pos |
| `UI/Presenters/LiveTelemetryPresenter.cs` | 16 欄即時 Telemetry |
| `Acquisition/AniloxCamera.cs` | 單台相機 MIL 資源封裝 |
| `Acquisition/CaptureTimestampCoordinator.cs` | 多相機存檔時間戳同步 |
| `UI/Managers/LiveCameraManager.cs` | 多台相機生命週期管理、連線數監控（OnCameraCountChanged） |
| `Settings/InspectionSettings.cs` | 根設定物件 |
| `Settings/Models/ChartSettings.cs` | 圖表 Y 軸範圍設定（ChartScaleMode + YMax）；StitchMode enum（Vertical / Global） |
| `Settings/Models/ImageViewSettings.cs` | 合圖方式設定（StitchMode） |
| `Settings/Models/MuraChartConfig.cs` | Mura 圖表閾值 PropertyGrid 展開代理 |
| `Settings/Stores/AcquisitionSettingsStore.cs` | 讀寫 acquisition-settings.json |
| `UI/State/UserSessionState.cs` | UI session 持久化 → session-state.json |
| `ImageCatalog/ImageRepository.cs` | 掃描目錄建立索引 |
| `Services/AoiService.cs` | C# ↔ Native P/Invoke wrapper（ProcessImage + ComputeColumnMean） |
| `Services/InspectionLogService.cs` | 每日 CSV 寫入；GrabId = `yyMMdd-HHmmss` 時間戳格式 |
| `Services/InspectionStatisticsService.cs` | CSV 統計服務；LoadConfigForDate（按日期載入 #CFG） |
| `Services/PlcState.cs` | PlcState enum（FSM 狀態）+ PlcIoSnapshot struct（IO 快照） |
| `Services/PlcGrabController.cs` | IO-Grab 連動：PlcState FSM、IO 追蹤、Watchdog keepalive；支援 IModbusTcpClient 注入測試 |
| `Services/CsvConfigSnapshot.cs` | 不可變設定快照 |
| `UI/Widgets/GrabImageStitcher.cs` | 多張影像垂直拼接 + MergeHorizontal 全域合圖；LoadCameraImage（internal） |
| `UI/Widgets/ProportionalScaler.cs` | Form 等比例縮放 |
| `sdk/AOI_SDK/src_dotnet/AOI.SDK/UI/SmartCanvas.cs` | PictureBox 子類：zoom/pan/edge/ClampPan；自訂白底黑邊十字游標 |

> 路徑前綴 `src_dotnet/AniloxRoll.Monitor/` 省略以節省空間。

### 測試專案

| 路徑 | 職責 |
|------|------|
| `src_dotnet/PlcBridge/PlcBridge.Core/IModbusTcpClient.cs` | Modbus TCP 介面（供 PlcGrabController mock 注入） |
| `tests/dotnet_test/AniloxRoll.Monitor.Tests/` | NUnit 3.x + Moq 4.x 測試專案 |
| `CsvConfigSnapshotTests.cs` | #CFG round-trip、ContentKey |
| `AcquisitionSettingsTests.cs` | Validate fallback、JSON Save/Load |
| `InspectionLogServiceTests.cs` | CSV 寫入、#CFG 插入、Pass/Fail 判定 |
| `InspectionStatisticsServiceTests.cs` | 時間/序號統計、veto 邏輯、Period 分組 |
| `PlcGrabControllerTests.cs` | FSM 狀態機：連線、邊緣偵測、故障恢復、CommLost |
| `StressTests.cs` | 長時間壓力：PLC 100 萬循環、CSV 50 萬筆、Settings 14.5 萬讀寫；STRESS_MINUTES 環境變數控制時長 |

---

## 文件路由索引

詳細架構與模式文件位於 `docs/dev/`（Claude Code 開發用），**不會自動載入**（節省 token），需要時再讀取。`docs/user-manual/` 保留給使用者操作說明書（功能穩定後撰寫）。

**user-manual 更新規則**：撰寫或更新 `docs/user-manual/` 時，必須同時比對 `docs/dev/` 和實際程式碼，確保說明書反映當前功能狀態，不可只參考 dev 文件（dev 文件本身可能滯後）。

### 按 Tab 查文件（快速入口）

| UI 區域 | 主要文件 | 涵蓋內容 |
|---------|---------|---------|
| **tabPageLiveView（即時監控）** | [`docs/dev/architecture-ui.md`](docs/dev/architecture-ui.md) § tabPageLiveView | btnCameraGrab/Free、GPU callback 鏈、Timer 驅動、panel 點擊、方向切換、背景預覽 |
| **tabPageReview（歷史查詢）** | [`docs/dev/architecture-ui.md`](docs/dev/architecture-ui.md) § tabPageReview | btnSelectFolder、Period/GrabId 導航、Gallery→Canvas→Chart 流程、V/H 決策矩陣、合圖模式 |
| **tabPageData（檢測報表）** | [`docs/dev/architecture-data-stats.md`](docs/dev/architecture-data-stats.md) | btnSelectDataFolder、統計三模式、listView、Period Charts、時間 cascade、cross-tab 同步 |
| **tabControlRight（右側設定面板）** | [`docs/dev/architecture-ui.md`](docs/dev/architecture-ui.md) § tabControlRight | PropertyGrid 變更效果、相機參數雙向繫結、系統 ListView |
| **panelStatusBar（上方狀態列）** | [`docs/dev/architecture-ui.md`](docs/dev/architecture-ui.md) § panelStatusBar | lblCamCount/PlcState/PlcConn/PlcIo 更新觸發源 |
| **lblPixelInfo（下方狀態列）** | [`docs/dev/architecture-ui.md`](docs/dev/architecture-ui.md) § 底部狀態列 | 3 條更新路徑（Live/Review/背景預覽） |
| **InitializeSystem（啟動流程）** | [`docs/dev/architecture-ui.md`](docs/dev/architecture-ui.md) § InitializeSystem | 完整初始化順序 + FormClosed 清理 |
| **Review ↔ Data 跨 Tab 同步** | [`docs/dev/architecture-data-stats.md`](docs/dev/architecture-data-stats.md) § 跨 Tab 同步 | 雙向 GrabId 同步、時間同步、Guard flags |

### 架構文件

| 文件 | 內容 | 何時讀取 |
|------|------|---------|
| [`docs/dev/architecture-ui.md`](docs/dev/architecture-ui.md) | 右側面板、tabMain、控制項觸發關係圖、Guard flags、V/H 顯示決策矩陣、ProportionalScaler | 修改 UI 控制項、事件流程、Form 佈局時 |
| [`docs/dev/architecture-image-pipeline.md`](docs/dev/architecture-image-pipeline.md) | GPU pipeline、Buffer 映射、V/H ridge、存檔格式、.bin 格式、ImageRepository、StandardBgSub | 修改影像處理、存檔格式、pipeline 參數、背景去除模式時 |
| [`docs/dev/architecture-acquisition.md`](docs/dev/architecture-acquisition.md) | MIL 取像、AniloxCamera、CLProtocol、Telemetry、SetGrabHeight、MilGrabSample、PLC 連動（PlcState FSM、IO 快照、Watchdog） | 修改相機控制、MIL 資源管理、Telemetry、PLC 整合時 |
| [`docs/user-manual/plc_diagrams.html`](docs/user-manual/plc_diagrams.html) | PLC FSM 視覺化：State Machine / SFC / Ladder / Timing 圖（純 SVG/HTML，離線可看） | 檢視 PLC 狀態機邏輯時（使用者用瀏覽器開啟；Claude 參考 `architecture-acquisition.md` 文字版） |
| [`docs/dev/architecture-data-stats.md`](docs/dev/architecture-data-stats.md) | tabPageData、統計模式、CSV 架構、Period Charts | 修改統計功能、CSV 格式、圖表時 |
| [`docs/dev/MIL_API_Reference.md`](docs/dev/MIL_API_Reference.md) | MIL .NET API 完整參考（常數、方法、範例） | 查詢 MIL API 用法時 |

### 模式文件

| 文件 | 內容 | 何時讀取 |
|------|------|---------|
| [`docs/dev/patterns-csharp.md`](docs/dev/patterns-csharp.md) | C# 命名、WinForms Designer、PropertyGrid、TrackBar、Settings 持久化、Anchor、Exception Handling | 開發 C#/WinForms 功能時 |
| [`docs/dev/patterns-performance.md`](docs/dev/patterns-performance.md) | SmartCanvas 拖曳、Chart sync 壓制、跨倍率 View 保存、MuraChart 軸線/閾值/InnerPlotPosition、全覽圖合併 | 效能問題排查、Chart 對齊修改時 |
| [`docs/dev/patterns-mil.md`](docs/dev/patterns-mil.md) | MIL 初始化順序、CLProtocol 時序、記憶體類型、Timer 競爭、資源釋放 | MIL 相關開發、資源管理時 |

---

## 實作指引

### 狀態機邏輯

實作 click counting、mode transition、status flag 等狀態機時：
1. 先列出完整的**狀態轉移表**（State + Event → Next State + Action）
2. 與使用者確認後再寫 code
3. 避免用 AND 條件做安全檢查（容易漏邊界情況），優先用比值/閾值比較

### Build 驗證

- 修改 `.cs`、`.csproj`、`.sln` 後**立即 build** 確認零錯誤
- 不得在 VS 的 reserved ImportGroup 放自訂 Import
- Build 命令：`"/c/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe"` + 專案路徑

### UI 開發

- Chart 對齊、座標換算等優先用**即時查詢**（如 MdispInquire、InnerPlotPosition），不要用靜態快取值
- 複雜 UI 行為（zoom/pan 聯動、多 chart 同步）修改前先讀 `docs/dev/patterns-performance.md`

### UI 輸入輸出文件同步

`docs/dev/architecture-ui.md` 和 `docs/dev/architecture-data-stats.md` 記錄的是 btn/cb/event 的觸發流程與輸出對照，用於**快速定位**改動影響範圍。但文件可能滯後於程式碼：
- **正常改動**：先查文件定位相關流程，再讀 code 確認，改完後更新文件
- **流程不如預期 / debug**：**不可只信文件**，必須直接讀 `.cs` 原始碼全面追蹤實際呼叫鏈，文件僅作為起點參考

修改任何 btn/cb/event handler 的觸發流程、Chart/ListView/Canvas/Label 的更新邏輯、Guard flag、跨 Tab 同步時，**必須同步更新對應文件**：

| 改動範圍 | 更新目標 |
|---------|---------|
| tabPageLiveView 的 btn/cb/timer/callback | `docs/dev/architecture-ui.md` § tabPageLiveView |
| tabPageReview 的 btn/cb/Gallery/Canvas/Chart | `docs/dev/architecture-ui.md` § tabPageReview |
| tabPageData 的 btn/cb/ListView/Period Charts | `docs/dev/architecture-data-stats.md` |
| tabControlRight 的 PropertyGrid/TrackBar/NUD | `docs/dev/architecture-ui.md` § tabControlRight |
| panelStatusBar / lblPixelInfo 更新路徑 | `docs/dev/architecture-ui.md` § panelStatusBar / 底部狀態列 |
| Guard flag 新增或移除 | `docs/dev/architecture-ui.md` § Guard flags |
| 跨 Tab 同步邏輯（Review ↔ Data） | `docs/dev/architecture-data-stats.md` § 跨 Tab 同步 |
| InitializeSystem 順序變更 | `docs/dev/architecture-ui.md` § InitializeSystem |
| 新增/移除控制項 | `CLAUDE.md` 按 Tab 查文件路口 + 對應 docs |

---

## Git Workflow 規則

**未經使用者明確說「commit/push」，不得主動執行任何 git commit 或 git push。**

**每次 commit / push 前，必須先更新相關文件：**

1. `CLAUDE.md` — 更新路由索引、關鍵檔案速查
2. `docs/dev/*.md` — 更新對應的架構或模式文件（根據改動內容選擇）
3. `README.md` — 更新對外專案說明

確保文件反映最新的程式碼狀態，讓下次對話能快速上手。
