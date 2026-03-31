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
| `UI/Form/AniloxRollForm.cs` | Form 邏輯：事件、InitializeSystem、Period Charts；內含 helpers: `BindBidirectionalSync`（TrackBar↔NUD 同步）、`GetCurveBasePath`、`PopulateAllGrabIdCombos`、`SetChartYRange`、`FindCameraById`、`MultiClickDetector`（inner class） |
| `UI/Form/AniloxRollForm.Designer.cs` | Form 控制項佈局（VS Designer） |
| `UI/Widgets/FormInteractionHelper.cs` | UI 互動、gallery 選擇、計時；ReviewConfig 代理 |
| `UI/Widgets/CanvasInteractionHelper.cs` | Canvas zoom/pan 事件、mm 座標換算；ReviewConfig → GetEffectiveOps/Pos |
| `UI/Presenters/LiveTelemetryPresenter.cs` | 16 欄即時 Telemetry |
| `Acquisition/AniloxCamera.cs` | 單台相機 MIL 資源封裝 |
| `Acquisition/CaptureTimestampCoordinator.cs` | 多相機存檔時間戳同步 |
| `UI/Managers/LiveCameraManager.cs` | 多台相機生命週期管理、連線數監控（OnCameraCountChanged） |
| `Settings/InspectionSettings.cs` | 根設定物件 |
| `Settings/Models/ChartSettings.cs` | 圖表 Y 軸範圍設定（ChartScaleMode + YMax） |
| `Settings/Models/MuraChartConfig.cs` | Mura 圖表閾值 PropertyGrid 展開代理 |
| `Settings/Stores/AcquisitionSettingsStore.cs` | 讀寫 acquisition-settings.json |
| `UI/State/UserSessionState.cs` | UI session 持久化 → session-state.json |
| `ImageCatalog/ImageRepository.cs` | 掃描目錄建立索引 |
| `Services/AoiService.cs` | C# ↔ Native P/Invoke wrapper（ProcessImage + ComputeColumnMean） |
| `Services/InspectionLogService.cs` | 每日 CSV 寫入 |
| `Services/InspectionStatisticsService.cs` | CSV 統計服務；LoadConfigForDate（按日期載入 #CFG） |
| `Services/PlcState.cs` | PlcState enum（FSM 狀態）+ PlcIoSnapshot struct（IO 快照） |
| `Services/PlcGrabController.cs` | PLC-Grab 連動：PlcState FSM、IO 追蹤、Watchdog keepalive；支援 IModbusTcpClient 注入測試 |
| `Services/CsvConfigSnapshot.cs` | 不可變設定快照 |
| `UI/Widgets/GrabImageStitcher.cs` | 多張影像垂直拼接 |
| `UI/Widgets/ProportionalScaler.cs` | Form 等比例縮放 |
| `sdk/AOI_SDK/src_dotnet/AOI.SDK/UI/SmartCanvas.cs` | PictureBox 子類：zoom/pan/edge/ClampPan |

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

詳細架構與模式文件位於 `docs/`，**不會自動載入**（節省 token），需要時再讀取：

### 架構文件

| 文件 | 內容 | 何時讀取 |
|------|------|---------|
| [`docs/architecture-ui.md`](docs/architecture-ui.md) | 右側面板、tabMain、控制項觸發關係圖、Guard flags、V/H 顯示決策矩陣、ProportionalScaler | 修改 UI 控制項、事件流程、Form 佈局時 |
| [`docs/architecture-image-pipeline.md`](docs/architecture-image-pipeline.md) | GPU pipeline、Buffer 映射、V/H ridge、存檔格式、.bin 格式、ImageRepository、StandardBgSub | 修改影像處理、存檔格式、pipeline 參數、背景去除模式時 |
| [`docs/architecture-acquisition.md`](docs/architecture-acquisition.md) | MIL 取像、AniloxCamera、CLProtocol、Telemetry、SetGrabHeight、MilGrabSample、PLC 連動（PlcState FSM、IO 快照、Watchdog） | 修改相機控制、MIL 資源管理、Telemetry、PLC 整合時 |
| [`docs/plc_diagrams.html`](docs/plc_diagrams.html) | PLC FSM 視覺化：State Machine / SFC / Ladder / Timing 圖（純 SVG/HTML，離線可看） | 檢視 PLC 狀態機邏輯時 |
| [`docs/architecture-data-stats.md`](docs/architecture-data-stats.md) | tabPageData、統計模式、CSV 架構、Period Charts | 修改統計功能、CSV 格式、圖表時 |
| [`docs/MIL_API_Reference.md`](docs/MIL_API_Reference.md) | MIL .NET API 完整參考（常數、方法、範例） | 查詢 MIL API 用法時 |

### 模式文件

| 文件 | 內容 | 何時讀取 |
|------|------|---------|
| [`docs/patterns-csharp.md`](docs/patterns-csharp.md) | C# 命名、WinForms Designer、PropertyGrid、TrackBar、Settings 持久化、Anchor、Exception Handling | 開發 C#/WinForms 功能時 |
| [`docs/patterns-performance.md`](docs/patterns-performance.md) | SmartCanvas 拖曳、Chart sync 壓制、跨倍率 View 保存、MuraChart 軸線/閾值/InnerPlotPosition、全覽圖合併 | 效能問題排查、Chart 對齊修改時 |
| [`docs/patterns-mil.md`](docs/patterns-mil.md) | MIL 初始化順序、CLProtocol 時序、記憶體類型、Timer 競爭、資源釋放 | MIL 相關開發、資源管理時 |

---

## Git Workflow 規則

**未經使用者明確說「commit/push」，不得主動執行任何 git commit 或 git push。**

**每次 commit / push 前，必須先更新相關文件：**

1. `CLAUDE.md` — 更新路由索引、關鍵檔案速查
2. `docs/*.md` — 更新對應的架構或模式文件（根據改動內容選擇）
3. `README.md` — 更新對外專案說明

確保文件反映最新的程式碼狀態，讓下次對話能快速上手。
