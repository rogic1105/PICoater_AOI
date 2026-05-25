# PICoater AOI — Claude Code Rules

## 架構原則：repo 分層（src / sdk / tools / tests / docs）

```
PICoater_AOI/
├── src/                  ← 應用程式（產品交付）
│   ├── dotnet/AniloxRoll.Monitor/  ← C# WinForms 主應用
│   └── native/                     ← C++ pipeline
├── sdk/                  ← 可獨立 split 的 library（純函式庫，無 GUI、無 exe）
│   ├── AOI/              ← 影像 SDK（native/{core_cv,cpp_utils,core_cv_api} + dotnet/AOI.SDK + benchmark/{framework,core_cv_benchmark} + third_party/stb；self-contained 可 split）
│   ├── Bridges/          ← 對外設備 / 系統橋接層
│   │   ├── IoBridge/                         ← ICP DAS ET-7044 IO module（Modbus TCP）
│   │   │   ├── IoBridge.Core/                ← library（IModbusTcpClient 介面 + ET-7044 實作）
│   │   │   └── examples/                     ← 可執行範例（ManualControl / Automation GUI）
│   │   ├── LightBridge/LightBridge.Core/     ← RS-232 LTS-3DPA24 光源
│   │   └── StorageBridge/StorageBridge.Core/ ← SMB + 檔案複製 + 循環儲存
│   ├── MIL/              ← MIL 集中區（MilGrab 取像+顯示範例 + docs：Matrox 規格書/CLProtocol）；隔離 MIL，換 grabber 整區換
│   └── docs/            ← 跨專案工程經驗（repo-style / testing pyramid / FSM）
├── tools/                ← 跨元件 / 應用層通用工具（不專屬單一 sdk 元件）
│   ├── ps/                   ← PowerShell 腳本
│   └── python/               ← Python 工具
├── tests/                ← 純自動化測試（.NET NUnit 三層 + TestRunner.bat/.ps1，量「對不對」）
├── benchmark（量「多快」）→ 不設頂層，跟被測對象住：
│       sdk/AOI/benchmark/core_cv_benchmark（通用 CV）、src/native/benchmark/picoater_pipeline_benchmark（pipeline）
├── algtest/              ← Python 演算法原型 / 可行性研究（讀 repo 外 05_QA_Validation；非自動化測試）
├── docs/                 ← 文件
│   ├── config/           ← 設定 JSON 範例
│   ├── dev/              ← 開發者參考（MIL API / 廠商規格書）
│   ├── user-manual/      ← 操作員說明（ui-flow.html / hardware-specs）
│   └── sample/           ← 範例程式（給 SDK 使用者參考的 demo）
├── assets/               ← 主程式品牌資源（AniloxRoll.ico）；sdk examples 的 icon 跟著元件走（sdk/Bridges/<X>/examples/assets/，self-contained 可帶走）
├── deploy/               ← 現場部署腳本（PowerShell + JSON）
└── .claude/skills/       ← Claude Code skills（按修改範圍觸發）

**examples/ vs tools/ 區分：**
- `sdk/<元件>/examples/` — **只服務單一 sdk 元件**的可執行範例（拿掉該元件就沒用）。展示「怎麼用這個 library」，self-contained 跟元件一起 split。如 IoBridge 的 ManualControl / Automation GUI。
- `tools/` — **跨元件 / 應用層**通用工具（log analyzer、部署 helper 等），不專屬單一元件。
- 判準：「這工具拿掉某個 sdk 元件還有用嗎？」沒用 → examples/；還有用 → tools/。
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
- `sdk/AOI/benchmark/core_cv_benchmark/` — 通用 CV micro-benchmark（C++）；跟 core_cv 同住，隨 sdk split 帶走
- `src/native/benchmark/picoater_pipeline_benchmark/` — pipeline 端到端速度（C++/CUDA：IO+傳輸+CV+resize+多相機吞吐）；緊鄰 src/native 演算法，供 agent loop 優化

**Python 演算法**：
- `algtest/` — 演算法原型 / 可行性研究（讀 repo 外 05_QA_Validation 資料；非自動化測試）

**未來擴充**：
- UI 自動化 → `tests/AniloxRoll.Monitor.UITests/`（拆 csproj）
- .NET micro-benchmark → 跟被測 .NET 元件同住（BenchmarkDotNet）

## 架構原則：SSoT 原子結構

所有「設定變更 → 副作用」流程必須遵守這個三層分工：

```
                   SettingsHub (state, SSoT)
                          │
                Changed event ↓
        ┌─────────────┬────┴────┬─────────────┐
        ↓             ↓         ↓             ↓
   PropertyGrid   image       chart 閾值    其他副作用
   (顯示)        (Mura on/off) (StripLines)  (save disk、reload、Live merge)
```

**規則：**

1. **state 集中在 `SettingsHub`** — 所有 setting 變更走 `Set` / `SetBatch` / `NotifyExternalChange`。沒有任何路徑直接 `_settings.X = ...`（bootstrap 階段例外，但加註解標記）。

2. **每個 UI 元件都是 view（搖桿）** — 按鈕、chart、滑桿、PropertyGrid 都是改 setting 的入口，不是邏輯擁有者。「點 chart 切 enhance」= 改 `EnableMuraEnhance`，副作用由 event 訂閱者跑，**不在 click handler 內 inline 跑副作用**。

3. **副作用是 view 對 event 的反應** — `FitToScreen`、`OnStitchModeChangedAsync`、`ApplyMuraEnhance`、save disk、PropertyGrid 同步顯示 — 全部訂閱 `Changed` event。view layer 自己決定怎麼更新，**不互相直接呼叫**。

4. **嚴格 transition 順序例外** — 多個 setting 同時變更且需要 atomic transition（如 chart click 同時改 StitchMode + EnableEnhance），用 `SetBatch`（save once、不 raise event），caller 自己 inline await transition 順序。這條 trade-off 要寫註解說明。

5. **變更來源要可區分** — `SettingChange.Source` 標示 `PropertyGrid`（UI 自己已 paint）vs `Programmatic`（程式碼路徑，view 要被動 refresh）。避免重複刷新造成閃爍。

**反模式：**

- `click handler` 內 inline 改多個 setting + 呼多個 apply（過去 chart click 邏輯）
- 跨層直接呼叫（如 chart click 直接 `await ApplyReviewEnhance(...)`，繞過 event）
- view 之間互相 invalidate（image view 知道 chart 存在）
- setting setter 寫 disk（save 屬 Hub 職責）

**討論 / 設計時的提問順序：**
「這是改哪個 setting？」 → 「副作用是什麼？」 → 「哪些 view 要更新？」— 而不是「按下按鈕跑哪些函式？」

## 專案結構

```
PICoater_AOI/
├── src/dotnet/AniloxRoll.Monitor/                 ← C# WinForms 應用程式
├── src/native/                                    ← C++ pipeline 實作
├── sdk/AOI/                                       ← 影像 SDK（native/{core_cv,cpp_utils,core_cv_api} + dotnet/AOI.SDK + benchmark/{framework,core_cv_benchmark}）
├── sdk/Bridges/IoBridge/IoBridge.Core/          ← Modbus TCP Client + IModbusTcpClient 介面
├── sdk/Bridges/LightBridge/LightBridge.Core/      ← LTS-3DPA24 RS-232 光源
├── sdk/Bridges/StorageBridge/StorageBridge.Core/  ← SMB 檔案複製 + 循環儲存
├── sdk/docs/                                      ← 跨專案工程經驗（atomic html）
├── tools/io-manual-control/IoBridge.ManualControl/  ← 手動 DI/DO GUI
├── tools/io-automation/IoBridge.Automation/         ← FSM 模擬 GUI
├── tests/AniloxRoll.Monitor.{Tests,Integration.Tests,Stress.Tests}/  ← NUnit 三層 + TestRunner.bat/.ps1
├── benchmark：sdk/AOI/benchmark/core_cv_benchmark/（通用 CV）+ src/native/benchmark/picoater_pipeline_benchmark/（pipeline）  ← 速度測試，跟被測對象住
├── algtest/                         ← Python 演算法原型 / 可行性研究
└── deploy/                          ← 現場部署腳本（PowerShell + JSON 參數）
    ├── storage-pc/                  ← 儲存機：固定 IP + SMB 共用 + 防火牆 + Guest 匿名（secedit）
    └── inspection-pc/               ← 檢測機：單 NIC 雙 IP 別名 + Client 端匿名 Guest SMB
```

## Native API

兩組 DLL，均宣告於 `src/dotnet/AniloxRoll.Monitor/Interop/NativeMethods.cs`：

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
| `UI/Form/AniloxRollForm.cs` | Form 邏輯：事件、InitializeSystem；內含 helpers: `BindBidirectionalSync`、`SetChartYRange`、`FindCameraById`、`CheckLiveMura`（Live 即時閾值→DO_MURA_DETECTED） |
| `UI/Form/AniloxRollForm.Designer.cs` | Form 控制項佈局（VS Designer） |
| `UI/Widgets/FormInteractionHelper.cs` | UI 互動、gallery 選擇、計時；ReviewConfig 代理 |
| `UI/Widgets/CanvasInteractionHelper.cs` | Canvas zoom/pan 事件、mm 座標換算；ReviewConfig → GetEffectiveOps/Pos |
| `UI/Widgets/EventGuard.cs` | 可重入 bool 旗標（EventGuard + EventGuardScope），using 語法自動還原 |
| `UI/Widgets/BaseCurveChartHelper.cs` | 曲線圖抽象基底：共用欄位、Build 骨架、閾值線、PostPaint 事件 |
| `UI/Widgets/ColumnCurveChartHelper.cs` | 切向（X 軸）mura 曲線圖：繼承 BaseCurveChartHelper，含 zoom 同步 |
| `UI/Widgets/RowCurveChartHelper.cs` | 法向（Y 軸）mura 曲線圖：繼承 BaseCurveChartHelper，座標反轉、InnerPlot 補償 |
| `UI/Widgets/TrackBarWheelInterceptor.cs` | TrackBar 滑鼠滾輪攔截器（從 AniloxRollForm 提取） |
| `UI/Widgets/ComboBoxWheelReverser.cs` | ComboBox 滑鼠滾輪方向反轉（從 AniloxRollForm 提取） |
| `UI/Widgets/MultiClickDetector.cs` | 多擊偵測器：雙擊/三擊辨識（從 AniloxRollForm 提取） |
| `UI/Widgets/CurveMergeHelper.cs` | 全覽圖合併演算法 + .bin 曲線讀取（UpdateOverviewChart、MergeCurves、MergeRowCurves、GetCurveBasePath） |
| `UI/Presenters/DataStatisticsPresenter.cs` | Data tab 統計邏輯：統計計算、combo 串聯、Period Charts、Mura 空間分布圖（chartMuraProfile）、跨 Tab 同步事件 |
| `UI/Presenters/ReviewStitchCoordinator.cs` | Review tab 拼接管理：LoadGrabStitchedViewAsync、合圖、ClearStitchedMode、overview chart 聯動 |
| `UI/Presenters/LiveTelemetryPresenter.cs` | 16 欄即時 Telemetry |
| `Acquisition/AniloxCamera.cs` | 單台相機 MIL 資源封裝；Global merge child-buffer 來源（其他細節見 LiveCameraManager）|
| `Acquisition/CameraFrameSaver.cs` | 存檔 I/O：SaveCapture（背景執行緒）、SaveJpegFromBytes、SaveCurveBinFromArray、Resource Log（CSV: CPU%/RAM/VRAM/GPU ms/Live/Review/StitchMode；啟動時 MergeOldResourceLogs 把「昨天以前」的小檔按日合併為 resource-monitor-yyyyMMdd.csv） |
| `Acquisition/CaptureTimestampCoordinator.cs` | 多相機存檔時間戳同步 |
| `UI/Managers/LiveCameraManager.cs` | 多台相機生命週期管理、連線數監控（OnCameraCountChanged）、即時全域合圖（EnableGlobalMerge / DisableGlobalMerge / RefreshGlobalMergeLayout）、合圖 zoom/pan（WheelZoomFilter）、合圖滑鼠座標（MergedMouseStatusHandler）、合圖 overview 聯動（TryGetMergedViewRange）、顯示同步 Timer（_mergedDisplayTimer） |
| `Settings/InspectionSettings.cs` | 根設定物件 |
| `Settings/Models/ChartSettings.cs` | 圖表 Y 軸範圍設定（ChartScaleMode + YMax）；StitchMode enum（Vertical / Global） |
| `Settings/Models/ImageViewSettings.cs` | 合圖方式設定（StitchMode） |
| `Settings/Models/MuraChartConfig.cs` | Mura 圖表閾值 PropertyGrid 展開代理 |
| `Settings/Models/CameraParamSettings.cs` | DCF 設定檔路徑 |
| `Settings/Models/LightSettings.cs` | 光源控制器設定（COM Port、Channel、Brightness） |
| `Settings/Stores/SettingsStoreHelper.cs` | Settings Load/Save 共用 helper：JSON 檔案 I/O、regex 解析工具方法 |
| `Settings/Stores/AcquisitionSettingsStore.cs` | 讀寫 acquisition-settings.json |
| `UI/State/UserSessionState.cs` | UI session 持久化 → session-state.json |
| `ImageCatalog/ImageRepository.cs` | 掃描目錄建立索引 |
| `Services/AoiService.cs` | C# ↔ Native P/Invoke wrapper（ProcessImage + ComputeColumnMean） |
| `Services/InspectionLogService.cs` | 每日 CSV 寫入；GrabId = `yyMMdd-HHmmss` 時間戳格式 |
| `Services/InspectionStatisticsService.cs` | CSV 統計服務；LoadConfigForDate（按日期載入 #CFG）；LoadConfigForGrabId / LoadImagePathsForGrabId（單 grab 取 #CFG 與 .bin 路徑，供 chartMuraProfile 對齊 chartOverview） |
| `Services/IoState.cs` | IoState enum（FSM 狀態）+ IoSnapshot struct（IO 快照） |
| `Services/IoGrabController.cs` | IO-Grab 連動：IoState FSM、IO 追蹤、Watchdog keepalive；支援 IModbusTcpClient 注入測試 |
| `Services/CsvConfigSnapshot.cs` | 不可變設定快照（CamOps/CamPos/CamGrabHeight/CamExposureUs/CamLineRateHz/Hessian/ErrorValue/TrimHead/TrimTail） |
| `Services/HessianRescaleHelper.cs` | View-time HM rescale 共用：Ratio / IsNoOp / RescaleInPlace1D\|2D / CloneAndRescale1D\|2D — 5 個公式單一來源 |
| `Services/StorageRetentionService.cs` | 循環儲存：事件驅動（grab 結束/每 10 grab/watchdog），磁碟可用空間低於門檻時刪最舊日期資料夾影像，保留 CSV |
| `Services/CleanupFlagWatcher.cs` | Storage PC 專用：每 10 秒自主查空間 + 清理；同時輪詢 cleanup-request.flag（Inspection PC 寫入）立即觸發 |
| `Settings/Models/AppModeConfig.cs` | 機台角色設定：Role（Inspection/Storage）、LocalConfigFolder、StorageFolderPath；Load/Save → Config\app-mode.json |
| `Services/RemoteCopyService.cs` | 背景遠端複製：ConcurrentQueue + 背景執行緒，File.Copy 含重試（3 次） |
| `Services/LightController.cs` | LTS-3DPA24 光源控制器 RS-232 通訊：AutoDetect（先試設定 COM 再掃描）、嚴格 probe（PDF §4.1.4 表-4 驗證：8-byte、cmd/ch echo、XOR checksum）、TurnOn/Off/SetBrightness，跟隨 IO Grab 開關 |
| `UI/Widgets/GrabImageStitcher.cs` | 多張影像垂直拼接 + MergeHorizontal 全域合圖；LoadCameraImage（internal） |
| `UI/Widgets/ProportionalScaler.cs` | Form 等比例縮放 |
| `sdk/AOI/dotnet/AOI.SDK/UI/SmartCanvas.cs` | PictureBox 子類：zoom/pan/edge/ClampPan；自訂白底黑邊十字游標 |

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

三方同步機制確保控制項命名在所有層級一致：

| 層 | 檔案 | 內容 |
|----|------|------|
| Form | `AniloxRollForm.Designer.cs` | `.Text` 畫面文字 |
| 速查表 | `CLAUDE.md` §控制項速查 | 標準名稱 + Name |
| 流程圖 | `docs/user-manual/ui-flow.html` | 【】包裹的控制項名稱 |

**規則：**
1. 改名時三層同時改，不可只改一層
2. 【】= 可操作控制項（使用者會點/選/拖的），被動顯示的不加
3. 同功能控制項共用標準名稱（如 Review/Data 的【讀取資料】）
4. Commit 前跑 `/update-docs` — 驗證速查表的 Name 全部存在於 Designer.cs

---

## 檢測參數速查（PropertyGrid 屬性）

使用者在【檢測設定】看到的參數。溝通格式：「屬性名-值」（例如「垂直正規值-0.2」「存檔-T」）。

**範圍**：此表只列「**what** — 參數是什麼 / 預設值 / 屬性名映射」。
**互動行為與 chart 聯動**統一在 `docs/user-manual/ui-flow.html`（單一真相）；不在表格內重複描述。

### 0. 機台設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| 機台角色 | `AppRole` | Inspection | Inspection / Storage；變更後寫 app-mode.json，重開程式生效 |

### 1. 機台佈局

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| ── OPS (um) ── | （分隔列，唯讀） | — | — |
| Cam 1~7 | `ab_OpsCam1~ah_OpsCam7` | 33.0 | 各相機像素尺寸 |
| A輪速度 (m/min) | `ai_OpsSpeed` → `AniloxRollSpeedMPerMin` | 40.0 | Anilox 輪速 |
| ── Start (mm) ── | （分隔列，唯讀） | — | — |
| Cam 1~7 | `bb_StartCam1~bh_StartCam7` | 0/400/800/1200/1600/2000/2400 | 各相機起始位置 |
| ── Crop (mm) ── | （分隔列，唯讀） | — | — |
| 去頭 | `cb_CropHead` → `Crop.TrimHeadMm` | 0.0 | CAM1 左側裁切 |
| 去尾 | `cc_CropTail` → `Crop.TrimTailMm` | 0.0 | CAM7 右側裁切 |

### 2. 檢測設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| ── 演算法 ── | （分隔列，唯讀） | — | — |
| 去背演算法 | `db_Algorithm` → `Algorithm` | SingleFrameBgSub | None / SingleFrameBgSub / StandardBgSub |
| 垂直正規值 | `dc_HessianMaxFactorV` → `HessianMaxFactorV` | 0.3 | V Hessian 正規化係數（capture-time baked-in） |
| 水平正規值 | `dd_HessianMaxFactorH` → `HessianMaxFactorH` | 0.3 | H Hessian 正規化係數（view-time only） |
| ── 檢出標準 ── | （分隔列，唯讀） | — | — |
| 檢出方向 | `eb_RidgeDir` → `RidgeDir` | Both | 垂直 / 水平 / 全部 |
| 垂直平均閾值 | `ec_ErrorValueMeanV` → `ErrorValueMeanV` | 0.2 | V chart Mean 閾值線 |
| 垂直最大閾值 | `ed_ErrorValueMaxV` → `ErrorValueMaxV` | 0.4 | V chart Max 閾值線 |
| 水平平均閾值 | `ee_ErrorValueMeanH` → `ErrorValueMeanH` | 0.2 | H chart Mean 閾值線 |
| 水平最大閾值 | `ef_ErrorValueMaxH` → `ErrorValueMaxH` | 0.4 | H chart Max 閾值線 |
| ── 背景校正 ── | （分隔列，唯讀） | — | — |
| 取時間 (sec) | `fb_BackgroundSampleSeconds` → `BackgroundSampleSeconds` | 3 | StandardBgSub 採集時間 |

### 3. 圖表設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| ── 檢測報表 ── | （分隔列，唯讀） | — | — |
| y座標 | `gb_ChartScaleMode` → `ChartScaleMode` | Auto | Auto / Fixed |
| 月產量 | `gc_YearlyYMax` → `ChartYearlyYMax` | 50000 | 良率年圖 Y 軸上限 |
| 日產量 | `gd_MonthlyYMax` → `ChartMonthlyYMax` | 2000 | 良率月圖 Y 軸上限 |
| 時產量 | `ge_DailyYMax` → `ChartDailyYMax` | 300 | 良率日圖 Y 軸上限 |
| ── 主畫面 ── | （分隔列，唯讀） | — | — |
| 合圖方式 | `hb_StitchMode` → `StitchMode` | Global | Vertical / Global |
| 監控強化 | `hc_EnableMuraEnhance` → `EnableMuraEnhance` | false | 即時影像強化 Mura |
| 回顧強化 | `hd_EnableReviewEnhance` → `EnableReviewEnhance` | false | 回顧影像強化 Mura |

### 4. 儲存設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| 存檔 | `EnableAutoCapture` | true | 取像時自動存檔 |
| 存原圖 | `SaveOriginalBmp` | false | 額外存原始 BMP |
| Anilox 根目錄 | `AniloxRootPath` | D:\Anilox | 資料根目錄；磁碟不存在時自動 fallback 到 C:\Anilox + MessageBox + 寫回 settings |
| 存檔目錄 | （computed）| `{AniloxRoot}\Captures` | 影像 + 統計 CSV；不顯示於 PropertyGrid |
| 存背景目錄 | （computed）| `{AniloxRoot}\Bg` | StandardBgSub 背景影像；不顯示 |
| Logs 目錄 | （computed）| `{AniloxRoot}\Logs` | Resource Log；不顯示 |
| Dcf 目錄 | （computed）| `{AniloxRoot}\Dcf` | MIL DCF；不顯示 |
| 預留空間 (GB) | `LocalMinFreeGB` | 100 | 磁碟可用空間低於此值觸發循環儲存，刪最舊日期影像（CSV 保留） |
| 遠端路徑 | `RemotePath` | \\192.168.10.20\Anilox\Captures | 遠端複製目標路徑（空=不複製）。單一 SMB share `Anilox` 子目錄 |
| 遠端設定路徑 | `RemoteConfigPath` | \\192.168.10.20\Anilox\Config | [Browsable(false)] 開發者設定；cleanup-request.flag 寫入位置 |

### 5. 相機設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| 設定檔 | `DcfPath` | D:\AniloxCaptures\dcf\Radient_Config.dcf | MIL Digitizer DCF 檔案路徑 |

### 6. 光源設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| 啟用光源 | `LightEnabled` | true | 啟用 LTS-3DPA24 光源控制器 |
| COM Port | `LightComPort` | COM17 | RS-232 連接埠；啟動時先試此 port，失敗則自動掃描所有 port（找到後更新此欄位） |
| 通道 | `LightChannel` | 1 | 使用通道（單通道機型固定 1） |
| 亮度 | `LightBrightness` | 255 | 亮度（0~255） |
| 暖機延遲 (ms) | `LightWarmupMs` | 300 | 開燈後等待光源穩定的延遲；Grab 啟動前插入此延遲 |

### 7. IO 設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| IO 型號 | `IoModel` | ET-7044 | 對應 `IoModuleFactory.Create(model)`；換型號改此值。目前支援 ET-7044 |
| 啟用 IO | `IoEnabled` | true | 啟用 IO Modbus TCP |
| IO IP | `IoIp` | 192.168.255.1 | ET-7044 IP |
| IO Port | `IoPort` | 502 | Modbus TCP port |

---

## 控制項速查（標準名稱 → 程式名稱）

每個控制項有一個**標準名稱**（中文），用於對話溝通。流程圖（`docs/user-manual/ui-flow.html`）也使用這些名稱。

### 即時監控（tabPageLiveView）

| 標準名稱 | Name | 類型 | 畫面文字 |
|---------|------|------|---------|
| 開始抓取 | `btnCameraGrab` | Button | 開始抓取 / 停止抓取 |
| 取得背景 | `btnGetBackground` | Button | 取得背景 |
| 預覽背景 | `btnViewBackground` | Button | 預覽背景 |
| 監控主畫面 | `panelMainDisplay` | Panel | — |
| 監控縮圖1~7 | `panelLiveCam1~7` | Panel | — |
| 監控切向曲線圖 | `muraChartVerticalLive` | Chart | — |
| 監控法向曲線圖 | `muraChartHorizontalLive` | Chart | — |
| 監控全覽圖 | `chartLiveOverview` | Chart | — |
| 暫停 Mura 檢測 | `lblIoDoMura`（點擊切換） | Label | DO1 MURA_DET / DO1 MURA ⏸（黃底=暫停中） |

### 歷史查詢（tabPageReview）

| 標準名稱 | Name | 類型 | 畫面文字 |
|---------|------|------|---------|
| 讀取資料 | `btnSelectFolder`（Review）/ `btnSelectDataFolder`（Data） | Button | 讀取資料 |
| 回顧縮圖1~7 | `pbCam1~7` | PictureBox | — |
| 回顧主畫面 | `canvasMain` | SmartCanvas | — |
| 回顧切向曲線圖 | `chartMuraVertical` | Chart | — |
| 回顧法向曲線圖 | `chartMuraHorizontal` | Chart | — |
| 回顧全覽圖 | `chartOverview` | Chart | — |
| 時段群組 | `grpReviewTimePeriod` | GroupBox | 時序 |
| 時段日期（時序cb） | `cbDate` | ComboBox | — |
| 時段時間（時序cb） | `cbTime` | ComboBox | — |
| 上一時段（上下鍵） | `btnPeriodPrev` | Button | < |
| 下一時段（上下鍵） | `btnPeriodNext` | Button | > |
| 單片群組 | `grpReviewGrabNav` | GroupBox | 單片 |
| 單片序號（序號cb） | `cbReviewGrabId` | ComboBox | — |
| 上一序號（上下鍵） | `btnGrabIdPrev` | Button | < |
| 下一序號（上下鍵） | `btnGrabIdNext` | Button | > |

### 檢測報表（tabPageData）

| 標準名稱 | Name | 類型 | 畫面文字 |
|---------|------|------|---------|
| 篩選異常 | `btnShowFail` | Button | 篩選異常 / 顯示全部 |
| 良率卡片1~7 | `panelStatCam1~7` | Panel | — |
| Mura 空間分布圖 | `chartMuraProfile` | Chart | — |
| 明細列表 | `listViewGrabDetail` | ListView | — |
| 序號範圍群組 | `groupBoxGrabIdRange` | GroupBox | 序號範圍 |
| 起始序號 | `cbGrabIdStart` | ComboBox | — |
| 結束序號 | `cbGrabIdEnd` | ComboBox | — |
| 序號選擇群組 | `grpDataSingleSheet` | GroupBox | 序號選擇 |
| 報表序號 | `cbDataGrabId` | ComboBox | — |
| 報表上一序號 | `btnGrabIdDataPrev` | Button | < |
| 報表下一序號 | `btnGrabIdDataNext` | Button | > |
| 時序範圍群組 | `groupBoxTimeRange` | GroupBox | 時序範圍 |
| 起始日期/時間 | `cbStartDate/cbStartTime` | ComboBox | — |
| 結束日期/時間 | `cbEndDate/cbEndTime` | ComboBox | — |
| 良率年圖 | `chartYearly` | Chart | — |
| 良率月圖 | `chartMonthly` | Chart | — |
| 良率日圖 | `chartDaily` | Chart | — |
| 年圖導航 | `cbChartYear` | ComboBox | — |
| 月圖導航 | `cbChartMonth` | ComboBox | — |
| 日圖導航 | `cbChartDay` | ComboBox | — |

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
| IO 狀態 | `lblIoState` | Label | ● 狀態: -- |
| IO 連線狀態 | `lblIoConn` | Label | ● IO: -- |
| 光源連線狀態 | `lblLightConn` | Label | ● 光源: -- |
| 儲存電腦連線狀態 | `lblStorageConn` | Label | ● 儲存電腦: -- |
| IO 燈號 | `lblIoDiAlive~lblIoDoPcBusy` | Label×5 | DI0~DO2 |

---

## Skills 路由（取代 docs/dev）

開發知識已整合至 `.claude/skills/`，按修改範圍觸發對應 skill：

| 修改範圍 | Skill | 涵蓋內容 |
|---------|-------|---------|
| UI 控制項、事件、Chart、Canvas | `/modify-ui` | Guard flags、V/H 決策矩陣、StitchMode、Chart 對齊、跨倍率 View、ProportionalScaler |
| Data tab 統計、CSV、Period Charts | `/modify-data-stats` | 統計三模式、CSV 格式、Period Charts、跨 Tab 同步 |
| GPU pipeline、Buffer、存檔格式 | `/modify-pipeline` | CUDA pinned memory、V/H ridge、.bin 格式、ImageRepository、StandardBgSub |
| MIL 取像、相機、CLProtocol、PLC | `/modify-acquisition` | 初始化順序、CLProtocol 延遲啟動、資源釋放、SetGrabHeight、IO FSM |
| C# / WinForms 通用開發 | `/csharp-patterns` | 命名規則、Settings 持久化、WinForms 陷阱、Designer 規則 |
| Native C API 新增/修改 | `/add-native-api` | P/Invoke 宣告、C++ 實作範本 |
| 效能瓶頸排查 | `/perf-diagnose` | Stopwatch 計時、IO/GPU/UI 分層診斷 |
| 追蹤 btn/cb/event I/O 流程 | `/review-flow <控制項>` | 完整 call chain 追蹤 + 文件比對 |
| Build 驗證 | `/build` | Release x64 完整 build（**一律 Release，不 build Debug**）|
| Commit 前文件更新 | `/update-docs` | 批次更新 CLAUDE.md + skills |
| 提交推送 | `/commit` | build + 文件 + conventional commit |
| 控制項別名記錄 | `/alias-log` | 對話中新稱呼 → 更新速查表 + 建議標準名稱 |
| 現場部署 / 網路 / SMB | `/deploy-network` | 雙網段架構、單 NIC 雙 IP、匿名 Guest SMB、編碼陷阱（bat ASCII / ps1 UTF-8 BOM / JSON UTF-8 讀法）、secedit SeDenyNetworkLogonRight |

### 參考文件（僅供查閱，不自動載入）

| 文件 | 用途 |
|------|------|
| [`docs/user-manual/ui-flow.html`](docs/user-manual/ui-flow.html) | **UI 互動流程設計檔（單一真相）**，1800+ 行，瀏覽器開啟 |
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
├── sample/        ← 示範程式（AOI.SDK.TestApp — 不參與 build，未來分離回 AOI_SDK repo）
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

### Build 驗證

- **一律 Release|x64**（主程式 + sdk + tools 全部）。**不要 build Debug** — 開發用 agent + `Trace.WriteLine` / `Console.WriteLine` 檢查（`Debug.WriteLine` 在 Release 是 no-op，不要用）。csproj 殘留的 Debug 配置請忽略，不要選用。
- 修改 `.cs`、`.csproj`、`.sln` 後**立即 build** 確認零錯誤
- 不得在 VS 的 reserved ImportGroup 放自訂 Import
- build 入口：產品 `PICoater_AOI.sln` / sdk 工具 `sdk/Tools.sln` / 單一 `xxx.csproj`（msbuild 直接 build，依賴自動拉）
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
