# PICoater AOI — Claude Code Rules

## 專案結構

```
PICoater_AOI/
├── src_dotnet/AniloxRoll.Monitor/   ← C# WinForms 應用程式
├── src_native/                      ← C++ pipeline 實作
└── sdk/AOI_SDK/                     ← 共用 SDK (core_cv_api / AOI.SDK)
```

## Native API

兩組 DLL，均宣告於 `src_dotnet/AniloxRoll.Monitor/Interop/NativeMethods.cs`：

| DLL | 函式 | 用途 |
|-----|------|------|
| `picoater_api.dll` | `PICoaterAPI_CreatePipeline` / `ProcessPipeline` / `DestroyPipeline` | GPU 檢測 pipeline |
| `core_cv_api.dll` | `CoreCV_AllocPinned` / `CoreCV_FreePinned` | CUDA pinned memory 管理 |
| `core_cv_api.dll` | `CoreCV_FastReadBMP` | 快速讀取 BMP（繞過 GDI+） |
| `core_cv_api.dll` | `CoreCV_Resize_GPU` | GPU 縮圖 |

## P/Invoke 架構規則

**所有 P/Invoke 宣告只能在 `AniloxRoll.Monitor/Interop/NativeMethods.cs`**，不得跨層使用 SDK 的 `AOI.SDK.Core.CoreCVWrapper`。

```csharp
// 正確
using AniloxRoll.Monitor.Core.Interop;
NativeMethods.CoreCV_FastReadBMP(...)

// 錯誤 — 跨層引用 SDK
using AOI.SDK.Core;
CoreCVWrapper.CoreCV_FastReadBMP(...)
```

## 效能設計原則

### 縮圖優先 / 延遲全解析度
1. `BatchInspectionService.ProcessBatch` → Parallel.For 平行產生 7 張縮圖（GPU resize）
2. 縮圖顯示於 `ThumbnailGridPresenter`
3. 使用者點選 → `OnGallerySelectionChanged` → `RunInspectionFullRes`（同步、on-demand）

目標：使用者感知延遲 ≤ 0.5 秒

### CUDA Pinned Memory

`NativeBufferPool` 的所有 buffer 使用 `CoreCV_AllocPinned`（cudaMallocHost），確保 H↔D memcpy 走 DMA 加速：

```
_inputBuffer, _muraBuffer, _ridgeBuffer, _thumbnailBuffer, _curveMeanBuffer, _curveMaxBuffer
```

### 影像處理順序

```
CoreCV_FastReadBMP  →  AoiService.ProcessImage  →  CoreCV_Resize_GPU  →  Create8bppBitmap
     IO (pinned)          GPU pipeline (選用)          GPU 縮圖 (選用)        CPU bitmap
```

## 效能計時

`InspectionEngine.RunInspectionFullRes` 和 `FormInteractionHelper.OnGallerySelectionChanged` 已有 Stopwatch 計時輸出：

```
[FullRes] mode=True  | IO=  17ms | GPU=  22ms | BMP=  28ms | Copy=  0ms | Total=   69ms  (14288x9003)
[OnSelect] Cam1 | FullRes=   69ms | Canvas=   0ms | Chart=  46ms | Total=  116ms
```

懷疑效能問題時，先看計時輸出，再對症下藥。使用 `/perf-diagnose` skill 協助分析。

## 關鍵檔案速查

| 路徑 | 職責 |
|------|------|
| `src_dotnet/AniloxRoll.Monitor/Interop/NativeMethods.cs` | 唯一 P/Invoke 宣告點 |
| `src_dotnet/AniloxRoll.Monitor/ImageProcessing/NativeBufferPool.cs` | CUDA pinned buffer 管理 |
| `src_dotnet/AniloxRoll.Monitor/ImageProcessing/InspectionEngine.ImageProcessing.cs` | 縮圖/全解析度影像處理 |
| `src_dotnet/AniloxRoll.Monitor/ImageProcessing/InspectionEngineConfig.cs` | MaxWidth=16384, MaxHeight=10000, DefaultSaveResizeScale=5, DefaultSaveJpgQuality=90 |
| `src_dotnet/AniloxRoll.Monitor/ImageProcessing/BatchInspectionService.cs` | Parallel.For 批次縮圖 |
| `src_dotnet/AniloxRoll.Monitor/UI/Widgets/FormInteractionHelper.cs` | UI 互動、gallery 選擇、計時；`NavigateToDateTime(DateTime)` → `_timeNavigator.NavigateTo(dt)` |
| `src_dotnet/AniloxRoll.Monitor/UI/Form/AniloxRollForm.cs` | Form 邏輯：事件、InitializeSystem、右側面板初始化、SyncCameraParamsFromHardware、TelemetryTimer、Period Charts (InitOneChart/FillPeriodChart/PopulateChartNavigators) |
| `src_dotnet/AniloxRoll.Monitor/UI/Presenters/LiveTelemetryPresenter.cs` | listViewCameras 16 欄即時 Telemetry（每 500ms 更新） |
| `src_dotnet/AniloxRoll.Monitor/UI/Form/AniloxRollForm.Designer.cs` | Form 控制項佈局（VS Designer 管理）；主要控制項已設 Anchor 支援視窗放大縮小 |
| `src_dotnet/AniloxRoll.Monitor/Acquisition/AniloxCamera.cs` | 單台相機 MIL 資源封裝（CLProtocol、曝光、GrabHeight、Telemetry）；`TrySaveCapture` 磁碟 I/O 走 `Task.Run` 非同步 |
| `src_dotnet/AniloxRoll.Monitor/Acquisition/CaptureTimestampCoordinator.cs` | 同 Line Rate 相機共用存檔時間戳（`Coordinate(lineRateHz, now)` → 100ms 內同組共用 `.fff`） |
| `src_dotnet/AniloxRoll.Monitor/UI/Managers/LiveCameraManager.cs` | 多台相機生命週期管理（Allocate/Grab/Free）；`OnLiveCurveData` 事件轉發至 Form |
| `src_dotnet/AniloxRoll.Monitor/Settings/InspectionSettings.cs` | 根設定物件（MachineLayout/Acquisition/Recipe/Storage） |
| `src_dotnet/AniloxRoll.Monitor/Settings/Models/AcquisitionSettings.cs` | 取像設定（各 7 台陣列：CameraGrabHeight[7]/CameraExposureTimeUs[7]/CameraLineRateHz[7]） |
| `src_dotnet/AniloxRoll.Monitor/Settings/Stores/AcquisitionSettingsStore.cs` | 讀寫 `Config\acquisition-settings.json`（tabPageCamera 的唯一持久化入口） |
| `src_dotnet/AniloxRoll.Monitor/Settings/System/SystemSettings.cs` | 相機硬體拓樸設定（CameraHardwareConfig 清單） |
| `src_dotnet/AniloxRoll.Monitor/UI/State/UserSessionState.cs` | UI session 狀態持久化（LastDataPath / 時間篩選 / LastEnableImageProcessing / LastGrabIdNum）→ `Config\session-state.json` |
| `src_dotnet/AniloxRoll.Monitor/UI/Navigators/DateTimeNavigator.cs` | 時間篩選 ComboBox cascade 管理；`NavigateTo(DateTime)` 將目標時間存入 SessionState 再重新 cascade 初始化 |
| `src_dotnet/AniloxRoll.Monitor/UI/Widgets/FormInteractionHelper.cs` | `SelectAndLoadFolder`：選擇資料夾後先 `SetLastDataPath`+`Save()` 再掃描檔案 |
| `src_dotnet/AniloxRoll.Monitor/Acquisition/Inspection/InspectionData.cs` | 檢測結果資料物件（Image/MuraCurveMean/MuraCurveMax/IsCompressedJpeg/ScaleFactor） |
| `src_dotnet/AniloxRoll.Monitor/ImageCatalog/ImageRepository.cs` | 掃描目錄建立索引，同時掃 `*_raw.jpg` + `*.bmp`；regex 解析 `yyyyMMdd_HHmmss.fff-CamId`；秒下拉顯示 `ss.fff` |
| `src_dotnet/AniloxRoll.Monitor/Services/CsvConfigSnapshot.cs` | 不可變設定快照（17 欄：7×CamOps + 7×CamPos + HessianMaxFactor + ErrorValueMean + ErrorValueMax）；`ToCsvLine()` → `#CFG,...`；`TryParse` 反序列化；`ContentKey` 變更偵測 |
| `src_dotnet/AniloxRoll.Monitor/Services/InspectionLogService.cs` | 抓圖事件編號（A00001 起）+ 每日 CSV 寫入；Header 9 欄 `Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs`；`#CFG` 列記錄設定快照（每日首筆 + 設定變更時插入）；`ForceWriteConfig` 供抓圖中途設定變更 |
| `src_dotnet/AniloxRoll.Monitor/Services/InspectionStatisticsService.cs` | CSV 統計服務：`Compute`、`ComputeByGrabIdRange`、`ComputeDetailedByGrabIdRange`、`LoadGrabIdInfos`、`LoadAvailableTimes`、**`LoadImagePathsForGrabId`**（回傳 `Dictionary<int,List<string>>` camId→排序路徑）、**`LoadConfigForGrabId`**（回傳該序號最近的 `#CFG` 快照） |
| `src_dotnet/AniloxRoll.Monitor/UI/Widgets/GrabImageStitcher.cs` | `StitchCamera(sortedPaths, bmpResizeScale, bmpLoader)`：同一相機多張影像垂直拼接；JPEG 直接載入；BMP 優先用 `bmpLoader`（CoreCV_FastReadBMP + GPU resize，比 GDI+ 快 ~10x），縮圖後 JPEG 95% 重編碼對齊品質；接縫修正：`NearestNeighbor` + `PixelOffsetMode.Half` |
| `src_dotnet/AniloxRoll.Monitor/UI/Presenters/InspectionStatsPresenter.cs` | tabPageData：7 個卡片 Panel（良率顏色）+ listViewStats（5 欄彙總表） |
| `src_dotnet/AniloxRoll.Monitor/UI/Widgets/ProportionalScaler.cs` | Form 等比例縮放 Helper；Initialize 記錄所有控制項比例 + 移除 Anchor；Resize 時按比例還原位置/大小/字體；TabControl 延遲頁面自動補記錄 |
| `sdk/AOI_SDK/src_dotnet/AOI.SDK/UI/SmartCanvas.cs` | PictureBox 子類：zoom/pan/edge；**拖曳效能**：拖曳中跳過 `GetPixel`，`TriggerStatusChange` 限流 32ms（chart ~30fps），`MouseUp` 補一次完整更新 |
| `sdk/AOI_SDK/core_cv_api/src/export_api.cpp` | CoreCV_Resize_GPU 實作 |
| `sdk/AOI_SDK/core_cv_api/include/export_c/export_api.h` | CoreCV_Resize_GPU 宣告 |

---

## AniloxRoll.Monitor 右側參數面板（tabControlRight）

頂部有 `panelStatusBar`（Dock=Top，Height=32），內含 `lblStatusGrab`（Dock=Fill，IEC 60073 訊號燈，待機=灰、抓取中=綠）。

Form 右側固定有 `tabControlRight`（Location=1209,37，Size=276×741），包含 3 個 Tab：

| Tab（Name） | Tab Text | 控制項 | 內容 |
|-------------|----------|--------|------|
| `tabPageInspSettings` | 檢測設定 | `propertyGridSettings`（Dock=Fill） | `InspectionSettings`（MachineLayout / Recipe / Storage）— **Acquisition 已隱藏（[Browsable(false)]）** |
| `tabPageCamera` | 相機參數 | `tabControlCamTabs`（嵌套） | 曝光時間 / 線掃速率 / 擷取高度 各 7 台（**唯一設定入口**） |
| `tabPageSystem` | 系統資訊 | `listViewCameras` + `listViewEngine` | SystemSettings.CameraDevices + InspectionEngineConfig 常數 |

`tabPageCamera` 內嵌 `tabControlCamTabs`，含 3 個子 Tab：

| Sub Tab（Name） | Sub Tab Text | 控制項範圍 |
|----------------|--------------|------------|
| `tabPageExposure` | 曝光時間 (μs) | `trackBarExpCam1`/`numExpCam1` (CAM1 master)；`trackBarExpCam2–7`/`numExpCam2–7` (CAM2–7)；min=1 μs，max=動態：`floor(900000/lrHz)` μs（隨 LR 更新） |
| `tabPageLineRate` | 線掃速率 (Hz) | `trackBarLrCam1`/`numLrCam1` (CAM1 master)；`trackBarLrCam2–7`/`numLrCam2–7` (CAM2–7)；範圍：100–10000 Hz；更改時自動更新 tabPageExposure 上限 |
| `tabPageGrabHeight` | 擷取高度 (px) | `trackBarHtCam1`/`numHtCam1` (CAM1 master)；`trackBarHtCam2–7`/`numHtCam2–7` (CAM2–7)；範圍：100–10000 px；預設 2048；拖動結束 MouseUp → `LiveCameraManager.RefreshMainDisplay()` |

主內容區為 `tabMain`，含 `tabPageLiveView`（即時監控）、`tabPageReview`（影像回顧）、`tabPageData`（檢測數據）。

**tabPageReview 主要控制項**：
- `canvasMain`：Location=(8, 123)，Size=(1070, 346)，直接置於 `tabPageReview`（已移除 `tlpReviewLayout`）
- `chartMura`：Location=(7, 426)，Size=(888, 96)，單台相機 Mean/Max 曲線（MuraChartHelper，可 zoom/pan 與 canvas 聯動）
- `chart1`：Location=(6, 529)，Size=(888, 96)，**全覽圖**（MuraChartHelper，Zoomable=false）：7 台相機曲線依 Cam_Pos 位置合併；重疊區域 Mean 取平均、Max 取最大值；合圖路徑（`UpdateStitchedOverviewChart`）和原圖路徑（`UpdateOverviewChartFromRepository`）皆更新

**tabPageReview 額外控制項（X=1084 右欄）**：
- `lblImageFormat`（Y=400）：顯示目前圖片格式，"壓縮 JPEG" / "原始 BMP"
- `lblImageScale`（Y=424）：顯示壓縮倍率，"縮放: 5x" / "縮放: 1x"
- 由 `FormInteractionHelper.OnGallerySelectionChanged` 透過 `data.IsCompressedJpeg` / `data.ScaleFactor` 更新
- `grpReviewGrabNav`（GroupBox，Location=1088,469，Size=109×96，Text="單片"）：含 `cbReviewGrabId`（DropDownList，選擇序號）、`btnGrabIdPrev`（"<"）、`btnGrabIdNext`（">"）；選擇後呼叫 `OnReviewGrabIdChanged` → `NavigateToDateTime(info.Earliest)` + **`LoadGrabStitchedViewAsync(grabId)`**（拼接模式，見下）；Prev/Next 透過 `StepReviewGrabId(±1)` 操作 `SelectedIndex` 觸發同一事件
- `grpReviewTimePeriod`（GroupBox，"時間段"）：時間段導航區塊，含 `btnPeriodPrev`/`btnPeriodNext`

**Grab ID 拼接模式（`_stitchedImages` 欄位）**：
- `_stitchedImages != null` 表示目前為拼接模式；`null` = 一般模式
- `LoadGrabStitchedViewAsync(grabId, hintFrom, hintTo, enableProcess)` → `LoadImagePathsForGrabId`（掃 CSV）→ 依模式分支：
  - **JPEG 路徑**：`StitchCamera(paths, useProcessed: enableProcess)`（_proc.jpg 若存在則替代 _raw.jpg）
  - **BMP 原圖**：`StitchCamera(paths, bmpLoader: LoadBmpAtScale)`
  - **BMP 處理**：`StitchCamera(paths, bmpLoader: ProcessBmpAtScale)`（逐張 GPU pipeline + resize，再拼接）
  - 完成後 `RotateFlip(RotateNoneFlipY)` 對齊取像時序
- `MergeCurves`：載入每張影像的 `_mean.bin`/`_max.bin`，全解析度合併（Mean=平均、Max=取最大），存入 `_stitchedCurveMean[7]`/`_stitchedCurveMax[7]`
- `ShowStitchedCameraInCanvas`：設 `_imageScaleFactor=DefaultSaveResizeScale` + `_currentCameraIndex` 後 `FitToScreen()`，再以合併曲線更新 chartMura
- `btnShowOriginal`/`btnShowProcessed`：若 `_stitchedImages != null` → `ReloadCurrentStitchedView(false/true)`；否則走原圖路徑
- `ClearStitchedMode()`：先 null `canvasMain.Image` + `_galleryManager.ClearImages()` 再 Dispose 所有 bitmaps；在所有一般載入路徑前呼叫（propertyGrid / btnSelectFolder / btnShowOriginal / btnShowProcessed / btnPeriodPrev / btnPeriodNext）
- `SelectionChanged` handler 以 `_stitchedImages != null` 分支：拼接模式呼叫 `ShowStitchedCameraInCanvas(idx)`，一般模式呼叫 `_interactionHelper.OnGallerySelectionChanged(idx)`

**cbReviewGrabId ↔ cbDataGrabId 雙向同步**：
- `OnReviewGrabIdChanged` → 同步 `cbDataGrabId` + 統計刷新（`_syncingGrabIdCross` guard 防迴圈）
- `OnSingleSheetComboChanged` → 同步 `cbReviewGrabId` + 導航時間 + 載入拼接圖（`_syncingGrabIdCross` guard）
- `SyncGrabIdFromTimeCombos`：時間 ComboBox 變更 → 找包含該時間的 grab ID → 同步 `cbReviewGrabId`（`_syncingGrabIdNav` guard）

**btnSelectFolder ↔ btnSelectDataFolder 資料共享**：
- `btnSelectFolder`（Review tab）：載入圖片後同時填充 Data tab 所有序號/時間 ComboBox + 圖表導航 + 統計
- `btnSelectDataFolder`（Data tab）：載入統計後同時設定 Review tab 的 `ImageRepository` + `TimeNavigator`（透過 `FormInteractionHelper.LoadDirectoryAndInitNavigator`）

**GroupBox 綠色高亮（活動模式指示）**：
- `SetGroupBoxActive(grp, active)`：透過 `ActiveGroupBox_Paint` 繪製綠色填充 `(220,248,225)` / 邊框 `(0,140,60)` / 標題文字；不設 `ForeColor` 避免子控制項繼承
- Review tab：`grpReviewTimePeriod`（原圖路徑）、`grpReviewGrabNav`（合圖路徑）二擇一
- Data tab：`groupBoxGrabIdRange`、`grpDataSingleSheet`、`groupBoxTimeRange` 三擇一（`SetActiveStatGroupBox`）

### 控制項觸發關係圖

```
btnSelectFolder ─┬─→ ImageRepository.LoadDirectory + TimeNavigator.Initialize
                 ├─→ 填充 cbReviewGrabId / cbGrabIdStart / cbGrabIdEnd / cbDataGrabId + 時間 ComboBox
                 ├─→ SyncGrabIdFromTimeCombos() → cbReviewGrabId.SelectedIndex
                 ├─→ PopulateChartNavigators() + RefreshStats()
                 ├─→ ClearStitchedMode() → 高亮 grpReviewTimePeriod
                 └─→ LoadImagesWithPeriodLockAsync(false)

btnSelectDataFolder ─┬─→ LoadDirectoryAndInitNavigator (Review tab)
                     ├─→ 填充 cbGrabIdStart / cbGrabIdEnd / cbDataGrabId / cbReviewGrabId + 時間 ComboBox
                     ├─→ PopulateChartNavigators() + RefreshStats()
                     └─→ 高亮 groupBoxGrabIdRange

btnShowOriginal / btnShowProcessed
  ├─ _stitchedImages != null（合圖路徑）→ ReloadCurrentStitchedView(false/true)
  │    → LoadGrabStitchedViewAsync(enableProcess)
  └─ _stitchedImages == null（原圖路徑）→ LoadImagesWithPeriodLockAsync(false/true)

btnPeriodPrev / btnPeriodNext → ClearStitchedMode() → MovePeriodAsync

btnGrabIdPrev / btnGrabIdNext → StepReviewGrabId(±1)
  → cbReviewGrabId.SelectedIndex → OnReviewGrabIdChanged()
    ├─→ NavigateToDateTime(info.Earliest)
    ├─→ LoadGrabStitchedViewAsync(grabId)
    └─→ (_syncingGrabIdCross) cbDataGrabId + cbGrabIdStart/End + 時間 + RefreshStats

btnGrabIdDataPrev / btnGrabIdDataNext → StepDataGrabId(±1)
  → cbDataGrabId.SelectedIndex → OnSingleSheetComboChanged()
    ├─→ cbGrabIdStart/End + 時間 + RefreshStats
    └─→ (_syncingGrabIdCross) cbReviewGrabId + NavigateToDateTime + LoadGrabStitchedViewAsync

cbReviewGrabId ←→ cbDataGrabId  （雙向同步，_syncingGrabIdCross 防迴圈）
cbYear~cbSec   →  SyncGrabIdFromTimeCombos() → cbReviewGrabId （_syncingGrabIdNav 防迴圈）

cbGrabIdStart / cbGrabIdEnd → OnGrabIdComboChanged → 時間同步 + RefreshStats → 高亮 groupBoxGrabIdRange
cbDataGrabId                → OnSingleSheetComboChanged → 上述 + 同步 cbReviewGrabId → 高亮 grpDataSingleSheet
cbStart/EndYear~Sec         → OnStartComboChanged / OnEndComboChanged → RefreshStats → 高亮 groupBoxTimeRange

Gallery SelectionChanged
  ├─ _stitchedImages != null → ShowStitchedCameraInCanvas(idx)
  └─ _stitchedImages == null → _interactionHelper.OnGallerySelectionChanged(idx) → FullRes + Canvas + Chart
```

Guard flags：
- `_syncingGrabIdNav`：防止 cbReviewGrabId ↔ 時間 ComboBox 迴圈
- `_syncingGrabIdCross`：防止 cbReviewGrabId ↔ cbDataGrabId 迴圈
- `_statComboUpdating`：防止 stat ComboBox cascade 重複觸發
- `_chartNavUpdating`：防止 chart 年月日 ComboBox cascade 重複觸發

### tabPageData 控制項

| 控制項 | Name | 說明 |
|--------|------|------|
| 7 個 Panel（X=6~930，Y=6） | `panelStatCam1`~`panelStatCam7` | 卡片式顯示（良率%、Pass/Total、顏色） |
| ListView | `listViewStats` | 5 欄彙總：相機/Pass/Fail/Total/良率（分母=唯一序號數） |
| ListView | `listViewGrabDetail` | 逐序號明細：序號 + CAM1~7 各欄 Pass/Fail/—（行紅底=任一 Fail） |
| ComboBox | `cbGrabIdStart` | 序號起（選擇後自動更新 cbStart 時間 + 統計） |
| ComboBox | `cbGrabIdEnd` | 序號迄（選擇後自動更新 cbEnd 時間 + 統計） |
| Start 時間 | `cbStartYear/Month/Day/Hour/Min/Sec` | 統計起始時間（cascading，僅顯示資料中存在的值） |
| End 時間 | `cbEndYear/Month/Day/Hour/Min/Sec` | 統計結束時間（cascading，start ≤ end 強制 clamp） |
| `btnSelectDataFolder` | "讀取資料夾" | 選擇 CaptureRootPath，載入後自動填充 cbGrabIdStart/End 及時間；同時填充 `cbReviewGrabId` |
| `btnQueryStats` | "統計數據" | 手動觸發 RefreshStats() |
| `btnShowFail` | "篩選異常" | Toggle：只顯示 listViewGrabDetail 中有 Fail 的序號（切換後文字改為"顯示全部"） |
| GroupBox | `grpDataSingleSheet`（Text="單片分類"） | 含 `lblDataGrabId` + `cbDataGrabId`（序號下拉）+ `btnGrabIdDataPrev`（"<"）/ `btnGrabIdDataNext`（">"）；選擇後自動設 cbGrabIdStart==cbGrabIdEnd，同步時間欄位，觸發 RefreshStats；與 `cbReviewGrabId` 雙向同步 |
| Chart × 3 | `chartYearly`/`chartMonthly`/`chartDaily` | StackedColumn（合格=綠/異常=紅）；Y 軸在右側（AxisY2 Labels）；AxisY（Primary）驅動水平 grid；X 軸水平標籤（Angle=0）；Y 軸預設：年=60000、月=2000、日=300 |
| 年月日導航 | `btnChartYearPrev/Next` + `cbChartYear`（同理月/日） | < ComboBox > 箭頭操作 SelectedIndex；Anchor=Bottom\|Left（跟圖表底部對齊） |

初始化：`InitializeSystem()` → `SetupDataTab()` → `InspectionStatsPresenter.Initialize()` + `InitGrabDetailListView()`

**Period Charts 資料流**：
- `BtnSelectDataFolder_Click` → `PopulateChartNavigators()` → 填充 `cbChartYear` items，觸發 cascade → `OnChartYearIndexChanged` → `OnChartMonthIndexChanged` → `OnChartDayIndexChanged`
- 年/月/日切換：`cbChartYear/Month/Day.SelectedIndexChanged` 觸發對應 `OnChartXxxIndexChanged()`；Prev/Next 按鈕操作 `cbChartXxx.SelectedIndex`
- `_chartNavUpdating = true` 在 `PopulateChartNavigators` / `OnChartYearIndexChanged` / `OnChartMonthIndexChanged` 填充子 ComboBox 時設置，防止 `SelectedIndexChanged` 重複觸發 cascade
- `InspectionStatisticsService.ComputeGroupedByMonthOfYear/DayOfMonth/HourOfDay`：固定 12/31/24 個 bucket，空的顯示 0
- `FillPeriodChart` 動態軸：`niceMax = max(5, ceil(maxTotal/5)*5)`，5 等分格線，只顯示 0 和 niceMax 兩個標籤；AxisY 與 AxisY2 兩軸 Maximum/Interval 必須同步

**Period Charts 軸線架構**（`InitOneChart` + `FillPeriodChart`）：
- `AxisY`（Primary，左）：隱藏 label，但**驅動 MajorGrid**（dotted，Light Gray），`Minimum=0`，`Maximum/Interval` 與 AxisY2 同步
- `AxisY2`（Secondary，右）：`Enabled=True`，顯示右側 label（只顯示 0 和 niceMax），`MajorGrid.Enabled=false`
- StackedColumn series 綁 `YAxisType.Secondary`
- **StackedColumn 空白啟動問題**：無資料時整個 chart area 空白（無 grid/軸/刻度），因為沒有 X 類別。解法：`InitOneChart` 傳入 `xCount`/`xStart` 預填 zero-value 資料點（月份=1–12，日期=1–31，小時=0–23）；`FillPeriodChart` 先 `Points.Clear()` 再填真實資料，佔位符不影響正式資料
- `InnerPlotPosition`：`X=0, Y=12, Width=93, Height=66`（左邊界貼齊，上邊留 12% 給標題）

**統計模式**：
- 序號模式（cbGrabIdStart/End 已選）→ `ComputeByGrabIdRange` + `ComputeDetailedByGrabIdRange`；分母 = 唯一序號數；同一序號同一相機任一張超標即 Fail
- 時間模式（fallback）→ `Compute`；分母 = 照片張數；每筆獨立判斷

### 參數分類（UI 可調 vs JSON 限定）

| 類別 | 參數 | 位置 |
|------|------|------|
| A. 相機硬體設定 | Id, SystemDescriptor, SystemNum, DevNum, DcfPath | JSON only（`system-settings.json`） |
| B. 取像設定（控制） | CameraExposureTimeUs[7], CameraLineRateHz[7], CameraGrabHeight[7] | **TrackBar 唯一入口**（`tabPageCamera`）+ `acquisition-settings.json`；PropertyGrid 不顯示；**任何時間調整都立即存檔**（不限 grab 期間） |
| C. 機台佈局 | Cam1–7_Ops, Cam1–7_Pos | PropertyGrid（`tabPageInspSettings`）+ JSON |
| D. 檢測配方 | HessianMaxFactor, ErrorValueMean, ErrorValueMax | PropertyGrid（`tabPageInspSettings`）+ JSON |
| E. 儲存設定 | EnableAutoCapture, CaptureRootPath, SaveOriginalBmp | PropertyGrid（`tabPageInspSettings`）+ JSON；`SaveResizeScale`/`SaveJpgQuality` 為 `[Browsable(false)]` 常數 |
| F. 影像引擎常數 | MaxWidth, MaxHeight, MaxThumbnailSide, Sigma 等 | ListView 唯讀（`tabPageSystem`） |

### 右側面板初始化流程（code-behind）

```
InitializeSystem()
  └─ InitializeRightPanelControls()
       ├─ SetupCameraTab()   ← 設定範圍 + 套用初始值 + 繫結事件（寫回 _settings + LiveCameraManager）
       └─ SetupSystemTab()   ← 填充 listViewCameras（SystemSettings）+ listViewEngine（InspectionEngineConfig）
```

**重要**：`tabControlRight` 的所有控制項**必須宣告在 `InitializeComponent()`**（Designer.cs），才能在 VS Designer 顯示。事件繫結（需要 `_settings`、`_liveCameraManager`）保留在 code-behind。

**tabPageLiveView 面板命名**：`panelLiveCam1–7`（各相機縮圖容器，148×111）；`panelMainDisplay`（主顯示，1072×347）；`muraChartLive`（即時 Mura 曲線圖，Anchor=Bottom|Left|Right，由 `_muraChartLiveHelper` 管理，`OnLiveCurveData` 事件驅動，只顯示 `SelectedMainCameraId` 的曲線）。`LiveCameraManager` 接收 `panelLiveCam1–7` 陣列與 `panelMainDisplay`。

**曝光上限計算**：`CalcExpMax(lrHz) = clamp(floor(900000/lrHz), 1, 10000)`。LR 改變時呼叫 `ApplyExpMax()` 更新所有 7 台曝光 TrackBar/NumericUpDown 的 Maximum 並夾緊現有值。

---

## MIL API 取像模組（AniloxCamera / LiveCameraManager）

### 架構對應

```
LiveCameraManager           ← 多台相機生命週期（Allocate/Grab/Free/Reinitialize）
    │
    └─ AniloxCamera × 7     ← 單台相機 MIL 資源（對應 MilGrabSample/MilCameraUnit）
            │
            └─ CameraSystemManager  ← MIL Application + System（對應 MilSystemManager）
```

### MIL 資源分配順序（AniloxCamera.Initialize）

```
MdigAlloc(systemId, devNum, dcfPath)          // 1. 開 Digitizer
  ↓
MdigControl(M_SOURCE_SIZE_Y, grabHeight)      // 2. 設 Grab 高度（查尺寸前先設）
  ↓
MdispAlloc × 2（primary + secondary）         // 3. 開 Display（AniloxCamera 有副顯）
  ↓
MdigInquire(M_SIZE_X / M_SIZE_Y)              // 4. 查實際尺寸
  ↓
new byte[W×H]  +  NativeBufferPool(W, H, 1)  // 5. CPU buffer + CUDA Pinned Memory
  ↓
MbufAlloc2d × 2（Grab，M_GRAB+M_PROC）        // 6. Grab Buffer（雙緩衝）
MbufAlloc2d（Display，M_DISP+M_PROC）         // 7. Display Buffer
MbufAlloc2d（Proc，M_PROC）                   // 8. Processing Buffer
  ↓
MdispSelectWindow(display, displayBuf, hwnd)  // 9. 綁定 Panel
MdispControl(M_SCALE_DISPLAY, M_ONCE)
MdispControl(M_CENTER_DISPLAY, M_ENABLE)
MdispControl(M_MOUSE_USE, M_ENABLE)
  ↓
MdispHookFunction(M_MOUSE_MOVE)               // 10. 掛 Mouse Hook
MdispHookFunction(M_MOUSE_LEFT_BUTTON_DOWN)
  ↓
SetExposureUs(_appliedExposureUs)             // 11. 套用初始曝光（CLProtocol 尚未啟用，走 legacy ns 路徑）
```

**注意**：步驟 2 必須在步驟 4 之前，否則 Buffer 大小錯誤。CLProtocol **不在此處啟動**（會與 MbufAlloc2d 競爭 MIL 內部鎖）。

### CLProtocol 啟動時序

```
SetUserGrabIntent(true)
  └─ ApplyGrabState()
       ├─ CheckPresence()                          // 確認相機在線
       ├─ MdigProcess(M_START, callback)           // 開始連續抓圖
       ├─ IsLive = true
       └─ StartCLProtocolAsync()                   // 所有資源就緒後才啟動
              └─ Task.Run(TryEnableCLProtocol)     // 背景執行，不阻塞 UI
                   ├─ MdigControl(M_GC_CLPROTOCOL_DEVICE_ID, "M_DEFAULT")
                   ├─ MdigControl(M_GC_CLPROTOCOL, M_ENABLE)  // 耗時 1–2 秒
                   ├─ _clProtocolEnabled = true
                   ├─ SetExposureUs(_appliedExposureUs)    // 重套曝光（改走 Feature API）
                   └─ SetLineRateHz(_appliedLineRateHz)    // 重套線掃速率（CLProtocol 才能設）
```

`_clProtocolInitStarted`（`volatile bool`）guard 防止 ToggleGrab 重複觸發。

### 曝光設定規則

| 條件 | Set | Get（量測） | 單位 |
|------|-----|------------|------|
| CLProtocol 啟用 | `MdigControlFeature("ExposureTime", M_TYPE_DOUBLE)` | `MdigInquireFeature("ExposureTime")` | **μs**（直接） |
| CLProtocol 未啟用 | `MdigControl(M_EXPOSURE_TIME, μs×1000)` | `MdigInquire(M_EXPOSURE_TIME)` ÷ 1000 | ns → μs |

- `_appliedExposureUs` 永遠記錄最後設定值，不依賴硬體回讀
- Camera Link 無 CLProtocol 時 `MdigInquire(M_EXPOSURE_TIME)` 通常回傳 0

### 線掃速率設定規則

| 條件 | 行為 |
|------|------|
| CLProtocol 啟用 | `MdigControlFeature("AcquisitionLineRate", M_TYPE_DOUBLE, Hz)` 立即生效 |
| CLProtocol 未啟用 | 僅記錄至 `_appliedLineRateHz`，待 CLProtocol 就緒後自動重套 |

`SetLineRateHz` 與 `SetExposureUs` 機制一致：先記錄、後套用，重新初始化後也能正確恢復。

### SetGrabHeight 完整流程（不可省略任何步驟）

```
1. MdigProcess(M_STOP)                           // 停止抓圖
2. MbufFree(GrabBuffers × 2)                     // 釋放舊 MIL Buffer
   MbufFree(DisplayBuffer)
   MbufFree(ProcBuffer)
3. NativeBufferPool.Dispose()                    // 釋放 CUDA Pinned Memory
4. MdigControl(M_SOURCE_SIZE_Y, newHeight)       // 設定新高度
5. MdigInquire(M_SIZE_X / M_SIZE_Y)              // 重查實際尺寸（硬體可能夾緊）
6. new byte[W×H]  +  new NativeBufferPool(W,H,1) // 重新分配 CPU + CUDA Pinned
7. MbufAlloc2d × 4                               // 重新分配 MIL Buffer
8. MdispSelectWindow(display, newDisplayBuf, hwnd)// 重新綁定 Display
9. MdigProcess(M_START)  ← if wasLive            // 恢復抓圖
```

**崩潰警告**：舊尺寸 Buffer 與新尺寸不符會導致 MIL 崩潰。必須先釋放再重新分配，不可省略步驟 2–3。

**Rollback 機制**：步驟 4–9 包在 try/catch 中，失敗時先再次呼叫 `FreeGrabBuffers()` 清除殘留，再嘗試以原高度 `AllocateAndBind(oldHeight)`。若 rollback 也失敗則設 `_userWantsGrab = false` 停用相機，防止 Timer 反覆重試。私有方法：`FreeGrabBuffers()`（步驟 2–3）、`AllocateAndBind(targetHeight, shouldRestart)`（步驟 4–9）。

### MIL 資源釋放順序（AniloxCamera.Dispose）

```
_isReleased = true                               // 立即阻止 ProcessingFunction 繼續
  ↓
MdigProcess(M_STOP)                              // 停止連續抓圖
  ↓
MdispHookFunction(M_MOUSE_MOVE + M_UNHOOK)       // 移除 Mouse Hook
MdispHookFunction(M_MOUSE_LEFT_BUTTON_DOWN + M_UNHOOK)
MdispSelectWindow(M_NULL, IntPtr.Zero)           // 取消 Display 綁定
  ↓
MbufFree(GrabBuffers × 2)                        // 釋放 Buffer
MbufFree(DisplayBuffer)
MbufFree(ProcBuffer)
  ↓
NativeBufferPool.Dispose()                       // 釋放 CUDA Pinned Memory
AoiService.Dispose()
  ↓
MdispFree(primary)  +  MdispFree(secondary)      // 釋放 Display
  ↓
MdigFree(digitizer)                              // 最後釋放 Digitizer
  ↓
GCHandle.Free()                                  // 釋放 GCHandle（Callback 安全鎖）
```

LiveCameraManager 釋放順序：
```
IsReleasing = true  →  Timer.Stop()
  →  cam.Free() × 7  →  MsysFree × n  →  MappFreeDefault
```

`IsReleasing = true` 必須在 `Timer.Stop()` 之前設定，防止 Tick 存取已釋放相機資源。

### Telemetry 查詢方法（AniloxCamera）

| 方法 | MIL API | 說明 |
|------|---------|------|
| `CurrentFps` | `MdigInquire(M_PROCESS_FRAME_RATE)` | 實際量測 FPS |
| `GetSelectedFrameRate()` | `MdigInquire(M_SELECTED_FRAME_RATE)` | DCF 設定目標 FPS |
| `GetFrameCount()` | `MdigInquire(M_PROCESS_FRAME_COUNT)` | 累計處理幀數 |
| `GetFrameMissed()` | `MdigInquire(M_PROCESS_FRAME_MISSED)` | Callback 遺漏幀數 |
| `GetGrabFrameMissed()` | `MdigInquire(M_GRAB_FRAME_MISSED)` | 硬體層遺漏幀數 |
| `GetScanMode()` | `MdigInquire(M_SCAN_MODE)` | "Line" / "Progressive" |
| `GetLineRateHz()` | `MdigInquireFeature("AcquisitionLineRate")` | 需 CLProtocol 啟用 |
| `GetMeasuredExposureUs()` | `MdigInquireFeature("ExposureTime")` | 需 CLProtocol 啟用 |
| `GetCameraTemperature()` | `MdigInquireFeature("DeviceTemperature")` | 相機本體溫度（°C） |
| `GetFpgaTemperature()` | `MsysInquire(M_TEMPERATURE_FPGA)` | 擷取卡 FPGA 溫度 |
| `GetMemoryFreeMB()` | `MsysInquire(M_MEMORY_FREE)` | 板卡可用記憶體（MB） |
| `GetPcieNumberOfLanes()` | `MsysInquire(M_PCIE_NUMBER_OF_LANES)` | PCIe 通道數 |
| `GetPcieSpeed()` | `MsysInquire(M_PCIE_SPEED)` | "Gen1"/"Gen2"/"Gen3" |

### 已知 MIL .NET Wrapper 限制

- `M_LINE_RATE` / `M_LINE_RATE_CURRENT` / `M_GRAB_SIZE_Y` 常數**不存在**於 .NET wrapper，不可使用。
  - Line Rate → CLProtocol Feature API `"AcquisitionLineRate"`
  - Grab Height → `MdigControl(M_SOURCE_SIZE_Y, height)`（`M_SOURCE_SIZE_Y` 存在）
- `MdigHookFunction(M_CAMERA_PRESENT)` 已移除，改用 Timer 每 500ms 輪詢 `MdigInquire(M_CAMERA_PRESENT)`。
- CLProtocol 初始化期間（約 1–2 秒）Line Rate、Exp Meas、Cam Temp 無法讀取，屬正常現象。

---

## MilGrabSample 模組（MIL 相機擷取參考實作）

### 路徑

```
sdk/AOI_SDK/src_dotnet/MilGrabSample/MilGrabSample/
├── Config/
│   └── CameraConfig.cs          ← 單台相機靜態設定（Id, DCF, ExposureUs, Panel, Label）
├── Hardware/
│   ├── MilCameraUnit.cs         ← 單台相機所有 MIL 資源（Digitizer/Display/Buffer）
│   ├── MilSystemManager.cs      ← MIL Application + System 分配與釋放（static）
│   └── MilImageProcessor.cs     ← MIL 影像處理（ColMeanSubtraction / Hessian / Binarize）
├── Session/
│   └── CameraSession.cs         ← 多台相機生命週期管理（Initialize/ToggleGrab/Release）
└── UI/
    ├── GrabForm.cs              ← 主視窗，僅含按鈕事件與 UI 更新
    ├── GrabForm.Designer.cs
    └── CameraListViewPresenter.cs ← ListView 欄位初始化與每 500ms 更新
```

### 架構原則

- `GrabForm` 不直接碰 MIL API，全部委由 `CameraSession`（相機）與 `CameraListViewPresenter`（ListView）
- `CameraSession` 不含任何 UI 依賴
- `MilCameraUnit` 不管 MIL System 生命週期，System 由 `CameraSession` 統一管理

### ListView 欄位索引（共 16 欄，0–15）

| 索引 | 欄位名稱 | 來源 | 格式 |
|------|---------|------|------|
| [0] | Camera | CameraConfig.Id | — |
| [1] | FPS | `MdigInquire(M_PROCESS_FRAME_RATE)` | F2 |
| [2] | Target FPS | `MdigInquire(M_SELECTED_FRAME_RATE)` | F2 |
| [3] | Line Rate(Hz) | `MdigInquireFeature("AcquisitionLineRate")` | F1 |
| [4] | Exp Set(μs) | `_appliedExposureUs`（不回讀硬體） | F1 |
| [5] | Exp Meas(μs) | `MdigInquireFeature("ExposureTime")` | F1 |
| [6] | Frames | `MdigInquire(M_PROCESS_FRAME_COUNT)` | — |
| [7] | Missed | `MdigInquire(M_PROCESS_FRAME_MISSED)` | — |
| [8] | Grab Miss | `MdigInquire(M_GRAB_FRAME_MISSED)` | — |
| [9] | Resolution | `MdigInquire(M_SIZE_X/Y)` | "W×H" |
| [10] | Scan Mode | `MdigInquire(M_SCAN_MODE)` | Line/Progressive |
| [11] | FPGA(°C) | `MsysInquire(M_TEMPERATURE_FPGA)`（擷取卡） | F1 |
| [12] | Cam Temp(°C) | `MdigInquireFeature("DeviceTemperature")`（相機） | F1 |
| [13] | Mem Free(MB) | `MsysInquire(M_MEMORY_FREE)` ÷ 1024² | — |
| [14] | PCIe Lanes | `MsysInquire(M_PCIE_NUMBER_OF_LANES)` | — |
| [15] | PCIe Speed | `MsysInquire(M_PCIE_SPEED)` | Gen1/2/3 |

**維護注意**：新增或移除欄位時，`Initialize()`（`for i < N`）、`Update()`（各 SubItems[n]）、`ResetAll()`（`for i <= N`）三處必須同步修改。

---

## 存檔格式（TrySaveCapture）

### 檔案命名與目錄結構

時間戳精確到毫秒（`.fff`），同 Line Rate 的相機由 `CaptureTimestampCoordinator` 協調共用同一 `.fff`（100ms 容差）。

**一律存檔的 4 個檔案**（不管 `SaveOriginalBmp` 設定）：
```
{CaptureRootPath}\{yyyy}\{yyyyMM}\{yyyyMMdd}\
    {yyyyMMdd_HHmmss.fff}-{CameraId}_raw.jpg   ← 縮小版原圖（GPU resize，1/SaveResizeScale）
    {yyyyMMdd_HHmmss.fff}-{CameraId}_proc.jpg  ← 縮小版處理圖
    {yyyyMMdd_HHmmss.fff}-{CameraId}_mean.bin  ← Mura Mean 曲線（全解析度長度）
    {yyyyMMdd_HHmmss.fff}-{CameraId}_max.bin   ← Mura Max 曲線（全解析度長度）
```

**額外檔案**（`SaveOriginalBmp = true` 時）：
```
    {yyyyMMdd_HHmmss.fff}-{CameraId}.bmp       ← 全解析度原圖（同步匯出，因 sourceBuffer 會被 MIL 回收）
```

### TrySaveCapture 非同步 I/O

GPU resize + Marshal.Copy 在 MIL callback 執行緒完成（快），JPEG/bin 寫入移至 `Task.Run` 背景執行緒，callback 立即返回不阻塞連續抓圖。`_lastCaptureKey` 在 Task.Run 前提前更新，防止重複觸發。`SaveOriginalBmp=true` 時 BMP 匯出必須在 callback 同步完成（`sourceBuffer` 會被 MIL 回收用於下一幀）。

### .bin 檔案格式

```
magic(4)="MCBF" | version(4=int) | scale_factor(4=float) | array_length(4=int) | float[]
```

- `scale_factor` 儲存縮圖倍率（`SaveResizeScale`），供 `ReadScaleFactorFromBin` 讀取
- 曲線長度 = 全解析度圖寬，`_raw.jpg` 寬度 = 全解析度 ÷ scale_factor

### InspectionData 格式欄位

| 欄位 | 類型 | 說明 |
|------|------|------|
| `IsCompressedJpeg` | bool | `true`=新格式，`false`=BMP |
| `ScaleFactor` | int | 縮圖倍率（1=BMP，5=JPEG 1/5） |

- 兩者由 `InspectionEngine.LoadFromPrecomputedFiles`（新格式）或 `RunInspectionFullRes` BMP 路徑設定
- 非處理模式下（curves=null）由 `ReadScaleFactorFromBin` 讀 .bin 標頭取得 ScaleFactor

### ImageRepository 掃描邏輯

同時掃 `*_raw.jpg` + `*.bmp`，兩種格式可在同一根目錄共存，`ParsePath` regex 兩種皆可 match。
`*_proc.jpg`、`*_mean.bin`、`*_max.bin` 不被收入（不符合 glob 模式）。

**JPG 優先規則**：`GetImages()` 同一相機同時存在 JPG 與 BMP 時，優先回傳 JPG（讀取速度快，走 `LoadFromPrecomputedFiles` 不需 GPU）。避免 `.ToDictionary()` 重複 key 崩潰。

### CanvasInteractionHelper 跨倍率 View 保存

`SaveViewIfNeeded` → 把 pixel viewport 轉換成 mm 世界座標存檔；
`UpdateCanvas` → 用新圖的 `_imageScaleFactor` 把 mm 反算回 pixel zoom/pan。
Y 軸用 `_savedYCenterFraction`（圖片高度中心分率）保持垂直位置。
`SaveViewIfNeeded` 在 `_imageScaleFactor` 更新前呼叫，`UpdateCanvas` 在更新後呼叫（呼叫順序由 `FormInteractionHelper.LoadImages` → `OnGallerySelectionChanged` 保證）。

---

## ProportionalScaler 等比例縮放

Form 使用 `AutoScaleMode = None`（Designer.cs），所有控制項的 Anchor 在 `ProportionalScaler.Initialize()` 時被移除（設為 `Top|Left`），由 Scaler 全權接管定位。

- **Initialize**：記錄每個控制項的 `Left/Top/Width/Height` 相對於 `Parent.ClientSize` 的比例 + `FontSize / FormHeight`
- **OnFormResize**：按比例重算位置/大小/字體（4–72pt 範圍，差距 > 0.5pt 才更新）
- **TabControl**：`SelectedIndexChanged` hook，首次切頁時補記錄延遲頁面的控制項
- **重要**：不可混用 Anchor 和 Scaler，否則 `ResumeLayout` 觸發 Anchor 重算會覆蓋 Scaler 設定

---

## Git Workflow 規則

**未經使用者明確說「commit/push」，不得主動執行任何 git commit 或 git push。**

**每次 commit / push 前，必須先更新以下兩個檔案：**

1. `CLAUDE.md` — 更新專案架構、設定規則、關鍵檔案速查等內容
2. `skills.md` — 更新開發過程累積的模式、陷阱、可重用知識

確保文件反映最新的程式碼狀態，讓下次對話能快速上手。
