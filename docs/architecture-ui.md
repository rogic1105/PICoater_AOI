# UI 架構 — 控制項、觸發關係、面板配置

## AniloxRoll.Monitor 右側參數面板（tabControlRight）

頂部有 `panelStatusBar`（Dock=Top，Height=32），內含 `lblStatusGrab`（Dock=Fill，IEC 60073 訊號燈，待機=灰、抓取中=綠）。

Form 右側固定有 `tabControlRight`（Location=1209,37，Size=276×741），包含 3 個 Tab：

| Tab（Name） | Tab Text | 控制項 | 內容 |
|-------------|----------|--------|------|
| `tabPageInspSettings` | 檢測設定 | `propertyGridSettings`（Dock=Fill） | `InspectionSettings`（MachineLayout / Recipe / Storage）— **Acquisition 已隱藏（[Browsable(false)]）** |
| `tabPageCamera` | 相機參數 | `tabControlCamTabs`（嵌套） | 曝光時間 / 線掃速率 / 擷取高度 各 7 台（**唯一設定入口**） |
| `tabPageSystem` | 系統資訊 | `listViewCameras` + `listViewEngine` + `listViewChartConst` + `listViewHardware` | SystemSettings.CameraDevices + InspectionEngineConfig 常數 + 圖表引擎常數 + 硬體參數（螢幕尺寸/解析度/DPI） |

`tabPageCamera` 內嵌 `tabControlCamTabs`，含 3 個子 Tab：

| Sub Tab（Name） | Sub Tab Text | 控制項範圍 |
|----------------|--------------|------------|
| `tabPageExposure` | 曝光時間 (μs) | `trackBarExpCam1`/`numExpCam1` (CAM1 master)；`trackBarExpCam2–7`/`numExpCam2–7` (CAM2–7)；min=1 μs，max=動態：`floor(900000/lrHz)` μs（隨 LR 更新） |
| `tabPageLineRate` | 線掃速率 (Hz) | `trackBarLrCam1`/`numLrCam1` (CAM1 master)；`trackBarLrCam2–7`/`numLrCam2–7` (CAM2–7)；範圍：100–10000 Hz；更改時自動更新 tabPageExposure 上限 |
| `tabPageGrabHeight` | 擷取高度 (px) | `trackBarHtCam1`/`numHtCam1` (CAM1 master)；`trackBarHtCam2–7`/`numHtCam2–7` (CAM2–7)；範圍：100–10000 px；預設 2048；拖動結束 MouseUp → `LiveCameraManager.RefreshMainDisplay()` |

主內容區為 `tabMain`，含 `tabPageLiveView`（即時監控）、`tabPageReview`（影像回顧）、`tabPageData`（檢測數據）。

---

## tabPageReview 主要控制項

- `canvasMain`：Location=(8, 123)，Size=(1070, 346)，直接置於 `tabPageReview`（已移除 `tlpReviewLayout`）
- `chartMura`：Location=(7, 426)，Size=(888, 96)，單台相機 Mean/Max 曲線（MuraChartHelper，可 zoom/pan 與 canvas 聯動）
- `chartOverview`：Location=(6, 529)，Size=(888, 96)，**回顧全覽圖**（MuraChartHelper，Zoomable=false）：7 台相機曲線依 Cam_Pos 位置合併；兩層合併演算法（per-camera max-window downsample → cross-camera overlap）；MaxOverviewPoints=2000 點上限；合圖路徑（`UpdateStitchedOverviewChart`）和原圖路徑（`UpdateOverviewChartFromRepository`）皆更新

### 額外控制項（X=1084 右欄）

- `lblImageFormat`（Y=400）：顯示目前圖片格式，"壓縮 JPEG" / "原始 BMP"
- `lblImageScale`（Y=424）：顯示壓縮倍率，"縮放: 5x" / "縮放: 1x"
- 由 `FormInteractionHelper.OnGallerySelectionChanged` 透過 `data.IsCompressedJpeg` / `data.ScaleFactor` 更新
- `grpReviewGrabNav`（GroupBox，Location=1088,469，Size=109×96，Text="單片"）：含 `cbReviewGrabId`（DropDownList，選擇序號）、`btnGrabIdPrev`（"<"）、`btnGrabIdNext`（">"）；選擇後呼叫 `OnReviewGrabIdChanged` → `NavigateToDateTime(info.Earliest)` + **`LoadGrabStitchedViewAsync(grabId)`**（拼接模式）；Prev/Next 透過 `StepReviewGrabId(±1)` 操作 `SelectedIndex` 觸發同一事件
- `grpReviewTimePeriod`（GroupBox，"時間段"）：時間段導航區塊，含 `btnPeriodPrev`/`btnPeriodNext`

---

## Grab ID 拼接模式（`_stitchedImages` 欄位）

- `_stitchedImages != null` 表示目前為拼接模式；`null` = 一般模式
- `LoadGrabStitchedViewAsync(grabId, hintFrom, hintTo, enableProcess)` → `LoadImagePathsForGrabId`（掃 CSV）→ 依模式分支：
  - **JPEG 路徑**：`StitchCamera(paths, useProcessed: enableProcess)`（_proc_v.jpg 若存在則替代 _raw.jpg）
  - **BMP 原圖**：`StitchCamera(paths, bmpLoader: LoadBmpAtScale)`
  - **BMP 處理**：`StitchCamera(paths, bmpLoader: ProcessBmpAtScale)`（逐張 GPU pipeline + resize，再拼接）
  - 完成後 `RotateFlip(RotateNoneFlipY)` 對齊取像時序
- `MergeCurves`：載入每張影像的 `_mean_v.bin`/`_max_v.bin`，全解析度合併（Mean=平均、Max=取最大），存入 `_stitchedCurveMean[7]`/`_stitchedCurveMax[7]`
- `MergeRowCurves`：載入 `_mean_h.bin`/`_max_h.bin`，垂直拼接（concatenation），存入 `_stitchedRowCurveMean[7]`/`_stitchedRowCurveMax[7]`
- `ShowStitchedCameraInCanvas`：設 `_imageScaleFactor=DefaultSaveResizeScale` + `_currentCameraIndex` 後 `FitToScreen()`，再以合併曲線更新 chartMura
- `checkBoxShowProcessed`（CheckBox）：勾選/取消 → `checkBoxShowProcessed_CheckedChanged` → 合圖路徑走 `ReloadCurrentStitchedView(checked)`，一般路徑走 `LoadImagesWithPeriodLockAsync(checked)`；`_syncingProcessedCheckbox` guard 防程式碼設定 `.Checked` 時重複觸發
- `ClearStitchedMode()`：先 null `canvasMain.Image` + `_galleryManager.ClearImages()` 再 Dispose 所有 bitmaps；在所有一般載入路徑前呼叫（propertyGrid / btnSelectFolder / btnPeriodPrev / btnPeriodNext）
- `SelectionChanged` handler 以 `_stitchedImages != null` 分支：拼接模式呼叫 `ShowStitchedCameraInCanvas(idx)`，一般模式呼叫 `_interactionHelper.OnGallerySelectionChanged(idx)`

---

## cbReviewGrabId ↔ cbDataGrabId 雙向同步

- `OnReviewGrabIdChanged` → 同步 `cbDataGrabId` + 統計刷新（`_syncingGrabIdCross` guard 防迴圈）
- `OnSingleSheetComboChanged` → 同步 `cbReviewGrabId` + 導航時間 + 載入拼接圖（`_syncingGrabIdCross` guard）
- `SyncGrabIdFromTimeCombos`：時間 ComboBox 變更 → 找包含該時間的 grab ID → 同步 `cbReviewGrabId`（`_syncingGrabIdNav` guard）

---

## btnSelectFolder ↔ btnSelectDataFolder 資料共享

- `btnSelectFolder`（Review tab）：載入圖片後同時填充 Data tab 所有序號/時間 ComboBox + 圖表導航 + 統計
- `btnSelectDataFolder`（Data tab）：載入統計後同時設定 Review tab 的 `ImageRepository` + `TimeNavigator`（透過 `FormInteractionHelper.LoadDirectoryAndInitNavigator`）

---

## GroupBox 綠色高亮（活動模式指示）

- `SetGroupBoxActive(grp, active)`：透過 `ActiveGroupBox_Paint` 繪製綠色填充 `(220,248,225)` / 邊框 `(0,140,60)` / 標題文字；不設 `ForeColor` 避免子控制項繼承
- Review tab：`grpReviewTimePeriod`（原圖路徑）、`grpReviewGrabNav`（合圖路徑）二擇一
- Data tab：`groupBoxGrabIdRange`、`grpDataSingleSheet`、`groupBoxTimeRange` 三擇一（`SetActiveStatGroupBox`）

---

## 控制項觸發關係圖

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

checkBoxShowProcessed.CheckedChanged
  ├─ _stitchedImages != null（合圖路徑）→ ReloadCurrentStitchedView(checked)
  └─ _stitchedImages == null（原圖路徑）→ LoadImagesWithPeriodLockAsync(checked)

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
- `_syncingProcessedCheckbox`：防止程式碼設定 `checkBoxShowProcessed.Checked` 時觸發 `CheckedChanged`
- `_statComboUpdating`：防止 stat ComboBox cascade 重複觸發
- `_chartNavUpdating`：防止 chart 年月日 ComboBox cascade 重複觸發

---

## SwitchRidgeDirection 三態切換

- 未勾選強化圖 → 自動勾選 `checkBoxShowProcessed` + 設方向
- 已勾選且點同方向 → 取消勾選（回原圖）
- 已勾選且點不同方向 → 切換方向，重新載入處理圖

### V/H 圖片與曲線顯示決策矩陣

**關鍵狀態變數**：
- `_activeRidgeDirection`（`"v"` / `"h"`）：控制處理圖方向
- `_lastReviewProcessedMode`（bool）：`checkBoxShowProcessed` 的邏輯狀態
- `_stitchedImages`（`Bitmap[]`）：`null`=一般模式，`!=null`=合圖模式

**canvasMain 顯示哪張圖**：

| 模式 | checkbox | direction | canvasMain 圖片 |
|------|:---:|:---:|---|
| 一般-原圖 | `false` | — | `_raw.jpg` 或 `.bmp` 原圖 |
| 一般-V處理 | `true` | `"v"` | `_proc_v.jpg`（JPEG）或 `_ridgeBuffer`（BMP GPU） |
| 一般-H處理 | `true` | `"h"` | `_proc_h.jpg`（JPEG）或 `_muraBuffer`（BMP GPU） |
| 合圖-原圖 | `false` | — | `_raw.jpg` 拼接 |
| 合圖-V處理 | `true` | `"v"` | `_proc_v.jpg` 拼接 |
| 合圖-H處理 | `true` | `"h"` | `_proc_h.jpg` 拼接 |

**Chart 曲線（永遠雙方向，不受 direction 影響）**：

| Chart | 一般模式 | 合圖模式 | 軸同步 |
|-------|---------|---------|--------|
| `chartMuraVertical` | `_mean_v.bin` / `_max_v.bin` | `MergeCurves` 合併結果 | X 軸 ↔ canvas 水平 viewport |
| `chartMuraHorizontal` | `_mean_h.bin` / `_max_h.bin` | `MergeRowCurves` 合併結果 | Y 軸 ↔ canvas 垂直 viewport |

**Chart 背景色**：`_activeRidgeDirection == "v"` → `chartMuraVertical` 淺藍；`"h"` → `chartMuraHorizontal` 淺藍；`checkBox=false` → 兩者皆預設色。

### 觸發流程

```
點擊 chartMuraVertical ──→ SwitchRidgeDirection("v") ──┐
點擊 chartMuraHorizontal → SwitchRidgeDirection("h") ──┤
                                                        ▼
                              ┌─────────────────────────────────────────┐
                              │ Case1: 未勾選 → checkbox=true           │
                              │ Case2: 同方向 → checkbox=false          │
                              │ Case3: 不同方向 → 直接重載（不改checkbox）│
                              └──────────────┬──────────────────────────┘
                                             ▼
                           checkBoxShowProcessed_CheckedChanged
                                     (Case1/2 觸發)
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      │ _stitchedImages != null                     │ _stitchedImages == null
                      ▼                                             ▼
          ReloadCurrentStitchedView               LoadImagesWithPeriodLockAsync
                      │                                             │
                      ▼                                             ▼
          LoadGrabStitchedViewAsync                    LoadImages → RunWorkflowAsync
              │                                                     │
              ├─ StitchCamera × 7                                   │
              │    enableProcess + ridgeDir                          ▼
              │    選擇 _raw / _proc_v / _proc_h                OnGallerySelectionChanged
              ├─ MergeCurves → V曲線                                │
              ├─ MergeRowCurves → H曲線                             ▼
              │                                         RunInspectionFullRes(ridgeDirection)
              ▼                                             │
  ShowStitchedCameraInCanvas                    ┌───────────┼───────────┐
      canvasMain ← _stitchedImages[idx]         │ JPEG      │ BMP+Proc  │ BMP+Orig
      chartV ← _stitchedCurveMean[idx]          ▼           ▼           ▼
      chartH ← _stitchedRowCurveMean[idx]  LoadFromPre.. GPU pipeline  直接讀buf
                                                │           │           │
                                                └─────┬─────┘           │
                                                      ▼                 │
                                            canvasMain.Image = bmp      │
                                            chartV ← _mean_v / _max_v  │
                                            chartH ← _mean_h / _max_h  │
```

**不變式**：
1. Canvas 圖片由 `_activeRidgeDirection` 決定（`"v"` → `_proc_v` / `_ridgeBuffer`，`"h"` → `_proc_h` / `_muraBuffer`）
2. Chart 曲線永遠雙方向同時更新（chartV=`_v` 曲線，chartH=`_h` 曲線），與 canvas 顯示的圖片無關
3. GPU Pipeline 永遠以 `"vertical+horizontal"` 模式執行，`_activeRidgeDirection` 只影響 UI 顯示選擇
4. 合圖模式的 V 曲線用 `MergeCurves`（per-column mean/max），H 曲線用 `MergeRowCurves`（垂直拼接）

---

## tabPageLiveView 面板

`panelLiveCam1–7`（各相機縮圖容器，148×111）；`panelMainDisplay`（主顯示，1072×347）；`muraChartVerticalLive`（即時切向 Mura 曲線圖，Anchor=Bottom|Left|Right，由 `_muraChartLiveHelper` 管理，`OnLiveCurveData` 事件驅動，只顯示 `SelectedMainCameraId` 的曲線）；`muraChartHorizontalLive`（即時法向 Mura 曲線圖，由 `_rowChartLiveHelper` 管理，`OnLiveRowCurveData` 事件驅動）；`chartLiveOverview`（即時全覽圖，Zoomable=false，由 `_liveOverviewHelper` 管理，`_liveOverviewTimer` 驅動，動態跟隨最大 FPS 50–500ms，兩層 max-window 合併）。`LiveCameraManager` 接收 `panelLiveCam1–7` 陣列與 `panelMainDisplay`。

**Live Chart ↔ panelMainDisplay 對齊**：`OnLiveCurveData` / `OnLiveRowCurveData` 透過 `AniloxCamera.TryGetSecondaryDisplayGeometry()` 查詢 MIL 副顯示器的即時 `M_ZOOM_FACTOR_X/Y` + `M_PAN_OFFSET_X/Y`，將 panel 邊緣轉換為 mm 後傳入 chart helper 的 `UpdateDataAndView` / `UpdateViewRange`，搭配 InnerPlotPosition 補償對齊。MIL `M_MOUSE_USE` 啟用後使用者滾輪操作會即時更新 zoom/pan，chart 每幀同步。per-camera OPS 在每幀呼叫 `SetOps` 更新（替代原先固定 `Cam1_Ops`）。

**曝光上限計算**：`CalcExpMax(lrHz) = clamp(floor(900000/lrHz), 1, 10000)`。LR 改變時呼叫 `ApplyExpMax()` 更新所有 7 台曝光 TrackBar/NumericUpDown 的 Maximum 並夾緊現有值。

### SwitchLiveDisplayDirection 三態切換（Live tab）

與 Review tab 的 `SwitchRidgeDirection` 相同邏輯，控制 Live 顯示的 V/H 處理圖方向：

- `_liveDisplayDirection`（`"v"` / `"h"`）：控制 Live 顯示方向
- `checkBoxEnableImageProcessing`：對應 Review 的 `checkBoxShowProcessed`

**三態邏輯**：
1. 未勾選 → 自動勾選 `checkBoxEnableImageProcessing` + 設方向
2. 已勾選且點同方向 → 取消勾選（回原圖）
3. 已勾選且點不同方向 → 切換方向（不改 checkbox）

**與 Review 的差異**：Live tab 不需重新載入影像，`AniloxCamera.ProcessingFunction` 每幀檢查 `EnableImageProcessing` 和 `LiveDisplayDirection`，下一幀自動更新 MIL 顯示 buffer。

**觸發流程**：
```
點擊 muraChartVerticalLive ──→ SwitchLiveDisplayDirection("v") ──┐
點擊 muraChartHorizontalLive → SwitchLiveDisplayDirection("h") ──┤
                                                                  ▼
                            ┌─────────────────────────────────────────────────┐
                            │ Case1: 未勾選 → 設方向 + checkbox=true          │
                            │ Case2: 同方向 → checkbox=false                  │
                            │ Case3: 不同方向 → SetLiveDisplayDirection（不改cb）│
                            └──────────────┬──────────────────────────────────┘
                                           ▼
                     checkBoxEnableImageProcessing_CheckedChanged
                                   (Case1/2 觸發)
                                           │
                         SetImageProcessingEnabled + UpdateLiveDirectionVisual
```

**Chart 背景色**：`_liveDisplayDirection == "v"` → `muraChartVerticalLive` 淺藍；`"h"` → `muraChartHorizontalLive` 淺藍；`checkbox=false` → 兩者皆預設色。

---

## 背景預覽模式（btnViewBackground / btnGetBackground）

### btnGetBackground

- 確認 `Algorithm == StandardBgSub`，確保相機已 allocate+grab
- 按鈕文字倒數顯示（`"採集中 {remaining}s"`）
- 採集完成 → 停止 grab → 自動呼叫 `btnViewBackground_Click`

### btnViewBackground（Toggle）

開啟預覽：
1. 載入 `bg_{width}_{camId}.bin` → `ExpandColMeanToBitmap`（float→8bpp Bitmap，高度=grabHeight）
2. `panelLiveCam1–7`：建立 PictureBox overlay（`Dock=Fill, StretchImage, BringToFront`），點擊切換主顯示
3. `panelMainDisplay`：建立 SmartCanvas overlay（`Dock=Fill, ClampPan=true, BringToFront`），支援 zoom/pan
4. 先 detach MIL secondary display（避免 native window z-order 衝突）
5. `lblPixelInfo` 顯示 `"背景預覽 [CAM N] | X: ..., Y: ... | 灰階值: ... | 縮放: ...x"`

關閉預覽（再次按 / btnCameraGrab）：
- `ClearBackgroundPreview()`：移除所有 overlay PictureBox/SmartCanvas，dispose bitmaps

### PictureBox Overlay 模式

MIL native display 使用 `MdispSelectWindow` 綁定 panel Handle，建立的子視窗在 z-order 最上層。
managed PictureBox 無法覆蓋它。解法：
1. 先 detach MIL display（`MdispSelectWindow(M_NULL)`）
2. 建立 PictureBox/SmartCanvas（`Dock=Fill`）
3. `BringToFront()` 確保覆蓋原有子控制項

### UI 鎖定邏輯（StandardBgSub）

`UpdateStandardBgSubLockState()`：
- `Algorithm == StandardBgSub` 且無 bin → `btnCameraGrab` disabled, `btnGetBackground` enabled
- `Algorithm == StandardBgSub` 且有 bin → 全部解鎖
- 其他算法 → `btnGetBackground` disabled

PropertyGrid 變更觸發重新檢查。

---

## 參數分類（UI 可調 vs JSON 限定）

| 類別 | 參數 | 位置 |
|------|------|------|
| A. 相機硬體設定 | Id, SystemDescriptor, SystemNum, DevNum, DcfPath | JSON only（`system-settings.json`） |
| B. 取像設定（控制） | CameraExposureTimeUs[7], CameraLineRateHz[7], CameraGrabHeight[7] | **TrackBar 唯一入口**（`tabPageCamera`）+ `acquisition-settings.json`；PropertyGrid 不顯示；**任何時間調整都立即存檔**（不限 grab 期間） |
| C. 機台佈局 | Cam1–7_Ops, Cam1–7_Pos | PropertyGrid（`tabPageInspSettings`）+ JSON |
| D. 檢測配方 | HessianMaxFactor, ErrorValueMean, ErrorValueMax | PropertyGrid（`tabPageInspSettings`）+ JSON |
| E. 儲存設定 | EnableAutoCapture, CaptureRootPath, SaveOriginalBmp | PropertyGrid（`tabPageInspSettings`）+ JSON；`SaveResizeScale`/`SaveJpgQuality` 為 `[Browsable(false)]` 常數 |
| F. 影像引擎常數 | MaxWidth, MaxHeight, MaxThumbnailSide, Sigma 等 | ListView 唯讀（`tabPageSystem`） |

---

## 右側面板初始化流程（code-behind）

```
InitializeSystem()
  └─ InitializeRightPanelControls()
       ├─ SetupCameraTab()   ← 設定範圍 + 套用初始值 + 繫結事件（寫回 _settings + LiveCameraManager）
       └─ SetupSystemTab()   ← 填充 listViewCameras（SystemSettings）+ listViewEngine（InspectionEngineConfig + SaveResizeScale/SaveJpgQuality）+ listViewChartConst（圖表引擎常數：MaxOverviewPoints/DownsampleMode 等）
```

**重要**：`tabControlRight` 的所有控制項**必須宣告在 `InitializeComponent()`**（Designer.cs），才能在 VS Designer 顯示。事件繫結（需要 `_settings`、`_liveCameraManager`）保留在 code-behind。

---

## CanvasInteractionHelper 跨倍率 View 保存 + Chart 視野同步

`SaveViewIfNeeded` → 把 pixel viewport 轉換成 mm 世界座標存檔；
`UpdateCanvas` → 用新圖的 `_imageScaleFactor` 把 mm 反算回 pixel zoom/pan。
Y 軸用 `_savedYCenterFraction`（圖片高度中心分率）保持垂直位置。
`SaveViewIfNeeded` 在 `_imageScaleFactor` 更新前呼叫，`UpdateCanvas` 在更新後呼叫（呼叫順序由 `FormInteractionHelper.LoadImages` → `OnGallerySelectionChanged` 保證）。

**Chart 視野同步**：`UpdateCanvasInfo`（canvas.StatusChanged 事件）同時更新：
- `MuraChartHelper`（chartMuraVertical）：X 軸 = canvas 水平 viewport mm
- `RowMuraChartHelper`（chartMuraHorizontal）：Y 軸 = canvas 垂直 viewport mm（`pixelTop/Bot * rowPitchMm`）
- Status bar：`位置:(X, Y) mm | X範圍 | Y範圍 | 座標 | 亮度 | 實體倍率`
  - 實體倍率 = `(zoom × screenMmPerPx) / (imageScaleFactor × opsInMm)`，1.0x = 螢幕 1cm = 實際 1cm
  - `screenMmPerPx` 由 `GetDeviceCaps(HORZSIZE/HORZRES)` 計算，啟動時傳入 CanvasInteractionHelper
  - panelMainDisplay 背景預覽 / MIL 即時影像也使用相同格式的 status bar
- `_suppressChartSync` flag 防止 FitToScreen/SetView 觸發的 StatusChanged 與手動 chart 更新衝突

**滑鼠手勢**：
- canvasMain / panelMainDisplay：雙擊 → FitToScreen，三擊 → 實體倍率 1x（畫面中心不動）
  - canvasMain：`CanvasInteractionHelper.SetPhysicalMagnification1x()` — `zoom = scaleFactor × opsInMm / screenMmPerPx`
  - panelMainDisplay（MIL live）：`LiveCameraManager.SetPhysicalMagnification1x()` — `zoom = opsInMm / screenMmPerPx`
  - panelMainDisplay（背景預覽）：`SetBgPreviewPhysicalMag1x()` — 同公式，scaleFactor=1

**Period Chart Y 軸 Auto/Fixed 切換**：
- 每張 chart 以 `chart.Tag = "auto"` / `null` 作為單一狀態源（不依賴全域 `_settings.Chart.ScaleMode`）
- `ApplyAutoScale` 設定 `chart.Tag = "auto"`，避免 Fixed→Auto 切換需要兩次點擊
- 500ms throttle（`_lastChartToggleTick`）防止快速連點

### RowMuraChartHelper InnerPlotPosition 補償

與 `MuraChartHelper` 同理：PostPaint 首次渲染後量測 InnerPlotPosition，凍結版面比例（`_cachedFTop`/`_cachedFBottom`），`GetAdjustedZoom` 套用補償使 chart 控件邊緣對齊 canvas 邊緣。

**Y 軸標籤反轉**：透過 `Customize` 事件將每個 Y 標籤文字替換為 `totalMm - value`，使視覺上 0 在上、max 在下（匹配 canvas Y=0 在頂部），而不使用 `IsReversed`（避免 X 軸跳到頂部的副作用）。

---

## ProportionalScaler 等比例縮放

Form 使用 `AutoScaleMode = None`（Designer.cs），所有控制項的 Anchor 在 `ProportionalScaler.Initialize()` 時被移除（設為 `Top|Left`），由 Scaler 全權接管定位。

- **Initialize**：記錄每個控制項的 `Left/Top/Width/Height` 相對於 `Parent.ClientSize` 的比例 + `FontSize / FormHeight`
- **OnFormResize**：按比例重算位置/大小/字體（4–72pt 範圍，差距 > 0.5pt 才更新）
- **TabControl**：`SelectedIndexChanged` hook，首次切頁時補記錄延遲頁面的控制項
- **重要**：不可混用 Anchor 和 Scaler，否則 `ResumeLayout` 觸發 Anchor 重算會覆蓋 Scaler 設定
