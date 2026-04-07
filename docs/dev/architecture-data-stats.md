# 檢測數據與統計 — tabPageData、CSV、Period Charts

## tabPageData 控制項

| 控制項 | Name | 說明 |
|--------|------|------|
| 7 個 Panel（X=6~930，Y=6） | `panelStatCam1`~`panelStatCam7` | 卡片式顯示（良率%、Pass/Total、顏色） |
| ListView | `listViewStats` | 5 欄彙總：相機(1~7)/Pass/Fail/Total/良率（分母=唯一序號數）；欄寬按標題文字比例自適配；全欄置中 |
| ListView | `listViewGrabDetail` | 逐序號明細：序號 + 1~7 各欄 Pass/Fail/—（行紅底=任一 Fail）；欄寬按標題文字比例自適配 |
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

---

## 統計模式（由 `_activeStatMode` 追蹤當前活動 GroupBox）

- **序號模式**（`_activeStatMode != groupBoxTimeRange` 且 cbGrabIdStart/End 已選）→ `ComputeByGrabIdRange` + `ComputeDetailedByGrabIdRange`；分母 = 唯一序號數；同一序號同一相機任一張超標即 Fail
- **時間模式**（`_activeStatMode == groupBoxTimeRange`）→ 找時間範圍內的 grab IDs → 同樣用 `ComputeByGrabIdRange`；分母 = 唯一序號數
- Period Charts（`ScanCsvByDateRange`）也使用序號分組邏輯：以 (GrabId, CamId) 為單位，一票否決

### InspectionStatsPresenter（tabPageData）

- 7 個 Panel 卡片：BackColor = 綠(≥95%) / 橙(80-95%) / 紅(<80%) / 灰(無資料)
- `listViewStats` 5 欄彙總：相機 / Pass / Fail / Total / 良率（序號模式下分母=唯一序號數）
- `listViewGrabDetail` 逐序號明細：序號 + CAM1~7（Pass/Fail/—），整行紅底 RGB(255,235,238) = 任一 CAM Fail，全 Pass 行綠底 RGB(232,245,233)

---

## 控制項觸發關係圖

```
btnSelectDataFolder_Click
  ├─→ FolderBrowserDialog → LoadAvailableTimes + LoadGrabIdInfos
  ├─→ _interactionHelper.LoadDirectoryAndInitNavigator（同步 Review tab）
  ├─→ 填充 cbGrabIdStart / cbGrabIdEnd / cbDataGrabId / cbReviewGrabId
  ├─→ 填充 cbStartDate~Sec / cbEndDate~Sec（全範圍）
  ├─→ PopulateChartNavigators() → cbChartYear cascade
  ├─→ RefreshStats()
  └─→ 高亮 groupBoxGrabIdRange

btnQueryStats_Click → RefreshStats()

btnShowFail_Click（Toggle）
  ├─ _showFailOnly = !_showFailOnly
  ├─ Text = "篩選異常" ↔ "顯示全部"
  ├─ BackColor = 淺紅 / 預設
  └─ ApplyFailFilter() → 重新填充 listViewGrabDetail

cbGrabIdStart / cbGrabIdEnd → OnGrabIdComboChanged(isStart)
  ├─ 強制 start ≤ end（index 倒序，自動 swap）
  ├─ 同步時間 ComboBox（SetCombosToDateTime）
  ├─ RefreshStats()
  └─ 高亮 groupBoxGrabIdRange

cbDataGrabId → OnSingleSheetComboChanged()
  ├─ 鎖定 cbGrabIdStart = cbGrabIdEnd = 同一序號
  ├─ 同步時間 ComboBox
  ├─ RefreshStats()
  ├─ 高亮 grpDataSingleSheet
  └─ 跨 tab 同步（_syncingGrabIdCross guard）：
      ├─ cbReviewGrabId.SelectedIndex = idx
      ├─ NavigateToDateTime(info.Earliest)
      └─ LoadGrabStitchedViewAsync(grabId)

cbStartDate/Time → OnStartComboChanged(fromLevel)
  ├─ Cascade：Date 變更 → 刷新 Time items
  ├─ ClampEndToStart（確保 end ≥ start）
  ├─ 高亮 groupBoxTimeRange
  └─ RefreshStats()

cbEndDate/Time → OnEndComboChanged(fromLevel)
  ├─ Cascade + ClampStartToEnd（確保 start ≤ end）
  ├─ 高亮 groupBoxTimeRange
  └─ RefreshStats()

cbChartYear → OnChartYearIndexChanged（_chartNavUpdating guard）
  ├─ 填充 cbChartMonth
  ├─ FillPeriodChart(chartYearly, ComputeGroupedByMonthOfYear)
  └─ Cascade → OnChartMonthIndexChanged

cbChartMonth → OnChartMonthIndexChanged
  ├─ 填充 cbChartDay
  ├─ FillPeriodChart(chartMonthly, ComputeGroupedByDayOfMonth)
  └─ Cascade → OnChartDayIndexChanged

cbChartDay → OnChartDayIndexChanged
  └─ FillPeriodChart(chartDaily, ComputeGroupedByHourOfDay)

btnChartYearPrev/Next → cbChartYear.SelectedIndex ± 1 → cascade
btnChartMonthPrev/Next → cbChartMonth.SelectedIndex ± 1 → cascade
btnChartDayPrev/Next → cbChartDay.SelectedIndex ± 1 → cascade

chartYearly/Monthly/Daily.MouseClick → PeriodChart_ToggleAutoScale
  ├─ 500ms throttle（_lastChartToggleTick）
  ├─ chart.Tag == "auto" → ApplyFixedScale（使用 settings Y-max）
  └─ chart.Tag == null → ApplyAutoScale（從資料計算 niceMax，設 Tag="auto"）
```

### RefreshStats() 三模式邏輯

```
RefreshStats()
  ├─ 序號模式（_activeStatMode != groupBoxTimeRange）
  │   └─ ComputeByGrabIdRange(startGrabId, endGrabId) + ComputeDetailedByGrabIdRange
  │
  ├─ 時間模式（_activeStatMode == groupBoxTimeRange）
  │   ├─ Parse start/end DateTime from combos
  │   ├─ 找時間範圍內的 GrabIds
  │   └─ ComputeByGrabIdRange（轉換為序號基準）
  │
  └─ Output：
      ├─ _statsPresenter.Update(summary) → panelStatCam1~7 + listViewStats
      ├─ _currentDetails = detailedList
      └─ ApplyFailFilter() → UpdateGrabDetailListView()
```

### 跨 Tab 同步（Review ↔ Data）

| 方向 | 觸發 | 同步內容 | Guard |
|------|------|---------|-------|
| Review → Data | `OnReviewGrabIdChanged` | cbDataGrabId + cbGrabIdStart/End + 時間 + RefreshStats + 高亮 grpDataSingleSheet | `_syncingGrabIdCross` |
| Data → Review | `OnSingleSheetComboChanged` | cbReviewGrabId + NavigateToDateTime + LoadGrabStitchedViewAsync | `_syncingGrabIdCross` |
| 時間 → GrabId | `SyncGrabIdFromTimeCombos` | 找包含該時間的 grabId → cbReviewGrabId | `_syncingGrabIdNav` |
| btnSelectFolder | Review tab 載入 | 同時填充 Data tab 所有 ComboBox + RefreshStats | — |
| btnSelectDataFolder | Data tab 載入 | 同時設定 Review tab ImageRepository + TimeNavigator | — |

---

## 檢測日誌 CSV 架構

### InspectionLogService

- 路徑：`{CaptureRootPath}\{YYYY}\{YYYYMM}\{YYYYMMDD}.csv`
- 欄位：`Id, FileName, MaxExceed, MeanExceed, MeanPeak, MaxPeak, GrabHeight, LineRateHz, ExposureUs`（Pass = MaxExceed + MeanExceed 均為 0）
- ID 格式：`yyMMdd-HHmmss`（時間戳，如 `260401-130511`），由 `FormatGrabId(DateTime)` 產生
- `btnCameraGrab_Click` 開始抓取時呼叫 `NextGrabId()` → `_currentGrabId`
- CSV 寫入時機：`AniloxCamera.TrySaveCapture()` 實際存檔後，透過 `OnInspectionResult` 事件逐層傳遞至 Form

### InspectionStatisticsService

- 遞迴掃描 `Directory.GetFiles(root, "*.csv", SearchOption.AllDirectories)`，兩種統計模式：
  - `Compute(root, start, end)`：時間範圍過濾，分母 = 張數，每筆獨立判斷
  - `ComputeByGrabIdRange(root, startGrabId, endGrabId)`：序號範圍過濾（字串 Ordinal 比較），分母 = 唯一序號數，一票否決
  - `ComputeDetailedByGrabIdRange(root, startGrabId, endGrabId)`：回傳 `List<GrabDetail>`（逐序號×CAM1~7 的 `bool?`）
- `LoadGrabIdInfos(root)` → `List<GrabIdInfo>`（每個序號的 Earliest/Latest 時間）
- `LoadAvailableTimes(root)` → `SortedSet<DateTime>`（全部時間戳，供 cascading comboBox 使用）
- **`LoadImagePathsForGrabId`**（回傳 `Dictionary<int,List<string>>` camId→排序路徑）
- **`LoadConfigForGrabId`**（回傳該序號最近的 `#CFG` 快照）
- CSV 格式：`Id,FileName,MaxExceed,MeanExceed`；FileName = `YYYYMMDD_HHMMSS-camId`；Id = `yyMMdd-HHmmss`
- 從 FileName 提取 CamId：`fileName.LastIndexOf('-')` 後的數字
- GrabId 為字串，字典序 = 時間序（`StringComparer.Ordinal`）

---

## CSV #CFG 設定快照

### 格式
```
#CFG,2025-03-23T14:30:00,Cam1_Ops=33,...,ErrorValueMax=2
```

### 變更偵測
`CsvConfigSnapshot.ContentKey`（所有 17 值的逗號字串）與 `_lastWrittenConfigKey` 比對，避免重複寫入。

### 讀取流程
`InspectionStatisticsService.LoadConfigForGrabId(root, grabId, hintFrom, hintTo)` → 逆向掃 CSV 找該序號上方最近的 `#CFG` → `CsvConfigSnapshot.TryParse` → 回傳快照。

### 用途
- 影像回顧拼接模式：`ShowStitchedCameraInCanvas` 用歷史 OPS/Pos/閾值更新 chartMura + chart1
- `ClearStitchedMode`：恢復 chart 為當前 `_settings`

---

## Period Charts 資料流

- `BtnSelectDataFolder_Click` → `PopulateChartNavigators()` → 填充 `cbChartYear` items，觸發 cascade → `OnChartYearIndexChanged` → `OnChartMonthIndexChanged` → `OnChartDayIndexChanged`
- 年/月/日切換：`cbChartYear/Month/Day.SelectedIndexChanged` 觸發對應 `OnChartXxxIndexChanged()`；Prev/Next 按鈕操作 `cbChartXxx.SelectedIndex`
- `_chartNavUpdating = true` 在 `PopulateChartNavigators` / `OnChartYearIndexChanged` / `OnChartMonthIndexChanged` 填充子 ComboBox 時設置，防止 `SelectedIndexChanged` 重複觸發 cascade
- `InspectionStatisticsService.ComputeGroupedByMonthOfYear/DayOfMonth/HourOfDay`：固定 12/31/24 個 bucket，空的顯示 0
- `FillPeriodChart` 動態軸：`niceMax = max(5, ceil(maxTotal/5)*5)`，5 等分格線，只顯示 0 和 niceMax 兩個標籤；AxisY 與 AxisY2 兩軸 Maximum/Interval 必須同步

### Period Charts 軸線架構（`InitOneChart` + `FillPeriodChart`）

- `AxisY`（Primary，左）：隱藏 label，但**驅動 MajorGrid**（dotted，Light Gray），`Minimum=0`，`Maximum/Interval` 與 AxisY2 同步
- `AxisY2`（Secondary，右）：`Enabled=True`，顯示右側 label（只顯示 0 和 niceMax），`MajorGrid.Enabled=false`
- StackedColumn series 綁 `YAxisType.Secondary`
- **StackedColumn 空白啟動問題**：無資料時整個 chart area 空白（無 grid/軸/刻度），因為沒有 X 類別。解法：`InitOneChart` 傳入 `xCount`/`xStart` 預填 zero-value 資料點（月份=1–12，日期=1–31，小時=0–23）；`FillPeriodChart` 先 `Points.Clear()` 再填真實資料，佔位符不影響正式資料
- `InnerPlotPosition`：`X=0, Y=12, Width=93, Height=66`（左邊界貼齊，上邊留 12% 給標題）

### FillPeriodChart 動態軸同步

```csharp
int maxTotal = data.Max(p => p.Pass + p.Fail);
int niceMax  = Math.Max(5, (int)(Math.Ceiling(maxTotal / 5.0) * 5));
double yMax  = niceMax * 1.05;
double yStep = niceMax / 5.0;
// AxisY 和 AxisY2 兩軸必須同步
area.AxisY.Maximum               = yMax;
area.AxisY.Interval              = yStep;
area.AxisY.MajorGrid.Interval    = yStep;
area.AxisY2.Maximum              = yMax;
area.AxisY2.Interval             = yStep;
area.AxisY2.LabelStyle.Interval  = niceMax;  // 只顯示 0 和 niceMax
```

預設 Y 軸最大值（無資料時）：chartYearly=60000，chartMonthly=2000，chartDaily=300。

### 年月日導航

`lblChartYear/Month/Day`（Label）已替換為 `cbChartYear/Month/Day`（`ComboBoxStyle.DropDownList`）：

```csharp
_chartNavUpdating = true;
cbChartYear.Items.Clear();
foreach (var y in _chartYears) cbChartYear.Items.Add(y.ToString());
cbChartYear.SelectedIndex = _chartYears.Count > 0 ? _chartYears.Count - 1 : -1;
_chartNavUpdating = false;
OnChartYearIndexChanged();   // 手動觸發一次 cascade
```

### Period Charts 統計邏輯
- `ScanCsvByDateRange` 以 `(GrabId, CamId)` 為單位分組，一票否決（任一張超標即 Fail）
- 與 `ComputeByGrabIdRange` 邏輯一致（序號基準統計）
