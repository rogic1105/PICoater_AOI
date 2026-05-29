# modify-data-stats

修改 Data tab 統計、CSV、Period Charts、跨 Tab 同步相關程式碼。

## 使用時機

修改 DataStatisticsPresenter、InspectionStatisticsService、InspectionLogService、Period Charts 或跨 Tab 同步邏輯時。

## 關鍵檔案

→ 見 `CLAUDE.md` §關鍵檔案速查（subset：`UI/Presenters/DataStatisticsPresenter` + `Services/Inspection*Service` + `Services/CsvConfigSnapshot`）。
→ Data tab 流程行為見 `docs/user-manual/ui-flow.html` §檢測報表（Data）。

## 注意事項

### listViewGrabDetail 點選 — MouseUp commit 模式

- **訂閱 `MouseUp` 不是 `SelectedIndexChanged`** — 按下時 PG 預設反白顯示「被選中」，放開（Left button）才 commit 切 grabId
- `OnGrabDetailRowCommitted` 內用 `_suppressRangeOnSingleSheetSync` flag 包住 `cbDataId.SelectedIndex = idx`
- `OnSingleSheetComboChanged` 看到 flag 跳過範圍 cb 同步（`cbDataIdStart`/`End` + `cbDataDateStart`/`Time` + `cbDataDateEnd`/`Time` 6 個）— **listView 點選時保留範圍 cb 不動**
- 其他路徑（< > 按鈕、直接改 cbDataId）正常同步範圍 cb
- `UpdateGrabDetailListView` 重填時不需要 unsubscribe/resubscribe（MouseUp 不被 Items.Clear/Add 觸發）

### Data tab 讀取資料 → Review tab 同步

- `btnDataSelectFolder` 觸發 `DataFolderSelected` event → `AniloxRollForm.OnDataFolderSelected`
- `OnDataFolderSelected` 是 async void，呼叫 `ResetAndLoadReviewAfterFolderChanged(dataPresenterAlreadySynced: true)` helper（與 Review tab `btnReviewSelectFolder_Click` 共用 helper）
- helper 內：state reset（合圖方式=全域、回顧強化=否）+ Live merge sync + chart series clear + DataPresenter `SyncGrabIdFromTime` + ClearStitchedMode + SetReviewGroupBoxes + SelectLatestInSingleSheetMode + LoadGrabStitchedViewAsync
- `dataPresenterAlreadySynced=true` 跳過 `SyncFromReviewFolder` 避免 duplicate load

### 統計模式（`_activeStatMode` 追蹤）
- 三模式：`GrpDataSingleSheet`（單片，cbDataId 驅動）、`GroupBoxGrabIdRange`（序號範圍，cbDataIdStart/End 驅動）、`GroupBoxTimeRange`（時序範圍，cbDataDateStart~cbDataTimeEnd 驅動）
- **三個 GroupBox 標題都可點切模式**（`SwitchActiveStatGroupBox`）— 與 Review tab 的 `grpReviewGrabNav.Click` 對等
- `btnDataSelectFolder` 預設進入 `GrpDataSingleSheet` 模式 + 最新一筆（descending [0]）— 與 Review tab `btnReviewSelectFolder` 行為對齊
- **序號模式**：`ComputeByGrabIdRange` — 分母=唯一序號數，同序號同相機一票否決
- **時間模式**：找時間範圍內 GrabIds → 同樣用 `ComputeByGrabIdRange`
- Period Charts（`ScanCsvByDateRange`）同樣用 (GrabId, CamId) 一票否決

### CSV 格式
- 路徑：`{Root}\{yyyy}\{yyyyMM}\{yyyyMMdd}.csv`
- 欄位：`Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,...`
- GrabId：`yyMMdd-HHmmss`（時間戳，字典序=時間序）
- CamId 從 FileName 提取：`fileName.LastIndexOf('-')` 後的數字
- `#CFG` 行格式：`#CFG,ISO-timestamp,key=value,...`

### Period Charts
- StackedColumn 綁 `YAxisType.Secondary`（AxisY2 右側顯示 label）
- AxisY（Primary）驅動 MajorGrid（只顯示 0 和 niceMax）
- 無資料時 `InitOneChart` 預填 zero-value 資料點（月1-12/日1-31/時0-23），防止整個 chart 空白
- `FillPeriodChart` 先 clear 再填真實資料
- `niceMax = max(5, ceil(maxTotal/5)*5)`，AxisY 和 AxisY2 兩軸 Maximum/Interval 必須同步
- chart.Tag = `"auto"` 代表 AutoScale 模式，null = FixedScale

### chartDataPatch（Mura 空間分布圖）
- **永遠**顯示「最新一筆」單 grab 的 stitch 視圖（不再多 grab 平均；多 grab 平均會稀釋峰值）
- 觸發點：`RefreshStats` → `UpdateMuraProfileChart(grabIds)` → 取 `grabIds[0]`（descending order = 最新）→ `UpdateMuraProfileForSingleGrab(info)`
- 資料來源：`LoadConfigForGrabId`（取該 grab 的 #CFG OPS/Pos）+ `LoadImagePathsForGrabId` + `CurveMergeHelper.MergeCurves`（合該 grab 內所有 capture）→ 與 `ReviewStitchCoordinator.UpdateStitchedOverviewChart` 同源 → chart 與 chartReviewPatch 對齊
- 不依賴 camReviewMain — Data tab 操作即時顯示對齊圖；Review tab 載入後 `SyncMuraProfileFromReview` 覆寫為同源資料，無視覺差

### CSV 讀寫並發保護
- 所有新增的 CSV reader 用 `InspectionStatisticsService.OpenCsvShared(path)` 而非 `new StreamReader(path)` — 內部用 `FileShare.ReadWrite` 對齊 writer 端，避免跨 process race（Storage PC 讀 vs Inspection PC 寫）。Writer 端 `InspectionLogService` 已同樣指定 `FileShare.ReadWrite`。
- 新增依時間掃 CSV 的方法時，要 `Array.Sort(csvFiles, StringComparer.Ordinal)` 使 captureHmV 跨日邊界正確沿用 — 路徑 `{yyyy}\{yyyyMM}\{yyyyMMdd}.csv` 字串序 = 時間序。

### PropertyGrid → Stats 重算的 debounce
- PropertyGrid 任何變更觸發 `RefreshStats + RefreshPeriodCharts` 都應走 `AniloxRollForm.ScheduleStatsRefresh()`（300ms debounce），而非直接呼叫 `_dataStatsPresenter.RefreshStats()`。
- 拖 slider 時連續變更會被合併為 1 次 CSV scan，避免每 tick full scan 造成 UI 凍結。
- 失敗時連續每 5 次彈 MessageBox 通知使用者（不淹沒對話框）。

### 跨 Tab 同步
| 方向 | Guard |
|------|-------|
| Review → Data | `_grabIdCrossGuard` |
| Data → Review | `_grabIdCrossGuard` |
| 時間 → GrabId | `_grabIdNavGuard` |
| `_chartNavGuard` | chart 年月日 cascade 填充時 |

### PropertyGrid 持久化
- 新增 Category/屬性 → 必須同步更新 `InspectionSettingsStore` 的 `SerializeJson`+`ParseJson`
- 不使用 `JavaScriptSerializer`（`user.config` 損毀時會拋例外）

## 步驟

1. 讀取 DataStatisticsPresenter 中相關方法
2. 確認統計模式是否受影響
3. 修改 + build 驗證
