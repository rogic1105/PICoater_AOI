---
name: modify-data-stats
description: Modify the Data tab, inspection CSV schema, statistics, report lists, period charts, range curves, or cross-tab selection synchronization. Use for report performance and data compatibility work.
---

# modify-data-stats

修改 Data tab 統計、CSV、Period Charts、跨 Tab 同步相關程式碼。

## 使用時機

修改 DataStatisticsPresenter、InspectionStatisticsService、InspectionLogService、Period Charts 或跨 Tab 同步邏輯時。

## 關鍵檔案

→ 見 repo 根 `AGENTS.md` §關鍵檔案速查（subset：`UI/Presenters/DataStatisticsPresenter` + `Services/Inspection*Service` + `Services/CsvConfigSnapshot`）。
→ Data tab 流程行為以 code 為準。

## 注意事項

### listViewGrabDetail 點選 — MouseUp commit 模式

- **訂閱 `MouseUp` 不是 `SelectedIndexChanged`** — 按下時 PG 預設反白顯示「被選中」，放開（Left button）才 commit 切 grabId
- `OnGrabDetailRowCommitted` 內用 `_suppressRangeOnSingleSheetSync` flag 包住 `cbDataId.SelectedIndex = idx`
- `OnSingleSheetComboChanged` 看到 flag 跳過範圍 cb 同步（`cbDataIdStart`/`End` + `cbDataDateStart`/`Time` + `cbDataDateEnd`/`Time` 6 個）— **listView 點選時保留範圍 cb 不動**
- 其他路徑（< > 按鈕、直接改 cbDataId）正常同步範圍 cb
- `UpdateGrabDetailListView` 重填時不需要 unsubscribe/resubscribe（MouseUp 不被列表重建觸發）

### 大量資料處理（預估常駐 ~1 萬筆）

- **`listViewGrabDetail` = `VirtualMode`**：不再逐筆建 `ListViewItem`，只存 `_visibleDetails`（List<GrabDetail>）+ 設 `VirtualListSize`；`RetrieveVirtualItem` → `BuildGrabDetailListViewItem(index)` 按需即時產生可見列。點選改用 `SelectedIndices[0]` 對應 `_visibleDetails[index].GrabId`（**不可用 `SelectedItems[0].Text`**，virtual 下不可靠）。
- **owner-draw 樣式不變**：`DrawSubItem` 照樣讀 `e.Item.Tag`（rowHasFail）畫紅綠底 + 選中外框；symbol 用 unicode `—`/`○`/`×`（無資料/正常/異常），**勿降級成 ASCII**。
- **欄寬 = `FitGrabDetailColumnsToContent`**：VirtualMode 下 `lv.Items` 為空，`AutoResizeColumns(ColumnContent)` / 量 Items 的 `FitListViewColumnsProportional` 都失效 → 改用 `_visibleDetails` 取樣量測，還原「貼齊內容緊湊欄寬」觀感。
- **4 個 grabId combo 批次填充**：`DataDateGrabIdNavigator.PopulateAllGrabIdCombos` 用 `BeginUpdate` + `Items.AddRange(object[])`（**非逐筆 Add**），一萬筆時重繪 4 萬次 → 4 次，避免每次載入/換日期 UI 凍住。

### camData1~7 良率色卡（`InspectionStatsPresenter`）

- **一張卡片 = sdk `TanukiCv.Controls.ColorTextCard`**（雙緩衝自繪控制項：底色 + 上/中/下三行字全部在單一 `OnPaint` 一次畫完）。`InspectionStatsPresenter` 算好門檻顏色 + `CAM{i}`/良率/Pass-Fail 字串 → `card.SetContent(back, top, center, bottom)`。換色 → 一次 `Invalidate` → **原子重繪**，不會半紅半綠。
- **反模式（已修，勿回退）**：舊版用 3 個 Dock 的 `Label` 疊在 `Panel` 上，換色時 panel + 3 label 各自非同步重繪 → 「上半綠下半紅」俄羅斯方塊 flicker。透明或實色 label 都一樣（多控制項 = 多重繪單位）。**單一自繪控制項才是根治**。
- **機制/政策邊界**：`ColorTextCard` = sdk 通用機制（純畫「一塊底色 + 三行字」，不知良率/相機）；顏色門檻（≥95%綠 / ≥80%橙 / <80%紅 / 無資料灰）+ CAM 命名 + 良率算法留 app（政策，`InspectionStatsPresenter`）。

### Data tab 讀取資料 → Review tab 同步

- `btnDataSelectFolder` 觸發 `DataFolderSelected` event → `AniloxRollForm.OnDataFolderSelected`
- `OnDataFolderSelected` 是 async void，呼叫 `ResetAndLoadReviewAfterFolderChanged(dataPresenterAlreadySynced: true)` helper（與 Review tab `btnReviewSelectFolder_Click` 共用 helper）
- helper 內：state reset（合圖方式=全域、回顧強化=否）+ Live merge sync + chart series clear + DataPresenter `SyncGrabIdFromTime` + ClearStitchedMode + SetReviewGroupBoxes + SelectLatestInSingleSheetMode + LoadGrabStitchedViewAsync
- `dataPresenterAlreadySynced=true` 跳過 `SyncFromReviewFolder` 避免 duplicate load

### 統計模式（`_activeStatMode` 追蹤）
- 三模式：`GrpDataSingleSheet`（單片，cbDataId 驅動）、`GroupBoxGrabIdRange`（序號範圍，cbDataIdStart/End 驅動）、`GroupBoxTimeRange`（時序範圍，cbDataDateStart~cbDataTimeEnd 驅動）
- **三個 GroupBox 標題都可點切模式**（`SwitchActiveStatGroupBox`）— 與 Review tab 的 `grpReviewGrabNav.Click` 對等
- `btnDataSelectFolder`/`btnReviewSelectFolder` 讀取資料後預設 `GrpDataSingleSheet` 模式：**單片顯示最新一筆**（`cbDataId` descending [0]）；**序號範圍預設「起始 cbDataIdStart=最舊、結束 cbDataIdEnd=最新」**（切範圍模式即涵蓋全部；明細列表隨 start/end 連動＝顯示全部）。兩路徑共用 `SelectLatestInSingleSheetMode()`（在 `StatComboGuard` 內設 start/end/cbDataId、不觸發 `OnSingleSheetComboChanged`；`SetActiveStatGroupBox` 顯式切模式、`RefreshStats` 由 caller 呼）。**勿在 `PopulateAllGrabIdCombos` guard 外再設 `cbDataId`＝會誤觸發 handler 把 start 拉回最新**
- **年/月/日期間可點設範圍**（`DataDateGrabIdNavigator`，`GrabIdRangeSource` enum：Global/Year/Month/Day/Custom）：點 `lblChartNavYear/Month/Day`（浮雕 Fixed3D 小晶片 + 手指游標）→ cbDataIdStart/End 只取該期間（值取自 `cbDataYieldYear/Month/Day`）；**範圍模式再點同一 active label → 轉 Custom 解除綁定，保留目前起訖且不重算**；點 `groupBoxGrabIdRange`→全局；手動拖範圍→Custom。**互斥高亮**：同時只有一個來源綠（`SetChipActive`/`SetGroupBoxActive` 同色 `_activeGrpFill/_activeGrpBorder`）；解除綁定時全滅。**單片 toggle 記憶**＝`_rangeSource` 保留 + 範圍 cb 不被單片動（見 [[cbDataId 取消同動]]）→ 回範圍自動還原。**active 來源的 cbDataYield 改變→範圍跟著更新**（`OnPeriodComboChangedForRange`，Custom/非 active 來源/串聯不觸發）
- **序號模式**：`ComputeByGrabIdRange` — 分母=唯一序號數，同序號同相機一票否決
- **時間模式**：找時間範圍內 GrabIds → 同樣用 `ComputeByGrabIdRange`
- Period Charts（`ScanCsvByDateRange`）同樣用 (GrabId, CamId) 一票否決

### CSV 格式
- 路徑：`{Root}\{yyyy}\{yyyyMM}\{yyyyMMdd}.csv`
- 欄位：`Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,...`
- GrabId：`yyMMdd-HHmmss`（時間戳，字典序=時間序）
- CamId 從 FileName 提取：`fileName.LastIndexOf('-')` 後的數字
- `#CFG` 行格式：`#CFG,ISO-timestamp,key=value,...`
- **Curve SSoT 與報表索引**：`MeanC/MaxC .bin` 是完整 Curve 樣本的真實來源；CSV
  `MeanPeak/MaxPeak/MaxCMean/MeanRPeak/MaxRPeak` 是 capture 當下從同份 bin 資料算出的可重建標量索引。
  報表明細第 8 判定欄固定為「列」：Row peaks 缺少時顯示 `—`，不得當 Pass；存在時以
  `HM_V_capture/HM_H_current` 與當前列門檻重判，同序號任一相機／capture Fail 即為 X。
  報表調整欄正規值或 Mean/Max 門檻時，以 CSV 索引重判全資料，不逐筆重讀 bin；
  不改寫歷史 CSV。若外部修改 bin，必須同步重建 CSV 索引，不得讓 Curve 與判定分歧。

### Period Charts
- StackedColumn 綁 `YAxisType.Secondary`（AxisY2 右側顯示 label）
- AxisY（Primary）驅動 MajorGrid（只顯示 0 和 niceMax）
- 無資料時 `InitOneChart` 預填 zero-value 資料點（月1-12/日1-31/時0-23），防止整個 chart 空白
- `FillPeriodChart` 先 clear 再填真實資料
- `niceMax = max(5, ceil(maxTotal/5)*5)`，AxisY 和 AxisY2 兩軸 Maximum/Interval 必須同步
- chart.Tag = `"auto"` 代表 AutoScale 模式，null = FixedScale

### chartDataColumn（Mura 空間分布圖）
- 單序號模式每格都顯示該 grab 的完整 Curve，不得用 debounce/latest-only 掠過中間序號；範圍模式則顯示 50 筆 Mean 均勻候選與 50 筆 MaxCMean 排名候選。
- 報表列 Curve 只屬單序號模式；與欄 Curve 共用單序號 profile/cache/prefetch，並沿用
  `RowCurveChartHelper + RowCurveDisplayAdapter`。切到序號範圍或年/月/日範圍必清空，不得保留上一筆。
- 單序號資料來源：`InspectionConfigRepository.LoadForGrabId`（#CFG OPS/Pos）+ `SingleGrabCurveSummaryStore`；匯總缺少／失效時才走 `InspectionImagePathRepository.LoadForGrabId`，最多 2 台相機並行執行欄／列 bin 合併，先顯示 Curve，再由單一背景 writer 原子寫回。
- `.mcsf` 匯總是可重建 materialized view，只保存 rescale 前的逐相機 MeanC 平均／MaxC 最大；ACAP curve record
  是新資料 SSoT，舊資料則 fallback 原始 MeanC/MaxC bins。格式版本、grab 時間範圍或相機數不符時不得使用舊匯總。
- 只有所有預期 capture 的 MeanC/MaxC 都成功讀取（`merged == captures`）才可落匯總；remote copy 未完成、
  ACAP record／舊 bin 損壞時記 `skip-incomplete`，避免固化部分資料。
- 匯總 writer 平常對序號互動讓路，pending raw profile 達 72 MB 才 pressure drain，96 MB 為硬上限；不可在每格同步 `Flush(true)`，也不可用無界背景 task 製造記憶體或磁碟競爭。單序號 raw Curve LRU 為 512 筆／256 MB，30,000 筆資料仍須維持固定上限。
- `SingleGrabCurveCache` 只保存 rescale 前的完整 Mean/Max 合併結果（LRU 64 筆／64 MB），相同 key 的前景與背景載入 single-flight；Presenter 依滾動方向只預讀下一個未命中相鄰序號，資料夾重載時必清空。
- cache 命中後必 clone 再做 view-time Hessian rescale，禁止直接修改 raw cache；設定變更可用同一 raw Curve 重新套目前比例與門檻。
- 不依賴 camReviewMain — Data tab 操作即時顯示對齊圖；Review tab 載入後 `SyncMuraProfileFromReview` 覆寫為同源資料，無視覺差

### CSV 讀寫並發保護
- 所有 CSV reader 共用 `InspectionCsvReader.OpenShared(path)`，資料列共用 `TryParseRecord`；不得在統計／回顧／curve 查詢另寫 `Split(',')` parser。Reader 內用 `FileShare.ReadWrite` 對齊 writer，避免 Storage PC 讀與 Inspection PC 寫的跨 process race；`InspectionLogService` writer 端同樣指定 `FileShare.ReadWrite`。
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
