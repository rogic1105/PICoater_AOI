# Code Review 報告 — 2026-05-15

對最近 5 個 commit（10f0b6d V/H 分離 / 9f9ee47 GroupBox 可點 / 866d02e repo 整理 / 0e01f95 listView fix / 9dbac3e dead code 清理）以及周邊整體進行 review。

## 摘要

- 總共找到 23 個問題
- Severity 分布：Critical 2 / High 6 / Medium 9 / Low 6

---

## Critical（壓力測試前必修）

### C1. `SwitchActiveStatGroupBox` 切到 TimeRange 不重設範圍 → 用單 grab 的時間切片掃整個 CSV

`DataStatisticsPresenter.cs:1313-1327`。`SwitchActiveStatGroupBox(_ctx.GroupBoxTimeRange)` 只切 active 旗標就 `RefreshStats()`，從未調整 `CbStartDate/EndDate/StartTime/EndTime`。
若使用者先在「單片」模式停留（PopulateAllGrabIdCombos 之後 SingleSheet 已把 start/end date-time combos 設為單 grab 的 Earliest/Latest，見 `OnSingleSheetComboChanged` line 408-409），再點「時序範圍」標題切過去，會用「單一秒級時間範圍」去 `TryParseStatDateTime` → 結果只命中 1 個 grab，**統計畫面假裝是「時序模式」但內容跟單片完全一樣**，使用者必然以為 bug。
建議：切到 TimeRange 時把 start = `_statAvailableTimes.Min`、end = `_statAvailableTimes.Max`（包 `StatComboGuard`），跟切到 GrabIdRange 時 spread 到 [最舊, 最新] 對稱處理。

### C2. `BatchInspectionService` 是 IDisposable 但 Form 從未 Dispose

`BatchInspectionService.cs:12` 宣告 `IDisposable`、line 133 有 `Dispose()`（內含 GPU pipeline 釋放）。Form 內 `OnFormClosing` (line 168) 與 `FormClosed` (line 1028) 都沒釋放 `_inspectionService`；commit 9dbac3e 刪掉 `FormInteractionHelper.CleanupSystem` 後沒人接手。
影響：每次關閉程式 CUDA pipeline / pinned buffer 不會主動釋放，雖然 process 結束 OS 會回收，但 hot-restart 或單元測試環境可能洩漏。
建議：`OnFormClosing` 或 `FormClosed` 加 `_inspectionService?.Dispose()`（在 `_liveCameraManager.FreeCameras()` 之後，依賴關係安全）。

---

## High（建議壓力測試前修）

### H1. `OnFormClosing` 與 `FormClosed` 雙路徑釋放資源，順序不一致

`AniloxRollForm.cs:168-177` 與 `:1028-1043` 各自釋放部分資源：
- `OnFormClosing` 釋放 `_plcGrabController` / `_telemetryTimer` / `_liveOverviewTimer` / `_lightController`。
- `FormClosed` 又重複釋放 `_plcGrabController` / `_lightController`，再多釋放 `_retentionService` / `_remoteCopyService` / `_cleanupFlagWatcher` / 相機。
雙路徑造成「`_lightController` 釋放 2 次」（雖然 `?.Dispose()` 容錯）、且 `FormClosed` 的 `await _plcGrabController.StopAsync()` 在 `OnFormClosing.Dispose()` 之後變成在 already-disposed 物件上 await。
建議：整合到單一路徑（Closing 早期停取像 + Stop Tasks，Closed 統一 Dispose），避免「同物件被 dispose 兩次」與「async 過 dispose」競態。

### H2. 跨 process race：InspectionLogService 寫 CSV 時 InspectionStatisticsService 讀 CSV 無 sharing 控制

`InspectionLogService.cs:103` 用 `new StreamWriter(csvPath, append: true, new UTF8Encoding(false))`（預設 FileShare.Read），但 `InspectionStatisticsService.Compute*`（多處）用 `new StreamReader(csvPath)`（預設 FileShare.Read）。同一 process 同一 lock 內寫，但 Storage PC 可能在 inspection 寫入時讀同份檔。
更嚴重：PropertyValueChanged 一次連動 `RefreshStats` + `RefreshPeriodCharts`，**Live grab 進行中觸發 `ForceWriteConfig`** 寫 CSV 的同時，Data tab UI 執行緒可能正在掃描同一個 CSV（Compute*），雖各自加 FileShare.Read 預設行為相容，但 .NET StreamWriter 行為在 append 並沒明確指定 share mode，**Windows 上是 FileShare.Read** 但仍是隱性依賴。
建議：所有 StreamReader 明確指定 `new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite)`，可在 `InspectionStatisticsService.TryLoadBin` (line 303) 已這樣做；CSV 讀路徑沒有同等保護。

### H3. PropertyValueChanged 連環觸發掃整個 CSV 的方法 5+ 次

`AniloxRollForm.cs:2089-2192`：一次 PropertyGrid 改值會呼叫：
1. `_liveCameraManager.SetCaptureSettings(_settings)`（包含 GPU pipeline 重設）
2. 8 個 chart helper `SetThresholds` / `SetOps`
3. `RefreshMuraProfileForSettingsChange`（單片模式 → 重讀 .bin + #CFG）
4. `UpdateStitchedOverviewChart` + `RefreshCurrentCameraChartsForSettingsChange`（rescale）
5. `RefreshStats()` — **掃整個 captureRoot 目錄所有 CSV**（GrabIdRange / Time 兩條路徑都掃）
6. `RefreshPeriodCharts()` → `OnChartYearIndexChanged` 連環掃 CSV（Year + Month + Day 三次）

對 100K+ 筆的歷史資料，改一個閾值會觸發 4 次 full CSV scan + 1 次目錄遞迴 `Directory.GetFiles`，UI 凍結可能 >1 秒。
建議：(a) RefreshStats 與 RefreshPeriodCharts 共用快取（如 ScanCsvByDateRange 結果），或 (b) `PropertyValueChanged` 用 debounce（250ms 內合併多次變更），或 (c) 把 RefreshStats 改成 `Task.Run` + 完成後 `BeginInvoke` 更新 UI。

### H4. PopulateStatDateCombos 包 Guard 但 manual 操作仍會誤觸 Time 模式

`DataStatisticsPresenter.cs:246-272` 已用 `StatComboGuard.Enter()` 防止程式化填充誤觸 `OnStartComboChanged → SetActiveStatGroupBox(TimeRange)`。但 `OnStartComboChanged` 與 `OnEndComboChanged` 開頭就 `SetActiveStatGroupBox(_ctx.GroupBoxTimeRange)`（line 299, 314）— 使用者**手動**選 cbStartDate 時會強制切到 TimeRange，這沒問題。但 `OnSingleSheetComboChanged`（line 408-409）內也會程式化更新 cbStartDate/Time 與 cbEndDate/Time，包在 StatComboGuard 內所以 OnStartComboChanged 不會跑 — 邏輯**正確**，但極脆弱：未來若有人在 `SetCombosToDateTime` 加新事件 hook，guard 邊界容易破。
建議：把 mode-switch 邏輯從 ComboChanged handler 移到一個更明確的「user-initiated only」函式（如 `SetCombosToDateTime` 改成不會 fire `SelectedIndexChanged`，採 silent assignment + 手動 RefreshStats 路徑）。

### H5. `InitializeRightPanelControls` / `SetupDataTab` 之間：listViewGrabDetail.SelectedIndexChanged 可能 row 0 自動選

`DataStatisticsPresenter.cs:634` `lv.SelectedIndexChanged += OnGrabDetailRowSelected`。`UpdateGrabDetailListView` 重填明細時 `lv.Items.Add(...)` 預設不選任何 item，所以理論上不會觸發。但 WinForms `ListView` 開啟 `MultiSelect=true`（預設）時，BeginUpdate/EndUpdate 之間若內部還原 selection（看 native 行為），會 fire 一次空的 SelectedIndexChanged。
`OnGrabDetailRowSelected` 開頭 `if (StatComboGuard.IsSet) return;` 沒防住這個情境（RefreshStats 不在 StatComboGuard 內）。極端情況：RefreshStats 後 listView 重填，剛好 user 已選了某 row，selection 跨重填保留 → 觸發 `OnGrabDetailRowSelected`，又把 cbDataGrabId 設一次。
建議：`UpdateGrabDetailListView` 開頭 `lv.SelectedIndexChanged -= OnGrabDetailRowSelected; ... lv.SelectedIndices.Clear(); lv.SelectedIndexChanged += OnGrabDetailRowSelected;`。

### H6. `ApplyHessianRescale` ratio=1 也跑了 noOp 判斷但只有 `ApplyHessianRescale` 有，`ApplySingleCurveRescale` 也有，但分散在三處：DataStatisticsPresenter / ReviewStitchCoordinator 各定義一份

`DataStatisticsPresenter.ApplyHessianRescale` (line 831) / `ReviewStitchCoordinator.ApplySingleCurveRescale` (line 313) / `ReviewStitchCoordinator.Clone1DAndRescale` (line 503) / `ReviewStitchCoordinator.CloneAndRescale` (line 366) — **4 個極相似的 rescale 函式**散在兩個檔案，公式雖一致但易漂移（未來若改 ratio 計算邏輯，必須 4 處同改）。
建議：抽出 `HessianRescaleHelper` 靜態類別到 `Core.Services` namespace，共用一份實作。

---

## Medium（後續迭代修）

### M1. ui-flow.html 沒描述 listViewGrabDetail row click → cbDataGrabId 同步

`docs/user-manual/ui-flow.html` grep 找不到 `listViewGrabDetail.*Click` 或 row-selection flow。commit 0e01f95 加了這個 feature，docs 未補。違反 CLAUDE.md 提到的「三方同步機制」。
建議：在 ui-flow 「點選明細列表」加一條 flow：`listViewGrabDetail row 點選 → cbDataGrabId 對齊序號 → OnSingleSheetComboChanged → 切換為單片模式 + 刷新 chartMuraProfile + sync cbReviewGrabId`。

### M2. ui-flow.html 沒描述「機台角色」AppRole 變更後重啟 + Storage PC 模式

`AniloxRollForm.cs:2127-2135` AppRole 變更 → MessageBox + 寫 app-mode.json + 重啟生效。ui-flow.html grep 找不到 `AppRole` 或「機台角色」。CLAUDE.md 速查表也只在「機台設定」一節提到。
建議：補一條 Storage / Inspection 雙模式啟動 flow。

### M3. CleanupFlagWatcher / CleanupSystem 路徑：Storage PC 的 cleanup 沒在 OnFormClosing 處理

`AniloxRollForm.cs:1042` `_cleanupFlagWatcher?.Dispose()` 在 `FormClosed`。但 `OnFormClosing` 先觸發、`FormClosed` 後觸發，Closing 階段 Watcher 還在跑（每 10 秒輪詢），若此時 form 已 dispose 控制項，Watcher callback 內若有 BeginInvoke 會炸。
建議：在 `OnFormClosing` 早期 Stop `_cleanupFlagWatcher`。

### M4. CsvConfigSnapshot 沒處理 V/H 都缺 + legacy 也缺的情況

`CsvConfigSnapshot.cs:213-218`：legacy fallback 邏輯有，但若新格式 V 寫了 H 沒寫（介於新舊間的混合資料）、且 legacy 鍵也沒有，**`hessianH` 會留 0**。`ThresholdContext.IsFail` 對 `captureHmV=0` 用 ratio=1，但 H 方向 chart rescale 若拿 currentHmH=0.3 ÷ captureHmV=0 → 仍 ratio=1（has 0 check），語意上 H 曲線退回顯示原 V baked-in 值（不對但不會炸）。
建議：fallback 規則明確：若 V 有 H 無，`hessianH = hessianV`（同公式）；若 V 無 H 無也無 legacy，0 → 用 `InspectionDefaults.HessianMaxFactorH`。

### M5. `ScanCsvByDateRange` 的 #CFG 跟蹤是「per-CSV-file」而非全域

`InspectionStatisticsService.cs:573-575` 每進入新 CSV 檔重設 `captureHmV = ctx?.CurrentHmV ?? 0f`。語意問題：若 day1.csv 結尾沒 #CFG（資料在 #CFG 前），day2.csv 開頭也沒 #CFG → day2 開頭資料用 `currentHmV` 重算（ratio=1），實際上應沿用 day1 最後的 #CFG。日邊界的 row 會誤判 Pass/Fail。
建議：(a) 每 CSV 開頭強制寫 #CFG（commit 描述提及但實作沒驗證每天第一筆都有 #CFG）；或 (b) 跨檔保留 captureHmV，CSV 列表先按日期排序。

### M6. `LoadGrabIdInfos` 與 `LoadImagePathsForGrabId` 兩處都 `Directory.GetFiles(...SearchOption.AllDirectories)` 沒過濾日期目錄

掃 CaptureRoot 整個樹（包含舊年資料夾、未來年資料夾）。100K 圖、500 天資料的場景，一次資料夾載入要掃 500+ CSV 並 parse。
建議：用 `LoadAvailableTimes` 結果做日期過濾，只掃涉及的 yyyyMMdd.csv。

### M7. `_currentDetails` / `_grabIdInfos` 跨 thread 沒保護

`DataStatisticsPresenter._currentDetails` 由 UI thread `RefreshStats` 寫（line 551, 578, 585），但 `ApplyFailFilter` 是 UI thread 讀（OK）。**不是直接 race**，但 commit 0e01f95 引入的 `OnGrabDetailRowSelected` 在 UI thread；如果未來把 RefreshStats 改 async + Task.Run（H3 建議），這裡會立刻變 race。
建議：現在 OK，但 H3 重構時要一併處理。

### M8. `_liveCurveMean` / `_liveCurveMax` 跨 thread + volatile bool

`AniloxRollForm.cs:157, 1312, 3073`：grab callback thread 寫 array reference，UI timer thread 讀（`LiveOverviewTimer_Tick`）。`_liveOverviewDirty = volatile bool` 但 array 元素本身沒 memory barrier，雖然 race window 小、結果只是「短暫看到舊曲線」可接受，但建議至少在 `OnLiveCurveData` 寫之後 `Interlocked.MemoryBarrier()`。

### M9. PopulateAllGrabIdCombos 不寫 cbDataGrabId 但 SyncFromReviewFolder 不傳 selectDataGrabId

`DataStatisticsPresenter.cs:202` `LoadDataFolder` 呼叫 `PopulateAllGrabIdCombos(selectDataGrabId: false)`，line 232 `SyncFromReviewFolder` 不傳參（用預設 false）。LoadDataFolder 之後立刻 line 219 又設 `_ctx.CbDataGrabId.SelectedIndex = 0`，看似冗餘但其實 `selectDataGrabId: false` 不會選 → 然後手動選 → 兩條路徑有微小語意差別。
建議：直接 `PopulateAllGrabIdCombos(selectDataGrabId: true)` 並移掉 line 217-220 的手動指派。

---

## Low（nice-to-have）

### L1. `LiveCameraManager.SetExposureForAll` / `SetGrabHeightForAll` 是 dead code

`find_dead_code.py` 偵測到，未被任何地方呼叫（不是 framework override）。commit 9dbac3e 應該刪除但漏掉。
建議：刪除（或加上 `[Obsolete]` 註記若未來有需要）。

### L2. `_propertyGrid_PropertyValueChanged` 太長（>100 行）

`AniloxRollForm.cs:2089-2192`，超過 100 行 + 8+ 個分支。建議拆成 `OnAppRoleChanged` / `OnRecipeChanged` / `OnStitchModeChanged` / `OnChartSettingsChanged` / `OnLightChanged` 5 個 sub-handler。

### L3. CLAUDE.md 仍寫 `BackgroundSampleRows` 在 InspectionSettingsStore.ParseJson 中是 legacy key — 但已不見

Grep 找不到 `BackgroundSampleRows` 在 ParseJson 中（`InspectionSettingsStore.cs:225` 改成 `obj.Contains("BackgroundSampleSeconds") ? ... : DefaultValue` — **沒有 fallback 到 BackgroundSampleRows**）。CLAUDE.md 對應段落正確（只說 BackgroundSampleSeconds），但 review 重點提到的「fallback 漏網」實際上不存在。
建議：（無動作；確認 Done。）

### L4. find_dead_code.py 偵測到 6 個候選，其中 `PreFilterMessage` / `GetEditStyle` / `EditValue` / `GetStandardValues` 是 framework override（之前文件已紀錄）

`docs/dev/dead-code-candidates.md` 已標記為「保留」。但 SetExposureForAll / SetGrabHeightForAll 該刪未刪（見 L1）。

### L5. CsvConfigSnapshot `ContentKey` 沒含 V/H 區分標記

`CsvConfigSnapshot.cs:73-93`：legacy CSV 與新 CSV 的 ContentKey 都會包含 6 個 ErrorValue 鍵與 2 個 HM 鍵，若 legacy CSV 解析後 V=H=同 legacy 值，ContentKey 仍能正確 hash。OK，但 `F4` 格式對極小值（如 0.0001）會截斷成相同 key。
建議：用 `R`（round-trip）或 `G9`/`G17` 格式。

### L6. `MIL_API_Reference.md` / `system-resources.md` 在 `docs/dev/` 但 CLAUDE.md 把它們列為「僅供查閱，不自動載入」— OK，但路徑前綴變了（多了 CLProtocol/ Grabber/ LTS_3DPA24/ 廠商文件）— CLAUDE.md 沒更新文件樹

CLAUDE.md `docs/` 目錄定位章節 `docs/dev/` 描述沒提到新加的子目錄。
建議：補一行 `├── dev/ (API、硬體規格、CLProtocol/Grabber/LTS_3DPA24 廠商文件)`。

---

## 沒問題的部分（review 後確認 OK 的）

- **View-time rescale 公式**：V chart 用 `HM_V_capture / HM_V_current`、H chart 用 `HM_V_capture / HM_H_current`、`chartMuraProfile` 單 grab 模式正確 rescale、aggregate 模式正確跳過。語意一致，註解清楚。
- **chartLiveOverview**：Live 自己產生資料 `_liveCurveMean[]`，無「baked-in」概念，rescale ratio=1 自然 OK。`LiveOverviewTimer_Tick` 不需要 rescale 路徑。
- **ThresholdContext 串通性**：`Compute` / `ComputeByGrabIdRange` / `ComputeDetailedByGrabIdRange` / `ScanCsvByDateRange` (via `ComputeGroupedByMonth/Day/Hour`) 全部接 optional `ctx`。`DataStatisticsPresenter.BuildThresholdContext` 在 4 條路徑都正確傳入（RefreshStats、OnChartYearIndexChanged、OnChartMonthIndexChanged、OnChartDayIndexChanged）。
- **CsvConfigSnapshot legacy `ErrorValueMean`/`HessianMaxFactor` fallback**：邏輯正確，V 與 H 都填 legacy 值（line 213-218）。新 CSV 寫 6+2 個鍵，向後相容完整。
- **InspectionLogService CSV 寫入 lock**：`_csvLock` 全 instance 共用，AppendRecord 與 ForceWriteConfig 都包進去，沒漏。
- **bin v1 vs v2**：v1（capture 曲線，無 light/exposure）vs v2（背景 bin，含 lightLevel/exposureUs）兩者用途不同，不是同一檔的不同版本，併存合理且 reader 已處理 backwards-compat。
- **`PopulateStatDateCombos` 加 StatComboGuard**：commit 0e01f95 修正生效，listViewGrabDetail 不再爆量。
- **`OnSingleSheetComboChanged` ↔ `OnReviewGrabIdChanged` 循環**：用 `GrabIdCrossGuard` 雙向防護 + `SyncDataGrabIdFromReview` 包 cross guard。實作正確、無無窮循環。
- **跨 thread BeginInvoke 都檢查 `IsHandleCreated || IsDisposed || Disposing`**：`OnLiveCurveData` (line 1326) / `OnLiveRowCurveData` (line 1373) / PLC callback (line 367-388) 都有檢查，OK。
- **`StorageRetentionService` 事件驅動清理 + Storage PC `CleanupFlagWatcher` 雙模式**：實作完整。
- **dead code 清理**：commit 9dbac3e 一次刪 31+1，find_dead_code.py 重跑只剩 framework override + 2 個遺漏（見 L1）。整體乾淨。

---

## 不確定 / 建議使用者親自確認

### U1. PropertyValueChanged 觸發 `_inspectionLogService.ForceWriteConfig` 的時機

`AniloxRollForm.cs:2120-2122` 只在 `IsLiveGrabbing` 時寫 #CFG。如果現場操作流程是「停止抓取後改參數，馬上重新抓取」，新的 #CFG 會在下一張圖寫入時（`AppendRecord` 內偵測 ContentKey 變更）才插入 — 行為正確但有 1 張圖的延遲。請確認這是否符合操作員期待。

### U2. V/H 分離後，native pipeline 只用 V — 但 H 方向的 capture-time 是否也用相同 V 的 HM？

native 註解（CsvConfigSnapshot.cs:18-21）寫「`HessianMaxFactorV` 是同時送 native 的單一 HM，H 為 view-time only」。意即即使 `RidgeDir=Horizontal`，capture 時送進 GPU 的 HM 仍是 V 的值。確認這是預期行為（V 永遠是 reference normalization、H 僅 view-time tunable）。

### U3. `SwitchActiveStatGroupBox` 切到 TimeRange 範圍應該 reset 到「資料夾全範圍」還是「上次 TimeRange 模式的設定」？

C1 建議重設到 [min, max]，但使用者習慣若是「上次 TimeRange 設定」，應該另存一份備份在 `_lastTimeRange` 而非每次 reset。請確認操作習慣。

### U4. `Hessian` 設為 0 的邊界

PropertyGrid 是否允許輸入 0？`ApplyHessianRescale` 有 `captureHm <= 0f` 檢查（會跳過 rescale），但若 currentHm=0 設定後，view-time 顯示曲線會保留 capture-time baked 值（不縮放）— 視覺上看起來「正規值無效」，可能讓使用者誤以為功能壞掉。
建議：PropertyGrid 加 `[Range(0.001f, 10f)]` 或 Validate 在 InspectionSettings 防 0。
