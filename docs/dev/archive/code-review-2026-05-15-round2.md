# Second-Pass Code Review — 2026-05-15

第二輪 review 對象：第一輪 18 個修法（commit `fee451f` C1+C2、`2294621` 批次 H/M/L）以及 11 個今日 commit 累積的整體狀態。

## 第一輪 18 個修法驗證（A）

| 編號 | 狀態 | 位置/註解 |
|---|---|---|
| **C1** SwitchActiveStatGroupBox TimeRange 重設 | ✅ 正確 | `DataStatisticsPresenter.cs:1314-1318`：切到 `GroupBoxTimeRange` 時呼叫 `PopulateStatDateCombos(_statAvailableTimes.Min, _statAvailableTimes.Max)`，內部已包 `StatComboGuard`。與 GrabIdRange 攤開行為對稱。⚠️ 小注意：若 `_statAvailableTimes.Count == 0`（資料夾空）則整段不做，後面 `RefreshStats()` 仍跑會用 stale combo 值 — 但這是 cold path、影響 <1%。 |
| **C2** `_inspectionService?.Dispose()` 在 FormClosed | ✅ 正確 | `AniloxRollForm.cs:1045-1047`：`FreeCameras` 之後、`_lightController?.Dispose()` 之前。依賴順序安全（相機釋放後再 dispose GPU pipeline 不會 race）。 |
| **H1** Form close 雙路徑分離 | ⚠️ 不完整 | `OnFormClosing` (168-184) 只 Stop 活動 + Dispose `_cleanupFlagWatcher`；`FormClosed` (1035-1051) 統一 Dispose。但 `RegisterStorageModeCleanup` (1269-1277) 又額外掛一個 `FormClosed`，再次 `_cleanupFlagWatcher?.Dispose()`。已 null 不會 NRE（OnFormClosing 已設 null），但兩個 FormClosed handler 並存且順序未定義（Storage mode 路徑），語意脆弱。**`_plcGrabController.StopAsync()` await 在 dispose 後**的問題：FormClosed 內現在先 `await StopAsync()` 再 `Dispose()`，OK，但 PLC 在 OnFormClosing 階段沒被 Stop（只 StopGrab on _liveCameraManager），若 PLC 在 await 期間有 callback 來，會走 BeginInvoke → 此時 Form Handle 可能已銷毀 → swallow（已有 `IsHandleCreated` 檢查），可接受。 |
| **H2** `OpenCsvShared` helper 替換 9 處 | ✅ 正確 | `InspectionStatisticsService.cs:84` 定義 helper、9 處 `OpenCsvShared(csvPath)` 全部到位。Reader 端 `FileShare.ReadWrite` 已達跨 process 讀寫 race 安全。**Writer 端（`InspectionLogService.cs:103, 163`）仍用 `new StreamWriter(csvPath, append: true, ...)`** — .NET 預設 FileShare.Read，意即 writer 持有時 reader 開 ReadWrite 仍會 sharing violation。但 .NET 5+/Framework 4.8 StreamWriter 內部用 `FileShare.Read` 對 reader 的 `FileShare.ReadWrite` 是不相容的（writer 端要 `ReadWrite` 才相容）。**真正的 race window**：寫 CSV 是 100ms 級瞬間（append 一行），讀則跨整檔，所以雖然有衝突可能，但只會偶發 IOException 被 catch 吞掉一個 CSV 的統計。可接受但**建議下一輪把 writer 也改成 `new FileStream(..., FileShare.ReadWrite)`+StreamWriter wrap**。 |
| **H3** PropertyValueChanged 300ms debounce | ⚠️ 不完整 | `AniloxRollForm.cs:2067-2085`：用 WinForms `Timer`（UI thread），`Stop/Start` 在同一 thread 不會 race。FormClosing 有 `_statsRefreshDebouncer?.Stop()` (182)，但**沒 Dispose** — Timer 在 form close 後仍持有 reference，process 退出才 GC。輕度資源洩漏但不影響功能。⚠️ Tick handler 內 `_dataStatsPresenter?.RefreshStats()` 在 form closed 後若還有 in-flight tick，可能在已釋放 presenter 上炸 — 但 null 條件運算子護住了。OK。 |
| **H4** SetCombosToDateTime guard 文件化 | ✅ 正確 | `DataStatisticsPresenter.cs:326-329`：加 doc comment 明示 caller 必須包 `StatComboGuard.Enter()`。屬於約定式契約，未根本解決脆弱但文件清楚。 |
| **H5** listView selection unsubscribe/subscribe | ⚠️ 不完整 | `DataStatisticsPresenter.cs:665, 692-694`：unsubscribe → BeginUpdate → ... → EndUpdate → subscribe。**沒包 try/finally** — 若 `lv.Items.Add(item)` 或 `AutoResizeColumns` 拋 exception，subscription 不會接回，從此 listView 點選失靈。建議 `try { ... } finally { lv.SelectedIndexChanged += OnGrabDetailRowSelected; }`。實務上 ListView.Items.Add 極少炸，可接受但不夠 robust。 |
| **H6** HessianRescaleHelper 集中化 | ✅ 正確 | `Services/HessianRescaleHelper.cs` 5 個 API；DataStatisticsPresenter.cs:822、ReviewStitchCoordinator.cs:306/307/343/344/424/425/448/449 全部改用 helper。邊界保留：`Ratio` 0 check (line 18)、`IsNoOp` epsilon 0.0001 check (line 23)、null 護衛 (line 38/47/63)。**語意正確**：原本散在 4 處的 in-place vs clone 區分維持不變（DataStatisticsPresenter 用 RescaleInPlace2D，ReviewStitchCoordinator 全部用 Clone — 後者保留 `_stitchedCurveMean/Max` 快取不變，是必要的）。 |
| **M1** ui-flow 補 listView click flow | ✅ 正確 | `ui-flow.html:1190-1203`：完整描述 OnGrabDetailRowSelected → cbDataGrabId → 單片模式 + chartMuraProfile 刷新 + sync cbReviewGrabId。H5 防護也 inline 註明。 |
| **M2** ui-flow 補 AppRole + Storage PC | ✅ 正確 | `ui-flow.html:1297-1304`：AppRole 變更 → app-mode.json → 重開生效 + Storage PC 模式說明（StorageRetentionService + CleanupFlagWatcher 隱藏 Live tab）。 |
| **M3** CleanupFlagWatcher 提前 stop | ✅ 正確 | `AniloxRollForm.cs:183`：OnFormClosing 內 Dispose + 設 null。FormClosed 路徑（line 1275 RegisterStorageModeCleanup）的 `?.Dispose()` 因為 null 安全。 |
| **M4** CsvConfigSnapshot V↔H 互補 | ✅ 正確 | `CsvConfigSnapshot.cs:202-209`：V/H 任一為 0 時鏡像到另一邊；新格式 CSV（V/H 都正常寫入）兩者皆 >0 不觸發鏡像 → 無副作用。 |
| **M5** #CFG 跨檔保留 + CSV 排序 | ✅ 正確 | `InspectionStatisticsService.cs:110-112, 188-190, 363-364, 591-592`：4 個 Compute/ScanCsvByDateRange 都用 `Array.Sort(csvFiles, StringComparer.Ordinal)`。**路徑排序等於時間排序**：`...\2026\202605\20260515.csv` < `...\2026\202605\20260516.csv` < `...\2026\202606\20260601.csv` < `...\2027\202701\20270101.csv` — 因為年份、月份資料夾名稱都是固定寬度零填，Ordinal byte-wise 比較等同數值比較。✅ 但 `LoadLatestConfig` (line 918) 用 `StringComparer.OrdinalIgnoreCase` 而其他用 `Ordinal`，雖然路徑都是 ASCII 數字結果一樣，**風格不一致**。 |
| **M8** Memory barrier | ✅ 正確 | `AniloxRollForm.cs:1324`：`Interlocked.MemoryBarrier()` 在 array reference 寫入後、`_liveOverviewDirty = true` 寫入前。`_liveOverviewDirty` 是 volatile bool → write release semantic。讀端 `LiveOverviewTimer_Tick:3102-3103` 讀 volatile bool（acquire）→ 後續讀 `_liveCurveMean/_liveCurveMax` 不會看到舊指標。**讀端不需額外 barrier**（volatile read 已是 acquire）。Note：MemoryBarrier 在 volatile write 前其實有點冗餘，但 explicit 表達意圖、不影響正確性。 |
| **L1** SetExposureForAll / SetGrabHeightForAll 刪除 | ✅ 正確 | Grep 全 repo 確認方法已不存在，僅 docs/ 還有歷史記錄文字。 |
| **L5** ContentKey 改 "R" 格式 | ✅ 正確 | `CsvConfigSnapshot.cs:80-92`：所有 float/double 欄位改 `ToString("R", inv)`。對 NaN 會輸出 "NaN"、Infinity 會輸出 "Infinity" — 不會崩，但若 PropertyGrid 允許輸入這些值會把 NaN key 與正常 key 區分（行為 OK）。**ContentKey 變長**（"R" 比 "F4" 平均長 2-3 字元），對 ContentKey 比對性能 negligible。 |
| **L6** CLAUDE.md docs 樹 | ✅ 正確 | CLAUDE.md 已更新（commit msg 提到）。 |
| **L3** BackgroundSampleRows 確認 | ✅ 正確 | Grep `BackgroundSampleRows` 不見於 src，僅文件殘留。 |

**小結**：18 個修法中 **15 個完全正確**、**3 個不完整但功能正確**（H1 雙 FormClosed handler / H3 Timer 未 Dispose / H5 缺 try/finally）。沒有 ❌ 真錯誤。

---

## 新引入的問題（B）

### Critical
（無）

### High

**B-H1. `StreamWriter` 端仍預設 `FileShare.Read`，與 `OpenCsvShared` (ReadWrite) 不完全相容**

`InspectionLogService.cs:103, 163` 用 `new StreamWriter(csvPath, append: true, ...)` — .NET 內部開檔模式是 `FileShare.Read`。reader 端用 `FileShare.ReadWrite` 在 writer 持有時嘗試開檔仍會拋 `IOException: 程序無法存取檔案`，被 catch 吞掉導致該 CSV 跳過。
**影響**：Live grab 寫 CSV 時，Data tab Refresh 偶發少一份檔案統計（每秒 7 張寫入時概率高）。
**建議**：writer 改 `new FileStream(csvPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite)` 再 wrap StreamWriter。

### Medium

**B-M1. `RegisterStorageModeCleanup` 殘留第二個 FormClosed handler**

`AniloxRollForm.cs:1269-1277` 仍存在於 Storage 模式 init。H1 修法把主要 FormClosed (1035-1051) 統一為單一路徑，但這個 handler 依然存在。執行時序：兩個 FormClosed handler 都會跑，先註冊先跑（一般是 RegisterStorageModeCleanup 在 InitServiceLayer 註冊較早）→ 它的 `_retentionService?.Dispose()` 跑完後，主 FormClosed handler 再跑 `_retentionService?.Dispose()` 第二次（null-safe 但意圖不清）。
**建議**：把 RegisterStorageModeCleanup 整個刪掉，邏輯併到主 FormClosed。

**B-M2. `_statsRefreshDebouncer` Tick handler 內 try/catch 吞所有 exception 但不重啟 timer**

`AniloxRollForm.cs:2072-2081`：Tick 內 RefreshStats 若連續炸 N 次，會被 swallow 但 user 看不到任何錯誤訊號，以為「改了設定但統計沒更新」。
**建議**：catch 內加 status bar 顯示「統計刷新失敗」或至少彈一次 MessageBox（debounce 防多次）。

**B-M3. `H5` listView re-subscribe 無 try/finally**

`DataStatisticsPresenter.cs:665-694` 若 ListView 操作中段拋 exception，subscription 永遠不接回。雖然 ListView.Items.Add 罕有炸，但 `AutoResizeColumns` 在 Disposed 控制項上會 ObjectDisposedException。
**建議**：包 `try { ... } finally { lv.SelectedIndexChanged += OnGrabDetailRowSelected; }`。

### Low

**B-L1. M5 sort comparer 不一致**

`InspectionStatisticsService.cs:918` 用 `StringComparer.OrdinalIgnoreCase`，其他 4 處用 `StringComparer.Ordinal`。對 ASCII 數字路徑結果相同，但風格不一致。

**B-L2. `RescaleInPlace1D` 對 null max 沒檢查 row chart**

`UpdateGlobalRowChart` (ReviewStitchCoordinator.cs:303) 已檢查 `mergedMean != null`，但若 `mergedMax` 為 null（理論上不會但 `CurveMergeHelper.MergeRowCurvesOverlap` 沒 contract）會 silent NoOp（helper line 38 有 null guard）。可接受。

---

## 未修 TODO 再評估（C）

### M6: `LoadGrabIdInfos AllDirectories` perf
**仍應 TODO**。`Directory.GetFiles(captureRoot, "*.csv", SearchOption.AllDirectories)` 對 500 天 / 100K 圖只掃 ~500 個 CSV（每天一個），不是 100K 個檔，比預期慢得多但仍是 cold path（btnSelectDataFolder 才呼叫，使用者操作會等待）。可接受。

### M7: `_currentDetails` 跨 thread
**H3 後仍安全**。`_statsRefreshDebouncer.Tick` 是 WinForms `System.Windows.Forms.Timer`（UI thread），handler 內 `RefreshStats()` 仍在 UI thread。`_currentDetails` 全部 UI thread 讀寫 — **不是 race**。✅ 但如同上一輪 M7 提醒：未來若把 Tick 改 `Task.Run` 模式（H3 進階優化），會立刻變 race。當前實作 OK。

### M9: `PopulateAllGrabIdCombos` vs `LoadDataFolder` 結構
**仍 TODO**。第一輪標 refactor 風險 > 收益，目前運作正確，沒新 bug 引入。

### L2: PropertyValueChanged > 100 行
**仍 TODO**。經 H3 加 debounce 後反而又長了 7 行（2121-2224）= 103 行。可拆但不影響正確性，壓力測試後再評估。

---

## ui-flow ↔ code 一致性（D）

| 位置 | 不一致 |
|---|---|
| `ui-flow.html:1167` SwitchActiveStatGroupBox flow | **缺述 C1 行為**：「切到 TimeRange 時自動把 cbStartDate/EndDate 攤開到資料夾全範圍」這個重要動作 ui-flow 沒提。建議在 line 1170 加一條 output：「切到時序範圍：cbStartDate/EndDate 重設到資料夾全範圍」。 |
| `ui-flow.html:1288` view-time 正規值 rescale 描述 | ✅ 正確且詳細（V 套 HM_V/HM_V，H 套 HM_V/HM_H）。HessianRescaleHelper 抽離後行為不變 — ui-flow 不需提及內部 helper 名。 |
| `ui-flow.html:1190-1203` listView click flow | ✅ M1 已補。H5 防護也內聯註明。 |
| `ui-flow.html:1297-1304` AppRole flow | ✅ M2 已補。 |
| `ui-flow.html:1202` H5 註：「unsubscribe SelectedIndexChanged」 | ⚠️ 與實作一致，但**沒提到缺 try/finally** 邊界 — 這是 B-M3 新發現的缺陷。 |

---

## 壓力測試 hot path（E）

按發生頻率 × 影響嚴重度排序：

### 1. `OnCameraInspectionResult` → `_inspectionLogService.AppendRecord` → CSV append
**路徑**：`AniloxRollForm.cs` callback → `InspectionLogService.cs:80-130`
**為什麼 hot**：每秒 7 張 × 8~24 小時 = 一天 ~6 × 10^5 次 CSV append + lock。
**壓力下可能**：
- 跨 process 寫 race（B-H1）：Inspection PC 寫的同時 Data tab UI 掃同份 CSV → IOException 吞 → 統計偶發少一份。
- `_csvLock` 全 instance 共享，若有兩個 Form instance（不應該但測試用），會 deadlock。
- `Directory.CreateDirectory` 每次都呼叫，雖然 OS 內部 cached，但 SMB share 上是 round-trip。
**建議監控**：CSV 寫入延遲（每張 grab 之間 < 10ms）、`Trace.WriteLine[InspectionLogService]` 出現頻率。

### 2. `OnLiveCurveData` → `_liveCurveMean/Max` 寫 + `CheckLiveMura` → DO_MURA_DETECTED
**路徑**：`AniloxRollForm.cs:1314-1326, 1284-1308`
**為什麼 hot**：每幀 callback thread × 7 相機；CheckLiveMura 內掃整個 mean/max array 找 max。
**壓力下可能**：
- Callback thread 不是 UI thread，array reference 寫入靠 `_liveOverviewDirty` volatile 同步（M8 已修）。
- 7 台相機並發進 `CheckLiveMura`，但 `_plcGrabController.TriggerDoMura()` 內部是否 thread-safe 沒在 review。
- ResourceLog CSV 寫入也在這條路徑上（CameraFrameSaver），與主 CSV 共用磁碟 IO。
**建議監控**：CPU% per cam thread、`_liveOverviewDirty` 抖動頻率、Mura DO trigger 與物理對齊延遲。

### 3. `_propertyGrid_PropertyValueChanged` → ScheduleStatsRefresh → RefreshStats
**路徑**：`AniloxRollForm.cs:2121-2224, 2067-2085` + `DataStatisticsPresenter.RefreshStats`
**為什麼 hot**：拖 slider 時連續觸發；單次 RefreshStats 掃整個 CaptureRoot 所有 CSV。
**壓力下可能**：
- Debounce 300ms 合併連續變更（H3），但**單次 Tick 仍掃整檔** — 100K 筆 + 500 CSV 預估 1~3 秒。
- 期間若 grab 仍在跑，UI 凍結會延遲視覺更新但不影響 grab callback（不同 thread）。
- B-M2: 若 RefreshStats 持續炸（如磁碟掉線），會吞 exception 但 user 不知。
**建議監控**：debounce 觸發頻率、RefreshStats 執行時長、UI thread 佔用率。

### 4. `ScanCsvByDateRange` 在 ChartYearly/Monthly/Daily Index 切換時
**路徑**：`DataStatisticsPresenter.OnChartYearIndexChanged` → `InspectionStatisticsService.ScanCsvByDateRange:570-632`
**為什麼 hot**：拖 Year/Month/Day combo 時連環觸發三次 ScanCsvByDateRange + RefreshStats。
**壓力下可能**：
- M5 修法把 captureHmV 跨檔保留 — **新行為的 side effect**：若使用者刪了中間某天 CSV，captureHmV 沿用前一天最後值，可能與該日實際 #CFG 不符（但這個 corner case 非常邊緣）。
- Year 圖切換時跨越多年資料，O(N) 掃描所有 CSV。
**建議監控**：chartNav 切換到 RefreshPeriodCharts 完成的延遲時間（目標 < 500ms for 100K 筆）。

### 5. `RemoteCopyService` 背景複製到 SMB share
**路徑**：`Services/RemoteCopyService.cs` ConcurrentQueue + 背景 thread
**為什麼 hot**：每張 grab 完成後 enqueue 一個檔到 Storage PC（每秒 7 張）。
**壓力下可能**：
- 1Gbps 網路理論吞吐 125MB/s；7 張 × 50MB（典型 BMP）= 350MB/s — **超過網路頻寬**，queue 會持續累積。
- Queue 無上限 → 24 小時後 queue 可能堆 10K+ pending → 進程 RAM 上升。
- Storage PC 端 cleanup-request.flag 與 CleanupFlagWatcher 觸發如太密集會與 RemoteCopy 競爭磁碟 IO。
**建議監控**：RemoteCopy queue 長度、SMB share 寫入吞吐、Storage PC 磁碟空閒空間下降率。

---

## 文件同步（F）

### CLAUDE.md 待補
1. **「關鍵檔案速查」表**沒列 `Services/HessianRescaleHelper.cs` — 集中 view-time rescale 公式的核心 helper，應列上。建議插在 `CsvConfigSnapshot.cs` 附近：
   `| Services/HessianRescaleHelper.cs | View-time HM rescale 共用：Ratio / IsNoOp / RescaleInPlace1D|2D / CloneAndRescale1D|2D — 5 個公式單一來源 |`

2. **`Services/StorageRetentionService.cs` / `CleanupFlagWatcher.cs`** 已列 ✅。

3. **`_statsRefreshDebouncer`** 是 Form 私有欄位 — 不必列關鍵檔案，但 `.claude/skills/modify-data-stats.md` 可補一段提醒：「PropertyGrid 觸發 RefreshStats 走 300ms debounce（`ScheduleStatsRefresh`），新增 caller 時應呼叫 helper 而非直接 `_dataStatsPresenter.RefreshStats()`，否則無 debounce 保護。」

4. **`OpenCsvShared`** 是 InspectionStatisticsService 私有 helper，不必列 — 但 `.claude/skills/modify-data-stats.md` 應提一句：「新增 CSV reader 用 OpenCsvShared（FileShare.ReadWrite）避免跨 process race」。

### dead-code-candidates.md
✅ 已更新（line 8 第二輪 SetExposureForAll/SetGrabHeightForAll 紀錄）。

### ui-flow.html
1. line 1170 加 output「切到時序範圍：cbStartDate/EndDate 攤開到資料夾全範圍」對應 C1。
2. line 1202 H5 註可加「未來改 async 模式時需注意 try/finally」（B-M3）— 但屬實作細節，非 user flow，建議移到 skills/modify-data-stats.md。

---

## 摘要

**第一輪修法**：18 個全部處理，**15 個正確、3 個不完整但無功能 bug**。
**新發現**：Critical 0 / High 1 / Medium 3 / Low 2 = 共 **6 個新問題**。

**最重要的新發現**：
1. **B-H1**：InspectionLogService writer 端仍 `FileShare.Read`，與 reader 端 `FileShare.ReadWrite` 不完全相容；跨 process race 仍有偶發 IOException 風險。
2. **B-M1**：RegisterStorageModeCleanup 殘留第二個 FormClosed handler，H1「單一 dispose 路徑」沒清乾淨。
3. **B-M3**：H5 unsubscribe/subscribe 缺 try/finally。

**壓力測試前必補**：B-H1（writer 端 FileShare 修正）— 其他可後續迭代。
**整體評估**：18 個第一輪修法品質高，新引入問題都是邊界/細節級，無 Critical 阻擋壓力測試。

預期 8~24 小時壓力測試後最可能浮現的問題依序：
1. RemoteCopy queue 累積（網路頻寬）
2. CSV 跨 process race 偶發少統計
3. RefreshStats UI 凍結（100K+ 筆時）
