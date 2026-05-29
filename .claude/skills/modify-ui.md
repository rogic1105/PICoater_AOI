# modify-ui

修改 UI 相關程式碼（Form、控制項、事件、Chart、Canvas）前的檢查清單。

## 使用時機

修改 AniloxRollForm、FormInteractionHelper、CanvasInteractionHelper、ReviewStitchCoordinator、EventGuard 或任何 UI 控制項行為時。

## 關鍵檔案

→ 見 `CLAUDE.md` §關鍵檔案速查（subset：`UI/Form/*` + `UI/Widgets/*` + `UI/Presenters/*`）。
→ UI 流程行為與控制項互動見 `docs/user-manual/ui-flow.html`（真相來源）。

## 注意事項

### L2 SSoT 原子結構（最重要 — 改 setting 必看 CLAUDE.md §架構原則）

所有 setting 變更走 `SettingsHub`（`Settings/Services/SettingsHub.cs`）：

| API | 用途 | 副作用 |
|---|---|---|
| `Hub.Set<V>(s => s.X, value)` | 程式碼路徑（AutoDetect 回寫、fallback、單 setting 變更）| save disk + raise event（Source=Programmatic）|
| `Hub.SetBatch(s => { ... })` | 多 setting 嚴格 transition 順序，caller 自己 await | save once，**不** raise event |
| `Hub.NotifyExternalChange(name, old, new)` | PropertyGrid 改值（setter 已寫 memory）| save disk + raise event（Source=PropertyGrid）|

`Form.OnSettingChanged(SettingChange c)` 是唯一訂閱者：共用前段（chart 閾值同步、Live SetCaptureSettings、ScheduleStatsRefresh 等）+ switch case（StitchMode/EnableEnhance/Algorithm/OPS-Start/Light 等個別 dispatch）。

**chart click / 按鈕 click 改 setting** — 永遠走 Hub，**不要** 直接 `_settings.X = ...`：
- 用 wrapper alias name（`s.hc_EnableMuraEnhance` 不是 `s.EnableMuraEnhance`）— PG GridItem.PropertyDescriptor.Name 是 wrapper name，OnSettingChanged case 也比對 wrapper name
- SetBatch 後手動 `RefreshGridItem(nameof(...))` 同步 PG 顯示（SetBatch 不 raise event）

**RefreshGridItem(name) trick** — 比 `propertyGridSettings.Refresh()` 精準：
- 找對應 GridItem → 暫時 `SelectedGridItem = found` 觸發 PG 內建 force re-read value → restore
- 只動單 cell、不全 Refresh、不閃、scroll 保留
- `_suppressGridSelChange` flag 抑制 SelectedGridItemChanged 副作用

**Commit-on-end** — 複雜 transition（chart click 切 StitchMode + 關 enhance + reload + fit）期間 `camReviewMain.Visible = false` 包 try/finally 避免中間幀閃。

**`async void OnSettingChanged` 防 race** — 用 SemaphoreSlim 序列化，連點時排隊處理。

bootstrap 例外（line 232 AppRole）：Hub 還沒建構，加註解標明合理 bypass。

### Guard Flags（EventGuard 模式）
- 所有 guard 使用 `EventGuard` 類 + `using (guard.Scope())` 自動還原
- `_statComboGuard` — stat ComboBox cascade 防重複
- `_grabIdNavGuard` — cbReviewId ↔ 時間 ComboBox 防迴圈
- `_grabIdCrossGuard` — cbReviewId ↔ cbDataId 防迴圈
- `_processedCheckboxGuard` — 程式碼設定 checkbox 時防觸發
- `_chartNavGuard` — chart 年月日 cascade 防重複
- `_suppressChartSync` — FitToScreen/SetView 時壓制 chart 同步
- `_syncingFromHw` — Telemetry 回讀硬體時防寫回
- `_liveOverviewDirty`（volatile bool）— 曲線回呼設 true，Timer 消費後清

### V/H 顯示決策矩陣
- Canvas 圖片由 `_activeRidgeDirection`（`"v"`/`"h"`）決定
- Chart 曲線永遠雙方向同時更新（不受 direction 影響）
- GPU Pipeline 永遠 `"vertical+horizontal"`，direction 只影響 UI 選圖

### StitchMode 行為
- **Global 模式**：
  - Live：`EnableGlobalMerge` → 監控主畫面即時合圖（MbufChild2d + MbufCopyClip，含 overlap 分割）；chartLiveVertical 不更新；chartLivePatch X 軸隨合併 display zoom 聯動（`LiveViewRangeProvider` → `TryGetMergedViewRange`）；lblPixelInfo 由 `_mergedDisplay` 的 `M_MOUSE_MOVE` hook 更新（mm 座標）
  - Live 手勢:雙擊=`ResetMainDisplayView`（fit-to-window）、三擊=`SetPhysicalMagnification1x`（1:1）、滾輪=`ApplyCustomZoom`（1.1x 步進），Global/單台共用邏輯
  - Live 初始化：`btnLiveGrab_Click` 首次分配相機後，若已為 Global 模式則立即 `EnableGlobalMerge`
  - Review：`ApplyGlobalMergeIfNeeded` → 回顧主畫面全域合圖；chartReviewVertical 清空；chartReviewHorizontal 正常載入
  - 切換時立即觸發（`_propertyGrid_PropertyValueChanged`）
  - OPS/Start 變更：Global 啟用中 → `RefreshGlobalMergeLayout`（暫停複製→重算 clip→buffer 大小變則重分配→恢復，下一幀生效）
- **Vertical 模式**：overview X 軸固定（不隨 canvas zoom）；muraVertical/Horizontal 隨動；Live 單台 MIL 顯示
- **Global → Vertical 切換**：必須呼叫 `ReviewStitchCoordinator.DisposeGlobalMergedImage()` 清掉殘留 `_globalMergedImage` / `_periodMergedImage`。否則 IsGlobalMerged 仍為 true，會誤觸發其他守門條件（`UpdateSelectedReviewCamFromViewCenter` 現已改用 `StitchMode == Global` 判斷，但 bitmap 仍須釋放避免佔記憶體）。

### SwitchRidgeDirection 三態切換
1. 未勾選 → 自動勾選 + 設方向
2. 已勾選 + 同方向 → 取消勾選（回原圖）
3. 已勾選 + 不同方向 → 切換方向（不改 checkbox）
- **視野保留**：所有切換路徑（含 `ApplyReviewEnhance` → `ReloadCurrentStitchedView`）在重載前先 `SaveCanvasView()` 存檔；`LoadGrabStitchedViewAsync` 內部以 `ShowStitchedCameraInCanvas(idx, resetView: false)` 顯示，不抹掉 saved view，由 `RestoreCanvasViewOrFit()` 還原。`ShowStitchedCameraInCanvas` 預設 `resetView=true`（camReview 切換相機時 Vertical 強制 fit to screen），呼叫端要保留視野必須明確傳 `false`。
- **視覺一致性**：Review tab `UpdateRidgeDirectionVisual(dir)` 與 Live tab `UpdateLiveDirectionVisual()` 兩版本邏輯對稱（淡藍底 + 橘色外框 BorderlineColor=`FromArgb(255,140,0)` Width=2 Solid）；新增/修改方向視覺時兩邊同步維護。Live 版自行從 `_settings.EnableMuraEnhance` + `_liveDisplayDirection` 推斷 dir；Review 版由呼叫端傳入。

### Chart 對齊與效能
- Chart X/Y 軸與 Canvas 對齊：用 `InnerPlotPosition`（PostPaint 首次快取後凍結），不用靜態值
- `_suppressChartSync`：`UpdateCanvas` 中壓制 `FitToScreen` 觸發的 chart sync，只在呼叫端做一次 `UpdateDataAndView`
- StripLines 必須放在 AxisY（Primary），AxisY2 初始化時不渲染
- `RefreshThresholds()` 不可放在 `UpdateDataAndView()` 末尾（會在 ResumeUpdates 後多一次 redraw）
- **`BaseCurveChartHelper` 預設值**：runtime 會被 owner 的 `RefreshThresholds(real values)` 覆寫，預設值只在 `Build()` 初始 Y 軸計算前短暫使用（`yMax = max(1.0, errorValueMax × 1.1)`）。目前為 `0.2f` / `0.4f`，與 `InspectionDefaults.ErrorValueMeanV/MaxV` 對齊。修改時兩邊同步即可，不影響 runtime 行為。
- AxisY2 必須有 anchor series 否則 scale 不初始化
- Y 軸標籤反轉用 `Customize` 事件替換文字，不用 `IsReversed`（會讓 X 軸跳頂部）

### 跨倍率 Canvas View 保存
- `SaveViewIfNeeded` 在 `_imageScaleFactor` 更新前呼叫（舊倍率），`UpdateCanvas` 在更新後呼叫
- 座標轉換：pixel → mm 世界座標 → 新 pixel（公式見 CanvasInteractionHelper）
- `SaveViewIfNeeded()` 在 `Image == null` 時 return 但不重置 `_shouldRestoreView`

### ProportionalScaler
- `AutoScaleMode = None`，Scaler 全權接管；不可混用 Anchor（Anchor 被 RecordRecursive 改成 `Top|Left`）
- `TabPage` 加入 `isLayoutContainer` 名單（同 TabControl/Panel/GroupBox/SplitContainer），Initialize 時遞迴記錄所有 TabPage 內 controls
- TabControl `SelectedIndexChanged` 觸發時除 `RecordRecursive(tab)` 外，主動 `ScaleRecursive(tab, _form.ClientSize)` 重 scale 該 tab。WinForms 對 inactive TabPage 有 lazy layout，maximize 時寫入的 Bounds 可能在 TabPage 變 active 時被 layout 引擎重設回 `Top|Left` Anchor 預設位置 → 不切 tab 直接 maximize 看不到放大；切 tab 主動 scale 解決此問題
- `_scaler.Initialize()` 在建構子內呼叫（非 Shown 內）：須早於任何 Resize 事件才能正確記錄 baseSize 與 Designer 預設 Bounds 作為 ratio 基準
- 機台角色 = Storage 時 `ApplyStorageModeUi` 在 Scaler 建立前移除 `tabPageLiveView`，Scaler 自動只處理剩下 2 個 tab

### Chart ZoomReset/Clear 禁忌
- 不可在 `await` 前執行 chart clear/reset（會被後續 `UpdateDataAndView` 覆蓋）
- 應由後續 `UpdateDataAndView` 原子性取代

### Designer.cs 規則
- 控制項必須在 `InitializeComponent()` 才能在 VS Designer 顯示
- 事件繫結（需 runtime 物件）放 code-behind
- 批次重命名：先替換較長數字（`comboBox12` 先於 `comboBox1`）

## 步驟

1. 讀取要修改的 handler 及其相關 guard flags
2. 確認是否涉及跨 Tab 同步（Review ↔ Data）
3. 修改 + build 驗證
4. 若新增/移除控制項，更新 Designer.cs + CLAUDE.md
