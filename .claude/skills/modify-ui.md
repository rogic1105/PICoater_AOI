# modify-ui

修改 UI 相關程式碼（Form、控制項、事件、Chart、Canvas）前的檢查清單。

## 使用時機

修改 AniloxRollForm、FormInteractionHelper、CanvasInteractionHelper、ReviewStitchCoordinator、EventGuard 或任何 UI 控制項行為時。

## 關鍵檔案

- `UI/Form/AniloxRollForm.cs` — 事件繫結、InitializeSystem、PropertyValueChanged
- `UI/Form/AniloxRollForm.Designer.cs` — 控制項佈局（VS Designer）
- `UI/Widgets/FormInteractionHelper.cs` — UI 互動、gallery 選擇、ReviewConfig 代理
- `UI/Widgets/CanvasInteractionHelper.cs` — zoom/pan、mm 座標、跨倍率 View 保存
- `UI/Widgets/EventGuard.cs` — using 語法 bool guard（取代散落 flag）
- `UI/Presenters/ReviewStitchCoordinator.cs` — Review 拼接管理、合圖、overview 聯動
- `UI/Presenters/DataStatisticsPresenter.cs` — Data tab 統計、跨 Tab 同步
- `UI/Widgets/OverviewChartManager.cs` — 全覽圖合併演算法
- `UI/Widgets/ProportionalScaler.cs` — Form 等比例縮放

## 注意事項

### Guard Flags（EventGuard 模式）
- 所有 guard 使用 `EventGuard` 類 + `using (guard.Scope())` 自動還原
- `_statComboGuard` — stat ComboBox cascade 防重複
- `_grabIdNavGuard` — cbReviewGrabId ↔ 時間 ComboBox 防迴圈
- `_grabIdCrossGuard` — cbReviewGrabId ↔ cbDataGrabId 防迴圈
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
  - Live：`EnableGlobalMerge` → 監控主畫面即時合圖（MbufChild2d + MbufCopyClip，含 overlap 分割）；muraChartVerticalLive 不更新；chartLiveOverview X 軸隨合併 display zoom 聯動（`LiveViewRangeProvider` → `TryGetMergedViewRange`）；lblPixelInfo 由 `_mergedDisplay` 的 `M_MOUSE_MOVE` hook 更新（mm 座標）
  - Review：`ApplyGlobalMergeIfNeeded` → 回顧主畫面全域合圖；chartMuraVertical 清空；chartMuraHorizontal 正常載入
  - 切換時立即觸發（`_propertyGrid_PropertyValueChanged`）
- **Vertical 模式**：overview X 軸固定（不隨 canvas zoom）；muraVertical/Horizontal 隨動；Live 單台 MIL 顯示

### SwitchRidgeDirection 三態切換
1. 未勾選 → 自動勾選 + 設方向
2. 已勾選 + 同方向 → 取消勾選（回原圖）
3. 已勾選 + 不同方向 → 切換方向（不改 checkbox）

### Chart 對齊與效能
- Chart X/Y 軸與 Canvas 對齊：用 `InnerPlotPosition`（PostPaint 首次快取後凍結），不用靜態值
- `_suppressChartSync`：`UpdateCanvas` 中壓制 `FitToScreen` 觸發的 chart sync，只在呼叫端做一次 `UpdateDataAndView`
- StripLines 必須放在 AxisY（Primary），AxisY2 初始化時不渲染
- `RefreshThresholds()` 不可放在 `UpdateDataAndView()` 末尾（會在 ResumeUpdates 後多一次 redraw）
- AxisY2 必須有 anchor series 否則 scale 不初始化
- Y 軸標籤反轉用 `Customize` 事件替換文字，不用 `IsReversed`（會讓 X 軸跳頂部）

### 跨倍率 Canvas View 保存
- `SaveViewIfNeeded` 在 `_imageScaleFactor` 更新前呼叫（舊倍率），`UpdateCanvas` 在更新後呼叫
- 座標轉換：pixel → mm 世界座標 → 新 pixel（公式見 CanvasInteractionHelper）
- `SaveViewIfNeeded()` 在 `Image == null` 時 return 但不重置 `_shouldRestoreView`

### ProportionalScaler
- `AutoScaleMode = None`，Scaler 全權接管；不可混用 Anchor
- TabControl 延遲頁面在首次切頁時才記錄

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
