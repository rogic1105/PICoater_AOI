# FSM Action Log

「點按鈕 → setting 變更 → state transition」對照系統。實作 Finite State Machine transition table 概念，自動驗證 UI 互動是否符合預期。

## 啟用方式

1. 編輯 `bin/x64/Release/Config/inspection-settings.json`，把
   ```json
   "DebugUiActionLog": false
   ```
   改成 `true`（PG 上不顯示，純 dev 用）。
   首次啟動 exe 後此欄位會自動補進 json（無需手動建立 key），第二次啟動才改成 true 也可以。
2. 啟動 `AniloxRoll.Monitor.exe`
3. 操作各種按鈕 / chart / PG 改值
4. 關閉 exe（log 寫到 `D:\Anilox\Logs\ui-actions-YYYYMMDD.jsonl`）
5. 用瀏覽器開 `docs/dev/fsm/viewer.html`
6. 拖入 jsonl → 自動標 ✓/❌/⚠

關閉時把 `DebugUiActionLog` 改回 `false` 即恢復零 overhead。

## 三個檔案

| 檔案 | 用途 |
|---|---|
| `state-catalog.csv` | 列出所有可能 state（setting 組合 → state_id 編號）|
| `transition-table.csv` | 列出所有預期 action transition `(action, from_state, to_state)` |
| `viewer.html` | fetch 上兩個 CSV + 拖入 log → 自動標 ✓/❌/⚠ + 顯示 coverage matrix |

## 三類標記

| 標記 | 意義 |
|---|---|
| ✓ Expected | action + from + to 在 transition-table 內 — 行為符合預期 |
| ❌ Unknown transition | from/to 都是 valid state，但 `(action, from, to)` **不在 transition-table** — bug 或漏列 |
| ⚠ Unknown state | from 或 to 算出的 state_id 不在 state-catalog — setting 組合超出預期範圍 |
| ○ View-only | 純視覺操作（拖曳、雙擊 FitToScreen），不改 setting，from == to |

## 怎麼加新的 transition

1. **加新 action**（如新按鈕）：在 `transition-table.csv` 加 row，列出該 action 在每個 state 應該轉到哪個 state
2. **加新 setting**（PG 多一個 bool / enum）：
   - 修改 `Services/UiActionLogger.cs` 的 `ComputeStateId()` — 加新 bit
   - `state-catalog.csv` 加新欄位 + 擴充 state 列舉
   - `transition-table.csv` 對應新增（舊 row 要更新 `from`/`to` 對應新 state_id）

## 當前覆蓋範圍（v1）

3 個 setting：`hb_StitchMode` × `hc_EnableMuraEnhance` × `hd_EnableReviewEnhance` = 8 state

5 個 action 已列：
- `chartLiveVertical.Click`
- `chartLivePatch.Click`
- `chartReviewVertical.Click`
- `chartReviewPatch.Click`
- `chartReviewHorizontal.Click` / `chartLiveHorizontal.Click` (view-only)
- `camReviewMain.DoubleClick` / `camReviewMain.Drag` (view-only)

**沒列在 transition-table 的 action 跑下去** → viewer 標 ❌，提示「這個 action 我沒寫到」。

## 加 ridge dir 維度（將來擴充）

如果要區分「強化方向 V vs H」，state 從 8 變 32（× v/h × Live ridge dir × Review ridge dir）。屆時：
1. `state-catalog.csv` 加 `liveRidgeDir`、`reviewRidgeDir` 欄
2. `ComputeStateId()` 加對應 bit
3. `chartLiveHorizontal.Click` / `chartReviewHorizontal.Click` 從 view-only 變成真 transition

## 整合 StorageRetentionService

`D:\Anilox\Logs\ui-actions-*.jsonl` 自動跟其他 capture / CSV 一樣按可用空間循環刪除（不需手動）。
