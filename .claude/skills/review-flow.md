# review-flow

追蹤指定控制項的完整輸入→動作→輸出流程，用於迴歸驗證。

## 使用時機

使用者輸入 `/review-flow <標準名稱>` 時（例如 `/review-flow 讀取資料`）。
也可在 refactor 或功能修改後，批次驗證受影響的流程。

## 參考資源

- **CLAUDE.md §控制項速查** — 標準名稱 → 程式碼 Name 對照
- **`docs/user-manual/ui-flow.html`** — 完整事件流程圖（瀏覽器開啟）

## 執行步驟

1. **查對照表** — 從 CLAUDE.md 控制項速查找到程式碼 Name

2. **定位 handler** — 在 AniloxRollForm.cs、DataStatisticsPresenter.cs、ReviewStitchCoordinator.cs 搜尋事件繫結

3. **追蹤呼叫鏈** — 從 handler 逐層追蹤：
   - 直接呼叫的方法（含 async/await）
   - 觸發的跨元件事件
   - Guard flag 的 enter/exit
   - 更新的 UI 控制項（用標準名稱）

4. **比對流程圖** — 對照 `docs/user-manual/ui-flow.html` 中該控制項的流程，逐項確認

5. **輸出驗證結果** — 格式：
   ```
   [觸發] 讀取資料 (btnSelectFolder)
     → [動作] ImageRepository.LoadDirectory
     → [動作] TimeNavigator.Initialize
     → [輸出] 時段日期/時間 → 填入最早值 ✅
     → [輸出] 單片序號 → 填入最早序號 ✅
     → [輸出] 回顧縮圖1~7 → 載入圖片 ✅
     → [輸出] 回顧主畫面 → 顯示影像 ✅
     → [同步] Data tab → 序號+時間+統計 ✅
   [結果] 7/7 通過
   ```

6. **若發現斷裂** — 標示 ❌ 並列出：
   - 哪個輸出沒有更新
   - 斷在哪個方法（行號）
   - 建議修復方式

## 批次驗證模式

`/review-flow --all` 驗證所有流程圖中的輸入控制項：

| Tab | 需驗證的輸入 |
|-----|------------|
| Live | 開始抓取、監控強化、監控切向曲線圖/法向曲線圖點擊、監控縮圖點擊、取得背景、預覽背景 |
| Review | 讀取資料、時段導航、單片序號、回顧縮圖點擊、回顧強化、回顧切向曲線圖/法向曲線圖點擊 |
| Data | 讀取資料、序號範圍、序號選擇、時序範圍、良率圖導航、篩選異常 |
| 右側 | 檢測設定（Recipe/StitchMode/Algorithm/ChartScale）、相機參數滑桿 |
| 跨Tab | Review→Data 同步、Data→Review 同步 |

## 流程圖維護

驗證過程中若發現流程圖與程式碼不一致，同時更新 `docs/user-manual/ui-flow.html`。
