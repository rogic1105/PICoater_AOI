# perf-diagnose

診斷切換相機/縮圖時的效能瓶頸

## 使用時機

使用者反映「切換圖片感覺慢」時。

## 步驟

1. **確認計時 log 已啟用**
   - `InspectionEngine.RunInspectionFullRes` → 輸出 `[FullRes]` 行
   - `FormInteractionHelper.OnGallerySelectionChanged` → 輸出 `[OnSelect]` 行
   - 若缺少，讀取這兩個檔案並加入 `Stopwatch` 計時

2. **要求使用者提供 console 輸出**，格式如下：
   ```
   [FullRes] mode=True  | IO=  17ms | GPU=  22ms | BMP=  28ms | Copy=  0ms | Total=   69ms
   [OnSelect] Cam1 | FullRes=   69ms | Canvas=   0ms | Chart=  46ms | Total=  116ms
   ```

3. **判斷瓶頸**

   | 慢的欄位 | 原因 | 對策 |
   |---------|------|------|
   | `IO` 大 | FastReadBMP 讀盤慢 | 確認 NativeBufferPool 用 pinned memory；確認 TanukiCv_FastReadBMP 有正確呼叫 |
   | `GPU` 大 | CUDA pipeline 慢 | 確認 AoiService 參數正確；確認 CUDA stream 沒有意外同步 |
   | `BMP` 大 | Create8bppBitmap 慢 | 確認用 ImageUtils.Create8bppBitmap（MemoryCopy 路徑），非 GDI+ pixel 迴圈 |
   | `Canvas` 大 | FitToScreen/SetView 慢 | 確認 _shouldRestoreView 邏輯正確；避免多次 Invalidate |
   | `Chart` 大 | ColumnCurveChartHelper 慢 | 確認 UpdateData/UpdateViewRange 不重繪整個 chart；考慮延遲更新 |

4. **目標值**：`[OnSelect] Total` ≤ 200ms（使用者感知 ≤ 0.5s 含縮圖階段）

## UI 渲染「不即時/閃爍/卡」診斷清單（2026-06-12 [ReviewSync] 實戰沉澱）

按序檢查（每項都有實際踩過的案例）：
1. **先上儀器再猜**：handler 包 Stopwatch（單次 >25ms 告警 + 每 N 次彙總 max），分段量各 chart/步驟。猜錯一次的成本 > 加 log。
2. **WM_PAINT 飢餓**：拖曳時滑鼠訊息佔滿佇列、paint 最低優先級 → Invalidate 排不到隊「放開滑鼠才動」。
   特徵＝量測 0~1ms 但體感 lag。解法＝跟隨後 `control.Update()` 強制同步畫。
3. **單次重繪太貴**：MSChart 全點數上 chart（法向=影像高上萬點）→ 重繪 22~67ms。解法＝顯示降採樣 ~2000 點
  （桶內 mean=平均/max=取大保峰值；資料判定不經此路）。
4. **MSChart axis Minimum/Maximum 重排版**：設這兩屬性觸發整張重排版（比 ScaleView.Zoom 貴一級）→ 變了才設。
5. **先算好位置再繪圖**：先顯示再 FitToScreen/SetView ＝ 閃一下。任何新圖/換 ID：佈局與視野算完才第一次 paint。
6. **回授迴圈**：A 動→B 跟→事件又回頭動 A。檢查事件鏈有無環；用 Source 區分（SSoT 規則 5）。
7. **chart 重畫必須原子帶視野**（BaseCurveChartHelper 不變量）：嚴禁先 Clear/重設→事後補視野（會閃回預設）。
8. **修法設計前先搜鏡像側**：Live↔Review 同源，同類問題先看另一側怎麼解（如 _liveViewLeftMm 快取模式）。

