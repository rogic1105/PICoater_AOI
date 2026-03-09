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
   | `IO` 大 | FastReadBMP 讀盤慢 | 確認 NativeBufferPool 用 pinned memory；確認 CoreCV_FastReadBMP 有正確呼叫 |
   | `GPU` 大 | CUDA pipeline 慢 | 確認 AoiService 參數正確；確認 CUDA stream 沒有意外同步 |
   | `BMP` 大 | Create8bppBitmap 慢 | 確認用 ImageUtils.Create8bppBitmap（MemoryCopy 路徑），非 GDI+ pixel 迴圈 |
   | `Canvas` 大 | FitToScreen/SetView 慢 | 確認 _shouldRestoreView 邏輯正確；避免多次 Invalidate |
   | `Chart` 大 | MuraChartHelper 慢 | 確認 UpdateData/UpdateViewRange 不重繪整個 chart；考慮延遲更新 |

4. **目標值**：`[OnSelect] Total` ≤ 200ms（使用者感知 ≤ 0.5s 含縮圖階段）
