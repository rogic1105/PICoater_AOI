# 效能模式與陷阱

## /perf-diagnose 排查流程

1. 先看 Stopwatch 計時輸出（`[FullRes]`、`[OnSelect]` 等）確認瓶頸在哪一段
2. 區分 IO / GPU / UI 三層，對症下藥
3. MIL 相關卡頓：優先排查是否有多執行緒競爭同一 MIL ID
4. CUDA 相關卡頓：確認是否為冷啟動（首次 `cudaMalloc` / `cudaMallocHost`）
5. UI 卡頓：確認 allocation 是否在 UI 執行緒同步執行，改為 `Task.Run` + `await`

---

## SmartCanvas 拖曳效能

### 問題
`OnMouseMove` 在每個 event 都執行：
1. `bmp.GetPixel(x, y)` — 每次 lock/unlock bitmap，大圖極慢
2. `TriggerStatusChange()` → `ScaleView.Zoom()` — chart 同步重繪
3. `_statusLabel.Text = ...` — label repaint + 字串分配

全部同步阻塞 UI thread → canvas `Invalidate()` 積壓 → 拖曳卡頓。

### 修正模式（throttle 分層策略）
```csharp
if (_isDragging)
{
    int now = Environment.TickCount;
    if (now - _lastStatusTickMs >= StatusThrottleMs) // 32ms
    {
        _lastStatusTickMs = now;
        TriggerStatusChange();
    }
}
else
{
    _lastColor = bmp.GetPixel(...); // 靜止 hover 才讀色
    TriggerStatusChange();
}

// OnMouseUp：補一次完整更新
protected override void OnMouseUp(...)
{
    _isDragging = false;
    TriggerStatusChange();
}
```

**效果**：canvas repaint 以滑鼠原生頻率執行；chart/statusbar ~30fps 更新；兩者不互相阻塞。

---

## Chart Sync 壓制（_suppressChartSync）

### 問題：FitToScreen/SetView 觸發 chart 雙次 redraw + range 錯誤

`SmartCanvas.FitToScreen()` 和 `SetView()` 末尾同步呼叫 `TriggerStatusChange()`，
觸發 `StatusChanged` → `UpdateCanvasInfo` → `UpdateViewRange`（chart redraw #1）。
之後呼叫端再呼叫 `UpdateDataAndView`（chart redraw #2）。

### 解法：UpdateCanvas 內部 suppress chart sync

```csharp
private bool _suppressChartSync = false;

public void UpdateCanvas(Bitmap newImage)
{
    _suppressChartSync = true;
    try { _canvas.FitToScreen(); }
    finally { _suppressChartSync = false; }
}

public void UpdateCanvasInfo(CanvasInfo info)
{
    if (!_suppressChartSync)
        _muraChartHelper?.UpdateViewRange(...);
}
```

### 呼叫端正確流程
```csharp
_canvasHelper.UpdateCanvas(data.Image);     // FitToScreen + chart sync 被壓制
_canvasHelper.TryComputeCurrentViewRange(index, out double leftMm, out double rightMm);
_muraChartHelper.UpdateDataAndView(mean, max, startPos, leftMm, rightMm);  // 唯一一次 redraw
```

### SmartCanvas FitToScreen/SetView 行為

`_zoom` 和 `_panOffset` 在返回前已正確設定，`TryComputeCurrentViewRange` 可在呼叫後立即讀取，無需 `BeginInvoke` 延遲。

---

## 跨倍率 Canvas View 保存（世界座標法）

### 問題
切換時間段（btnPeriodPrev/PeriodNext）若前後圖片倍率不同（BMP 1x vs JPEG 5x），pixel zoom/pan 直接還原會造成視窗跳位。

### 解法：mm 世界座標存/取

**Save（切換前，`_imageScaleFactor` 仍為舊圖）**：
```csharp
double pixelLeft  = (0             - pan.X) / zoom * _imageScaleFactor;
double pixelRight = (_canvas.Width - pan.X) / zoom * _imageScaleFactor;
_savedViewLeftMm  = startPosMm + pixelLeft  * opsInMm;
_savedViewRightMm = startPosMm + pixelRight * opsInMm;
_savedYCenterFraction = (canvas.Height/2 - pan.Y) / zoom / image.Height;
```

**Restore（新圖載入後，`_imageScaleFactor` 已更新）**：
```csharp
double leftPx  = (savedViewLeftMm  - startPosMm) / (opsInMm * _imageScaleFactor);
double rightPx = (savedViewRightMm - startPosMm) / (opsInMm * _imageScaleFactor);
float zoom = (float)(canvas.Width / (rightPx - leftPx));
float panX = (float)(-leftPx * zoom);
float panY = (float)(canvas.Height / 2 - savedYCenterFraction * newImage.Height * zoom);
canvas.SetView(zoom, new PointF(panX, panY));
```

**呼叫時序保證**：
```
LoadImages() → SaveViewIfNeeded()       ← 舊 scaleFactor
             → RunWorkflowAsync()
                  → OnGallerySelectionChanged()
                       → SetImageScaleFactor(newSf)  ← 更新
                       → UpdateCanvas(newImage)       ← 用新 sf 還原
```

**注意**：`SaveViewIfNeeded()` 在 `Image == null` 時直接 return，**不會**重置
`_shouldRestoreView` 旗標。這是因為 `ClearStitchedMode()` 會先清空 Image，
若 reset flag 則之前存好的 view 會被覆蓋。

**適用場景**：不只 `checkBoxShowProcessed` 切換，所有導覽控制項都保留 view：
- `btnPeriodPrev/Next`、`cbDate/cbTime`（period 模式）
- `cbReviewGrabId`、`btnGrabIdPrev/Next`（stitched 模式）
- `checkBoxShowProcessed`（兩種模式都適用）

**Stitched 模式路徑**：Form 端呼叫 `SaveCanvasView()` → `LoadGrabStitchedViewAsync` →
`ShowStitchedCameraInCanvas` 呼叫 `RestoreViewOrFitToScreen()`（不經 ClearCanvas，
因為 stitched 圖片由 `_stitchedImages[]` 管理，不應被 dispose）。

Fallback：settings 不可用時退回 pixel zoom/pan 直接還原（同倍率仍正確）。

---

## MuraChartHelper — Chart 軸線設定

### Y 軸移到右側

**陷阱**：`Axis` 沒有 `IsOnTheRightSide` 屬性。

正確做法：
- `AxisY`（左）：隱藏 label/刻度，保留 grid 和 StripLines（Primary axis 才能渲染 StripLines）
- `AxisY2`（右）：顯示刻度 label，不畫 grid
- 資料 Series 綁 `AxisType.Primary`
- 加 anchor series（transparent，Y=[0,2.2]）強制 AxisY/AxisY2 初始化 scale

**陷阱：AxisY2 不顯示 label**：AxisY2 沒有 bound series 時 scale 不初始化。解法：加 `_anchorY2`（Secondary，transparent）。

---

## MuraChart 閾值參考線

`MuraChartHelper.SetThresholds(float mean, float max)` 畫兩條水平參考線：
- `ErrorValueMax` → **紅色實線**
- `ErrorValueMean` → **紅色虛線**

- **StripLines 必須放在 `AxisY`（Primary）**：AxisY2 上初始化時不渲染
- **陷阱：Series + ±1e9 X 座標 → `OverflowException`**
- **`RefreshThresholds()` 不可放在 `UpdateDataAndView()` 末尾**：會在 `ResumeUpdates()` 之後再觸發一次 chart redraw（StripLines/Axis 修改不受 SuspendUpdates 壓制），造成閃爍。只在 `Build()` 和 `SetThresholds()` 呼叫。

---

## MuraChart X 軸與 Canvas 對齊（InnerPlotPosition 補償）

### 問題
`ScaleView.Zoom(leftMm, rightMm)` 對應 plot 內部區域邊緣，而非控制項邊緣。右側 AxisY2 標籤 margin 導致曲線向右偏移。

### 解法：反推 zoom 範圍

讀取 `ChartArea.InnerPlotPosition`（百分比），反推控制項邊緣對應所需的 zoom 值：

```csharp
private void GetAdjustedZoom(double leftMm, double rightMm,
                             out double zoomMin, out double zoomMax)
{
    double s = rightMm - leftMm;
    zoomMin = leftMm + _cachedFLeft  * s;
    zoomMax = leftMm + _cachedFRight * s;
}
```

### 正確讀取時機：PostPaint 事件快取

`InnerPlotPosition` 只在渲染後才有效值。應在 `PostPaint` 事件快取：

```csharp
private double _cachedFLeft  = 0.0;  // 預設無補償
private double _cachedFRight = 1.0;

private void OnPostPaint(object sender, ChartPaintEventArgs e)
{
    if (_innerPlotPositionFrozen) return;
    var inner = _chart.ChartAreas[0].InnerPlotPosition;
    if (inner.Width < 1.0) return;
    _cachedFLeft  = (inner.X + 0.5f) / 100.0;  // 0.5f = 左邊界留白
    _cachedFRight = (inner.X + inner.Width) / 100.0;
    _innerPlotPositionFrozen = true;
    // 凍結 InnerPlotPosition（Auto=false）
    // 若 cache 改變且已有邏輯視野 → BeginInvoke 補正 zoom
}
```

### 初次載入跳動問題修法（三件事同時做）

1. **凍結 InnerPlotPosition**（`Auto=false`）：防止後續 zoom/data 改變版面
2. **記錄邏輯視野**（`_logicalLeftMm/_logicalRightMm`）
3. **首次凍結時補正**：`BeginInvoke` 非同步重算 zoom

**為何用 `BeginInvoke`**：PostPaint 在 render pipeline 中，直接修改 zoom 會觸發遞迴 render。

---

## RowMuraChartHelper InnerPlotPosition 補償

與 `MuraChartHelper` 同理：PostPaint 首次渲染後量測，凍結比例。

**Y 軸標籤反轉**：透過 `Customize` 事件將標籤文字替換為 `totalMm - value`，不使用 `IsReversed`（避免 X 軸跳到頂部的副作用）。

---

## WinForms Chart IsReversed 陷阱

**問題**：`AxisY.IsReversed = true` 會讓 X 軸跳到 chart 頂部。

**解法**：若只需 Y 軸標籤「0 在上、max 在下」，用 `Customize` 事件攔截標籤，替換文字為 `totalValue - originalValue`。

---

## WinForms Chart X 軸更多 label

`LabelAutoFitMinFontSize = 6`（預設 ~8），縮小後可顯示更多 label。配合 MinorGrid 提供視覺密度。

---

## 全覽圖（7 台合併曲線）

三個全覽圖控制項皆用 `MuraChartHelper`（Zoomable=false）：
- `chartOverview`（tabPageReview）
- `chartLiveOverview`（tabPageLiveView），由 `_liveOverviewTimer` 驅動（50–500ms）

### 兩層合併演算法
1. 格點間距：`gridMm = max(minOpsUm/1000, (globalMax-globalMin)/MaxOverviewPoints)`
2. 全域 X 範圍：`min(Cam_Pos)` ~ `max(Cam_Pos + curveLen × opsMm)`
3. **第一層（per-camera max-window）**：同一相機多點映射到同一 bin → 取最大值
4. **第二層（cross-camera overlap）**：不同相機重疊區域 → Mean 取平均、Max 取最大值
5. X 軸 OPS 傳入 `gridMm * 1000.0`

### MuraChartHelper mm 單位標籤
- 使用 `PostPaint` 事件在圖表右下角繪製 "mm"（不使用 Chart Title）
- 靜態 Font/Brush 避免重複建立
