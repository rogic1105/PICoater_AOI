# modify-pipeline

修改影像處理 pipeline、Buffer 管理、存檔格式相關程式碼。

## 使用時機

修改 GPU pipeline、NativeBufferPool、InspectionEngine、AoiService、存檔格式或 ImageRepository 時。

## 關鍵檔案

→ 見 `CLAUDE.md` §關鍵檔案速查（subset：`ImageProcessing/*` + `Services/AoiService` + `Interop/NativeMethods` + `ImageCatalog/ImageRepository`）。
→ Native pipeline 細節見 `src/native/modules/GetPICoaterBackground/` 與 `docs/dev/MIL_API_Reference.md`。

## 注意事項

### GPU Pipeline（永遠雙方向）
- Pipeline 永遠以 `"vertical+horizontal"` 執行，`RidgeDirection` 只影響 UI 顯示
- 流程：`calcColumnMeans_RemoveOutliers` → `calcColumnBackground` → `gaussianBlur`（一次共用）→ `computeHessianResponse(V)` → `computeHessianResponse(H)`
- `_ridgeBuffer` = vertical ridge，`_muraBuffer` = horizontal ridge（步驟 5 覆蓋去背圖）

### 曲線中性化（峰值保留，無 clamp）
- 曲線（`mura_curve_mean/max`、`mura_row_curve_mean/max`）從 **`d_hessian_resp_` 原始 float Hessian response** 計算，**在 scale+clamp 到 u8 之前**
- 計算後套用 `scale_f32_inplace_gpu(..., scale_factor)` 做純 scalar 乘法（`scale_factor = 255/正規值`），**不 clamp** → 峰值保留、.bin 值可超過 255
- u8 ridge 影像（`d_ridge_out` / `d_mura_out`）仍走 `scale_clamp_f32_to_u8_gpu` 顯示路徑，不影響 `.bin`
- inline kernel `k_scale_f32_inplace` 定義在 `Module_GetPICoaterBackground.cu` 本檔內（不暴露到 SDK header）

### 正規值 V/H 分離（C# 層）
- Settings 拆分為 `HessianMaxFactorV`（垂直）+ `HessianMaxFactorH`（水平），native 端介面（`AoiAlgorithmParams.HessianMaxFactor`）維持單一欄位
- **Capture 時** 送進 native 的單一 HM = `HessianMaxFactorV` → bin 中 baked-in 的縮放係數是 `255/HessianMaxFactorV`
- **View 時** rescale 公式：
  - V 曲線（chartMuraVertical / chartOverview / chartMuraProfile / muraChartVerticalLive）：`display = (bin/255) × (HM_V_capture / HM_V_current)` — 改 V 即時生效
  - H 曲線（chartMuraHorizontal / row chart）：`display = (bin/255) × (HM_V_capture / HM_H_current)` — 改 H 即時生效；公式 numerator 用 V_capture 因為 bin 是被 V baked-in
- 改 PropertyGrid 正規值 V/H 時，Form 的 `_propertyGrid_PropertyValueChanged` 呼叫 `RefreshMuraProfileForSettingsChange` + `_stitchCoordinator.UpdateStitchedOverviewChart` 立即重畫
- CSV `#CFG` 記錄兩個欄位 `HessianMaxFactorV`、`HessianMaxFactorH`；舊單一 `HessianMaxFactor` 欄位讀檔時 fallback 到 V=H=該值

### CUDA Pinned Memory
- 所有 NativeBufferPool buffer 使用 `CoreCV_AllocPinned`（cudaMallocHost）
- NativeBufferPool.Dispose：`_isDisposed = true` 必須在所有 `FreePinned` 之前設定（防重複 Free）

### 存檔格式（TrySaveCapture）
- 7 個固定檔案：`_raw.jpg`, `_proc_v.jpg`, `_proc_h.jpg`, `_mean_v.bin`, `_max_v.bin`, `_mean_h.bin`, `_max_h.bin`
- 額外（`SaveOriginalBmp=true`）：全解析度 `.bmp`（必須在 callback 同步完成）
- JPEG 寫入移至 `Task.Run` 背景執行；BMP 匯出同步（`sourceBuffer` 會被 MIL 回收）
- 時間戳精確到毫秒（`.fff`），同 Line Rate 相機由 `CaptureTimestampCoordinator` 協調

### .bin 檔案格式
```
magic(4)="MCBF" | version(4=int) | scale_factor(4=float) | array_length(4=int) | float[]
```
- `scale_factor` = `SaveResizeScale`；曲線長度 = 全解析度圖寬

### ImageRepository 掃描
- 同時掃 `*_raw.jpg` + `*.bmp`，兩格式共存
- **JPG 優先**：同相機同時存在 JPG+BMP 時，回傳 JPG（避免 duplicate key 崩潰）
- `_proc_*.jpg` 和 `*.bin` 不被收入索引

### StandardBgSub 模式
- `precomputed_col_mean != nullptr` → 跳過動態 column mean 計算
- bg bin 路徑：`bg_{width}_{cameraId}.bin`
- `TryComputeColumnMean` 從 `_milLastGrabBuffer`（原始）讀取，不可從 `_milProcBuffer`（已處理，近乎全零）

### ProcessingFunction 行為
- **不管 `EnableImageProcessing` 一律執行 GPU 處理**
- `EnableImageProcessing` 只控制「顯示原圖還是處理圖」
- 目的：即使 checkbox 未勾選也能計算 Mura peak 值供 CSV 判斷

## 步驟

1. 讀取要修改的 pipeline 階段
2. 確認 buffer 映射是否正確
3. 修改 + build 驗證（Release|x64）
4. 若修改 native API，同步更新 `/add-native-api` skill 的範本
