---
name: modify-pipeline
description: Modify image-processing pipelines, native processing calls, buffer management, curve generation, background subtraction, resize behavior, or inspection output persistence.
---

# modify-pipeline

修改影像處理 pipeline、Buffer 管理、存檔格式相關程式碼。

## 使用時機

修改 GPU pipeline、NativeBufferPool、InspectionEngine、AoiService、存檔格式或 ImageRepository 時。

## 關鍵檔案

→ 見 repo 根 `AGENTS.md` §關鍵檔案速查（subset：`ImageProcessing/*` + `Services/AoiService` + `Interop/NativeMethods` + `ImageCatalog/ImageRepository`）。
→ Native pipeline 細節見 `sdk/TanukiCv/native/tanuki_pipeline/`（framework/modules/pipelines/api 分層；modules/background_sub + ridge_hessian）；MIL 取像 API 屬 `$modify-acquisition`。

## 注意事項

### GPU Pipeline（永遠雙方向）
- Pipeline 永遠以 `"vertical+horizontal"` 執行，`RidgeDirection` 只影響 UI 顯示
- 流程：`calcColumnMeans_RemoveOutliers` → `calcColumnBackground` → `gaussianBlur`（一次共用）→ `computeHessianResponse(V)` → `computeHessianResponse(H)`
- `_ridgeBuffer` = vertical ridge，`_muraBuffer` = horizontal ridge（步驟 5 覆蓋去背圖）

### 曲線中性化（峰值保留，無 clamp）
- 曲線（`mura_curve_mean/max`、`mura_row_curve_mean/max`）從 **`d_hessian_resp_` 原始 float Hessian response** 計算，**在 scale+clamp 到 u8 之前**
- 計算後套用 `scale_f32_inplace_gpu(..., scale_factor)` 做純 scalar 乘法（`scale_factor = 255/正規值`），**不 clamp** → 峰值保留、.bin 值可超過 255
- u8 ridge 影像（`d_ridge_out` / `d_mura_out`）仍走 `scale_clamp_f32_to_u8_gpu` 顯示路徑，不影響 `.bin`
- inline kernel `k_scale_f32_inplace` 定義在 `modules/ridge_hessian/src/ridge_hessian.cu` 本檔內（不暴露到 SDK header）

### 正規值 V/H 分離（C# 層）
- Settings 拆分為 `HessianMaxFactorV`（垂直）+ `HessianMaxFactorH`（水平），native 端介面（`AoiAlgorithmParams.HessianMaxFactor`）維持單一欄位
- **Capture 時** 送進 native 的單一 HM = `HessianMaxFactorV` → bin 中 baked-in 的縮放係數是 `255/HessianMaxFactorV`
- **View 時** rescale 公式：
  - V/欄曲線（`chartLiveColumn` / `chartReviewColumn` / `chartDataColumn`）：`display = (bin/255) × HM_V_capture × HM_V_current` — 先還原 neutral Hessian，再套目前欄正規值
  - H/列曲線（`chartLiveRow` / `chartReviewRow`）：`display = (bin/255) × HM_V_capture × HM_H_current` — bin 仍以 V_capture 寫入，但顯示套目前列正規值
  - CSV 與 `#CURVE-C` 已保存拍攝當時的正規化摘要，不可再套 raw bin 公式；摘要換算為
    `displaySummary = capturedSummary × HM_current / HM_capture`。兩種表示由
    `HessianRescaleHelper.RawCurveToDisplayScale` 與 `NormalizedValueToDisplayScale` 明確分流。
- **Live 強化圖**：neutral half map 每幀依自身最大值編成 8-bit，並把 `sourceGain` 隨幀送到
  `ImageDisplayView` / `WaterfallView` / `ThumbStrip`；顯示倍率為 `currentHm/sourceGain`。不得以 capture HM
  當固定 byte 基準，否則高峰會先截成 255、低峰則受量化誤差放大。
- 改 PropertyGrid 正規值 V/H 時，Form 的 `_propertyGrid_PropertyValueChanged` 呼叫 `RefreshMuraProfileForSettingsChange` + `_stitchCoordinator.UpdateStitchedOverviewChart` 立即重畫
- CSV `#CFG` 記錄 `HessianMaxFactorV`、`HessianMaxFactorH` 與 capture-time `RidgeSigma`；
  細線濾除變更必須產生新版快照。舊資料缺少 `RidgeSigma` 時維持可讀，該值視為未知（0）。

### CUDA Pinned Memory
- 所有 NativeBufferPool buffer 使用 `TanukiCv_AllocPinned`（cudaMallocHost）
- NativeBufferPool.Dispose：`_isDisposed = true` 必須在所有 `FreePinned` 之前設定（防重複 Free）

### 存檔格式（TrySaveCapture）
- 7 個固定檔案：`_raw.jpg`, `_proc_c.jpg`, `_proc_r.jpg`, `_mean_c.bin`, `_max_c.bin`, `_mean_r.bin`, `_max_r.bin`
- 額外（`SaveOriginalBmp=true`）：全解析度 `.bmp`（必須在 callback 同步完成）
- JPEG 寫入移至 `Task.Run` 背景執行；BMP 匯出同步（`sourceBuffer` 會被 MIL 回收）
- 時間戳精確到毫秒（`.fff`），同 Line Rate 相機由 `CaptureTimestampCoordinator` 協調

### 存檔縮圖 fused（一進多出，2026-07）
- 縮圖**不再**在 `TrySaveCapture` 呼 `TanukiCv_Resize_GPU`（那會把已在 GPU 的圖拉回 host 再二次 H2D）。
- 改成：`TryApplyPicoaterRidge` 呼 `ProcessImage` 時，output struct 帶 3 個 pinned dst（raw/V/H）+ 目標尺寸 →
  pipeline（`export_api.cpp`）在**檢測同一次 device 停留**、用 resident `d_input/d_ridge/d_mura` + 可重用 `d_resize`
  就地縮 → D2H。`TrySaveCapture` 直接讀預縮好的 buffer。
- **兩個獨立 gate**：存檔縮圖仍由 `wantResize = EnableAutoCapture && !SuppressCapture && CaptureRootPath && scale>1 && buffers` 決定；
  neutral half Hessian 標準圖則每個成功處理幀都產生，監控與回顧共用相同來源，不得綁在存檔開關；
  純 live 幀傳 0 → pipeline 跳過縮圖（不浪費）。存/不存以 grab 為單位決定，不做 per-frame。
- **防呆**：`_lastFrameResized`（TryApplyPicoaterRidge 開頭清 false、ProcessImage 成功才設 wantResize）→
  detection 失敗幀不讀舊縮圖。resize 失敗 pipeline 回 -2（該幀不存，不崩）。
- `TanukiCv_Resize_GPU` 仍保留給 **GPU LOD 顯示**（它的輸入本來就在 host，非重複上傳）。
- 上傳次數：原圖 H2D 從「檢測+存檔各 1」降為只有檢測 1 次；省掉每幀 3 次 resize 的 cudaMalloc/H2D。

### .bin 檔案格式
```
magic(4)="MCBF" | version(4=int) | scale_factor(4=float) | array_length(4=int) | float[]
```
- `scale_factor` = `SaveResizeScale`；曲線長度 = 全解析度圖寬
- 讀端唯一入口為 `CurveBinFile.Load`：header 逐欄驗證、float payload 一次 bulk read；禁止回退成逐元素 `ReadSingle()`。
- 多 capture 欄曲線由 `CurveMergeHelper.MergeCurves` 邊讀邊累加 Mean／Max，不保留全部來源陣列做第二輪掃描；每個 bin 仍完整讀取，不得用 latest-only 掠過序號。
- 報表單序號可使用 `SingleGrabCurveSummaryStore` 的 `.mcsf` 可重建匯總，避免冷磁碟開啟數百個小 bin；匯總不改統計公式、不取代原始 bin。重建後先回 UI，互動停止 750ms 才由 bounded 單一 writer 採同目錄暫存檔原子替換。

### Capture archive compatibility
- `{grabId}.acap` stores independent raw/processed JPEG and C/R curve records for one grab.
- `CaptureArchiveStore` owns the container index, CRC validation, virtual paths, and random record reads.
- All readers must accept both archive virtual paths and legacy individual files; UI code must not
  parse the binary format or treat the archive as one stitched image.
- Rebuildable preview caches live inside the same archive. New archives prefer one raw/C/R
  `PreviewAtlas` record per mode: camera strips are packed into one JPEG bounded by 1920x1080,
  with camera rectangles and source dimensions in the payload. Review reads and decodes one atlas,
  then splits camera images in memory. Older per-frame thumbnail records remain read fallback only.
  Full JPEG records remain authoritative; settled full-image load invalidates older preview output.
- CSV remains the selection/index truth. A selected grab with `images=0` or `cams=0` is a
  data-health failure (`R2.assets`), not a valid blank review state.

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
4. 若修改 native API，同步更新 `$add-native-api` skill 的範本
