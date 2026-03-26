# 影像處理 Pipeline — GPU、Buffer 映射、存檔格式

## 效能設計原則

### 縮圖優先 / 延遲全解析度
1. `BatchInspectionService.ProcessBatch` → Parallel.For 平行產生 7 張縮圖（GPU resize）
2. 縮圖顯示於 `ThumbnailGridPresenter`
3. 使用者點選 → `OnGallerySelectionChanged` → `RunInspectionFullRes`（同步、on-demand）

目標：使用者感知延遲 ≤ 0.5 秒

### CUDA Pinned Memory

`NativeBufferPool` 的所有 buffer 使用 `CoreCV_AllocPinned`（cudaMallocHost），確保 H↔D memcpy 走 DMA 加速：

```
_inputBuffer, _muraBuffer, _ridgeBuffer, _thumbnailBuffer, _curveMeanBuffer, _curveMaxBuffer,
_curveRowMeanBuffer, _curveRowMaxBuffer
```

### 影像處理順序

```
CoreCV_FastReadBMP  →  AoiService.ProcessImage  →  CoreCV_Resize_GPU  →  Create8bppBitmap
     IO (pinned)          GPU pipeline (選用)          GPU 縮圖 (選用)        CPU bitmap
```

---

## CUDA Pipeline V/H Ridge 分離（永遠雙方向）

**Pipeline 永遠以 `"vertical+horizontal"` 模式執行**，確保 V/H 影像與曲線皆存檔。`RidgeDirection` recipe 設定僅影響 UI 顯示，不影響 pipeline 計算。

`PICoaterDetector::Run`（`Module_GetPICoaterBackground.cu`）流程：
1. Column mean background removal → `d_mura_out`（去背圖）
2. Gaussian blur（一次，V/H 共用）→ `d_hessian_f32_`
3. `computeHessianResponse_gpu(VERTICAL)` → `d_ridge_out` + col curves（`calcColumnMeans`/`calcColumnMax`）
4. `computeHessianResponse_gpu(HORIZONTAL)` → `d_mura_out`（覆蓋去背圖）+ row curves（`calcRowMeans`/`calcRowMax`）

C# 端對應：`_ridgeBuffer` = vertical ridge，`_muraBuffer` = horizontal ridge

---

## GPU 強化圖 Pipeline（PICoaterDetector）詳細

Raw image 經過 GPU pipeline 產生強化圖（ridge image）+ Mean/Max 曲線。

### 完整流程

```
Raw Image (8-bit grayscale, Pinned Memory)
  │
  ├─ CoreCV_FastReadBMP → _inputBuffer
  ├─ cudaMemcpy (Host → Device)
  │
  ╔══ PICoaterDetector::Run("vertical+horizontal") ═════════════╗
  ║                                                             ║
  ║  1. calcColumnMeans_RemoveOutliers_gpu()                    ║
  ║     → 每行(column)平均值，去除離群值 (σ_col=1)              ║
  ║                                                             ║
  ║  2. calcColumnBackground_u8_gpu()                           ║
  ║     → d_mura_out = 原圖 - 行平均（去背景）                  ║
  ║                                                             ║
  ║  3. gaussianBlur_gpu (σ=RidgeSigma)（一次，V/H 共用）       ║
  ║     → d_hessian_f32_                                        ║
  ║                                                             ║
  ║  4. computeHessianResponse_gpu(VERTICAL)                    ║
  ║     → d_ridge_out（vertical ridge）                         ║
  ║     → calcColumnMeans_gpu → MuraCurveMean                   ║
  ║     → calcColumnMax_gpu  → MuraCurveMax                     ║
  ║                                                             ║
  ║  5. computeHessianResponse_gpu(HORIZONTAL)                  ║
  ║     → d_mura_out（horizontal ridge，覆蓋去背圖）            ║
  ║     → calcRowMeans_gpu → MuraRowCurveMean                   ║
  ║     → calcRowMax_gpu  → MuraRowCurveMax                     ║
  ║                                                             ║
  ╚═════════════════════════════════════════════════════════════╝
  │
  ├─ cudaMemcpy (Device → Host)
  │   → _ridgeBuffer(V), _muraBuffer(H), col/row curve buffers
  │
  └─ Create8bppBitmap + Marshal.Copy
     → InspectionData { Image, MuraCurveMean/Max, MuraRowCurveMean/Max }
```

### 參數對照

| 參數 | 值 | 來源 | 可調 |
|------|------|------|------|
| σ_col（離群值去除） | 1 | hardcoded in .cu | ✗ |
| BgSigmaFactor | 2.0 | `InspectionEngineConfig.DefaultBgSigma` | ✗ |
| RidgeSigma | 9.0 | `InspectionEngineConfig.DefaultRidgeSigma` | ✗ |
| RidgeMode | "vertical+horizontal" | hardcoded（永遠雙方向） | ✗ |
| HessianMaxFactor | 2.0 (default) | `InspectionRecipe.HessianMaxFactor` → PropertyGrid | ✓ |
| ErrorValueMean | 0.3 (default) | `InspectionRecipe` → PropertyGrid | ✓（閾值，不影響 GPU） |
| ErrorValueMax | 0.5 (default) | `InspectionRecipe` → PropertyGrid | ✓（閾值，不影響 GPU） |

### 關鍵檔案

| 檔案 | 職責 |
|------|------|
| `src_native/modules/GetPICoaterBackground/src/Module_GetPICoaterBackground.cu` | GPU kernel：5 步驟實作 |
| `src_native/c_api/picoater_api/src/export_api.cpp` | Native C API：cudaMemcpy + pipeline 調度 |
| `sdk/AOI_SDK/core_cv/include/core_cv/imgproc/core_background.hpp` | mean/background/max 函式宣告 |
| `sdk/AOI_SDK/core_cv/include/core_cv/imgproc/core_features.hpp` | hessianRidge 函式宣告 |
| `src_dotnet/AniloxRoll.Monitor/ImageProcessing/InspectionEngine.ImageProcessing.cs` | C# 入口：ProcessImage / RunInspectionFullRes |
| `src_dotnet/AniloxRoll.Monitor/Services/AoiService.cs` | C# ↔ Native P/Invoke wrapper |

---

## MIL 與 GPU 記憶體類型對照

| 類型 | API | 說明 | 適用場景 |
|------|-----|------|---------|
| MIL Buffer | `MbufAlloc2d` | MIL 管理的 Host 記憶體 | MdigProcess 抓圖、MdispSelect 顯示 |
| GPU Device | `CoreCV_MallocGPU`（cudaMalloc） | GPU 顯示卡上的記憶體 | CUDA kernel 直接讀寫 |
| Pinned Host | `CoreCV_AllocPinned`（cudaMallocHost） | CPU 側 DMA 加速記憶體 | H↔D memcpy 高吞吐，如 NativeBufferPool |

MilGrabSample 使用 **GPU Device** 記憶體（二值化 kernel）。
AniloxRoll.Monitor 使用 **Pinned Host** 記憶體（picoater pipeline 大量 DMA 傳輸）。

---

## 效能計時

`InspectionEngine.RunInspectionFullRes` 和 `FormInteractionHelper.OnGallerySelectionChanged` 已有 Stopwatch 計時輸出：

```
[FullRes] mode=True  | IO=  17ms | GPU=  22ms | BMP=  28ms | Copy=  0ms | Total=   69ms  (14288x9003)
[OnSelect] Cam1 | FullRes=   69ms | Canvas=   0ms | Chart=  46ms | Total=  116ms
```

懷疑效能問題時，先看計時輸出，再對症下藥。使用 `/perf-diagnose` skill 協助分析。

---

## 存檔格式（TrySaveCapture）

### 檔案命名與目錄結構

時間戳精確到毫秒（`.fff`），同 Line Rate 的相機由 `CaptureTimestampCoordinator` 協調共用同一 `.fff`（100ms 容差）。

**一律存檔的 7 個檔案**（不管 `SaveOriginalBmp` 設定）：
```
{CaptureRootPath}\{yyyy}\{yyyyMM}\{yyyyMMdd}\
    {yyyyMMdd_HHmmss.fff}-{CameraId}_raw.jpg      ← 縮小版原圖（GPU resize，1/SaveResizeScale）
    {yyyyMMdd_HHmmss.fff}-{CameraId}_proc_v.jpg   ← 切向（vertical）ridge 處理圖
    {yyyyMMdd_HHmmss.fff}-{CameraId}_proc_h.jpg   ← 法向（horizontal）ridge 處理圖
    {yyyyMMdd_HHmmss.fff}-{CameraId}_mean_v.bin   ← 切向 Mura Mean 曲線（全解析度長度）
    {yyyyMMdd_HHmmss.fff}-{CameraId}_max_v.bin    ← 切向 Mura Max 曲線（全解析度長度）
    {yyyyMMdd_HHmmss.fff}-{CameraId}_mean_h.bin   ← 法向 Mura Mean 曲線（全解析度長度）
    {yyyyMMdd_HHmmss.fff}-{CameraId}_max_h.bin    ← 法向 Mura Max 曲線（全解析度長度）
```
- `_v` 後綴 = vertical（切向），對應 `chartMuraVertical`（chartMura）
- `_h` 後綴 = horizontal（法向），對應 `chartMuraHorizontal`
- `_proc_h.jpg` 來源：CUDA `d_mura_out` 存放水平 ridge 圖（pipeline 永遠跑雙方向），resize 後存檔

**額外檔案**（`SaveOriginalBmp = true` 時）：
```
    {yyyyMMdd_HHmmss.fff}-{CameraId}.bmp       ← 全解析度原圖（同步匯出，因 sourceBuffer 會被 MIL 回收）
```

### TrySaveCapture 非同步 I/O

GPU resize + Marshal.Copy 在 MIL callback 執行緒完成（快），JPEG/bin 寫入移至 `Task.Run` 背景執行緒，callback 立即返回不阻塞連續抓圖。`_lastCaptureKey` 在 Task.Run 前提前更新，防止重複觸發。`SaveOriginalBmp=true` 時 BMP 匯出必須在 callback 同步完成（`sourceBuffer` 會被 MIL 回收用於下一幀）。

### .bin 檔案格式

```
magic(4)="MCBF" | version(4=int) | scale_factor(4=float) | array_length(4=int) | float[]
```

- `scale_factor` 儲存縮圖倍率（`SaveResizeScale`），供 `ReadScaleFactorFromBin` 讀取
- 曲線長度 = 全解析度圖寬，`_raw.jpg` 寬度 = 全解析度 ÷ scale_factor

---

## InspectionData 格式欄位

| 欄位 | 類型 | 說明 |
|------|------|------|
| `IsCompressedJpeg` | bool | `true`=新格式，`false`=BMP |
| `ScaleFactor` | int | 縮圖倍率（1=BMP，5=JPEG 1/5） |

- 兩者由 `InspectionEngine.LoadFromPrecomputedFiles`（新格式）或 `RunInspectionFullRes` BMP 路徑設定
- 非處理模式下（curves=null）由 `ReadScaleFactorFromBin` 讀 .bin 標頭取得 ScaleFactor

---

## ImageRepository 掃描邏輯

同時掃 `*_raw.jpg` + `*.bmp`，兩種格式可在同一根目錄共存，`ParsePath` regex 兩種皆可 match。
`*_proc_v.jpg`、`*_proc_h.jpg`、`*_v.bin`、`*_h.bin` 不被收入（不符合 glob 模式）。

**JPG 優先規則**：`GetImages()` 同一相機同時存在 JPG 與 BMP 時，優先回傳 JPG（讀取速度快，走 `LoadFromPrecomputedFiles` 不需 GPU）。避免 `.ToDictionary()` 重複 key 崩潰。

### 混格式掃描

```csharp
// 同時掃兩種格式，讓不同時期的資料共存
Directory.GetFiles(root, "*_raw.jpg", AllDirectories)
    .Concat(Directory.GetFiles(root, "*.bmp", AllDirectories))
    .ToArray()
```
**陷阱**：舊的 either/or 邏輯（先掃 jpg，有就不掃 bmp）會讓混合資料夾丟失 BMP 檔。

**陷阱**：`GetImages()` 原本用 `.ToDictionary(x => x.CameraId, x => x.FullPath)`，當 BMP+JPG 同時存在時 duplicate key 會拋 `ArgumentException`，導致 `RunWorkflowAsync` 靜默失敗（UI 無反應）。改用手動迭代 + JPG 優先覆寫邏輯。

### SaveOriginalBmp（原 UseCompressedCapture 反轉）

- `SaveOriginalBmp=false`（預設）：存 7 檔（`_raw.jpg`/`_proc_v.jpg`/`_proc_h.jpg`/`_mean_v.bin`/`_max_v.bin`/`_mean_h.bin`/`_max_h.bin`）
- `SaveOriginalBmp=true`：額外同步匯出全解析度 `.bmp`
- JSON 向後相容：讀取時若有 `SaveOriginalBmp` key 直接用，否則讀 `UseCompressedCapture` 並反轉

---

## 壓縮存檔格式細節

### 檔名時間戳：`yyyyMMdd_HHmmss.fff`

- 毫秒精度（`.fff`），支援每秒多幀存檔
- 同 Line Rate 相機由 `CaptureTimestampCoordinator` 在 100ms 容差內共用同一 `.fff`
- `_lastCaptureKey` 以毫秒粒度去重（原為秒級）
- 相關解析點：`ImageRepository` regex `(\d{6})\.(\d{3})-(\d)`、`TryParseFileNameDateTime` 解析 19 碼、review tab 秒下拉顯示 `ss.fff`

### 捕獲端（AniloxCamera.TrySaveCapture）

- `UseCompressedCapture=true`：GPU resize + `Marshal.Copy` 在 callback 執行緒完成 → 磁碟 I/O（`SaveJpegFromBytes` + `SaveCurveBinFromArray`）移至 `Task.Run` 背景執行
- `UseCompressedCapture=false`：`MbufExport(.bmp)`（同步，舊行為）
- JPEG 需要 24bpp：GDI+ JPEG encoder 不支援 8bpp indexed；用 `GCHandle.Alloc` pin byte[] → `Create8bppBitmap` → `Graphics.DrawImage` 至 24bpp → `Save(JpegCodecInfo)`

### 回顧端（InspectionEngine.ImageProcessing.cs）

- 路徑末尾 `_raw.jpg` → `LoadFromPrecomputedFiles`；否則 BMP+GPU 路徑（向下相容）
- 非處理模式（curves=null）的 ScaleFactor：`ReadScaleFactorFromBin` 只讀 16 bytes 標頭，不載入整個 float[]
- `IsCompressedJpeg` / `ScaleFactor` 統一由 engine 設定，UI 層直接讀取，**不再從 curve/image 比例推斷**

---

## NativeBufferPool Dispose 安全模式

`_isDisposed = true` 必須在所有 `FreePinned` 呼叫**之前**設定：

```csharp
public void Dispose()
{
    if (_isDisposed) return;
    _isDisposed = true;  // ← 先設，即使後續 Free 拋例外也不會重複釋放
    FreePinned(ref _inputBuffer);
    // ... 其餘 Free
}
```

若先 Free 再設旗標，中途拋例外會導致 `_isDisposed` 永遠是 `false`，下次 `Dispose()` 重複 Free 同一 pointer。
