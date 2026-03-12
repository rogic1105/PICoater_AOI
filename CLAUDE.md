# PICoater AOI — Claude Code Rules

## 專案結構

```
PICoater_AOI/
├── src_dotnet/AniloxRoll.Monitor/   ← C# WinForms 應用程式
├── src_native/                      ← C++ pipeline 實作
└── sdk/AOI_SDK/                     ← 共用 SDK (core_cv_api / AOI.SDK)
```

## Native API

兩組 DLL，均宣告於 `src_dotnet/AniloxRoll.Monitor/Interop/NativeMethods.cs`：

| DLL | 函式 | 用途 |
|-----|------|------|
| `picoater_api.dll` | `PICoaterAPI_CreatePipeline` / `ProcessPipeline` / `DestroyPipeline` | GPU 檢測 pipeline |
| `core_cv_api.dll` | `CoreCV_AllocPinned` / `CoreCV_FreePinned` | CUDA pinned memory 管理 |
| `core_cv_api.dll` | `CoreCV_FastReadBMP` | 快速讀取 BMP（繞過 GDI+） |
| `core_cv_api.dll` | `CoreCV_Resize_GPU` | GPU 縮圖 |

## P/Invoke 架構規則

**所有 P/Invoke 宣告只能在 `AniloxRoll.Monitor/Interop/NativeMethods.cs`**，不得跨層使用 SDK 的 `AOI.SDK.Core.CoreCVWrapper`。

```csharp
// 正確
using AniloxRoll.Monitor.Core.Interop;
NativeMethods.CoreCV_FastReadBMP(...)

// 錯誤 — 跨層引用 SDK
using AOI.SDK.Core;
CoreCVWrapper.CoreCV_FastReadBMP(...)
```

## 效能設計原則

### 縮圖優先 / 延遲全解析度
1. `BatchInspectionService.ProcessBatch` → Parallel.For 平行產生 7 張縮圖（GPU resize）
2. 縮圖顯示於 `ThumbnailGridPresenter`
3. 使用者點選 → `OnGallerySelectionChanged` → `RunInspectionFullRes`（同步、on-demand）

目標：使用者感知延遲 ≤ 0.5 秒

### CUDA Pinned Memory

`NativeBufferPool` 的所有 buffer 使用 `CoreCV_AllocPinned`（cudaMallocHost），確保 H↔D memcpy 走 DMA 加速：

```
_inputBuffer, _muraBuffer, _ridgeBuffer, _thumbnailBuffer, _curveMeanBuffer, _curveMaxBuffer
```

### 影像處理順序

```
CoreCV_FastReadBMP  →  AoiService.ProcessImage  →  CoreCV_Resize_GPU  →  Create8bppBitmap
     IO (pinned)          GPU pipeline (選用)          GPU 縮圖 (選用)        CPU bitmap
```

## 效能計時

`InspectionEngine.RunInspectionFullRes` 和 `FormInteractionHelper.OnGallerySelectionChanged` 已有 Stopwatch 計時輸出：

```
[FullRes] mode=True  | IO=  17ms | GPU=  22ms | BMP=  28ms | Copy=  0ms | Total=   69ms  (14288x9003)
[OnSelect] Cam1 | FullRes=   69ms | Canvas=   0ms | Chart=  46ms | Total=  116ms
```

懷疑效能問題時，先看計時輸出，再對症下藥。使用 `/perf-diagnose` skill 協助分析。

## 關鍵檔案速查

| 路徑 | 職責 |
|------|------|
| `src_dotnet/AniloxRoll.Monitor/Interop/NativeMethods.cs` | 唯一 P/Invoke 宣告點 |
| `src_dotnet/AniloxRoll.Monitor/ImageProcessing/NativeBufferPool.cs` | CUDA pinned buffer 管理 |
| `src_dotnet/AniloxRoll.Monitor/ImageProcessing/InspectionEngine.ImageProcessing.cs` | 縮圖/全解析度影像處理 |
| `src_dotnet/AniloxRoll.Monitor/ImageProcessing/InspectionEngineConfig.cs` | MaxWidth=16384, MaxHeight=10000 |
| `src_dotnet/AniloxRoll.Monitor/ImageProcessing/BatchInspectionService.cs` | Parallel.For 批次縮圖 |
| `src_dotnet/AniloxRoll.Monitor/UI/Widgets/FormInteractionHelper.cs` | UI 互動、gallery 選擇、計時 |
| `sdk/AOI_SDK/core_cv_api/src/export_api.cpp` | CoreCV_Resize_GPU 實作 |
| `sdk/AOI_SDK/core_cv_api/include/export_c/export_api.h` | CoreCV_Resize_GPU 宣告 |

---

## Envision_MdigGrab 模組（MIL 相機擷取工具）

### 路徑

```
sdk/AOI_SDK/src_dotnet/Envision_MdigGrab/Envision_MdigGrab/
├── Config/
│   └── CameraConfig.cs          ← 單台相機靜態設定（Id, DCF, ExposureUs, Panel, Label）
├── Hardware/
│   ├── MilCameraUnit.cs         ← 單台相機所有 MIL 資源（Digitizer/Display/Buffer）
│   ├── MilSystemManager.cs      ← MIL Application + System 分配與釋放（static）
│   └── MilImageProcessor.cs     ← MIL 影像處理（ColMeanSubtraction / Hessian / Binarize）
├── Session/
│   └── CameraSession.cs         ← 多台相機生命週期管理（Initialize/ToggleGrab/Release）
└── UI/
    ├── GrabForm.cs              ← 主視窗，僅含按鈕事件與 UI 更新
    ├── GrabForm.Designer.cs
    └── CameraListViewPresenter.cs ← ListView 欄位初始化與每 500ms 更新
```

### 架構原則

- `GrabForm` 不直接碰 MIL API，全部委由 `CameraSession`（相機）與 `CameraListViewPresenter`（ListView）
- `CameraSession` 不含任何 UI 依賴
- `MilCameraUnit` 不管 MIL System 生命週期，System 由 `CameraSession` 統一管理

### CLProtocol（GenICam Camera Link）

**啟用方式**（`MilCameraUnit.Initialize()` 完成後在背景執行緒自動執行）：
```csharp
MIL.MdigControl(MilDigitizer, MIL.M_GC_CLPROTOCOL_DEVICE_ID, "M_DEFAULT");
MIL.MdigControl(MilDigitizer, MIL.M_GC_CLPROTOCOL, M_ENABLE);
// → _clProtocolEnabled = true
```

**重要**：`M_GC_CLPROTOCOL, M_ENABLE` 會載入 CLProtocol DLL 並讀取相機 GenICam XML，**耗時數秒**。
必須用 `Task.Run` 在背景執行，不可同步呼叫，否則 Init MIL 按鈕會卡頓。

CLProtocol 就緒後自動重新套用 `_appliedExposureUs`（`SetExposureUs` 重呼叫）。

### 曝光設定規則

| 條件 | Set | Get（量測） | 單位 |
|------|-----|------------|------|
| CLProtocol 啟用 | `MdigControlFeature("ExposureTime")` | `MdigInquireFeature("ExposureTime")` | **μs**（直接） |
| CLProtocol 未啟用 | `MdigControl(M_EXPOSURE_TIME, μs×1000)` | `MdigInquire(M_EXPOSURE_TIME)` ÷ 1000 | ns → μs |

- Camera Link 無 CLProtocol 時 `MdigInquire(M_EXPOSURE_TIME)` 通常回傳 0（硬體不支援讀回）
- `_appliedExposureUs` 永遠記錄最後設定值，不依賴硬體回讀

### Feature API（CLProtocol 啟用後可用）

| Feature 名稱 | 用途 | 單位 |
|---|---|---|
| `"ExposureTime"` | 曝光讀寫 | μs |
| `"AcquisitionLineRate"` | Line Rate 讀寫 | Hz |
| `"DeviceTemperature"` | 相機本體溫度（唯讀） | °C |

使用 `MdigInquireFeature` / `MdigControlFeature`，均需先確認 `_clProtocolEnabled == true`。

### ListView 欄位索引（共 16 欄，0–15）

| 索引 | 欄位名稱 | 來源 | 格式 |
|------|---------|------|------|
| [0] | Camera | CameraConfig.Id | — |
| [1] | FPS | `MdigInquire(M_PROCESS_FRAME_RATE)` | F2 |
| [2] | Target FPS | `MdigInquire(M_SELECTED_FRAME_RATE)` | F2 |
| [3] | Line Rate(Hz) | `MdigInquireFeature("AcquisitionLineRate")` | F1 |
| [4] | Exp Set(μs) | `_appliedExposureUs`（不回讀硬體） | F1 |
| [5] | Exp Meas(μs) | `MdigInquireFeature("ExposureTime")` | F1 |
| [6] | Frames | `MdigInquire(M_PROCESS_FRAME_COUNT)` | — |
| [7] | Missed | `MdigInquire(M_PROCESS_FRAME_MISSED)` | — |
| [8] | Grab Miss | `MdigInquire(M_GRAB_FRAME_MISSED)` | — |
| [9] | Resolution | `MdigInquire(M_SIZE_X/Y)` | "W×H" |
| [10] | Scan Mode | `MdigInquire(M_SCAN_MODE)` | Line/Progressive |
| [11] | FPGA(°C) | `MsysInquire(M_TEMPERATURE_FPGA)`（擷取卡） | F1 |
| [12] | Cam Temp(°C) | `MdigInquireFeature("DeviceTemperature")`（相機） | F1 |
| [13] | Mem Free(MB) | `MsysInquire(M_MEMORY_FREE)` ÷ 1024² | — |
| [14] | PCIe Lanes | `MsysInquire(M_PCIE_NUMBER_OF_LANES)` | — |
| [15] | PCIe Speed | `MsysInquire(M_PCIE_SPEED)` | Gen1/2/3 |

**維護注意**：新增或移除欄位時，`Initialize()`（`for i < N`）、`Update()`（各 SubItems[n]）、`ResetAll()`（`for i <= N`）三處必須同步修改。

### MIL 資源釋放順序

```
MdigProcess(M_STOP) → MdispHookFunction(M_UNHOOK) → MbufFree × n
→ MdispFree → MdigFree     （MilCameraUnit.Free()）
→ MsysFree × n             （CameraSession.ReleaseResources()）
→ MappFreeDefault           （MilSystemManager.FreeApplication()）
```

`IsReleasing = true` 必須在 Timer Stop 之前設定，防止 Tick 存取已釋放資源。

### 已知限制

- `M_LINE_RATE` / `M_LINE_RATE_CURRENT` / `M_GRAB_SIZE_Y` 常數在 MIL .NET wrapper 中**不存在**，不可使用。
  - Line Rate → CLProtocol Feature API `"AcquisitionLineRate"`
  - Grab Height → `MdigControl(M_SOURCE_SIZE_Y, height)`（`M_SOURCE_SIZE_Y` 存在），且必須走完整 Stop → Free Buffers → Set → Realloc → Restart 流程，否則舊尺寸 Buffer 與新尺寸不符會崩潰。
- `MdigHookFunction(M_CAMERA_PRESENT)` 已移除，相機連線狀態改由 Timer 每 500ms 輪詢 `MdigInquire(M_CAMERA_PRESENT)`。
- CLProtocol 初始化期間（約 1–2 秒）`Exp Meas`、`Line Rate`、`Cam Temp` 欄位會顯示 N/A，屬正常現象。
