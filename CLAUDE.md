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
