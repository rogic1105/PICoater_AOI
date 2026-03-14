# MilGrabSample — Claude Code Rules

Matrox MIL 多相機擷取工具，支援 Camera Link + CLProtocol（GenICam）。
作為 AniloxRoll.Monitor 取像模組的**參考實作**，兩者共用相同的 MIL API 邏輯。

## 專案結構

```
MilGrabSample/
├── Config/
│   └── CameraConfig.cs            ← 單台相機靜態設定（Id, DCF, ExposureUs, Panel, Label）
├── Hardware/
│   ├── MilCameraUnit.cs           ← 單台相機所有 MIL 資源（Digitizer/Display/Buffer）
│   ├── MilSystemManager.cs        ← MIL Application + System 分配與釋放（static）
│   └── MilImageProcessor.cs       ← MIL 影像處理（ColMeanSubtraction / Hessian / Binarize）
├── Session/
│   └── CameraSession.cs           ← 多台相機生命週期管理（Initialize/ToggleGrab/Release）
└── UI/
    ├── GrabForm.cs                ← 主視窗，僅含按鈕事件與 UI 更新
    ├── GrabForm.Designer.cs
    └── CameraListViewPresenter.cs ← ListView 16 欄初始化與每 500ms 更新
```

## 架構層次與職責

```
GrabForm                    ← 按鈕事件、UI 更新（不直接碰 MIL API）
    │
    ├─ CameraSession        ← 多台相機生命週期（Initialize/ToggleGrab/Release）
    │       │
    │       └─ MilCameraUnit × N   ← 每台相機的 MIL 資源封裝
    │               │
    │               └─ MilSystemManager  ← MIL Application + System（static）
    │
    └─ CameraListViewPresenter  ← ListView 顯示更新（從 MilCameraUnit 查詢數據）
```

**原則**：`GrabForm` 只做「使用者按了什麼」；硬體細節由 `CameraSession` 和 `MilCameraUnit` 處理；`CameraSession` 不含任何 UI 依賴。

---

## MIL API 初始化順序

### 階段 1：MIL Application 與 System（MilSystemManager）

```
MappAlloc("M_DEFAULT", M_DEFAULT, ref milApp)       // 1. 開 MIL Application（全局唯一）
  ↓
MsysAlloc(systemDescriptor, systemNum, M_DEFAULT, ref sysId)  // 2. 開擷取卡（每卡一次）
```

**釋放順序**（必須反向）：
```
MsysFree(sysId) × n   →   MappFreeDefault()
```

### 階段 2：單台相機資源（MilCameraUnit.Initialize）

```
MdigAlloc(sysId, devNum, dcfPath, M_DEFAULT, ref digitizer)  // 3. 開 Digitizer（相機通道）
  ↓
MdispAlloc(sysId, M_DEFAULT, "M_DEFAULT", M_DEFAULT, ref display)  // 4. 開 Display
  ↓
MdigInquire(digitizer, M_SIZE_X / M_SIZE_Y)          // 5. 查影像尺寸（由 DCF 決定）
  ↓
CoreCV_MallocGPU(out gpuInput, W, H)                 // 6. 分配 GPU Buffer（Input）
CoreCV_MallocGPU(out gpuOutput, W, H)                //    分配 GPU Buffer（Output）
new byte[W × H]（CPU host buffer × 2）               //    分配 CPU Buffer
  ↓
MbufAlloc2d(sysId, W, H, 8+M_UNSIGNED,
    M_IMAGE+M_GRAB+M_PROC, ref grabBuf[i]) × 2       // 7. Grab Buffer（雙緩衝）
MbufAlloc2d(... M_IMAGE+M_DISP+M_PROC, ref displayBuf)  // 8. Display Buffer
MbufAlloc2d(... M_IMAGE+M_PROC, ref procBuf)         // 9. Processing Buffer
  ↓
MdispSelectWindow(display, displayBuf, panelHwnd)    // 10. 綁定 Panel
MdispControl(display, M_SCALE_DISPLAY, M_ONCE)
MdispControl(display, M_CENTER_DISPLAY, M_ENABLE)
MdispControl(display, M_MOUSE_USE, M_ENABLE)
  ↓
MdispHookFunction(display, M_MOUSE_MOVE, callback)   // 11. 掛 Mouse Hook
```

**CLProtocol 不在 Initialize 啟動**（見下方說明）。

### 階段 3：開始抓圖（ApplyGrabState / CameraSession.StartGrab）

```
CameraSession.StartGrab()
  └─ cam.SetUserGrabIntent(true)
       └─ ApplyGrabState()
            ├─ CheckPresence()  // MdigInquire(M_CAMERA_PRESENT)
            ├─ MdigProcess(digitizer, grabBufs, 2, M_START, M_DEFAULT, callback, userData)
            ├─ IsLive = true
            └─ StartCLProtocolAsync()    // ← 所有資源就緒後才啟動 CLProtocol
```

### 階段 4：CLProtocol 初始化（背景執行緒）

```
Task.Run(TryEnableCLProtocol)
    ├─ MdigControl(digitizer, M_GC_CLPROTOCOL_DEVICE_ID, "M_DEFAULT")
    ├─ MdigControl(digitizer, M_GC_CLPROTOCOL, M_ENABLE)  // 耗時 1–2 秒，載入 GenICam XML
    ├─ _clProtocolEnabled = true
    ├─ SetExposureUs(_appliedExposureUs)     // 重套曝光（改走 Feature API μs 路徑）
    └─ SetLineRateHz(_appliedLineRateHz)     // 重套線掃速率
```

**為何不能在 Initialize() 裡啟動 CLProtocol**：`M_GC_CLPROTOCOL, M_ENABLE` 會載入 CLProtocol DLL 並讀取相機 GenICam XML，耗時數秒，且其背景執行緒的 `MdigControl` 會與主執行緒的 `MbufAlloc2d` 競爭 MIL 內部鎖，導致 UI 卡頓。必須等 `MdigProcess(M_START)` 確認圖像串流已建立後才啟動。

`_clProtocolInitStarted`（`volatile bool`）防止 ToggleGrab 重複觸發。

---

## 參數設定規則

### 曝光時間（SetExposureUs）

| 條件 | Set | Get（量測） | 單位 |
|------|-----|------------|------|
| CLProtocol 啟用 | `MdigControlFeature("ExposureTime", M_TYPE_DOUBLE)` | `MdigInquireFeature("ExposureTime")` | **μs**（直接） |
| CLProtocol 未啟用 | `MdigControl(M_EXPOSURE_TIME, μs × 1000)` | `MdigInquire(M_EXPOSURE_TIME)` ÷ 1000 | ns → μs |

- `_appliedExposureUs` 永遠記錄最後設定值，不依賴硬體回讀
- Camera Link 無 CLProtocol 時 `MdigInquire(M_EXPOSURE_TIME)` 通常回傳 0

### 線掃速率（SetLineRateHz）

- 僅能透過 CLProtocol Feature API `"AcquisitionLineRate"` 設定（Hz）
- CLProtocol 未啟用時：記錄至 `_appliedLineRateHz`，CLProtocol 就緒後自動重套
- `_appliedLineRateHz` 機制與 `_appliedExposureUs` 相同

### 擷取高度（SetGrabHeight）— 完整流程，不可省略

```
1.  MdigProcess(M_STOP)                          // 停止抓圖
2.  MbufFree(GrabBuffers × 2)                    // 釋放舊 MIL Buffer
    MbufFree(DisplayBuffer)
    MbufFree(ProcBuffer)
3.  CoreCV_FreeGPU(gpuInput)                     // 釋放舊 GPU Buffer
    CoreCV_FreeGPU(gpuOutput)
4.  MdigControl(M_SOURCE_SIZE_Y, newHeight)      // 設定新高度
5.  MdigInquire(M_SIZE_X / M_SIZE_Y)             // 重查實際尺寸（硬體可能夾緊）
6.  CoreCV_MallocGPU(out gpuInput, W, H)         // 重新分配 GPU Buffer
    CoreCV_MallocGPU(out gpuOutput, W, H)
    new byte[W × H] × 2                          // 重新分配 CPU Buffer
7.  MbufAlloc2d × 4                              // 重新分配 MIL Buffer
8.  MdispSelectWindow(display, newBuf, hwnd)     // 重新綁定 Display
9.  MdigProcess(M_START)  ← if wasLive           // 恢復抓圖
```

**崩潰警告**：跳過步驟 2–3 直接 `M_SOURCE_SIZE_Y` 再重用舊 Buffer，尺寸不符會導致 MIL 崩潰。

### UI 控制項範圍（CameraParamPanel 常數）

| 參數 | Min | Max | Default | SmallChange | LargeChange |
|------|-----|-----|---------|-------------|-------------|
| 曝光時間（μs） | 1 | 10000 | — | — | — |
| 線掃速率（Hz） | 1 | 10000 | — | — | — |
| 擷取高度（px） | 1 | 10000 | 2048 | 64 | 512 |

---

## MIL 資源釋放順序（MilCameraUnit.Free）

```
_isReleased = true                               // 立即阻止 ProcessingFunction 繼續
  ↓
MdigProcess(M_STOP)
  ↓
MdispHookFunction(M_MOUSE_MOVE + M_UNHOOK)
MdispSelectWindow(M_NULL, IntPtr.Zero)
  ↓
MbufFree(GrabBuffers × 2)
MbufFree(DisplayBuffer)
MbufFree(ProcBuffer)
  ↓
CoreCV_FreeGPU(gpuInput)
CoreCV_FreeGPU(gpuOutput)
  ↓
MdispFree(display)
MdigFree(digitizer)
  ↓
GCHandle.Free()
```

CameraSession 釋放順序：
```
IsReleasing = true  →  Timer.Stop()
  →  cam.Free() × N  →  MsysFree × n  →  MappFreeDefault
```

`IsReleasing = true` 必須在 `Timer.Stop()` 之前，防止 Tick 存取已釋放相機資源。

---

## Telemetry 查詢方法

| 方法 | MIL API | CLProtocol 必要 |
|------|---------|----------------|
| `CurrentFps` | `MdigInquire(M_PROCESS_FRAME_RATE)` | 否 |
| `GetSelectedFrameRate()` | `MdigInquire(M_SELECTED_FRAME_RATE)` | 否 |
| `GetFrameCount()` | `MdigInquire(M_PROCESS_FRAME_COUNT)` | 否 |
| `GetFrameMissed()` | `MdigInquire(M_PROCESS_FRAME_MISSED)` | 否 |
| `GetGrabFrameMissed()` | `MdigInquire(M_GRAB_FRAME_MISSED)` | 否 |
| `GetScanMode()` | `MdigInquire(M_SCAN_MODE)` | 否 |
| `GetLineRateHz()` | `MdigInquireFeature("AcquisitionLineRate")` | **是** |
| `GetMeasuredExposureUs()` | `MdigInquireFeature("ExposureTime")` | **是** |
| `GetCameraTemperature()` | `MdigInquireFeature("DeviceTemperature")` | **是** |
| `GetFpgaTemperature()` | `MsysInquire(M_TEMPERATURE_FPGA)` | 否 |
| `GetMemoryFreeMB()` | `MsysInquire(M_MEMORY_FREE)` | 否 |
| `GetPcieNumberOfLanes()` | `MsysInquire(M_PCIE_NUMBER_OF_LANES)` | 否 |
| `GetPcieSpeed()` | `MsysInquire(M_PCIE_SPEED)` | 否 |

## ListView 欄位索引（共 16 欄，0–15）

| 索引 | 欄位名稱 | 來源 |
|------|---------|------|
| [0] | Camera | CameraConfig.Id |
| [1] | FPS | `MdigInquire(M_PROCESS_FRAME_RATE)` |
| [2] | Target FPS | `MdigInquire(M_SELECTED_FRAME_RATE)` |
| [3] | Line Rate(Hz) | `MdigInquireFeature("AcquisitionLineRate")` |
| [4] | Exp Set(μs) | `_appliedExposureUs`（不回讀硬體） |
| [5] | Exp Meas(μs) | `MdigInquireFeature("ExposureTime")` |
| [6] | Frames | `MdigInquire(M_PROCESS_FRAME_COUNT)` |
| [7] | Missed | `MdigInquire(M_PROCESS_FRAME_MISSED)` |
| [8] | Grab Miss | `MdigInquire(M_GRAB_FRAME_MISSED)` |
| [9] | Resolution | `MdigInquire(M_SIZE_X/Y)` |
| [10] | Scan Mode | `MdigInquire(M_SCAN_MODE)` |
| [11] | FPGA(°C) | `MsysInquire(M_TEMPERATURE_FPGA)` |
| [12] | Cam Temp(°C) | `MdigInquireFeature("DeviceTemperature")` |
| [13] | Mem Free(MB) | `MsysInquire(M_MEMORY_FREE)` ÷ 1024² |
| [14] | PCIe Lanes | `MsysInquire(M_PCIE_NUMBER_OF_LANES)` |
| [15] | PCIe Speed | `MsysInquire(M_PCIE_SPEED)` |

**維護注意**：新增或移除欄位時，`Initialize()`（`for i < N`）、`Update()`（各 SubItems[n]）、`ResetAll()`（`for i <= N`）三處必須同步修改。

---

## 已知 MIL .NET Wrapper 限制

- `M_LINE_RATE` / `M_LINE_RATE_CURRENT` / `M_GRAB_SIZE_Y` 常數**不存在**，不可使用。
  - Line Rate → CLProtocol Feature API `"AcquisitionLineRate"`
  - Grab Height → `MdigControl(M_SOURCE_SIZE_Y, height)`（存在）
- `MdigHookFunction(M_CAMERA_PRESENT)` 已移除，改用 Timer 每 500ms 輪詢 `MdigInquire(M_CAMERA_PRESENT)`。
- CLProtocol 初始化期間（約 1–2 秒）[3][5][12] 欄位顯示 N/A，屬正常現象。
