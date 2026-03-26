# MIL 取像模組 — AniloxCamera / LiveCameraManager / MilGrabSample

## 架構對應

```
LiveCameraManager           ← 多台相機生命週期（Allocate/Grab/Free/Reinitialize）
    │
    └─ AniloxCamera × 7     ← 單台相機 MIL 資源（對應 MilGrabSample/MilCameraUnit）
            │
            └─ CameraSystemManager  ← MIL Application + System（對應 MilSystemManager）
```

---

## MIL 資源分配順序（AniloxCamera.Initialize）

```
MdigAlloc(systemId, devNum, dcfPath)          // 1. 開 Digitizer
  ↓
MdigControl(M_SOURCE_SIZE_Y, grabHeight)      // 2. 設 Grab 高度（查尺寸前先設）
  ↓
MdispAlloc × 2（primary + secondary）         // 3. 開 Display（AniloxCamera 有副顯）
  ↓
MdigInquire(M_SIZE_X / M_SIZE_Y)              // 4. 查實際尺寸
  ↓
new byte[W×H]  +  NativeBufferPool(W, H, 1)  // 5. CPU buffer + CUDA Pinned Memory
  ↓
MbufAlloc2d × 2（Grab，M_GRAB+M_PROC）        // 6. Grab Buffer（雙緩衝）
MbufAlloc2d（Display，M_DISP+M_PROC）         // 7. Display Buffer
MbufAlloc2d（Proc，M_PROC）                   // 8. Processing Buffer
  ↓
MdispSelectWindow(display, displayBuf, hwnd)  // 9. 綁定 Panel
MdispControl(M_SCALE_DISPLAY, M_ONCE)
MdispControl(M_CENTER_DISPLAY, M_ENABLE)
MdispControl(M_MOUSE_USE, M_ENABLE)
  ↓
MdispHookFunction(M_MOUSE_MOVE)               // 10. 掛 Mouse Hook
MdispHookFunction(M_MOUSE_LEFT_BUTTON_DOWN)
  ↓
SetExposureUs(_appliedExposureUs)             // 11. 套用初始曝光（CLProtocol 尚未啟用，走 legacy ns 路徑）
```

**注意**：步驟 2 必須在步驟 4 之前，否則 Buffer 大小錯誤。CLProtocol **不在此處啟動**（會與 MbufAlloc2d 競爭 MIL 內部鎖）。

---

## CLProtocol 啟動時序

```
SetUserGrabIntent(true)
  └─ ApplyGrabState()
       ├─ CheckPresence()                          // 確認相機在線
       ├─ MdigProcess(M_START, callback)           // 開始連續抓圖
       ├─ IsLive = true
       └─ StartCLProtocolAsync()                   // 所有資源就緒後才啟動
              └─ Task.Run(TryEnableCLProtocol)     // 背景執行，不阻塞 UI
                   ├─ MdigControl(M_GC_CLPROTOCOL_DEVICE_ID, "M_DEFAULT")
                   ├─ MdigControl(M_GC_CLPROTOCOL, M_ENABLE)  // 耗時 1–2 秒
                   ├─ _clProtocolEnabled = true
                   ├─ SetExposureUs(_appliedExposureUs)    // 重套曝光（改走 Feature API）
                   └─ SetLineRateHz(_appliedLineRateHz)    // 重套線掃速率（CLProtocol 才能設）
```

`_clProtocolInitStarted`（`volatile bool`）guard 防止 ToggleGrab 重複觸發。

### 為什麼 CLProtocol 必須延遲到第一次抓圖？

`MdigControl(M_GC_CLPROTOCOL, M_ENABLE)` 載入 CLProtocol DLL + 讀取相機 GenICam XML，耗時 2–5 秒。
若在 `Initialize()` 期間以 `Task.Run` 啟動，背景執行緒的 `MdigControl` 會與主執行緒的
`MbufAlloc2d`、`MdispAlloc` 競爭 MIL 內部鎖，造成 Init 按鈕卡頓。

**Guard 寫法**：
```csharp
private volatile bool _clProtocolInitStarted = false;

private void StartCLProtocolAsync()
{
    if (_clProtocolInitStarted) return;
    _clProtocolInitStarted = true;
    Task.Run(() => TryEnableCLProtocol());
}
```

### CLProtocol 初始化逾時保護

`TryEnableCLProtocol()` 在硬體異常時可能無限等待。使用 `Task.WhenAny` + `Task.Delay` 實現非阻塞逾時：

```csharp
private void StartCLProtocolAsync()
{
    if (_clProtocolInitStarted) return;
    _clProtocolInitStarted = true;
    var initTask    = Task.Run((Action)TryEnableCLProtocol);
    var timeoutTask = Task.Delay(TimeSpan.FromSeconds(10));
    Task.WhenAny(initTask, timeoutTask).ContinueWith(_ =>
    {
        if (!initTask.IsCompleted)
            Trace.WriteLine($"[CAM{CameraId}] CLProtocol 初始化逾時（>10s）...");
    });
}
```

- **不取消 initTask**：即使逾時，initTask 仍可在後台完成（MIL 不支援安全取消）
- **ContinueWith 在 ThreadPool 執行**：不阻塞 UI 或 CLProtocol 執行緒

---

## 曝光設定規則

| 條件 | Set | Get（量測） | 單位 |
|------|-----|------------|------|
| CLProtocol 啟用 | `MdigControlFeature("ExposureTime", M_TYPE_DOUBLE)` | `MdigInquireFeature("ExposureTime")` | **μs**（直接） |
| CLProtocol 未啟用 | `MdigControl(M_EXPOSURE_TIME, μs×1000)` | `MdigInquire(M_EXPOSURE_TIME)` ÷ 1000 | ns → μs |

- `_appliedExposureUs` 永遠記錄最後設定值，不依賴硬體回讀
- Camera Link 無 CLProtocol 時 `MdigInquire(M_EXPOSURE_TIME)` 通常回傳 0

### SetExposureUs 夾緊上限（曝光 × 線掃速率）

曝光時間上限 = `floor(900000 / lineRateHz)`，與 UI 的 `CalcExpMax` 公式一致：

```csharp
if (_appliedLineRateHz > 0)
{
    double maxUs = Math.Max(1.0, Math.Min(10000.0,
                   Math.Floor(900000.0 / _appliedLineRateHz)));
    if (exposureUs > maxUs) exposureUs = maxUs;
}
```

- 公式來源：一行掃描時間 = 1/lineRateHz 秒；曝光必須 < 90% × 掃描時間（安全係數）
- `_appliedLineRateHz = 0` 時跳過（CLProtocol 尚未設定）

---

## 線掃速率設定規則

| 條件 | 行為 |
|------|------|
| CLProtocol 啟用 | `MdigControlFeature("AcquisitionLineRate", M_TYPE_DOUBLE, Hz)` 立即生效 |
| CLProtocol 未啟用 | 僅記錄至 `_appliedLineRateHz`，待 CLProtocol 就緒後自動重套 |

`SetLineRateHz` 與 `SetExposureUs` 機制一致：先記錄、後套用，重新初始化後也能正確恢復。

---

## SetGrabHeight 完整流程（不可省略任何步驟）

```
1. MdigProcess(M_STOP)                           // 停止抓圖
2. MbufFree(GrabBuffers × 2)                     // 釋放舊 MIL Buffer
   MbufFree(DisplayBuffer)
   MbufFree(ProcBuffer)
3. NativeBufferPool.Dispose()                    // 釋放 CUDA Pinned Memory
4. MdigControl(M_SOURCE_SIZE_Y, newHeight)       // 設定新高度
5. MdigInquire(M_SIZE_X / M_SIZE_Y)              // 重查實際尺寸（硬體可能夾緊）
6. new byte[W×H]  +  new NativeBufferPool(W,H,1) // 重新分配 CPU + CUDA Pinned
7. MbufAlloc2d × 4                               // 重新分配 MIL Buffer
8. MdispSelectWindow(display, newDisplayBuf, hwnd)// 重新綁定 Display
9. MdigProcess(M_START)  ← if wasLive            // 恢復抓圖
```

**崩潰警告**：舊尺寸 Buffer 與新尺寸不符會導致 MIL 崩潰。必須先釋放再重新分配，不可省略步驟 2–3。

**Rollback 機制**：步驟 4–9 包在 try/catch 中，失敗時先再次呼叫 `FreeGrabBuffers()` 清除殘留，再嘗試以原高度 `AllocateAndBind(oldHeight)`。若 rollback 也失敗則設 `_userWantsGrab = false` 停用相機，防止 Timer 反覆重試。私有方法：`FreeGrabBuffers()`（步驟 2–3）、`AllocateAndBind(targetHeight, shouldRestart)`（步驟 4–9）。

---

## MIL 資源釋放順序（AniloxCamera.Dispose）

```
_isReleased = true                               // 立即阻止 ProcessingFunction 繼續
  ↓
MdigProcess(M_STOP)                              // 停止連續抓圖
  ↓
MdispHookFunction(M_MOUSE_MOVE + M_UNHOOK)       // 移除 Mouse Hook
MdispHookFunction(M_MOUSE_LEFT_BUTTON_DOWN + M_UNHOOK)
MdispSelectWindow(M_NULL, IntPtr.Zero)           // 取消 Display 綁定
  ↓
MbufFree(GrabBuffers × 2)                        // 釋放 Buffer
MbufFree(DisplayBuffer)
MbufFree(ProcBuffer)
  ↓
NativeBufferPool.Dispose()                       // 釋放 CUDA Pinned Memory
AoiService.Dispose()
  ↓
MdispFree(primary)  +  MdispFree(secondary)      // 釋放 Display
  ↓
MdigFree(digitizer)                              // 最後釋放 Digitizer
  ↓
GCHandle.Free()                                  // 釋放 GCHandle（Callback 安全鎖）
```

LiveCameraManager 釋放順序：
```
IsReleasing = true  →  Timer.Stop()
  →  cam.Free() × 7  →  MsysFree × n  →  MappFreeDefault
```

`IsReleasing = true` 必須在 `Timer.Stop()` 之前設定，防止 Tick 存取已釋放相機資源。

---

## Telemetry 查詢方法（AniloxCamera）

| 方法 | MIL API | 說明 |
|------|---------|------|
| `CurrentFps` | `MdigInquire(M_PROCESS_FRAME_RATE)` | 實際量測 FPS |
| `GetSelectedFrameRate()` | `MdigInquire(M_SELECTED_FRAME_RATE)` | DCF 設定目標 FPS |
| `GetFrameCount()` | `MdigInquire(M_PROCESS_FRAME_COUNT)` | 累計處理幀數 |
| `GetFrameMissed()` | `MdigInquire(M_PROCESS_FRAME_MISSED)` | Callback 遺漏幀數 |
| `GetGrabFrameMissed()` | `MdigInquire(M_GRAB_FRAME_MISSED)` | 硬體層遺漏幀數 |
| `GetScanMode()` | `MdigInquire(M_SCAN_MODE)` | "Line" / "Progressive" |
| `GetLineRateHz()` | `MdigInquireFeature("AcquisitionLineRate")` | 需 CLProtocol 啟用 |
| `GetMeasuredExposureUs()` | `MdigInquireFeature("ExposureTime")` | 需 CLProtocol 啟用 |
| `GetCameraTemperature()` | `MdigInquireFeature("DeviceTemperature")` | 相機本體溫度（°C） |
| `GetFpgaTemperature()` | `MsysInquire(M_TEMPERATURE_FPGA)` | 擷取卡 FPGA 溫度 |
| `GetMemoryFreeMB()` | `MsysInquire(M_MEMORY_FREE)` | 板卡可用記憶體（MB） |
| `GetPcieNumberOfLanes()` | `MsysInquire(M_PCIE_NUMBER_OF_LANES)` | PCIe 通道數 |
| `GetPcieSpeed()` | `MsysInquire(M_PCIE_SPEED)` | "Gen1"/"Gen2"/"Gen3" |
| `TryGetSecondaryDisplayGeometry()` | `MdispInquire(M_ZOOM_FACTOR_X/Y, M_PAN_OFFSET_X/Y)` | 副顯示器 zoom/pan 狀態（隨使用者滾輪變化） |

---

## 已知 MIL .NET Wrapper 限制

- `M_LINE_RATE` / `M_LINE_RATE_CURRENT` / `M_GRAB_SIZE_Y` 常數**不存在**於 .NET wrapper，不可使用。
  - Line Rate → CLProtocol Feature API `"AcquisitionLineRate"`
  - Grab Height → `MdigControl(M_SOURCE_SIZE_Y, height)`（`M_SOURCE_SIZE_Y` 存在）
- `MdigHookFunction(M_CAMERA_PRESENT)` 已移除，改用 Timer 每 500ms 輪詢 `MdigInquire(M_CAMERA_PRESENT)`。
- CLProtocol 初始化期間（約 1–2 秒）Line Rate、Exp Meas、Cam Temp 無法讀取，屬正常現象。

---

## CUDA 冷啟動

第一次呼叫 `CoreCV_MallocGPU`（`cudaMalloc`）會初始化 CUDA context，耗時約 1–2 秒。
若要減少此開銷，可在 `MilCameraUnit.Initialize()` 前先呼叫任意 CUDA 熱身操作
（例如 `AoiService.Initialize()`，如 AniloxRoll.Monitor 的做法）。

---

## MIL Display Zoom/Pan 查詢（Live Chart 對齊）

`panelMainDisplay` 使用 MIL `M_SCALE_DISPLAY` + `M_CENTER_DISPLAY` + `M_MOUSE_USE`，使用者可用滾輪縮放/平移。

### 解法：MdispInquire 即時查詢

```csharp
MIL.MdispInquire(displayId, MIL.M_ZOOM_FACTOR_X, ref zoomX);
MIL.MdispInquire(displayId, MIL.M_ZOOM_FACTOR_Y, ref zoomY);
MIL.MdispInquire(displayId, MIL.M_PAN_OFFSET_X, ref panX);
MIL.MdispInquire(displayId, MIL.M_PAN_OFFSET_Y, ref panY);
```

Panel 邊緣 → buffer pixel → mm：
```csharp
double leftPixel  = panX;
double rightPixel = panX + panelWidth / zoomX;
double viewLeftMm  = startPos + leftPixel  * opsInMm;
double viewRightMm = startPos + rightPixel * opsInMm;
```

**陷阱**：滑鼠在 MIL 顯示的非影像區域（黑邊）時，`MouseStatusHandler` 回傳 `pixelValue = -1`。座標不可靠，需靠 `MdispInquire` 反推。

---

## CaptureTimestampCoordinator（同步多相機存檔時間戳）

7 台相機各自在 MIL callback 中取 `DateTime.Now`，同 FPS 相機有 5-15ms 時間差，導致檔名不同、無法直覺配對。

`CaptureTimestampCoordinator` 以 `(int)lineRateHz` 為 group key：
- 同一 rate group 內，第一台 callback 建立時間戳
- 後續同組在 100ms 內到達的共用同一時間戳
- 不同 rate 的相機各自獨立

### 關鍵接線
- `LiveCameraManager` 持有 `_timestampCoordinator`，注入每台 `AniloxCamera.TimestampCoordinator`
- **`AllocateCameras` 必須呼叫 `cam.SetLineRateHz()`**，否則 `_appliedLineRateHz = 0` 導致 coordinator 被跳過
- Grab 中途改 Line Rate → `_appliedLineRateHz` 立即更新 → 下一輪 callback 自動改組

---

## WinForms Timer + Task.Run 競爭條件修復

**問題**：Timer Tick（UI 執行緒）與 FreeCameras（Task.Run 背景執行緒）並發。若 `FreeCameras` 先釋放相機，Tick 仍在存取已釋放的資源。

**三步修復**：

```csharp
// 1. ReleaseAsync：先停 Timer，再 Task.Run
public async Task ReleaseAsync()
{
    _cameraStatusTimer.Stop();   // ← 先停
    IsReleasing = true;
    await Task.Run(() => FreeCameras());
}

// 2. Tick：快照相機清單，防止遍歷中被 Free
void CameraStatusTimer_Tick(object sender, EventArgs e)
{
    AniloxCamera[] snapshot;
    try { snapshot = _cameras.ToArray(); }
    catch { return; }

    foreach (var cam in snapshot)
    {
        if (IsReleasing) return;   // ← mid-loop 檢查
        // ... 讀取 Telemetry
    }
}

// 3. AniloxCamera：CheckPresence / ApplyGrabState 先查 _isReleased
private void CheckPresence()
{
    if (_isReleased) return;
}
```

**關鍵順序**：`Timer.Stop()` 必須在 `Task.Run` **之前**，因為 `Timer.Stop()` 發生在 UI 執行緒，確保下一個 Tick 不會被排入 message queue。

---

## Hardware → UI 反向同步（SyncFromCamera 5% hysteresis）

每 500ms 從相機硬體讀回實際值，超過 5% 才更新 UI，防止 CLProtocol 就緒後 UI 顯示舊值：

```csharp
private bool _syncingFromHw = false;  // 防止 ValueChanged 再回寫硬體

if (!_dragging.Contains(_expBars[idx]))
{
    double hw = cam.GetMeasuredExposureUs();
    if (hw > 0)
    {
        int clamped = Math.Max(bar.Minimum, Math.Min(bar.Maximum, (int)hw));
        double diff = Math.Abs(clamped - bar.Value) / (double)Math.Max(1, bar.Value);
        if (diff > 0.05)
        {
            _syncingFromHw = true;
            bar.Value = clamped; num.Value = clamped;
            acq.CameraExposureTimeUs[idx] = clamped;
            _syncingFromHw = false;
        }
    }
}
```

- `GetMeasuredExposureUs()` 只在 CLProtocol 就緒後回傳非零值
- `GetLineRateHz()` 同理

---

## 相機參數即時存檔

### 原則
`ConfigManager.SaveAcquisitionSettings(acq)` 在 `acq.CameraXxx[idx] = value` 之後**立即呼叫**，不受 `!_dragging` 或硬體呼叫阻擋。

### 順序
```
acq.CameraXxx[idx] = value;           // 1. 更新記憶體
ConfigManager.SaveAcquisitionSettings; // 2. 立即持久化
_liveCameraManager?.SetXxx();          // 3. 硬體（可能失敗，不影響存檔）
```

MouseUp 只負責補送硬體寫入 + `SwitchToCamera`，不再重複存檔。

---

## 即時 Telemetry ListView 架構

`LiveTelemetryPresenter`（移植自 MilGrabSample.CameraListViewPresenter）：

- **16 欄**：Camera / FPS / Target FPS / Line Rate / Exp Set / Exp Meas / Frames / Missed / GrabMiss / Resolution / Scan Mode / FPGA°C / Cam Temp°C / Mem Free / PCIe Lanes / PCIe Speed
- `Initialize(IList<CameraHardwareConfig>)` — 建立欄位 + 初始列（Tag = camId）
- `Update(IReadOnlyList<AniloxCamera>)` — 每 500ms 讀取所有 Telemetry 更新 SubItems
- `ResetAll()` — FreeCameras 後呼叫，所有欄位還原為 "N/A"
- `listViewCameras` 完全由 `LiveTelemetryPresenter` 管理
- Telemetry Timer 在 `SetupSystemTab()` 建立（`Interval=500`，永遠運行），Tick 同時呼叫 `Update` 和 `SyncCameraParamsFromHardware`

---

## AniloxCamera 影像處理 + 日誌整合

- `ProcessingFunction` **不管 `EnableImageProcessing` 一律執行 GPU 處理**（`TryApplyPicoaterRidge`）
  - `EnableImageProcessing` 只控制「顯示原圖還是處理圖」
  - 目的：即使 checkbox 未勾選也能計算 Mura 曲線 peak 值供 CSV 判斷
- `TryApplyPicoaterRidge` 傳入 `_nativeBufferPool.CurveMeanBuffer` + `CurveMaxBuffer`，讀回 peak 值（`max / 255f`），存入 `_lastMeanPeak` / `_lastMaxPeak`
- `TrySaveCapture` 存檔後觸發 `OnInspectionResult(camId, fileNameNoExt, meanPeak, maxPeak)`
- 事件鏈：`AniloxCamera.OnInspectionResult` → `LiveCameraManager.OnInspectionResult` → `AniloxRollForm.OnCameraInspectionResult` → `InspectionLogService.AppendRecord`
- **前提**：`EnableAutoCapture = true` 才會存檔，才有日誌

---

## MilGrabSample 模組（MIL 相機擷取參考實作）

### 路徑

```
sdk/AOI_SDK/src_dotnet/MilGrabSample/MilGrabSample/
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

### MilCameraUnit 初始化順序（正確）

```
Initialize()：
  MdigAlloc
  MdispAlloc
  CoreCV_MallocGPU × 2      ← GPU device 記憶體（第一次呼叫會觸發 CUDA context init）
  MbufAlloc2d × 4           ← MIL buffer
  MdispSelectWindow / MdispControl / MdispHookFunction
  ← Initialize() 結束，UI 立刻響應

ApplyGrabState()（第一次 MdigGrab）：
  MdigProcess(M_START)
  IsLive = true
  StartCLProtocolAsync()    ← 最後才啟動，避免競爭 MIL 內部鎖
```

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
