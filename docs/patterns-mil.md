# MIL 開發模式與陷阱

本文件涵蓋 MIL（Matrox Imaging Library）相關的開發模式、初始化順序、資源管理陷阱。
完整 MIL API 常數與方法參考另見 `docs/MIL_API_Reference.md`。

---

## MIL 初始化效能原則

### MilCameraUnit 初始化順序（正確）

```
Initialize()：
  MdigAlloc
  MdispAlloc
  CoreCV_MallocGPU × 2      ← GPU device 記憶體（第一次會觸發 CUDA context init）
  MbufAlloc2d × 4           ← MIL buffer
  MdispSelectWindow / MdispControl / MdispHookFunction
  ← Initialize() 結束，UI 立刻響應

ApplyGrabState()（第一次 MdigGrab）：
  MdigProcess(M_START)
  IsLive = true
  StartCLProtocolAsync()    ← 最後才啟動，避免競爭 MIL 內部鎖
```

### 為什麼 CLProtocol 必須延遲到第一次抓圖？

`MdigControl(M_GC_CLPROTOCOL, M_ENABLE)` 載入 CLProtocol DLL + 讀取相機 GenICam XML，耗時 2–5 秒。
若在 `Initialize()` 期間以 `Task.Run` 啟動，背景執行緒的 `MdigControl` 會與主執行緒的
`MbufAlloc2d`、`MdispAlloc` 競爭 MIL 內部鎖，造成 Init 按鈕卡頓。

### CLProtocol 初始化逾時保護

使用 `Task.WhenAny` + `Task.Delay` 實現非阻塞逾時，不需 `CancellationToken`：

```csharp
var initTask    = Task.Run((Action)TryEnableCLProtocol);
var timeoutTask = Task.Delay(TimeSpan.FromSeconds(10));
Task.WhenAny(initTask, timeoutTask).ContinueWith(_ =>
{
    if (!initTask.IsCompleted)
        Trace.WriteLine($"[CAM{CameraId}] CLProtocol 初始化逾時（>10s）...");
});
```

- **不取消 initTask**：MIL 不支援安全取消
- **ContinueWith 在 ThreadPool 執行**：不阻塞 UI

---

## MIL 與 GPU 記憶體類型對照

| 類型 | API | 說明 | 適用場景 |
|------|-----|------|---------|
| MIL Buffer | `MbufAlloc2d` | MIL 管理的 Host 記憶體 | MdigProcess 抓圖、MdispSelect 顯示 |
| GPU Device | `CoreCV_MallocGPU`（cudaMalloc） | GPU 顯示卡上的記憶體 | CUDA kernel 直接讀寫 |
| Pinned Host | `CoreCV_AllocPinned`（cudaMallocHost） | CPU 側 DMA 加速記憶體 | H↔D memcpy 高吞吐 |

MilGrabSample 使用 **GPU Device** 記憶體。
AniloxRoll.Monitor 使用 **Pinned Host** 記憶體。

---

## CUDA 冷啟動

第一次 `CoreCV_MallocGPU`（`cudaMalloc`）會初始化 CUDA context，耗時約 1–2 秒。
若要減少開銷，在 `MilCameraUnit.Initialize()` 前先呼叫任意 CUDA 熱身操作。

---

## MIL 資源釋放順序

### AniloxCamera.Dispose

```
_isReleased = true           → MdigProcess(M_STOP)
→ MdispHookFunction(UNHOOK)  → MdispSelectWindow(M_NULL)
→ MbufFree × 4               → NativeBufferPool.Dispose() → AoiService.Dispose()
→ MdispFree × 2              → MdigFree
→ GCHandle.Free()
```

### LiveCameraManager

```
IsReleasing = true  →  Timer.Stop()
  →  cam.Free() × 7  →  MsysFree × n  →  MappFreeDefault
```

`IsReleasing = true` 必須在 `Timer.Stop()` 之前。

---

## WinForms Timer + Task.Run 競爭條件

**問題**：Timer Tick（UI 執行緒）與 FreeCameras（背景執行緒）並發存取已釋放資源。

**三步修復**：
1. `ReleaseAsync`：先 `Timer.Stop()` 再 `Task.Run`（`Stop()` 在 UI 執行緒確保下一 Tick 不排入 queue）
2. Tick：快照相機清單 `_cameras.ToArray()`，mid-loop 檢查 `IsReleasing`
3. `AniloxCamera`：`CheckPresence` / `ApplyGrabState` 先查 `_isReleased`

---

## SetGrabHeight 完整流程（不可省略任何步驟）

```
1. MdigProcess(M_STOP)
2. MbufFree × 4 + NativeBufferPool.Dispose()     // 釋放舊 Buffer
3. MdigControl(M_SOURCE_SIZE_Y, newHeight)
4. MdigInquire(M_SIZE_X/Y)                       // 重查（硬體可能夾緊）
5. new byte[] + new NativeBufferPool              // 重新分配
6. MbufAlloc2d × 4
7. MdispSelectWindow(newDisplayBuf)
8. MdigProcess(M_START) if wasLive
```

**崩潰警告**：舊尺寸 Buffer 與新尺寸不符 → MIL 崩潰。必須先釋放再重新分配。

**Rollback**：失敗時 `FreeGrabBuffers()` 清殘 → `AllocateAndBind(oldHeight)`。若 rollback 也失敗 → `_userWantsGrab = false`。

---

## CaptureTimestampCoordinator

7 台相機各自 `DateTime.Now` 有 5-15ms 差。以 `(int)lineRateHz` 為 group key：
- 同 rate 第一台建時間戳，100ms 內同組共用
- 不同 rate 各自獨立

**關鍵接線**：`AllocateCameras` 必須呼叫 `cam.SetLineRateHz()`，否則 coordinator 被跳過。

---

## 相機參數即時存檔

```
acq.CameraXxx[idx] = value;           // 1. 更新記憶體
ConfigManager.SaveAcquisitionSettings; // 2. 立即持久化
_liveCameraManager?.SetXxx();          // 3. 硬體（可能失敗，不影響存檔）
```

MouseUp 只補送硬體寫入 + `SwitchToCamera`，不重複存檔。

---

## Hardware → UI 反向同步（5% hysteresis）

每 500ms 從硬體讀回值，差距超 5% 才更新 UI（`_syncingFromHw` flag 防回寫硬體）。
`GetMeasuredExposureUs()` / `GetLineRateHz()` 只在 CLProtocol 就緒後回傳非零。

---

## 已知 MIL .NET Wrapper 限制

- `M_LINE_RATE` / `M_LINE_RATE_CURRENT` / `M_GRAB_SIZE_Y` 常數**不存在**
  - Line Rate → Feature API `"AcquisitionLineRate"`
  - Grab Height → `MdigControl(M_SOURCE_SIZE_Y, height)`
- `MdigHookFunction(M_CAMERA_PRESENT)` 已移除 → Timer 500ms 輪詢 `MdigInquire(M_CAMERA_PRESENT)`
- CLProtocol 初始化期間（1–2 秒）Line Rate、Exp Meas、Cam Temp 無法讀取，屬正常

---

## MIL Display Zoom/Pan 即時查詢

```csharp
MIL.MdispInquire(displayId, MIL.M_ZOOM_FACTOR_X, ref zoomX);
MIL.MdispInquire(displayId, MIL.M_PAN_OFFSET_X, ref panX);
```

Panel 邊緣 → buffer pixel → mm：
```csharp
double rightPixel = panX + panelWidth / zoomX;
double viewRightMm = startPos + rightPixel * opsInMm;
```

**陷阱**：黑邊區域 `MouseStatusHandler` 回傳 `pixelValue = -1`。
