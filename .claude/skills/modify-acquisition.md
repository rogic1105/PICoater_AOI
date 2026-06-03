# modify-acquisition

修改 MIL 取像、相機控制、CLProtocol、Telemetry、PLC 連動相關程式碼。

## 使用時機

修改 AniloxCamera、LiveCameraManager、CameraSystemManager、IoGrabController 或 MIL API 呼叫時。

## 架構（2026-05-26 重構後）

MIL 取像/顯示資源已抽到 **`sdk/MIL/MilGrabber.Core/MilCamera.cs`**（純 MIL 封裝 library，一台相機=一個 `MilCamera`）。本文以下的 **MIL 初始化順序 / CLProtocol / 曝光線掃 / 資源釋放** 細節**現在都在 `MilCamera`**，不在 `AniloxCamera`。

- **改 MIL 取像/顯示/參數/CLProtocol/telemetry → 改 `MilCamera`**。注意：`SetExposureUs` 先存設定值再 early return（Initialize 前設也記得）；`AppliedLineRateHz` 暴露「設定值」供時間戳協調用（`GetLineRateHz` 在 CLProtocol 未就緒時回 0）
- **`AniloxCamera` = composition**：持 `MilCamera _mil` + 訂閱 `_mil.FrameReady`，在 `OnMilFrameReady` 跑檢測(picoater_api)/存檔/合圖/曲線（非 MIL）；自己不再有 MIL 資源/hook
- 換非 MIL grabber = 整包換 `sdk/MIL`，AniloxCamera 的檢測邏輯不動

## 關鍵檔案

→ 見 `CLAUDE.md` §關鍵檔案速查（subset：`Acquisition/*` + `UI/Managers/LiveCameraManager` + `Services/IoGrabController` + `Services/IoState` + `UI/Presenters/LiveTelemetryPresenter` + `Settings/Stores/AcquisitionSettingsStore`）。
→ MIL .NET API 完整參考 `docs/dev/MIL_API_Reference.md`。
→ IO FSM 視覺化 `docs/user-manual/io_diagrams.html`。

## MIL 初始化順序（不可打亂）

```
MdigAlloc → MdigControl(M_SOURCE_SIZE_Y) → MdispAlloc × 2 → MdigInquire(SIZE)
→ CPU+CUDA buffer → MbufAlloc2d × 4 → MdispSelectWindow → MdispHookFunction
→ SetExposureUs
```
- `M_SOURCE_SIZE_Y` 必須在 `MdigInquire` 之前（否則 buffer 大小錯誤）
- CLProtocol **不在此處啟動**

## CLProtocol 啟用（重要）— 分配後、grab 前，只對在線相機

- `BeginCLProtocolInit()`（public，原 `StartCLProtocolAsync`）在**相機分配完成後、第一次 grab 之前**背景啟用（不在 grab 期間 enable + 重套線掃 → 否則首抓掉幀，cam1 最明顯）。
- 觸發點：`LiveCameraManager.AllocateCameras` 迴圈後 `foreach cam: if (cam.CheckPresence()) cam.BeginCLProtocolInit();`
  —— **只對在線相機**。對斷線相機 enable 會卡住 MIL 內部鎖（全 0/7 時 7 台全卡 → 逾時翻 true 後 timer 輪詢搶鎖 → UI 凍死）。斷線相機 `_clProtocolInitStarted=false` → `IsHwParamsStable=true`（不擋就緒判定），之後連上走 legacy 參數路徑。
- 耗時 2-5 秒/台；完成前 `IsHwParamsStable=false`。上層就緒判定：`LiveCameraManager.AreCamerasHwReady`（全相機 `IsHwParamsStable`）+ 一次性 `OnHwReady` 事件 → 解鎖「開始抓取」鈕、建立全域合圖。
- **就緒前 UI 不可碰 MIL**：`CameraStatusTimer_Tick` 在 `!AreCamerasHwReady` 時跳過 `CheckPresence` 輪詢；全域合圖 `EnableGlobalMerge` 延後到 `OnCamerasHwReady`（都避免與背景 CLProtocol 搶 MIL 鎖造成凍結）。
- **Quad 卡 DevNum>=2 必須明確列舉 Device ID**，`"M_DEFAULT"` 無效
- `_clProtocolInitLock`（static）序列化同卡多 digitizer；`_clProtocolInitStarted`（volatile）防重複；`IsHwParamsStable => !_clProtocolInitStarted || _clProtocolInitDone`
- 逾時保護：`Task.WhenAny(initTask, Task.Delay(10s))`，不取消（MIL 不支援安全取消）—— 只設 `_clProtocolInitDone` 旗標，故**斷線相機不可啟動**（MIL 呼叫實際仍會卡，逾時無法中止）
- **光源 `InitLightController` 的 `AutoDetect`（掃 COM 阻塞數秒）必須背景 `Task.Run`**（不可在 UI 執行緒）；四硬體（相機/IO/光源/儲存）為**平行**初始化，非依序

## 曝光/線掃設定

| 條件 | Set API | 單位 |
|------|---------|------|
| CLProtocol 啟用 | `MdigControlFeature("ExposureTime")` | μs |
| CLProtocol 未啟用 | `MdigControl(M_EXPOSURE_TIME, μs×1000)` | ns→μs |

- 曝光上限 = `floor(900000 / lineRateHz)`
- `_appliedExposureUs` 永遠記錄最後設定值，不依賴硬體回讀

## MIL 資源釋放順序

```
_isReleased = true → MdigProcess(M_STOP)
→ MdispHookFunction(UNHOOK) → MdispSelectWindow(M_NULL)
→ MbufFree × 4 → NativeBufferPool.Dispose → AoiService.Dispose
→ MdispFree × 2 → MdigFree → GCHandle.Free
```
- `IsReleasing = true` 必須在 `Timer.Stop()` 之前
- Timer Tick 快照相機清單 `_cameras.ToArray()`，mid-loop 檢查 `IsReleasing`

## SetGrabHeight（不可省略步驟）

`M_STOP → Free buffers+pool → M_SOURCE_SIZE_Y → Inquire → Realloc → MdispSelectWindow → M_START`
- 舊尺寸 buffer ≠ 新尺寸 → MIL 崩潰
- Rollback：失敗時 FreeGrabBuffers → AllocateAndBind(oldHeight)

## 已知 MIL .NET Wrapper 限制

- `M_LINE_RATE` / `M_GRAB_SIZE_Y` 常數不存在 → 用 Feature API 或 `M_SOURCE_SIZE_Y`
- `MdigHookFunction(M_CAMERA_PRESENT)` 已移除 → Timer 500ms 輪詢
- CLProtocol 初始化期間 Telemetry 查詢回傳 0，屬正常

## 即時全域合圖（Live Global Merge）

- `EnableGlobalMerge(opsUm, startPosMm)`：在第一台相機的 System 上 `MbufAlloc2d` 合併 buffer，計算各相機 X 偏移
- **Overlap 分割**：相鄰相機重疊區域取中點分界（與 `GrabImageStitcher.MergeHorizontal` 一致），每台相機存 `_mergedSrcClipLeft` / `_mergedSrcClipWidth`
- 每幀 `ProcessingFunction` callback：`MbufChild2d` 建立裁切子 buffer → `MbufCopyClip(childBuf, mergedBuf, dstX, 0)` → `MbufFree(childBuf)`
- `MdispSelectWindow(mergedDisplay, mergedBuffer, mainPanel.Handle)` 綁定顯示
- **Zoom/Pan**：`WheelZoomFilter` 攔截滾輪，`IsGlobalMergeActive` 時直接操作 `_mergedDisplay`（`MdispZoom` / `MdispPan`）
- **滑鼠座標**：`MdispHookFunction(M_MOUSE_MOVE)` 掛在 `_mergedDisplay`，buffer 座標 → mm 座標（`_mergedMinStartMm + x * _mergedRefOpsMm`）
- **Overview 聯動**：`TryGetMergedViewRange()` 從 `_mergedDisplay` 的 zoom/pan 計算 X 視野 mm 範圍，供 `LiveViewRangeProvider` 使用
- `DisableGlobalMerge()`：先清除各相機 `_mergedTargetBuffer = M_NULL`，unhook 滑鼠，再 `MbufFree` + `MdispFree`
- `FreeCameras()` 會先呼叫 `DisableGlobalMerge()`
- CAM1-4 在 System0、CAM5-7 在 System1，合併 buffer 在 System0，MbufCopyClip 跨 System 由 MIL 處理 DMA
- Global 模式下 `SwitchMainDisplay` 只切換高亮縮圖，不切換主畫面

## Hardware → UI 反向同步

- 每 500ms 讀硬體值，差距 > 5% 才更新 UI
- `_syncingFromHw` flag 防止 ValueChanged 再回寫硬體
- 拖曳中不同步（`_dragging` HashSet）

## CaptureTimestampCoordinator

- `(int)lineRateHz` 為 group key，同 rate 100ms 內共用時間戳
- `AllocateCameras` 必須呼叫 `cam.SetLineRateHz()`，否則 coordinator 被跳過

## PLC 連動

- IoState FSM：Idle → Running → Stopping → Faulted / CommLost
- IO 快照 5 LED：DI0(GRAB), DI1(STOP), DI2(MURA_ACK), DO0(READY), DO1(MURA)
- DO_MURA_DETECTED 不中斷取像（MIL callback 仍持續運作）
- ET-7044 模組 IP 可設，PollTick ~500ms

## 步驟

1. 讀取要修改的 MIL 相關方法
2. 確認資源分配/釋放順序
3. 修改 + build 驗證（Release|x64）
