# modify-acquisition

修改 MIL 取像、相機控制、CLProtocol、Telemetry、PLC 連動相關程式碼。

## 使用時機

修改 AniloxCamera、LiveCameraManager、CameraSystemManager、PlcGrabController 或 MIL API 呼叫時。

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

## CLProtocol 延遲啟動（重要）

- `StartCLProtocolAsync()` 在第一次 `MdigProcess(M_START)` 之後才呼叫
- 耗時 2-5 秒，若在 Initialize 期間啟動會與 MbufAlloc/MdispAlloc 競爭 MIL 內部鎖
- **Quad 卡 DevNum>=2 必須明確列舉 Device ID**，`"M_DEFAULT"` 無效
- `_clProtocolInitLock`（static）序列化同卡多 digitizer
- `_clProtocolInitStarted`（volatile bool）防重複觸發
- 逾時保護：`Task.WhenAny(initTask, Task.Delay(10s))`，不取消（MIL 不支援安全取消）

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

- PlcState FSM：Idle → Running → Stopping → Faulted / CommLost
- IO 快照 5 LED：DI0(GRAB), DI1(STOP), DI2(MURA_ACK), DO0(READY), DO1(MURA)
- DO_MURA_DETECTED 不中斷取像（MIL callback 仍持續運作）
- ET-7044 模組 IP 可設，PollTick ~500ms

## 步驟

1. 讀取要修改的 MIL 相關方法
2. 確認資源分配/釋放順序
3. 修改 + build 驗證（Release|x64）
