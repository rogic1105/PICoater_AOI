# modify-acquisition

修改 MIL 取像、相機控制、CLProtocol、Telemetry、PLC 連動相關程式碼。

## 使用時機

修改 AniloxCamera、LiveCameraManager、CameraSystemManager、IoGrabController 或 MIL API 呼叫時。

## 架構（2026-05-26 重構後）

MIL 取像/顯示資源已抽到 **`sdk/MIL/MilGrabber.Core/MilCamera.cs`**（純 MIL 封裝 library，一台相機=一個 `MilCamera`）。本文以下的 **MIL 初始化順序 / CLProtocol / 曝光線掃 / 資源釋放** 細節**現在都在 `MilCamera`**，不在 `AniloxCamera`。

- **改 MIL 取像/顯示/參數/CLProtocol/telemetry → 改 `MilCamera`**。注意：`SetExposureUs` 先存設定值再 early return（Initialize 前設也記得）；`AppliedLineRateHz` 暴露「設定值」供時間戳協調用（`GetLineRateHz` 在 CLProtocol 未就緒時回 0）
- **`AniloxCamera` = composition**：持 `MilCamera _mil` + 訂閱 `_mil.FrameReady`，在 `OnMilFrameReady` 跑檢測(tanuki_pipeline_api)/存檔/合圖/曲線（非 MIL）；自己不再有 MIL 資源/hook
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

`同值守門(高度未變+buffer已配→直接 return) → M_STOP+M_WAIT → MdigControl(M_GRAB_ABORT) → Free buffers+pool → M_SOURCE_SIZE_Y → Inquire → Realloc → MdispSelectWindow → settle → M_START`
- **同值守門必須有**（2026-06-24）：套設定時會對每台呼 `SetGrabHeight(同值)`；不擋會做多餘 free+realloc 撞背景 CLProtocol enable → **CAM1 stall**（見下）。
- 舊尺寸 buffer ≠ 新尺寸 → MIL 崩潰
- Rollback：失敗時 FreeGrabBuffers → AllocateAndBind(oldHeight)
- **高度硬上限 `AcquisitionDefaults.MaxGrabHeightPx=12000`**（固定值，**不分台數**）：12062 是「grab 中把單台高度往上拉」的真硬體上限（on-board 兼 PCIe latency 緩衝），cap 12000 避開。換相機(寬)/grabber 須重新實測（grab 中往上拉找 stall 邊界）。

### ⚠ 改高度會讓相機永久 stall — 根因與修法（2026-06-18 實機確認）

**症狀**：改高度後某台（甚至兩台）相機 `M_PROCESS_FRAME_COUNT` 凍在 0、fps=0，永不恢復；停/開（含停止抓取→開始抓取）救不回，**只有重開程式**。穩態不改參數＝0 stall。鐵證：「停→改高度→開」重複多次會 stall；「停→改高度多次→開一次」不會。

**根因**（Grok 第二意見 + Matrox 官方文件確認）：
1. `MdigProcess(M_STOP, M_DEFAULT)` 是**優雅停止**、只取消佇列、**不 hard-drain** CL 接收器/DMA → 殘留狀態跨「重複 free+realloc+re-arm」累積成永久壞狀態。
2. **`M_SOURCE_SIZE_Y` 純 digitizer 端**（幾條線切一幀）。**絕不可寫相機 GenICam `Height` feature 去「同步」**（試過→相機輸出尺寸錯亂、兩台 stall + FPS 算錯）。line-scan 相機 Height ≠ grab 高度。
3. 沒有 in-app reset digitizer 的 API（無 `MdigReset`）→ 證實「只有重開程式（=MdigFree/MdigAlloc）能救」。

**修法（已實作，實機驗證 stall 消失）**：改尺寸前加
- `MdigProcess(…, M_STOP + M_WAIT, …)`：等佇列 grab 全跑完才返回（drain，非只取消）。
- `MdigControl(dig, M_GRAB_ABORT, M_DEFAULT)`：立即中止 in-flight + 佇列（eV-CL 支援；guard try/catch 防 wrapper 不支援）。

**Matrox 官方文件路徑（查證來源，未來再查從這裡）**：
- `C:\Program Files\Matrox Imaging\MIL\DOC\mil_help\content\Reference\dig\MdigProcess.htm`（M_STOP 預設取消佇列；M_STOP+M_WAIT 等佇列跑完）
- `…\Reference\dig\MdigControl.htm`（**M_GRAB_ABORT**＝立即中止 in-flight+佇列；M_COMMAND_QUEUE_MODE M_QUEUED/M_IMMEDIATE；無 reset）
- `…\Reference\dig\MdigHalt.htm`（MdigHalt 是 MdigGrabContinuous 的夥伴，**非** MdigProcess）
- `…\UserGuide\grabbing\Grabbing_and_processing.htm`（尺寸會變的官方做法＝MdigProcess bufarray=M_NULL 自動配，或 max-size buffer 配一次）
- `…\UserGuide\grabbing\Linescan_cameras.htm`、`…\Readme\milRadienteVCL\milRadienteVCL.htm`（eV-CL 無 SOURCE_SIZE 改尺寸 stall 的 release note）
- `Mil.h:3693`（`M_GRAB_ABORT = 6643L`）

**階段 2（max-buffer / auto-allocate）已試 → 棄用**：max-buffer（一次配 max、改高度只改 `M_SOURCE_SIZE_Y`）7 台 host ~3.6GB 逼爆非分頁池、且非根因；auto-allocate（MdigProcess bufarray=M_NULL）官方確認對 on-board 占用沒幫助。兩者都不採用。`MilCamera.UseMaxHeightBuffers` flag/scaffold 留著當紀錄但預設 false。

### ⚠⚠ 改高度 stall 的「主因」是啟動競態，非記憶體/高度（2026-06-24 dropdiag 定案，推翻 6/18~6/23 部分推論）

**鐵證**：高度 12000 時 dropdiag 顯示 **CAM2 正常 grab（FPS 0.83）、CAM1 frameCount 卡 0**（兩台都接都 CLProtocol enabled）→ 12000 不是硬限，是 CAM1 單獨 stall。

**真根因**：套設定時對每台呼 `SetGrabHeight(同值)` → **多餘 free+realloc（UI 執行緒）撞上 CAM1 CLProtocol enable（背景執行緒）** → MIL 並發 → CAM1 stall。trace log 抓到 realloc 插在 CAM1「using device ID」與「enabled successfully」之間。

**修法**：`SetGrabHeight` 開頭同值守門（見上）→ 同值不 realloc → 不撞 CLProtocol。**+ 改高度熱路徑禁止 MsysInquire/MdigInquire**（含診斷 `GetMemoryFreeMB()`）：會插進相機 MIL 序列 → cam1 stall（本次診斷一度自污染中招）。板載記憶體看背景寫的 resource-monitor CSV。

**離線判讀工具**：`Program.cs` 已加**檔案 trace listener** → `D:\Anilox\Logs\trace-*.log`（AutoFlush，含 `[HtRealloc]`）。配 dropdiag（每台 frame 數）就能分辨「哪台 stall、stall 在配 buffer 還是 re-arm」。**難重現 stall 一律先看這兩個檔再下結論**（這次差點被「板載上限」帶歪一整天）。

> 註：12062 硬體上限 + 「板載每 path temporary buffer 在 MdigAlloc 預留、占用顯示為 4 台總和」仍是真的（官方 `Grabbing_large_images.htm` / `Minimum_latency…`），但**不是**那些 stall 的原因。完整脈絡見 `docs/dev/grabheight-max-buffer-stage2.md`「★ 2026-06-24 ②」。

### 改參數掉偵診斷 log（Logs\）
- `phaselog-yyyyMMdd.csv`：每幀硬體 frame-start tick（`MilCamera.PhaseLog.cs` Data Latch）→ 真實相位/掉幀位置。
- `dropdiag-yyyyMMdd_HHmmss.csv`：每 500ms 背景記 frames/procMissed/grabMissed（`LiveTelemetryPresenter.DropDiagLogPath`）→ 分層（host vs 硬體）。
- `paramchange-yyyyMMdd_HHmmss.csv`：每次改參數 time,scope,cam,param,value（`AniloxRollForm.ParamChangeLogPath`）→ 對齊 `_ticks.csv` 看掉偵 vs 改參數。
- 結論：**穩態 0 掉偵；掉偵 100% 來自改參數的重啟空檔**。

## Grab 中改參數（協調套用 + 參數鎖 + stall）

- **`LiveCameraManager.ApplyParamCoordinated(camId, write)`**＝只停/寫/開**被改的那一台**（曝光/線掃/高度單滑桿走此）。相位已用 phaselog 證明不重要（free-run 2-3 條線）→ **不再全部相機一起停/開**（會連累沒被改的 cam2 反覆 stop/start → stall）。All 滑桿才用無參數版 `ApplyParamCoordinated(write)`（全停全開）。
- **絕不在套用後同步 `Thread.Sleep` 等出幀** → 會凍 UI（Windows 變灰 Not Responding）→ 拉滑桿被排隊、解凍後 replay「跳到空拉位置」＝暴力漏洞。stop→write→start 本身已被 MouseUp handler 序列化。
- **參數鎖**（`AniloxRollForm.SettingsTabs.cs` `ApplyCamParam`/`SetParamControlsLocked`）：套用期間 `Enabled=false` 所有參數控制項（拒輸入不排隊）→ 非阻塞 `Forms.Timer` 輪詢 `LiveCameraManager.AllAdvancedSince`（恢復出幀）**且**至少鎖滿 **2 個完整幀週期**（`GetMaxFramePeriodMs`＝高度/線掃率，實測 2 週期才不 stall）或 5s 逾時才解鎖 → 逼「改完馬上又改」慢下來。
- **改參數窗口暫停存檔**（`LiveCameraManager.SetCaptureSuppressed` → `AniloxCamera.SuppressCapture` → `TrySaveCapture` 早退）：套用時全相機暫停存檔、等全部恢復同步（解鎖時）才恢復 → 存出的序列不含重啟空檔、各台齊全（不影響 grab/檢測/顯示）。
- **stall 偵測**（`LiveCameraManager.Telemetry.cs` status timer 500ms）：IsLive 但 `CurrentFps < 0.05` 持續 2s ＝ stall（縮圖紅「STALL」）。**FPS 0.1 是合法慢速（一幀 10s）不算**，只認真正的 0。stall＝硬體層 CL 失鎖，**停/開（含停止抓取→開始抓取）救不回、只有重開程式** → 不做無效自動 thrash（會 stall 的最大宗＝改線掃；高度也會）。深度救援（MdigFree+MdigAlloc+CLProtocol）暫不做，先靠 prevention。

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
