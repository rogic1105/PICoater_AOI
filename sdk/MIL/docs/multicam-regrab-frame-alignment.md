# 多相機 free-run 取像：hot standby、啟動同步與全域收幀 gate

> 適用：Matrox Radient eV-CL grabber + Camera Link **line-scan** 相機，多台同板、**free-run（TriggerMode=Off、無外部/硬體 trigger/encoder）**。
> 症狀出現在「監控瀑布圖（WaterfallView，逐幀往下接的即時長圖）」上最明顯，因為它把「每台第 N 幀」並排顯示 → 兩台差一幀立刻看得到。但**根因在取像層（MIL grab 啟停），不是顯示層**。

---

## 1. 症狀（兩個，方向相反）

| # | 症狀 | 何時出現 |
|---|------|---------|
| A | **起頭錯位**：re-grab（不關程式、停止再開始抓取）後，瀑布最頂端 cam1 單獨一條、cam2 晚一格才出現（看起來「少一偵」）。**第一次 grab 不會**。 | 第二次以後的 grab 啟動 |
| B | **結尾多跑**：停止抓取後，cam2 比 cam1 **多吐出幾偵**（瀑布末端 cam2 單獨幾條）。 | 每次停止抓取 |

兩個症狀都是「兩台 free-run 相機的 frame 邊界沒對齊」在 grab **啟動/停止的瞬間**被放大。

---

## 2. 根因（free-run + 無 trigger 的本質）

**核心事實：free-run line-scan 相機的 frame/line phase 由相機自己的內部時脈決定，grabber 的 `MdigProcess(M_START/M_STOP)` 不能定義相機的 frame phase，只能決定「從哪一個完整 frame 開始/停止收」。** 真正要硬體同步兩台的 frame 邊界，需要**外部 trigger / encoder / 共用 line 或 frame reset**（本機型明確是 free-run + trigger off，所以做不到相機層同步）。

量測方式：每幀的 `M_GRAB_FRAME_START` 由板載硬體 latch 一個 `M_TIME_STAMP`（板載時鐘 ticks，零 callback jitter），見 [`MilCamera.PhaseLog.cs`](../MilGrabber.Core/MilCamera.PhaseLog.cs)。同板兩台 ticks 可直接相減＝相位差 φ。實測 φ ≈ 6 萬 ticks（< 0.5ms）≪ 一個 frame period（實測 ~1.25 億 ticks，1fps 測試）。**φ 很小 → 兩台本來幾乎同步**，問題只在 grab 啟停瞬間的「量化到哪一個完整 frame」。

### 2A. 起頭錯位根因 — `M_START` 接「下一個完整 frame」

- `MdigProcess(M_START)` 只是讓 digitizer 開始接收**後續的完整 frame**。若 `M_START` 發生在某 frame 已經開始掃描之後，grabber **不會把半截 frame 交成一張** → 第一個 callback 等**下一個**完整 frame。
- re-grab 逐台 `M_START`：cam1 剛好趕在 frame N 開始前 arm → 第一幀＝N；cam2 晚一點、落在 frame N 已開始 → 第一幀＝**N+1** → 差一格。之後兩台仍只差 φ，但序號已差一格。
- **為何第一次 grab 對齊、第二次不齊**：第一次是「乾淨狀態」起手 —— digitizer/buffer/latch 剛建立、CLProtocol 剛 enable、沒有前一輪 in-flight/queue 殘留 → 兩台剛好都等到同一個完整 frame。第二次是「在連續 free-run 流中重新插入 grabber」，而舊的停止**沒有乾淨 drain**（見 4A），狀態不乾淨 → 兩台 re-arm 跨 frame 邊界時序不一。

### 2B. 結尾多跑根因 — `M_STOP` 阻塞 ~1 frame × 逐台序列

- `MdigProcess(M_STOP)` 是**優雅停止**：會**阻塞等「目前 in-progress 那一幀 + 其 hook」跑完**才返回（最多 ~1 個 frame 時間）。
- 主程式**逐台序列**呼叫停止：`foreach cam → M_STOP`。cam1 的 `M_STOP` 阻塞那 ~1 frame 期間，**cam2 還在 free-run 繼續收幀**（它的 `M_STOP` 還沒輪到）→ 等輪到 cam2 時它已多收幾偵。
- log 證實這些是**新產生的幀**（tick 在 cam1 最後一幀之後、逐 period 遞增），不是舊 buffer drain。

> ❌ **`M_GRAB_ABORT` 不能拿來「快速停」**：它只中止「當前這一張 grab」，`MdigProcess` 連續迴圈會**自動 re-arm** 繼續收下一張 → 相機沒停。（曾誤用它做 pass-1 快停，無效。）

---

## 3. 現行修法（待機持續、開始時實測同步、產品收幀一次切換）

### 3A. ReadyIdle 保持 hot standby

CLProtocol 與 processing buffers 就緒後，產品對在線相機啟用 `KeepAcquiringWhenIdle`。每台相機真的從
raw callback 觀測到第一個 frame-start tick 後才算 Ready，不用固定延遲猜測不同電腦要等多久。

這裡的「CLProtocol 就緒」必須是初始化工作及曝光／線掃參數寫入均已實際返回。10 秒 timeout
只能留下診斷，不能把仍在執行的工作視為完成；若其中一台在 `MdigProcess` 已運行後才重寫
線掃率，該台的實體 frame phase 會被改變，全域產品 gate 無法補救。

```csharp
cam.EnableHotStandby();
if (cam.IsAcquisitionWarm) { /* camera is ReadyIdle */ }
```

- raw callback 在 standby 只更新 readiness／tick；不進 GPU、顯示、CSV 或存檔。
- idle frame 不寫 phase log，避免程式整天待機造成診斷檔無限增長。
- SDK samples 預設不啟用此功能，保留原本的 Start/Stop 語意。

### 3B. Start 先做實體同步，Stop 只切產品 gate

hot standby 能避免 Stop 時逐台阻塞，但不能證明多台 free-run 相機仍在相同 frame phase。每次
Start 都先在產品 gate 關閉時執行 `SynchronizeAcquisitionAsync`：

1. 平行 `PauseAcquisition`，完整 drain 全部在線相機。
2. 在停止狀態重套各台**現行** `AcquisitionLineRate`。實機證據顯示，只有
   `M_STOP/M_START` 不一定改變相機自由運轉相位；重寫 Line Rate 才會重建 timing。
3. 從同一 worker back-to-back `ResumeAcquisition`。
4. 等各台第一個新 raw callback，讀 Data Latch frame-start tick。
5. 同板相機 spread ≤ 5ms 才成功；超限最多重試 3 次，仍超限就保持 gate 關閉。

同步成功後，`StartGrabAsync(deferCaptureGate:true)` 完成顯示 reset 與各相機產品意圖設定；Form
再建立新的 GrabId、capture plan 與 duration guard，最後呼 `OpenCaptureGate()` 做一次全域 gate
寫入。這個順序避免 standby 的立即首幀沿用上一輪資料 owner。`StopGrab` 的第一個動作則關閉同一個
gate，再清各相機產品意圖：

```csharp
_captureGateOpen = true;   // Start: all callbacks gain product acceptance together
_captureGateOpen = false;  // Stop: all callbacks lose product acceptance together
```

- `OnMilFrameReady` 必須同時看到 `UserWantsGrab && CaptureGateOpen` 才可進產品流程。
- Stop 不再逐台 `M_STOP`，所以沒有「停 cam1 時 cam2 又多跑幾幀」。
- Start 不使用固定等待；是否可開始由實際首幀 tick 決定，所以不同電腦速度只影響等待多久，
  不影響成功條件。

### 3C. 實體 Pause 的合法入口

Start 同步、停止狀態的高度重配置或程式釋放 MIL 資源時，使用已驗證的：

`PauseAcquisition → M_STOP+M_WAIT → M_GRAB_ABORT → 修改／釋放`

Grab 中曝光只在背景寫入 integration time，不改 Line Rate／幀高，因此沿用目前 acquisition
generation，不走實體 pause/resume，也不關產品 gate。線掃速度與擷取高度只能在停止 Grab 後修改，
避免運轉中改變 frame timing 或重配 buffer。Start 同步仍以 raw callback 與硬體 tick 實測，不用 sleep。

### 3D. 顯示層配套

- **重 grab 時 Reset 瀑布**（`WaterfallView.Reset()`，於 `StartGrab` 呼叫）：清舊圖 + 重置對齊狀態，下次幀重新 bootstrap。符合「重 grab 該清舊圖」的預期，也避免新幀接在舊網格上。
- **tick 網格錨定**（`WaterfallView`）：每幀獨立 `seq = round((tick − origin) / period)`，同一掃描各台（φ≪半週期）落同格。3A/3B 把取像層的幀邊界對齊後，顯示層就如實呈現齊頭。

---

## 4. 失敗的嘗試（記錄，避免再走）

| 嘗試 | 為何失敗 |
|------|---------|
| **每次 Stop 做並行 M_STOP+M_WAIT，再於 Start 逐台 M_START** | 比序列停止好，但 free-run 的 re-arm 仍可能跨 frame boundary，且低線掃時停止會有秒級等待；現已由 hot standby + gate 取代。 |
| **顯示層「啟動等湊齊預期相機數才畫第一條」** | 加了明顯 lag、且沒解根本（取像層仍錯位）。使用者要求退回。 |
| **`M_GRAB_ABORT` 當 pass-1 快停** | 只中止當前一張，`MdigProcess` loop 自動 re-arm → 相機沒停，cam2 照樣多跑。 |
| **per-camera 序號累加（第 N 幀對第 N 幀，用 round(delta/period) 偵掉幀）** | 重建/re-grab 時某台第一幀是舊幀 → 整條序號歪。改純 tick 網格錨定才穩。 |
| **指望 CLProtocol re-enable 解 phase** | 會擾動相機 timing，是副作用不是可靠同步機制。 |
| **`MdigGrabContinuous` / `M_SYNCHRONOUS`** | 改的是 API 等待語意，不會把兩台 free-run 的 frame 邊界綁在一起。 |

---

## 5. 官方 MIL 實作 / API 語意（來源見 §7）

- **連續抓圖啟停**＝`MdigProcess(dig, bufs, n, M_START / M_STOP, ...)`（官方標準雙緩衝 pattern，見 in-repo MIL 參考）。
- **`M_STOP`（不帶 `M_WAIT`）**＝優雅停止：等「目前 in-progress 的 grab + 關聯處理」完成才停（≈ 阻塞 1 個 frame）；**不保證**把整個佇列 drain。
- **`M_STOP + M_WAIT`**＝等佇列中的 grab **全部**跑完才返回（完整 drain）。
- **`M_GRAB_ABORT`（`MdigControl`）**＝立即中止 in-flight + 佇列；但**對 `MdigProcess` 連續模式不等於「停止」**，loop 會 re-arm。要停 loop 只能 `M_STOP`。
- **free-run + TriggerMode Off**：相機 frame phase 不受 grabber 控制；grabber 只能選「接哪一個完整 frame」。真正硬體同步兩台 frame 邊界需 HW trigger / encoder / 共用 line-frame reset。

> 本專案另有 B-linesync 路線嘗試用硬體 trigger 同步 frame phase；本分支維持 free-run，
> 因此 §3 解決的是產品收幀邊界與反覆 re-arm 問題，不宣稱改變相機的硬體 phase。

---

## 6. 限制與後續

- 全域 gate 保證「產品何時開始／停止接受幀」只有一個決策點；Start 同步器再保證每個同板群組
  放行前的首幀 spread ≤ 5ms。
- 這仍不是硬體 trigger。重套 Line Rate 是本機型實測有效的 free-run timing reset；更換相機、
  grabber 或 firmware 後必須重跑 phase DVT。
- 不同板（cam1-4 板0 / cam5-7 板1）的 tick epoch 不同、不可直接相減；目前只能各板內驗證。
- 若產品要求七台 callback 組成不可分割 frame set，下一階段是硬體 trigger，或
  **owned-buffer frame-set barrier**：依硬體 tick 對齊完整 frame set，整組到齊才發布；不能在
  MIL callback 裡持有原始 buffer 等其他相機。

---

## 7. Source（來源標註）

- **MIL API 語意**：Matrox MIL `MdigProcess`（`M_START`/`M_STOP`/`M_WAIT`）、`MdigControl(M_GRAB_ABORT)` 官方行為。專案內對照：[MIL API reference](../../../.agents/skills/modify-acquisition/references/mil-api-reference.md)（MdigProcess 連續抓圖、釋放逆序、SetGrabHeight buffer 重配流程）。
- **已驗證的乾淨 drain pattern 出處**：[`MilCamera.Params.cs`](../MilGrabber.Core/MilCamera.Params.cs) `SetGrabHeight`（改尺寸前 `M_STOP+M_WAIT` + `M_GRAB_ABORT` drain，註解「M_STOP 只取消佇列、M_GRAB_ABORT 才立即中止 in-flight+佇列」）。相關坑文件：[`grab-height-param-stall.md`](grab-height-param-stall.md)。
- **frame-start tick 量測機制**：[`MilCamera.PhaseLog.cs`](../MilGrabber.Core/MilCamera.PhaseLog.cs)（Data Latch `M_GRAB_FRAME_START` + `M_TIME_STAMP`；參考 Matrox BoardSpecific/DataLatch 範例）。
- **實證資料**：現場 trace log `{AniloxRoot}\Logs\trace-*.log` 的 `[Waterfall]` 行（每條 band 記各台 frame-start tick；fresh grab 兩台差 φ≈6 萬、re-grab cam2 晚 1 period≈1.25 億、停止時 cam2 多出 tick 在 cam1 之後）。
- **第二模型 review**：以 Codex（OpenAI CLI，read-only 讀本 repo 實際 code）交叉驗證根因與修法優先序（乾淨 drain > 並行/back-to-back > 共同 frame epoch；排除 `M_GRAB_ABORT` 快停、`MdigGrabContinuous`、CLProtocol re-enable）。
- **程式碼落點**：取像層 [`MilCamera.cs`](../MilGrabber.Core/MilCamera.cs)
  `EnableHotStandby/PauseAcquisition`；產品閘門
  [`LiveCameraManager.cs`](../../../src/dotnet/AniloxRoll.Monitor/UI/Managers/LiveCameraManager.cs)
  `StartGrab/StopGrab`；顯示層
  [`WaterfallView.cs`](../../../src/dotnet/AniloxRoll.Monitor/UI/Widgets/WaterfallView.cs)（tick 網格錨定）。
