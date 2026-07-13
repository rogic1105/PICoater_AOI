# 多相機 free-run 取像：re-grab 起頭錯位 + 停止 cam2 多跑幾偵 — 根因與修法定稿

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

## 3. 修法（取像層，兩個一起才頭尾都齊）

### 3A. 停止改「乾淨 drain」→ 修起頭錯位（A）

`MilCamera.ApplyGrabState()` 的停止分支，從裸 `M_STOP` 改成乾淨 drain（**鏡像 `SetGrabHeight` 已驗證的 pattern**）：

```csharp
// MilCamera.cs ApplyGrabState() 停止分支
MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
    MIL.M_STOP + MIL.M_WAIT, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
try { MIL.MdigControl(_milDigitizer, MIL.M_GRAB_ABORT, MIL.M_DEFAULT); } catch { }
IsLive = false;
```

- `M_STOP + M_WAIT`：等佇列中的 grab 全部跑完才返回（**drain**，非只取消），之後沒有 FrameReady 在跑。
- `M_GRAB_ABORT`：再立即中止任何 in-flight + 佇列殘留（防「優雅停止留殘留」）。
- 效果：下次 re-grab 從**乾淨狀態 re-arm，接近第一次 grab** → 兩台較易等到同一個完整 frame、起頭齊。

### 3B. 停止改「並行」→ 修結尾多跑（B）

`LiveCameraManager.StopGrab()`：多相機停止從**逐台序列**改成**並行**：

```csharp
// 各台 M_STOP+M_WAIT 各自阻塞等自己 in-progress 幀（~1 frame）；
// 並行 → 同時阻塞、彼此只差 φ → 兩台幾乎「同一幀」停下。
System.Threading.Tasks.Parallel.ForEach(_cameras, cam => cam.SetUserGrabIntent(false));
```

- 不同 digitizer 互不干擾，可並行（每台各自 try/catch 於 ApplyGrabState）。
- 啟動 `M_START` 不阻塞（只是 arm），逐台序列即可、不必並行。

### 3C. 顯示層配套（不是主修，但要對）

- **重 grab 時 Reset 瀑布**（`WaterfallView.Reset()`，於 `StartGrab` 呼叫）：清舊圖 + 重置對齊狀態，下次幀重新 bootstrap。符合「重 grab 該清舊圖」的預期，也避免新幀接在舊網格上。
- **tick 網格錨定**（`WaterfallView`）：每幀獨立 `seq = round((tick − origin) / period)`，同一掃描各台（φ≪半週期）落同格。3A/3B 把取像層的幀邊界對齊後，顯示層就如實呈現齊頭。

---

## 4. 失敗的嘗試（記錄，避免再走）

| 嘗試 | 為何失敗 |
|------|---------|
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

> 本專案另有 B-linesync 路線（`feat`/`fix` 別的分支）嘗試用「兩階段叢集啟動 TIMER1」收斂啟動相位差，屬硬體 trigger 同步方向；**本分支（free-run）不含**，故只能用 §3 的「乾淨 drain + 並行停止」把啟停瞬間的量化誤差壓到最小。

---

## 6. 限制與後續

- §3 是把「啟停瞬間量化到不同 frame」的**機率壓到最低**（乾淨狀態 + 同時停），對 free-run 已足夠（實測頭尾皆齊）。但**free-run 本質**上仍無法 100% 保證 `M_START` 不跨 frame 邊界 —— 若未來偶發再現，codex 建議的「保證解」是 **交付前建立共同 frame epoch**：start 後讀兩台第一筆 `M_GRAB_FRAME_START` tick，差≈1 period 就丟掉早的那台一幀，兩台從同一 scan 才放行（仍是取像層修，不是顯示層補）。
- 跨板（cam1-4 板0 / cam5-7 板1）tick epoch 不同、不可直接相減；7 台上線需各板自錨 period/origin。

---

## 7. Source（來源標註）

- **MIL API 語意**：Matrox MIL `MdigProcess`（`M_START`/`M_STOP`/`M_WAIT`）、`MdigControl(M_GRAB_ABORT)` 官方行為。專案內對照：[MIL API reference](../../../.agents/skills/modify-acquisition/references/mil-api-reference.md)（MdigProcess 連續抓圖、釋放逆序、SetGrabHeight buffer 重配流程）。
- **已驗證的乾淨 drain pattern 出處**：[`MilCamera.Params.cs`](../MilGrabber.Core/MilCamera.Params.cs) `SetGrabHeight`（改尺寸前 `M_STOP+M_WAIT` + `M_GRAB_ABORT` drain，註解「M_STOP 只取消佇列、M_GRAB_ABORT 才立即中止 in-flight+佇列」）。相關坑文件：[`grab-height-param-stall.md`](grab-height-param-stall.md)。
- **frame-start tick 量測機制**：[`MilCamera.PhaseLog.cs`](../MilGrabber.Core/MilCamera.PhaseLog.cs)（Data Latch `M_GRAB_FRAME_START` + `M_TIME_STAMP`；參考 Matrox BoardSpecific/DataLatch 範例）。
- **實證資料**：現場 trace log `{AniloxRoot}\Logs\trace-*.log` 的 `[Waterfall]` 行（每條 band 記各台 frame-start tick；fresh grab 兩台差 φ≈6 萬、re-grab cam2 晚 1 period≈1.25 億、停止時 cam2 多出 tick 在 cam1 之後）。
- **第二模型 review**：以 Codex（OpenAI CLI，read-only 讀本 repo 實際 code）交叉驗證根因與修法優先序（乾淨 drain > 並行/back-to-back > 共同 frame epoch；排除 `M_GRAB_ABORT` 快停、`MdigGrabContinuous`、CLProtocol re-enable）。
- **程式碼落點**：取像層 [`MilCamera.cs`](../MilGrabber.Core/MilCamera.cs) `ApplyGrabState`（乾淨 drain）+ [`LiveCameraManager.cs`](../../../src/dotnet/AniloxRoll.Monitor/UI/Managers/LiveCameraManager.cs) `StopGrab`（並行）/`StartGrab`（Reset 瀑布）；顯示層 [`WaterfallView.cs`](../../../src/dotnet/AniloxRoll.Monitor/UI/Widgets/WaterfallView.cs)（tick 網格錨定）。
