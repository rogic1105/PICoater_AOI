# 階段二：改高度 no-realloc（max-buffer / auto-allocate）研究與 scaffold

> 分支 `feat/grabheight-max-buffer`。階段一（`M_GRAB_ABORT` drain，已 merge main）已修「改高度重啟空檔 stall」；
> 此分支研究 grab 記憶體 + max-buffer。**下方「★ 2026-06-23 釐清」是最終真相，前面舊章節有被推翻處（保留當推理過程）。**

## ★ 2026-06-23 釐清（MILConfig 截圖 + 實測 + agent 文件，這是最終結論）

**記憶體配置真相（推翻前面多個假設）：**
- grab buffer 配在 **host 非分頁記憶體池**（MILConfig「Non-Paged Memory」），**不是板載**。本機 MILConfig 實際值：
  Requested/Allocated = **4096MB（4GB）**；**Chunk Size = 4096MB（Automatic，整池一塊）** → **「單 buffer ≤ 256MB chunk」是文件舉例值、本機不適用**（chunk=4GB，單 buffer 幾乎無上限）。
  設定存 registry `HKLM\SYSTEM\CurrentControlSet\Services\ailmemmanager\Parameters`（DmaSize/MaxChunk=4096）；MILConfig.exe 可改；RAM 127GB 有餘裕。
- 板載 **1GB DDR3 硬體固定**（x8/DF/QB 板型；`M_MEMORY_SIZE=1024` 正確、非 bug）。不可軟體調。
- **DCF 是純相機格式（Intellicam binary），不含記憶體配置。**

**真正的 grab 高度瓶頸 = 板載 1GB（實測），不是 host 4GB**：
- 實測：9000×4台 板載剩 174（用 850MB）；10000×4=944MB OK；15000×4=1416MB 超 → 撞。
- 每台每行 ≈ **0.0236 MB**（= 寬16384 × ~1.5 ring ÷ 1MB；850÷4÷9000 反推）。
- grab acquisition 用板載（host buffer + 板載 FIFO），板載 1GB 較小先撞。

**stall 的三個獨立原因（釐清）：**
1. **改參數重啟空檔**（階段一已解：M_GRAB_ABORT drain，在 main）。
2. **grab 中 realloc 配比「初始」大的 buffer**（初始 7000→改 10000 stall；初始 10000→來回 OK）→ **max-buffer 解**（初始配大、改高度只改 M_SOURCE_SIZE_Y 不 realloc；實測「buffer 9000 改小到任意值幀正常」證明 buffer>source size OK，推翻「幀填滿 buffer」疑慮）。
3. **autoMax 自動管道引入的 stall**（cam1 一開機 stall）—— **不是高度值**（json 直接設 8736 不 stall，證明值無辜；同值經 autoMax 管道才 stall）。最可能是管道在 **第一台相機 MIL init 序列前/中插入額外 MdigInquire/MsysInquire**（實測加/移會變）。**教訓：autoMax 計算絕不可碰 MIL init 序列；高度值要從「設 CameraGrabHeight 的同位置」當純數字進去（跟 json 同路徑）。**

**怎麼「自動分配最大高度」（可靠、不矇）：**
- 公式：`max高度 = 板載1024 × safety% ÷ 同板台數 ÷ 0.0236`（板0 4台 85%≈9200、板1 3台≈12000）。驗證：10000×4=92% OK、15000×4 超。
- 但板載量要 `MsysInquire`（查 MIL）→ **不可在 cam init 序列查**（會 stall）；須在「所有相機 init 完、grab 前」查一次。
- **更穩健 = 實配驗證**：開機從目標高度往下試配 `MbufAlloc2d`，**配不出回 M_NULL（不 stall）就降**，找實際配得出的最大。完全不靠係數、自動適應。
- 配置的高度要走「跟 json 同路徑」（純數字設 CameraGrabHeight），不可經會碰 MIL init 的管道。

**現狀（2026-06-23 commit）：** 改回「grab 高度用 json 合法值、不主動 clamp」→ cam1 OK、8736 也 OK。autoMax 主動 clamp 整套已移除。max-buffer / 自動分配（用上面實配驗證）列為下一步。下方診斷（板載查詢/Height feature log/dropdiag/paramchange）+ settings（safety/BoardTotalMemMB 預留）保留供下一步用。

---

> 以下為 6/18–6/22 推理過程（部分結論已被上方釐清推翻，如「256MB chunk」「依板載 clamp autoMax」，保留當紀錄）。

## 目標

改高度現行流程很重：`STOP+WAIT → ABORT → FreeGrabBuffers → MdigControl(M_SOURCE_SIZE_Y) → MbufAlloc×N → settle → START`。
鐵證（操作員）：「停→改高度→開」重複會 stall；「停→改高度多次→開一次」不會 → **每次「realloc→re-arm」累積壞狀態**。
階段二想**拿掉每次 free/realloc**，讓改高度只改 `M_SOURCE_SIZE_Y`。

## 關鍵語意（Matrox 官方文件查證，2026-06-18）

**`UserGuide/grabbing/Linescan_cameras.htm`**：
> "When acquiring data from a line-scan camera, each line of each destination buffer band is filled from top to bottom.
> **The operation will only end once the entire buffer has been filled.**"

→ **line-scan 幀在「整個 destination buffer 填滿」才完成**。這暗示：若 grab buffer 配 max 高度、`M_SOURCE_SIZE_Y`
設小值，**幀可能要填滿整個 max buffer 才完成 → 幀變成 max 高度（壞）**。即 buffer 高度與 `M_SOURCE_SIZE_Y` 可能
必須一致 → **max-buffer 方案可能不可行**。

**`UserGuide/grabbing/Grabbing_and_processing.htm`**：MIL 對「尺寸會變」的官方做法是
> "MdigProcess() can **automatically allocate an appropriate image buffer** to store each grabbed frame...
> useful when grabbing from a camera configured to transmit images with **different dimensions for each grab**.
> ...set DestContainerOrImageBufArrayPtr to **M_NULL**."

→ **auto-allocate（bufarray=M_NULL）才是 MIL 變尺寸的正路**：MIL 每幀自動配符合當前 source size 的 buffer，
改 `M_SOURCE_SIZE_Y` 後下一幀自動跟上，**呼叫端不手動 free/realloc grab buffer**。

## 兩個方案

| 方案 | 做法 | 風險 |
|---|---|---|
| **A. max-buffer**（本分支 scaffold） | grab/display buffer 配一次 max 高度，改高度只改 `M_SOURCE_SIZE_Y` | doc 暗示**可能不可行**（幀變 max 高度）。記憶體大（max×width×N×cam）。需上機驗 M_SIZE_Y |
| **B. auto-allocate** | `MdigProcess(bufarray=M_NULL)` MIL 自動配每幀 buffer | 核心 grab buffer 管理大改（hook 取 MIL 配的 buffer）；display/native buffer 仍要自管。較大工程 |

## 本分支的 scaffold（flag 預設關，可上機測 A 的可行性）

- `MilCamera.UseMaxHeightBuffers`（static bool，預設 false）：true → Initialize 配 buffer 時用 `MaxGrabHeightPx`
  (=10000，與滑桿 HtMax 一致)，`SetGrabHeight` 走 no-realloc 路徑（只改 `M_SOURCE_SIZE_Y` + 更新 FrameHeight）。
- **開關**：環境變數 `PICOATER_MAXBUF=1`（不用重編）。在 `AniloxRollForm.AutoAllocateCameras`（AllocateCameras 前）讀。
- **上機驗證步驟**：
  1. 設 `PICOATER_MAXBUF=1` 啟動程式、抓圖。
  2. 改高度，看 Trace log：`[CAMx] 階段二 no-realloc 改高度：req=H M_SIZE_Y=? buf=10000`。
  3. **判讀**：
     - `M_SIZE_Y == req`（小高度）→ **max-buffer 可行！** 幀高度正確、不 realloc → 改高度 near-instant。再驗影像/曲線高度對。
     - `M_SIZE_Y == 10000`（=buf）或影像變 max 高度 → **doc 疑慮成立、A 不可行** → 走方案 B（auto-allocate）。

## 已知 caveat（A 方案）

- **display buffer 也 max 高度**：MIL 直繪會顯示 max（下方 stale）。SmartCanvas 路徑讀 `FrameWidth×FrameHeight`
  （actual）故正確。若 A 可行，MIL 直繪需用 actual 高度的 child buffer（`MbufChild2d`）顯示。
- **記憶體**：max(10000)×width×2 buffer×cam 數。7 cam 約 GB 級，需確認板載/host 記憶體夠（telemetry `Mem Free`）。
- **native/GPU buffer**：`onStoppedBeforeRestart` 仍會依 FrameHeight 重配上層 buffer（不變）。

## 未測的「大獎」

若 A 可行，下一步試 **grab 中改 `M_SOURCE_SIZE_Y` 不停機**（`M_COMMAND_QUEUE_MODE=M_QUEUED` 幀邊界套用）→
改高度像曝光一樣 live、不掉幀。本 scaffold 仍 stop+drain（保守），確認 A 可行後再試 live。

## ★ 板載記憶體約束 — 高度 stall 的真根因 + 安全 max 高度算法（2026-06-22 實機數據）

### 真根因（已確認）
「高度拉太高就 stall」**不是相機高度上限**（相機 GenICam `Height` 的 M_FEATURE_MAX 回 `4294967295`＝
無限，line-scan 本質），而是**「在運行中 realloc 配比初始更大的 grab buffer」撞到 grabber 板載記憶體**。
鐵證（操作員 2 台在線測）：
- 初始高度 **1893**（buffer 配 1893）→ 往上拉超過 → realloc 配更大 → **stall**。
- 初始高度 **10000**（buffer 配 10000）→ 來回拉都 ≤10000 → realloc 配同等/更小 → **不 stall**。
→ 初始化時配的 buffer 高度＝該 session 的「安全上限」。**這正是階段二 max-buffer（一開始配 max、之後不配更大）會根治 stall 的手動驗證。**

### 記憶體共用範圍：每張板（不是全系統、不是每相機獨立）
`GetMemoryFreeMB`（板載可用）對同板相機回**同值** → 同一張板的 channel **共用該板記憶體池**。
本機 7 台拓樸（`SystemDefaults.NewCameraDevices`）：
- **板 0**（SystemNum=0）：CAM 1,2,3,4 — **4 台共用板0**
- **板 1**（SystemNum=1）：CAM 5,6,7 — **3 台共用板1**

### 反推板載容量（2 台在線實測）
| 高度 | 板載剩餘 |
|---|---|
| 1893 | 779MB |
| 10000 | 395MB |

高度差 8107 行多吃 384MB（2 台）→ **每台每行 ≈ 0.0237MB**（寬 16384）→ 板載每張板「可給 buffer」≈ **869MB**。
（係數含 ring buffer 數/display 是否在板等假設 → 要精確需 code 查板載**總量** + 配置前後各量一次得真係數。）

### 安全 max 高度表（每板，留 15% 餘裕；`H_max ≈ 31250 / 每板台數`）
| 板 | 相機 | 每板台數 | 理論最高 | clamp 10000 |
|---|---|---|---|---|
| 板0 | CAM1-4 | 4 | 7812 | **7812** |
| 板1 | CAM5-7 | 3 | 10416 | **10000** |

→ **整機統一高度上限 = 取最緊板 = ~7812**。日常作業高度（如 3500）7 台 ≈ 580MB/板 < 869MB → 安全。
（先前誤算「7 台共用一池→4464」是錯的，記憶體是每板共用。）

### 待辦（精確版）
1. code 查**板載總記憶體**（非剩餘）+ 配置前後量真係數。
2. 依**每板實際相機數**算安全 max 高度 → 設 max-buffer 高度 + clamp 高度滑桿上限（拉不到爆區＝根治）。

## 決策

**先不合入 main**。上機翻 flag 驗一次再定：A 可行 → 完成 A（加 child-buffer 顯示 + 板載安全 max 高度）；
A 不可行 → 重做成 B。階段一已解 stall，階段二不急。
