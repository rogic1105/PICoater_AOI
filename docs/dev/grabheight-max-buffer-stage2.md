# 階段二：改高度 no-realloc（max-buffer / auto-allocate）研究與 scaffold

> 分支 `feat/grabheight-max-buffer`。**未上機驗證** —— 此文件記錄研究結論 + 一個 flag 預設關的可測 scaffold，
> 供回來上機決策。階段一（`M_GRAB_ABORT` drain，已 merge main）**已修好改高度 stall**；階段二純屬**最佳化**
> （改高度更快、結構性免疫），非必需。

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

## 決策

**先不合入 main**。上機翻 flag 驗一次再定：A 可行 → 完成 A（加 child-buffer 顯示）；A 不可行 → 重做成 B。
階段一已解 stall，階段二不急。
