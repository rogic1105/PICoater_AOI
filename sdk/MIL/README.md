# MIL — Matrox 取像/顯示集中區

MIL 相關全圈在這一區（隔離 Matrox 依賴）。未來換 grabber（不需 MIL）時，整區替換/移除最乾淨，也避免 MIL 依賴/license 散落全 repo。

## Contents

- `MilGrabber.Core/` — **MIL 取像/顯示封裝 library**（純 MIL、無 GUI）：`MilCamera`（一台相機）+ `MultiCameraMerger`（多相機即時合圖工頭）+ `MilCameraParams`（參數公式）。詳見其 README。
- `MilGrabber.Monitor/` — **多相機監控範例 exe**（WinExe，可獨立跑）：用 `MilGrabber.Core` 組裝佈局 / 選相機 / 參數面板 / 抓取相機資訊。
- `docs/` — Matrox 廠商規格書 + CLProtocol 範例（純參考，不參與 build）

## 定位（sdk 範例庫 + Agent 組裝）

sdk 範例庫的「取像/顯示」一塊。新專案 / Agent 可拿 `MilGrab` 當範例組裝即時監控。`AniloxRoll.Monitor` 的「即時監控」tab 是這個範例的**完整應用版**（多相機 + 全域合圖 + 即時曲線）。

## 換 grabber 時

換成非 MIL 的 grabber = **新增另一包範例**（新 grabber 的 grab+顯示+UI），不碰這包；要完全免 MIL 就整區移除 `sdk/MIL/`。

## 約束

維持現有 MIL 函式集（grab / display 基本款），**不引入新 MIL 函式** —— 目前的 MIL 不需額外 USB runner，避免觸發需要額外 license 的功能。範例擴充（如即時曲線）用 GPU/CV 算、不碰 MIL API。

## 工程紀錄：批次套用（display 巨圖刷新）

> 原則見全域 `~/.claude/CLAUDE.md`「架構原則：批次套用（先算好，最後一次提交）」。此處記 MIL display 的具體案例。

**問題**：合圖巨圖（7 相機 ≈ 89112×3001 ≈ 2.7 億像素）重繪很貴。`EnableGlobalMerge` 早期把 `MdispSelectWindow` + `M_SCALE_DISPLAY` + `M_CENTER_DISPLAY` 在 `M_UPDATE` 仍 ENABLE 下連續執行 → 每個 control 各觸發一次巨圖重繪（切換 lag）；且 select 瞬間顯示「grab hook 尚未貼滿的合併 buffer」→ 橫條半貼殘影閃一下。

**修法（批次）**：先 `M_UPDATE M_DISABLE` → 再 `MdispSelectWindow` + 所有 control（此時都不重繪）→ 33ms timer `M_UPDATE M_NOW` 統一刷一次。**多次重繪 → 一次**，殘影 / lag 同時消。

**通用規則**：MIL 巨圖 display 的多個 control 操作，一律包在 `M_UPDATE M_DISABLE … 刷新` 之間，不要邊設定邊重繪。

**成本分類（決定要不要批次）：**
- 昂貴 → 批次：grabber / digitizer（影像 + PCIe + 重繪 / 重啟）。如相機參數寫入（拖曳放掉才寫、會中斷 grab）、合圖巨圖刷新。
- 便宜 → 即時：RS232（光源 `LightBridge`）/ RJ45（IO `IoBridge`）的控制指令。如光源亮度拖滑桿即時套用。
- 不涉及：讀取 / 查詢（telemetry、IO 狀態）— 不改狀態、不需批次。

**待辦（未統一）**：`_mergedDisplay` 的 `M_UPDATE` 有兩種管理並存 — `EnableGlobalMerge` 常駐 DISABLE（靠 timer M_NOW）vs zoom/pan handler 結尾 ENABLE。混用可能讓 zoom/pan 後自動刷新被打開、多相機又閃。值得統一成單一模式（需實機驗證防回歸）。

> 註：`MilGrabber.Monitor` 是 WinExe，要跑就設為起始專案。`MilGrabber.Core` 是 library（ProjectReference 引用）。
