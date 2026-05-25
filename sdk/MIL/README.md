# MIL — Matrox 取像/顯示集中區

MIL 相關全圈在這一區（隔離 Matrox 依賴）。未來換 grabber（不需 MIL）時，整區替換/移除最乾淨，也避免 MIL 依賴/license 散落全 repo。

## Contents

- `MilGrab/` — **MIL 取像+顯示可執行範例**（WinExe，可獨立跑）：MApp 初始化 → MdigAlloc（digitizer）→ MbufAlloc2d（grab buffer）→ MdispSelectWindow（顯示到 WinForms panel）→ MdigGrabContinuous（連續取像）+ 相機在線 hook / 自動重連 / 釋放。雙相機 demo。
- `docs/` — Matrox 廠商規格書 + CLProtocol 範例（純參考，不參與 build）

## 定位（sdk 範例庫 + Agent 組裝）

sdk 範例庫的「取像/顯示」一塊。新專案 / Agent 可拿 `MilGrab` 當範例組裝即時監控。`AniloxRoll.Monitor` 的「即時監控」tab 是這個範例的**完整應用版**（多相機 + 全域合圖 + 即時曲線）。

## 換 grabber 時

換成非 MIL 的 grabber = **新增另一包範例**（新 grabber 的 grab+顯示+UI），不碰這包；要完全免 MIL 就整區移除 `sdk/MIL/`。

## 約束

維持現有 MIL 函式集（grab / display 基本款），**不引入新 MIL 函式** —— 目前的 MIL 不需額外 USB runner，避免觸發需要額外 license 的功能。範例擴充（如即時曲線）用 GPU/CV 算、不碰 MIL API。

> 註：`MilGrab` 是 WinExe，要跑就設為起始專案；它 self-contained（無 ProjectReference，只依賴 MIL + WinForms）。
