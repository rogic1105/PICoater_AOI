# MIL — Matrox 取像/顯示集中區

MIL 相關全圈在這一區（隔離 Matrox 依賴）。未來換 grabber（不需 MIL）時，整區替換/移除最乾淨，也避免 MIL 依賴/license 散落全 repo。

## Contents

- `MilGrabber.Core/` — **MIL 取像/顯示封裝 library**（純 MIL、無 GUI）：`MilCamera`（一台相機）+ `MultiCameraMerger`（多相機即時合圖工頭）+ `MilCameraParams`（參數公式）。詳見其 README。
- `samples/MilGrabber.Monitor/` — **多相機監控範例 exe**（WinExe，可獨立跑）：用 `MilGrabber.Core` 組裝佈局 / 選相機 / 參數面板 / 抓取相機資訊。
- `docs/` — Matrox 廠商規格書 + CLProtocol 範例（純參考，不參與 build）

## 定位（sdk 範例庫 + Agent 組裝）

sdk 範例庫的「取像/顯示」一塊。新專案 / Agent 可拿 `samples/MilGrabber.Monitor` 當範例組裝即時監控。`AniloxRoll.Monitor` 的「即時監控」tab 是這個範例的**完整應用版**（多相機 + 全域合圖 + 即時曲線）。

## 換 grabber 時

換成非 MIL 的 grabber = **新增另一包範例**（新 grabber 的 grab+顯示+UI），不碰這包；要完全免 MIL 就整區移除 `sdk/MIL/`。

## 約束

維持現有 MIL 函式集（grab / display 基本款），**不引入新 MIL 函式** —— 目前的 MIL 不需額外 USB runner，避免觸發需要額外 license 的功能。範例擴充（如即時曲線）用 GPU/CV 算、不碰 MIL API。

## 顯示邊界

`MilGrabber.Core` 保留通用的 MIL display API，供 SDK 範例或其他 MIL 應用使用。產品
`AniloxRoll.Monitor` 不建立 MIL 原生顯示視窗：MIL 負責取像與合併 buffer，產品顯示、縮放、平移與
滑鼠座標全部走 CPU 的 `ImageDisplayView`／`WaterfallView`／`ImageCanvas`。

`MultiCameraMerger.MergedBuffer` 是合圖資料源，不代表呼叫端必須用 `MdispSelectWindow` 顯示。
SDK 範例若使用 MIL display，批次變更多個 display control 時仍應先停更新、一次套用後再刷新；
這是範例端的 display policy，不得回流成產品主畫面的第二條顯示路。

> 註：`MilGrabber.Monitor` 是 WinExe，要跑就設為起始專案。`MilGrabber.Core` 是 library（ProjectReference 引用）。
