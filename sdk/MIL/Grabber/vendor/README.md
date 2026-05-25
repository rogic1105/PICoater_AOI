# Grabber — Vendor Reference

廠商（Matrox）給的東西（不參與 build），純參考用。

## Contents

- `matrox-grabber/` — Matrox Radient grabber 系列規格書 + 範例 code
- `matrox-clprotocol/` — Matrox CLProtocol（GenICam over Camera Link）DLL 範例
  - `C++/CLProtocol.cpp` — 廠商示範如何寫 CLProtocol.dll
  - `C++/vs2022/` — VS2022 build project
  - `CLProtocol.xml` — GenICam 描述檔範例

## Source

- 廠商：Matrox Electronic Systems
- 取得方式：Matrox MIL SDK 安裝包附帶
- 版本：Copyright Matrox 1992-2025（CLProtocol.cpp 註明）

## Used by

- `src/dotnet/AniloxRoll.Monitor/Acquisition/AniloxCamera.cs` — `MdigControl(M_GC_CLPROTOCOL, M_ENABLE)` 流程對照
- 偵錯時對照 `matrox-clprotocol/C++/CLProtocol.cpp` 看 Device ID 列舉、GenICam API 使用
- 相關坑：Quad card 必須列舉 Device ID（`M_DEFAULT` 對 DevNum&gt;=2 無效）

## When to update

廠商發新 SDK 版本時整包替換。**本專案的 code 不會直接 build 這份 vendored copy**，
runtime 載入的是 Matrox SDK 安裝後系統內的 CLProtocol.dll。

## Note

本專案目前**沒有自己寫 CLProtocol.dll**（用廠商隨相機提供的現成 .dll）。
這份範例純對照用，未來若決定自己寫，會複製到本元件成為我們的 code（取代廠商現成 .dll）。
