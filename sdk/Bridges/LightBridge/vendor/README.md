# LightBridge — Vendor Reference

廠商給的東西（不參與 build），純參考用。需要更新時整包替換，**不要**在這目錄內手動 patch。

## Contents

- `lts3dpa24/` — LTS-3DPA24 系列光源控制器
  - `20220923 LTS_3DPA24系列控制器说明书 （全电压）.pdf` — RS-232 protocol 完整規格（§4.1.4 8-byte 命令格式）
  - `光源控制 DEMO VC(3DPA) 20250513/` — 廠商 Visual C++ 示範程式（含 XOR checksum 計算範例）

## Source

- 廠商：LTS（Linkpark Technology Suzhou）
- 取得方式：廠商隨機附 USB / 廠商網站
- 取得日期：2022-09-23（manual）/ 2025-05-13（DEMO）

## Used by

- `../LightBridge.Core/LightController.cs` — protocol §4.1.4 命令格式對照、AutoDetect probe 邏輯
- 偵錯時對照 `光源控制 DEMO/` 看 XOR checksum 計算

## When to update

- 廠商發新規格書 / 新 DEMO 時整包替換
- **不要**在這目錄內手動改任何檔案（會被下次升級洗掉）

需要改廠商範例邏輯時，複製到 `../LightBridge.Core/` 變「我們的 code」再改。
