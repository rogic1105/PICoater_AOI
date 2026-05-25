# GrabberBridge — 殼目錄

**目前沒有 code**，只有 `vendor/`。預留未來抽出 `GrabberBridge.Core/` 的位置。

## 為什麼有這個殼

Matrox MIL 是本專案主要的影像擷取 SDK。目前 `src/dotnet/AniloxRoll.Monitor/` 直接呼叫 MIL，但未來若把 Matrox 互動抽成獨立 wrapper（如 `GrabberBridge.Core` 接相機/Grabber 跟 CLProtocol），這個殼目錄就現成的。

當前內容只有廠商規格書 + 範例（`vendor/`），讓修改 Matrox 相關 code 時容易找到參考資料。

## Contents

- `vendor/matrox-grabber/` — Matrox Radient grabber 規格書 + 範例
- `vendor/matrox-clprotocol/` — Matrox CLProtocol DLL 範例（GenICam ↔ Camera Link）

## Future GrabberBridge.Core

如果未來抽出 `GrabberBridge.Core/`，結構會變：

```
sdk/Bridges/GrabberBridge/
├── GrabberBridge.Core/          ← 新增（純 .NET wrapper）
└── vendor/                      ← 既有（廠商規格 + 範例）
    ├── matrox-grabber/
    └── matrox-clprotocol/
```

引用方（如 AniloxRoll.Monitor）改成 `<ProjectReference Include="..\..\..\sdk\Bridges\GrabberBridge\GrabberBridge.Core\..." />`，符合 [style-layout](../../docs/style-layout.html) §4.3 分界。
