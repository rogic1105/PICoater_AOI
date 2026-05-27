# TanukiCv.Controls

TanukiCv 的 WinForms 控制項 library。獨立 assembly，僅依賴
`System.Windows.Forms` + `System.Drawing`，**不依賴 `TanukiCv.Core`**（控制項不需要 GPU 引擎型別）。

## Namespace

| Namespace | 內容 |
|-----------|------|
| `TanukiCv.Controls` | `SmartCanvas`（zoom / pan / edge-trigger / ClampPan 的 `PictureBox` 子類，含自訂白底黑邊十字游標）、`CanvasInfo`（StatusChanged 事件資料） |

## 歷史

`SmartCanvas` 原先因 `TanukiCv.Core`（舊名 `AOI.SDK`）csproj 帶有一個死的
Matrox `<Reference>`，被迫 source-link 進各 consumer（主程式 + benchmark UI）以避開
design-time 載入失敗。移除死引用後 `SmartCanvas` 是純 WinForms 控制項，遂獨立成本 assembly，
consumer 改以 `ProjectReference` 引用。

## 定位

self-contained，未來可隨 `sdk/TanukiCv` 一起 split 為獨立 repo。
