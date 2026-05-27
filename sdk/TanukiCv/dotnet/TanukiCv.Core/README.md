# TanukiCv.Core

TanukiCv 的純 .NET library（無 GUI、無 exe、無 Matrox）。對上層 .NET 應用封裝
GPU 引擎 `core_cv_api.dll`，並提供影像 / 檔案輔助工具。

## Namespace

| Namespace | 內容 |
|-----------|------|
| `TanukiCv.Core` | `CoreCVWrapper`（`core_cv_api.dll` 的 P/Invoke 封裝）、`GPUHelper`、`GPUProcessor` |
| `TanukiCv.Core.Models` | `TimedResult`（計時結果容器） |
| `TanukiCv.Utils` | `FileUtils`、`ImageUtils` |

## 依賴

- `core_cv_api.dll`（runtime；由 P/Invoke 載入，不在 build 期參考）
- `System.Drawing` / `System.Windows.Forms`（`ImageUtils` / `FileUtils` 用到）

## 定位

- 引用 `TanukiCv.Core` 不會被迫拉入任何 UI 控制項相依（控制項在 `TanukiCv.Controls`）。
- self-contained，未來可隨 `sdk/TanukiCv` 一起 split 為獨立 repo。
