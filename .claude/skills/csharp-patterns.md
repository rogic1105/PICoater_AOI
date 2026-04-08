# csharp-patterns

C# / WinForms 開發模式與陷阱速查。

## 使用時機

開發新 C# 功能、新增控制項、修改 Settings 持久化、或遇到 WinForms 陷阱時查閱。

## 命名規則

| 對象 | 規則 | 範例 |
|------|------|------|
| Namespace/Project | PascalCase 無底線 | `AniloxRoll.Monitor` |
| 3+ 字元縮寫 | PascalCase | `Mil`, `Aoi` |
| 2 字元縮寫 | 全大寫 | `IO`, `UI` |
| 知名 SDK 縮寫 | 保留全大寫 | `MIL` |

## Settings 持久化

| 檔案 | 對應 |
|------|------|
| `Config\inspection-settings.json` | PropertyGrid |
| `Config\acquisition-settings.json` | TrackBar |
| `Config\system-settings.json` | 唯讀硬體拓樸 |
| `Config\session-state.json` | UI session state |

- **不使用 `JavaScriptSerializer`**（`user.config` 損毀時拋例外），改用手刻 `SerializeJson/ParseJson`
- 新增 PropertyGrid 屬性 → 必須同步更新 Store 的序列化/反序列化
- `AcquisitionSettings` 索引 0=CAM1…6=CAM7
- 存檔在 `ValueChanged` 觸發（不用 `MouseUp`，因為拖曳放開不在控制項範圍內時 MouseUp 不觸發）

## WinForms 陷阱

### PictureBox Bitmap Dispose
先 `galleryManager.ClearImages()`（PictureBox.Image = null）再 `Dispose()`，否則 Paint 事件爆炸。

### TrackBar 滾輪
Windows 原生每 notch 送 3 格。解法：`NativeWindow` 子類攔截 `WM_MOUSEWHEEL`，自行 `+Math.Sign(delta)`。
不要用 `MouseWheel` 事件（無法阻止 DefWndProc 的 3 格行為）。攔截器必須存放防 GC 回收。

### GDI+ 無縫拼接
```csharp
g.InterpolationMode = InterpolationMode.NearestNeighbor;
g.PixelOffsetMode   = PixelOffsetMode.Half;
```
否則 0.5px 偏移 + 色彩擴散 → 拼接接縫線。

### AOI.SDK AllowUnsafeBlocks
Solution 映射 `Debug|x64` → `Debug|Any CPU`（含空格），`AllowUnsafeBlocks` 必須放無條件全域 PropertyGroup。

### Exception Handling
絕不裸 `catch {}`，最少 `Trace.WriteLine($"[Class.Method] {ex.GetType().Name}: {ex.Message}")`。

### PropertyGrid
- `[Browsable(false)]` 隱藏但仍可讀寫
- `PropertySort.Categorized`（不加 Alphabetical）保留宣告順序
- 欄位寬度用反射 `MoveSplitterTo`（indent=16 + rightMargin=8）

### ListView AutoFit
`AutoResizeColumn(ColumnContent)` vs `AutoResizeColumn(HeaderSize)` 取 max。
動態更新的 ListView 只在第一次 Tick 後 fit 一次（`_fitDone` flag）。

### IEC 60073 訊號燈
綠 `#388E3C`（正常）、灰 `#757575`（待機）、紅 `#C62828`（故障）、黃 `#F9A825`（警告）。

## .NET Framework 4.8 + C# 7.3 注意

- 新增 .cs 檔必須手動加 `<Compile Include>` 到 .csproj
- 無 `switch expression`、`is not`、`record`、`init`
- `string interpolation` 可用；`??=` 不可用
