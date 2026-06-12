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
- `AcquisitionSettings` 索引 0=CAM1…6=CAM7
- 存檔在 `ValueChanged` 觸發（不用 `MouseUp`，因為拖曳放開不在控制項範圍內時 MouseUp 不觸發）

### ★ 新增一個設定參數的 5 步清單（缺一步就出 bug）

> 預設值唯一來源在 `Settings/Models/Defaults/*Defaults.cs`；model 初始值與 Store fallback **都引用它**（檔頭註解明寫）。曾踩坑：`MainDisplay`/`LiveLod` 新增時直接在 model 寫死字面值、又沒進 JSON → 跳過 Defaults 架構 + 改了不持久化。照這張清單就不會漏：

1. **進 Defaults** — `InspectionDefaults.cs`（或對應 `*Defaults.cs`）加常數；enum 用 `static readonly`，值型別用 `const`。
2. **model 初始值引用 Defaults** — `= InspectionDefaults.X`，**不寫死字面值**（如 `= LiveLodMode.CPU` ❌ → `= InspectionDefaults.LiveLod` ✅）。
3. **Store `SerializeJson` 加一行** — 寫進對應區塊（注意前一行尾要補逗號）。
4. **Store `Parse*` 加讀取 + fallback** — fallback 一律 `InspectionDefaults.X`（不寫死字面值）；enum 用 `Enum.TryParse(..., out var v)` 失敗回 Defaults。
5. **PropertyGrid attribute**（若要露出給使用者）— `[Category]`/`[DisplayName]`/`[Description]`；enum 中文用 `[Description]` + `EnumDescConverter`。

驗收：刪 `bin\...\Config\inspection-settings.json` 重啟 → 應以新預設重生並寫回（含步驟 3 的新欄位）。⚠ 若參數漏了步驟 3/4（沒進 JSON），它**永遠只吃 model 預設、PropertyGrid 改了重開不留存**。

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

### TanukiCv.Core AllowUnsafeBlocks
Solution 映射 `Debug|x64` → `Debug|Any CPU`（含空格），`AllowUnsafeBlocks` 必須放無條件全域 PropertyGroup。

### Exception Handling
絕不裸 `catch {}`，最少 `Trace.WriteLine($"[Class.Method] {ex.GetType().Name}: {ex.Message}")`。

### PropertyGrid
- `[Browsable(false)]` 隱藏但仍可讀寫
- `PropertySort.Categorized`（不加 Alphabetical）保留宣告順序
- 欄位寬度用反射 `MoveSplitterTo`（indent=16 + rightMargin=8）
- **scroll 捲到頂**：`ExpandAllGridItems()` 後 scroll 停在最後一項。`VScrollBar` 是內部 `GridView` 的子控制項，外層 `Controls` 找不到。Fix：在 `Shown` 事件設 `SelectedGridItem` 到第一個屬性（PropertyGrid 會自動捲動）。層級：`SelectedGridItem.Parent` = 當前 category，`.Parent.Parent` = 根節點，`GridItems[0].GridItems[0]` = 第一 category 第一屬性。不可在 constructor 用 `BeginInvoke`（handle 尚未建立）。

### PropertyGrid 動態說明欄（TypeDescriptionProvider）

點選屬性時在底部說明欄動態顯示目前值，用 `TypeDescriptionProvider` + `PropertyDescriptor` wrapper 實作。

**⚠️ 陷阱：`[TypeDescriptionProvider]` 屬性的循環參考**
```csharp
// ❌ 錯誤：_base 在 class 載入時呼叫 GetProvider，
//         因屬性已存在，GetProvider 又回傳自己 → instance 永遠 null
[TypeDescriptionProvider(typeof(MyProvider))]
class MySettings { }

class MyProvider : TypeDescriptionProvider {
    private static readonly TypeDescriptionProvider _base =
        TypeDescriptor.GetProvider(typeof(MySettings)); // ← 循環！
    public MyProvider() : base(_base) { }
}
```

**✅ 正確：per-instance 模式（parent 明確傳入）**
```csharp
// 不加 [TypeDescriptionProvider] 屬性
class MySettings { }

class MyProvider : TypeDescriptionProvider {
    private readonly MySettings _s;
    public MyProvider(TypeDescriptionProvider parent, MySettings s)
        : base(parent) { _s = s; }
    public override ICustomTypeDescriptor GetTypeDescriptor(Type t, object inst)
        => new DynamicDescriptor(base.GetTypeDescriptor(t, inst), _s);
}

// 在 Form 初始化，SelectedObject 設定前呼叫：
TypeDescriptor.AddProvider(
    new MyProvider(TypeDescriptor.GetProvider(_settings), _settings), _settings);
propertyGridSettings.SelectedObject = _settings;
```

**實作重點**
- `DynamicDescriptor : CustomTypeDescriptor` — 覆寫 `GetProperties()` 與 `GetProperties(Attribute[])` 兩個多載，用 `new PropertyDescriptorCollection(arr)` 回傳包裝後的陣列
- `ValueDescriptor : PropertyDescriptor` — 用 `base(inner)` 複製原本屬性（名稱、屬性集），只覆寫 `Description`
- `Description` 取目前值：`_inner.GetValue(_s)`；bool → `"是"/"否"`；enum → 反射讀 `[Description]` 屬性
- 標題列（`[ReadOnly(true)]` 分隔符）名稱以 `"Header"` 結尾，**不包裝**，說明欄保持空白
- `GetValue` 用 `try/catch` 包裹，失敗回傳 `""`
- 多個 per-instance provider 可疊加（如同時有過濾器 `StorageModeSettingsFilter`），後加的為最外層

### ListView AutoFit
`AutoResizeColumn(ColumnContent)` vs `AutoResizeColumn(HeaderSize)` 取 max。
動態更新的 ListView 只在第一次 Tick 後 fit 一次（`_fitDone` flag）。

### IEC 60073 訊號燈
綠 `#388E3C`（正常）、灰 `#757575`（待機）、紅 `#C62828`（故障）、黃 `#F9A825`（警告）。

## .NET Framework 4.8 + C# 7.3 注意

- 新增 .cs 檔必須手動加 `<Compile Include>` 到 .csproj
- 無 `switch expression`、`is not`、`record`、`init`
- `string interpolation` 可用；`??=` 不可用

## GDI+ Bitmap 跨執行緒鐵則（2026-06-12 race 實戰）

- **Bitmap/Image 是單執行緒物件**：任一執行緒 LockBits 中，另一執行緒連 `get_Width` 都會炸
  `InvalidOperationException: 其他地方正在使用物件`（快速換 ID 時背景轉換 vs 下一輪載入相撞）。
- **修法選架構不選鎖**：要跨執行緒傳影像 → 在「Bitmap 仍獨佔、未發布」的階段（如解碼當下）
  轉成**不可變 byte[]**再傳；Bitmap 本體不跨出擁有者。`lock(bmp)` 只防自己人，防不了外部讀取。
- 案例：ReviewStitchCoordinator 解碼 Parallel.For 內轉灰階 → 事件只傳 bytes+尺寸。

