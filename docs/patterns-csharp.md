# C# / WinForms 開發模式與陷阱

## C# 命名規則（專案標準）

### 命名格式
| 對象 | 規則 | 範例 |
|------|------|------|
| Namespace / Project / Assembly | PascalCase，**不使用底線** | `MilGrabSample`、`AniloxRoll.Monitor` |
| 3 字元以上縮寫 | PascalCase | `Mil`、`Aoi`、`Sdk` |
| 2 字元縮寫 | 全大寫 | `IO`、`UI` |
| 知名 SDK 縮寫（慣例） | 保留全大寫可接受 | `MIL`（Matrox Imaging Library） |

### 重新命名 C# 專案的完整步驟
1. `git mv` 外層資料夾（solution 層）
2. `git mv` 內層資料夾（project 層）
3. `git mv` `.csproj`、`.sln`
4. 修改 `.sln` — project 名稱 + 路徑
5. 修改 `.csproj` — `<RootNamespace>`、`<AssemblyName>`
6. 修改 `Properties/AssemblyInfo.cs` — `AssemblyTitle`、`AssemblyProduct`
7. 修改所有 `.cs` — `namespace OldName` → `namespace NewName`
8. 修改 `Properties/Resources.Designer.cs` — namespace + resource 字串（`"OldName.Properties.Resources"`）
9. 修改 `Properties/Settings.Designer.cs` — namespace
10. 更新 `CLAUDE.md` 路徑引用

> `Backup/`、`obj/`、`bin/` 為建置產物，不需手動修改。

---

## WinForms Designer 控制項規則

### 控制項必須在 InitializeComponent() 才能在 VS Designer 顯示

動態在 code-behind 建立的控制項（`new TrackBar()`、`new ListView()` 等）**不會出現在 VS Designer**。
若需要 Designer 能看到，必須：

1. 在 `InitializeComponent()` 頂端加 `this.xxx = new ...`
2. 加 `SuspendLayout()` / `BeginInit()`（ISupportInitialize 控制項：TrackBar、NumericUpDown）
3. 加 container 的 `Controls.Add(this.xxx)`
4. 加控制項屬性設定區塊
5. 加 `ResumeLayout()` / `EndInit()`
6. 加 `private System.Windows.Forms.Xxx xxx;` 欄位宣告

然後在 code-behind 的 `InitializeSystem()` 只做：
- 從 runtime 資料套用初始值（`trackBar.Value = _settings.Xxx`）
- 繫結事件（需要 `_settings`、service 等 runtime 物件的部分）

### Designer.cs 控制項批次重命名

`Edit` 工具的 `replace_all: true` 可安全批次替換 Designer.cs 中的控制項名稱：

```
old_string: "panel7"  →  new_string: "panelStatCam1"
```

**安全性確認**：`panel7` 不是 `panel70` / `panel17` 等的子字串。

**順序**：先替換較長的數字（`comboBox12` 先於 `comboBox1`），避免 `comboBox1` 誤替換 `comboBox12` 中的部分字元。

### Designer.cs 移除 TableLayoutPanel（取出子控制項）

1. 移除 `this.tlp = new System.Windows.Forms.TableLayoutPanel();`
2. 移除 `this.tlp.SuspendLayout();` / `this.tlp.ResumeLayout(false);`
3. 將 `parent.Controls.Add(this.tlp)` 換成直接 `parent.Controls.Add(child1); parent.Controls.Add(child2);`
4. 刪除整個 TLP 設定區塊
5. 子控制項移除 `Dock = Fill`，設定絕對 Location / Size
6. 移除欄位宣告

位置計算（百分比 → 絕對）：TLP Location=(8, 123), Size=(1070, 495), Row0=70%, Row1=30%：
- Row0 子控制項：Location=(8, 123), Size=(1070, 346)  → 346 = floor(495 × 0.7)
- Row1 子控制項：Location=(8, 469), Size=(1070, 149)  → 469 = 123 + 346

---

## AOI.SDK.csproj AllowUnsafeBlocks 陷阱

Solution 將 `Debug|x64` 映射為 `Debug|Any CPU`（Platform="Any CPU" 含空格），導致：
- `.csproj` 中 `Condition="'Debug|AnyCPU'"` 的 PropertyGroup **不被套用**（名稱不符）
- `AllowUnsafeBlocks` 必須放在**無條件的全域 PropertyGroup**

---

## Settings 分檔持久化架構

| 檔案 | 對應 Tab | 內容 |
|------|---------|------|
| `Config\inspection-settings.json` | tabPageInspSettings（PropertyGrid） | MachineLayout + Recipe + Storage |
| `Config\acquisition-settings.json` | tabPageCamera（TrackBar） | CameraGrabHeight[7] + CameraExposureTimeUs[7] + CameraLineRateHz[7] |
| `Config\system-settings.json` | tabPageSystem（唯讀） | 相機硬體拓樸 |

- 兩個 JSON 首次執行自動建立（`Load()` 讀不到時 `Save(defaults)` 建檔）
- `AcquisitionSettings` 陣列索引 0 = CAM1 … 6 = CAM7
- 存檔在 **`ValueChanged`** 觸發（不用 `MouseUp`）：TrackBar 快速拖曳放開不在控制項範圍內時 `MouseUp` 不一定觸發
- **`AcquisitionSettingsStore` 不使用 `JavaScriptSerializer`**：改用手刻 `SerializeJson` / `ParseJson`（Regex 解析陣列，`InvariantCulture` 解析 double），原因：`user.config` 損毀時 `new JavaScriptSerializer()` 拋 `ConfigurationErrorsException`
- **`InspectionSettingsStore` 同樣不使用 `JavaScriptSerializer`**：手刻 `SerializeJson` + `ParseJson`

### AcquisitionSettings 初始值與 Validate fallback 一致性

`AcquisitionSettings.cs` 的屬性初始值必須與 `Validate()` fallback 一致，確保首次執行與 JSON 損毀回退行為相同。

---

## PropertyGrid 模式

### 隱藏特定屬性

`[Browsable(false)]` 讓屬性不在 PropertyGrid 顯示，但仍可正常讀寫（序列化、code-behind 不受影響）。

### 欄位寬度自動適配

```csharp
private void AutoFitPropertyGridLabelColumn(PropertyGrid grid)
{
    var gridView = grid?.Controls?.OfType<Control>()
                      .FirstOrDefault(c => c.GetType().Name == "PropertyGridView");
    var moveSplitter = gridView.GetType()
        .GetMethod("MoveSplitterTo", BindingFlags.Instance | BindingFlags.NonPublic);
    using (var g = gridView.CreateGraphics())
    {
        float maxTextWidth = 0;
        // ... MeasureString each label
        const int indent = 16, rightMargin = 8;
        moveSplitter?.Invoke(gridView, new object[] { indent + (int)maxTextWidth + rightMargin });
    }
}
```

**陷阱**：padding 設為 6 太小，最長 label 仍被截。正確值 indent=16 + rightMargin=8。

### 屬性排列順序

`PropertySort.Categorized`（不加 Alphabetical）保留 `.cs` 宣告順序。

---

## TrackBar + NumericUpDown 雙向同步

避免互觸無窮迴圈的 pattern（用 captured local bool）：

```csharp
bool syncing = false;
trackBar.ValueChanged += (s, e) => {
    if (syncing) return;
    syncing = true;
    numericUpDown.Value = trackBar.Value;
    syncing = false;
};
numericUpDown.ValueChanged += (s, e) => {
    if (syncing) return;
    syncing = true;
    trackBar.Value = Math.Max(trackBar.Minimum, Math.Min(trackBar.Maximum, (int)numericUpDown.Value));
    syncing = false;
};
```

`syncing` 是 lambda 捕獲的 local 變數，兩個 lambda 共用同一個 heap slot。

### TrackBar 拖曳偵測模式

拖曳期間抑制硬體寫入（避免每個中間值都呼叫 SetGrabHeight / SetExposureUs）：

```csharp
private readonly HashSet<TrackBar> _dragging = new HashSet<TrackBar>();

bar.MouseDown  += (s, e) => _dragging.Add(bar);
bar.MouseUp    += (s, e) =>
{
    _dragging.Remove(bar);
    _liveCameraManager?.SetXxxForCamera(camId, bar.Value);
};
bar.ValueChanged += (s, e) =>
{
    if (sync || _syncingFromHw) return;
    if (!_dragging.Contains(bar))
        _liveCameraManager?.SetXxxForCamera(camId, bar.Value);
};
```

- `HashSet<TrackBar>` per Form，7 台 TrackBar 共用
- `SetGrabHeight` 特別受益：拖曳期間完全不執行（Buffer 重分配代價高）

### TrackBar 滑鼠滾輪每格 = 1

Windows 原生 TRACKBAR 控制項 WM_MOUSEWHEEL 行為：每個滾輪 notch 送出 **3 × TB_LINEUP/TB_LINEDOWN**。
解法：`NativeWindow` 子類攔截 `WM_MOUSEWHEEL`，自行 `+Math.Sign(delta)` 後 `return`。

- **不要用 `bar.MouseWheel` 事件**：無法阻止 DefWndProc 的原生 3 格行為
- 攔截器引用必須存放於 `List<NativeWindow>` 防止 GC 回收

### Exposure 夾緊視覺回饋

LR 改變導致曝光被夾緊時，以 `Color.OrangeRed` BackColor 提醒（比 ForeColor 視覺更強）。

---

## WinForms Form Resize — Anchor 策略

| 控制項位置 | Anchor | 效果 |
|-----------|--------|------|
| 主內容區（跨全寬全高） | `Top\|Bottom\|Left\|Right` | 四方向等比延伸 |
| 右側固定寬度面板 | `Top\|Bottom\|Right` | 保持寬度，跟右邊緣 |
| 底部圖表（全寬） | `Bottom\|Left\|Right` | 保持在底部，水平延伸 |
| 右欄操作按鈕 | `Top\|Right` 或 `Bottom\|Right` | 跟右邊緣 |
| 頂部固定高 ListView | `Top\|Bottom\|Left` | 只垂直延伸 |
| 導航按鈕（年/月/日） | `Bottom\|Left` | 跟圖表底部 |

---

## ProportionalScaler — Form 等比例縮放

Form 使用 `AutoScaleMode = None`（Designer.cs），所有控制項的 Anchor 在 `ProportionalScaler.Initialize()` 時被移除。

- **Initialize**：記錄每個控制項的比例 + `FontSize / FormHeight`
- **OnFormResize**：按比例重算位置/大小/字體（4–72pt 範圍，差距 > 0.5pt 才更新）
- **TabControl**：`SelectedIndexChanged` hook，首次切頁時補記錄延遲頁面
- **重要**：不可混用 Anchor 和 Scaler，否則 `ResumeLayout` 觸發 Anchor 重算會覆蓋 Scaler 設定

---

## UserSessionState 持久化架構

- 檔案路徑：`Config\session-state.json`
- **不使用 `JavaScriptSerializer`**：其構造函數存取 `ConfigurationManager`，若 `user.config` 損毀則拋例外
- 改用自建 `ParseJson` / `SerializeJson` / `EscapeJson`（零外部依賴）
- `Load()` 讀不到時回傳空 dict；檔案在第一次 `Save()` 時才建立
- `WriteToFile` 使用 `FileStream(FileMode.Create, FileShare.ReadWrite)`

### 已知陷阱：user.config 損毀

**症狀**：`session-state.json` 永遠停在 `{}`。
**根因**：`user.config` 含 null bytes → `ConfigurationManager` 失敗 → `JavaScriptSerializer()` 拋例外 → `catch {}` 靜默吞掉。
**修復**：刪除損毀的 `user.config` 並 Rebuild。

---

## Exception Handling 標準

### 專案規則：絕不裸 `catch {}`

```csharp
// ✅ 最小代價：保留 type + message
catch (Exception ex)
{
    Trace.WriteLine($"[ClassName.MethodName] {ex.GetType().Name}: {ex.Message}");
}
```

- `Trace.WriteLine` 在 Debug 和 Release 都有效
- 含路徑加來源，含相機 ID 加 `[CAM{CameraId}]`

---

## PictureBox Bitmap Dispose 陷阱

```csharp
// ❌ 先 Dispose，PictureBox 還持有引用 → Paint 事件爆炸
foreach (var img in cache) img.Dispose();

// ✅ 先清 PictureBox 引用，再 Dispose
galleryManager.ClearImages();          // 所有 PictureBox.Image = null
foreach (var img in cache) img.Dispose();
```

---

## ListView AutoFit 欄寬

```csharp
private static void AutoFitListViewColumns(ListView lv)
{
    for (int i = 0; i < lv.Columns.Count; i++)
    {
        lv.AutoResizeColumn(i, ColumnHeaderAutoResizeStyle.ColumnContent);
        int contentWidth = lv.Columns[i].Width;
        lv.AutoResizeColumn(i, ColumnHeaderAutoResizeStyle.HeaderSize);
        if (contentWidth > lv.Columns[i].Width)
            lv.Columns[i].Width = contentWidth;
    }
}
```

各 ListView 觸發時機：
- **靜態資料**（`listViewEngine`）：`SetupSystemTab()` 末尾一次
- **每 500ms 動態更新**（`listViewCameras`）：第一次 Tick 後一次（`_telemetryFitDone` flag）
- **統計資料**（`listViewStats`、`listViewGrabDetail`）：每次 `RefreshStats()` 後

### FitListViewColumnsProportional

用 `Graphics.MeasureString` 量測標題文字寬度，按比例分配欄寬填滿控制項（無水平捲軸）。
WinForms 限制：Column 0 強制左對齊；Column 1+ 支援 `HorizontalAlignment.Center`。

---

## IEC 60073 訊號燈（WinForms）

| 顏色 | 含義 | 本專案用途 |
|------|------|-----------|
| 綠 `#388E3C` | 正常運轉中 | 相機抓取中 |
| 灰 `#757575` | 待機/中性 | 待機 |
| 紅 `#C62828` | 危險/故障 | 預留：異常 |
| 黃 `#F9A825` | 警告 | 預留：警告 |

架構：`panelStatusBar`（Dock=Top）→ `lblStatusGrab`（Dock=Fill，TextAlign=MiddleRight）。
用 Label 而非 Panel（天生有 Text + BackColor），不用 PictureBox（長條形文字不需圓形 LED）。

---

## GDI+ 無縫拼接圖片

```csharp
g.InterpolationMode = InterpolationMode.NearestNeighbor;
g.PixelOffsetMode   = PixelOffsetMode.Half;
g.DrawImage(src,
    new Rectangle(destX, destY, destW, destH),
    new Rectangle(0, 0, src.Width, src.Height),
    GraphicsUnit.Pixel);
```

GDI+ 有內部 0.5px 偏移 + `HighQualityBicubic` 色彩擴散 → 拼接接縫線。
用 `NearestNeighbor` + `PixelOffsetMode.Half` 修正。
