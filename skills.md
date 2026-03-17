# PICoater AOI — Skills & Patterns

專案開發過程累積的可重用知識，補充 `CLAUDE.md` 的規則。

---

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

## MIL 初始化效能原則

### MilCameraUnit 初始化順序（正確）

```
Initialize()：
  MdigAlloc
  MdispAlloc
  CoreCV_MallocGPU × 2      ← GPU device 記憶體（第一次呼叫會觸發 CUDA context init）
  MbufAlloc2d × 4           ← MIL buffer
  MdispSelectWindow / MdispControl / MdispHookFunction
  ← Initialize() 結束，UI 立刻響應

ApplyGrabState()（第一次 MdigGrab）：
  MdigProcess(M_START)
  IsLive = true
  StartCLProtocolAsync()    ← 最後才啟動，避免競爭 MIL 內部鎖
```

### 為什麼 CLProtocol 必須延遲到第一次抓圖？

`MdigControl(M_GC_CLPROTOCOL, M_ENABLE)` 載入 CLProtocol DLL + 讀取相機 GenICam XML，耗時 2–5 秒。
若在 `Initialize()` 期間以 `Task.Run` 啟動，背景執行緒的 `MdigControl` 會與主執行緒的
`MbufAlloc2d`、`MdispAlloc` 競爭 MIL 內部鎖，造成 Init 按鈕卡頓。

**Guard 寫法**：
```csharp
private volatile bool _clProtocolInitStarted = false;

private void StartCLProtocolAsync()
{
    if (_clProtocolInitStarted) return;
    _clProtocolInitStarted = true;
    Task.Run(() => TryEnableCLProtocol());
}
```

### CUDA 冷啟動
第一次呼叫 `CoreCV_MallocGPU`（`cudaMalloc`）會初始化 CUDA context，耗時約 1–2 秒。
若要減少此開銷，可在 `MilCameraUnit.Initialize()` 前先呼叫任意 CUDA 熱身操作
（例如 `AoiService.Initialize()`，如 AniloxRoll.Monitor 的做法）。

---

## MIL 與 GPU 記憶體類型對照

| 類型 | API | 說明 | 適用場景 |
|------|-----|------|---------|
| MIL Buffer | `MbufAlloc2d` | MIL 管理的 Host 記憶體 | MdigProcess 抓圖、MdispSelect 顯示 |
| GPU Device | `CoreCV_MallocGPU`（cudaMalloc） | GPU 顯示卡上的記憶體 | CUDA kernel 直接讀寫 |
| Pinned Host | `CoreCV_AllocPinned`（cudaMallocHost） | CPU 側 DMA 加速記憶體 | H↔D memcpy 高吞吐，如 NativeBufferPool |

MilGrabSample 使用 **GPU Device** 記憶體（二值化 kernel）。
AniloxRoll.Monitor 使用 **Pinned Host** 記憶體（picoater pipeline 大量 DMA 傳輸）。

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

### TrackBar + NumericUpDown 雙向同步

避免互觸無窮迴圈的 pattern（用 captured local bool）：

```csharp
bool syncing = false;
trackBar.ValueChanged += (s, e) => {
    if (syncing) return;
    syncing = true;
    numericUpDown.Value = trackBar.Value;
    // ... 寫回設定
    syncing = false;
};
numericUpDown.ValueChanged += (s, e) => {
    if (syncing) return;
    syncing = true;
    trackBar.Value = Math.Max(trackBar.Minimum, Math.Min(trackBar.Maximum, (int)numericUpDown.Value));
    // ... 寫回設定
    syncing = false;
};
```

`syncing` 是 lambda 捕獲的 local 變數，兩個 lambda 共用同一個 heap slot，C# closure 保證正確。

### AOI.SDK.csproj AllowUnsafeBlocks 陷阱

Solution 將 `Debug|x64` 映射為 `Debug|Any CPU`（Platform="Any CPU" 含空格），導致：
- `.csproj` 中 `Condition="'Debug|AnyCPU'"` 的 PropertyGroup **不被套用**（名稱不符）
- `AllowUnsafeBlocks` 必須放在**無條件的全域 PropertyGroup**：

```xml
<PropertyGroup>
  <StartupObject />
  <AllowUnsafeBlocks>true</AllowUnsafeBlocks>   ← 這裡
</PropertyGroup>
```

---

## Settings 分檔持久化架構

| 檔案 | 對應 Tab | 內容 |
|------|---------|------|
| `Config\inspection-settings.json` | tabPageInspSettings（PropertyGrid） | MachineLayout + Recipe + Storage |
| `Config\acquisition-settings.json` | tabPageCamera（TrackBar） | CameraGrabHeight[7] + CameraExposureTimeUs[7] + CameraLineRateHz[7] |
| `Config\system-settings.json` | tabPageSystem（唯讀） | 相機硬體拓樸 |

- 兩個 JSON 首次執行自動建立（`Load()` 讀不到時 `Save(defaults)` 建檔）
- `AcquisitionSettings` 陣列索引 0 = CAM1 … 6 = CAM7
- 存檔在 **`ValueChanged`** 觸發（不用 `MouseUp`）：TrackBar 快速拖曳放開不在控制項範圍內時 `MouseUp` 不一定觸發，導致遺漏存檔
- **`AcquisitionSettingsStore` 不使用 `JavaScriptSerializer`**：改用手刻 `SerializeJson` / `ParseJson`（Regex 解析陣列，`InvariantCulture` 解析 double），`FileStream(FileMode.Create, FileShare.ReadWrite)`，`Trace.WriteLine` 記錯誤。原因同 `UserSessionState`：`user.config` 損毀時 `new JavaScriptSerializer()` 拋 `ConfigurationErrorsException`，`catch { Debug.WriteLine }` 在 Release 靜默失敗，導致 JSON 永不更新
- **`InspectionSettingsStore` 同樣不使用 `JavaScriptSerializer`**：手刻 `SerializeJson`（逐屬性展開 MachineLayout/Recipe/Storage）+ `ParseJson`（`ExtractObject` Regex 提取巢狀物件，再用 `GetDouble`/`GetFloat`/`GetBool`/`GetString` 逐屬性解析），同樣 `FileStream(FileMode.Create, FileShare.ReadWrite)` + `Trace.WriteLine`

---

## PropertyGrid 隱藏特定屬性

若某個屬性不應在 PropertyGrid 顯示（例如：已有其他 UI 專門控制），在屬性上加 `[Browsable(false)]`：

```csharp
// InspectionSettings.cs
[Browsable(false)]
public AcquisitionSettings Acquisition { get; set; } = new AcquisitionSettings();
```

- Category / DisplayName 等 attribute 可同時移除，`[Browsable(false)]` 優先生效
- 屬性本身仍可正常讀寫（序列化、code-behind 存取不受影響）
- 適用場景：同一設定有兩個 UI 入口時，保留一個入口、隱藏另一個，避免重複設定造成混亂

---

## Git Workflow 規則

**每次 commit / push 前，必須先更新以下兩個檔案：**

1. `CLAUDE.md` — 更新專案架構、設定規則、關鍵檔案速查等內容
2. `skills.md` — 更新開發過程累積的模式、陷阱、可重用知識

確保文件反映最新的程式碼狀態，讓下次對話能快速上手。

---

## UserSessionState 持久化架構

### 設計規則

- 檔案路徑：`Config\session-state.json`（`AppDomain.CurrentDomain.BaseDirectory` 同目錄）
- **不使用 `JavaScriptSerializer`**：其構造函數會存取 `ConfigurationManager`，若 `user.config` 損毀則拋出 `ConfigurationErrorsException`，導致 `WriteToFile` 靜默失敗
- 改用自建 `ParseJson` / `SerializeJson` / `EscapeJson`（純 Regex + StringBuilder，零外部依賴）
- `Load()` 讀不到檔案時回傳空 dict，**不**預建空檔（避免「建立後立刻覆寫」的競爭）；檔案在第一次 `Save()` 時才建立
- `WriteToFile` 使用 `FileStream(FileMode.Create, FileShare.ReadWrite)`，允許其他 process 持有讀取 handle 時仍能寫入

### 儲存時機

| 觸發點 | 呼叫位置 |
|--------|---------|
| 選擇資料夾 | `FormInteractionHelper.SelectAndLoadFolder` — `SetLastDataPath` + `Save()` 在 `LoadDirectory` **之前** |
| 時間篩選確認 | `DateTimeNavigator.SaveCurrentSelection` → `SaveDateTimeSelection` + `Save()` |
| 影像處理開關 | `AniloxRollForm.checkBoxEnableImageProcessing_CheckedChanged` → `SetLastEnableImageProcessing` + `Save()` |

### 已知陷阱：user.config 損毀

**症狀**：`session-state.json` 永遠停在 `{}` 或舊內容，只有重新 Build 後第一次選擇資料夾才寫入。

**根本原因**：
1. `user.config`（`%APPDATA%\Local\AniloxRoll\...\user.config`）含 null bytes（0x00），使 `ConfigurationManager` 初始化失敗
2. `new JavaScriptSerializer()` 構造函數呼叫 `ConfigurationManager.GetSection()`，拋出 `ConfigurationErrorsException`
3. `WriteToFile` 的 `catch {}` 靜默吞掉例外，`session-state.json` 不更新

**修復**：刪除損毀的 `user.config` 並 Rebuild。為防止再次發生，`UserSessionState` 改用不依賴 ConfigurationManager 的自建 JSON 實作。

---

## MuraChart 閾值參考線

`MuraChartHelper.SetThresholds(float mean, float max)` 在 chartMura 畫兩條水平參考線：

- `ErrorValueMax` → **紅色實線**
- `ErrorValueMean` → **紅色虛線**

實作要點：
- **必須用 `StripLine` on `AxisY`（非 Series）**：`StripLine` 自動橫跨全圖，不需 X 座標
- **陷阱：Series + ±1e9 X 座標 → `OverflowException`**：Chart 計算刻度時嘗試覆蓋整個 X 範圍，整數溢位崩潰。每次讀取圖片就會發生
- StripLine 寫法：`IntervalOffset = threshold, StripWidth = 0, Interval = 0, BorderColor = Color.Red`
- 資料曲線：Mean = `DeepSkyBlue` 虛線，Max = `Blue` 實線
- Y 軸上限自動擴展：`AxisY.Maximum = Math.Max(1.0, threshTop * 1.1)`
- 初始化在 `InitializeSystem()` 呼叫 `SetThresholds`；`_propertyGrid_PropertyValueChanged` 亦呼叫

---

## TrackBar 拖曳偵測模式

拖曳期間抑制硬體寫入（避免每個中間值都呼叫 SetGrabHeight / SetExposureUs）：

```csharp
private readonly HashSet<TrackBar> _dragging = new HashSet<TrackBar>();

bar.MouseDown  += (s, e) => _dragging.Add(bar);
bar.MouseUp    += (s, e) =>
{
    _dragging.Remove(bar);
    // 拖曳結束：補送一次硬體寫入
    _liveCameraManager?.SetXxxForCamera(camId, bar.Value);
    ConfigManager.SaveAcquisitionSettings(acq);
};
bar.ValueChanged += (s, e) =>
{
    if (sync || _syncingFromHw) return; sync = true;
    // ... UI 同步 ...
    if (!_dragging.Contains(bar))
    {
        _liveCameraManager?.SetXxxForCamera(camId, bar.Value);
        ConfigManager.SaveAcquisitionSettings(acq);
    }
    sync = false;
};
```

- `HashSet<TrackBar>` per Form（不是 per camera），7 台 TrackBar 共用
- `SetGrabHeight` 特別受益：拖曳期間完全不執行（Buffer 重分配代價高）

---

## Hardware → UI 反向同步（SyncFromCamera 5% hysteresis）

每 500ms 從相機硬體讀回實際值，超過 5% 才更新 UI，防止 CLProtocol 就緒後 UI 顯示舊值：

```csharp
private bool _syncingFromHw = false;  // 防止 ValueChanged 再回寫硬體

// 在 ValueChanged 中加入 _syncingFromHw guard：
if (sync || _syncingFromHw) return;

// SyncCameraParamsFromHardware（Telemetry Timer Tick 呼叫）：
if (!_dragging.Contains(_expBars[idx]))
{
    double hw = cam.GetMeasuredExposureUs();
    if (hw > 0)
    {
        int clamped = Math.Max(bar.Minimum, Math.Min(bar.Maximum, (int)hw));
        double diff = Math.Abs(clamped - bar.Value) / (double)Math.Max(1, bar.Value);
        if (diff > 0.05)
        {
            _syncingFromHw = true;
            bar.Value = clamped; num.Value = clamped;
            acq.CameraExposureTimeUs[idx] = clamped;
            _syncingFromHw = false;
        }
    }
}
```

- `GetMeasuredExposureUs()` 只在 CLProtocol 就緒後回傳非零值
- `GetLineRateHz()` 同理

---

## 即時 Telemetry ListView 架構

`LiveTelemetryPresenter`（移植自 MilGrabSample.CameraListViewPresenter）：

- **16 欄**：Camera / FPS / Target FPS / Line Rate / Exp Set / Exp Meas / Frames / Missed / GrabMiss / Resolution / Scan Mode / FPGA°C / Cam Temp°C / Mem Free / PCIe Lanes / PCIe Speed
- `Initialize(IList<CameraHardwareConfig>)` — 建立欄位 + 初始列（Tag = camId）
- `Update(IReadOnlyList<AniloxCamera>)` — 每 500ms 讀取所有 Telemetry 更新 SubItems
- `ResetAll()` — FreeCameras 後呼叫，所有欄位還原為 "N/A"
- `listViewCameras` 完全由 `LiveTelemetryPresenter` 管理，舊的靜態 5 欄設定已移除
- Telemetry Timer 在 `SetupSystemTab()` 建立（`Interval=500`，永遠運行），Tick 同時呼叫 `Update` 和 `SyncCameraParamsFromHardware`

---

## Exposure 夾緊視覺回饋

LR 改變導致曝光被夾緊時，以 OrangeRed 背景色提醒：

```csharp
private void UpdateExpMaxAndClampColor(int idx, int newMax)
{
    _expBars[idx].Maximum = newMax;
    _expNums[idx].Maximum = newMax;
    if (_expBars[idx].Value > newMax)
    {
        _expBars[idx].Value = newMax; _expNums[idx].Value = newMax;
        _expNums[idx].BackColor = Color.OrangeRed;
    }
    else { _expNums[idx].BackColor = SystemColors.Window; }
}
```

- NUD BackColor（比 ForeColor 視覺更強，容易辨識）
- LR ValueChanged（TrackBar + NUD 兩側）均呼叫此方法

---

## 檢測日誌 CSV 架構

### InspectionLogService

- 路徑：`{CaptureRootPath}\{YYYY}\{YYYYMM}\{YYYYMMDD}.csv`
- 欄位：`Id, FileName, MaxExceed, MeanExceed`（Pass = 兩者均為 0）
- ID 格式：`A00001`（5 位數字，跨日不重置），計數器持久化至 `session-state.json` 的 `LastGrabIdNum`
- `btnCameraGrab_Click` 開始抓取時呼叫 `NextGrabId()` → `_currentGrabId`
- CSV 寫入時機：`AniloxCamera.TrySaveCapture()` 實際存檔後，透過 `OnInspectionResult` 事件逐層傳遞至 Form

### AniloxCamera 影像處理 + 日誌整合

- `ProcessingFunction` **不管 `EnableImageProcessing` 一律執行 GPU 處理**（`TryApplyPicoaterRidge`）
  - `EnableImageProcessing` 只控制「顯示原圖還是處理圖」
  - 目的：即使 checkbox 未勾選也能計算 Mura 曲線 peak 值供 CSV 判斷
- `TryApplyPicoaterRidge` 傳入 `_nativeBufferPool.CurveMeanBuffer` + `CurveMaxBuffer`，讀回 peak 值（`max / 255f`），存入 `_lastMeanPeak` / `_lastMaxPeak`
- `TrySaveCapture` 存檔後觸發 `OnInspectionResult(camId, fileNameNoExt, meanPeak, maxPeak)`
- 事件鏈：`AniloxCamera.OnInspectionResult` → `LiveCameraManager.OnInspectionResult` → `AniloxRollForm.OnCameraInspectionResult` → `InspectionLogService.AppendRecord`
- **前提**：`EnableAutoCapture = true` 才會存檔，才有日誌

### InspectionStatisticsService

- 遞迴掃描 `Directory.GetFiles(root, "*.csv", SearchOption.AllDirectories)`，兩種統計模式：
  - `Compute(root, start, end)`：時間範圍過濾，分母 = 張數，每筆獨立判斷
  - `ComputeByGrabIdRange(root, startNum, endNum)`：序號範圍過濾，分母 = 唯一序號數，一票否決
  - `ComputeDetailedByGrabIdRange(root, startNum, endNum)`：回傳 `List<GrabDetail>`（逐序號×CAM1~7 的 `bool?`）
- `LoadGrabIdInfos(root)` → `List<GrabIdInfo>`（每個序號的 Earliest/Latest 時間）
- `LoadAvailableTimes(root)` → `SortedSet<DateTime>`（全部時間戳，供 cascading comboBox 使用）
- CSV 格式：`Id,FileName,MaxExceed,MeanExceed`；FileName = `YYYYMMDD_HHMMSS-camId`；Id = `A00001`
- CSV 路徑：`{root}\{YYYY}\{YYYYMM}\{YYYYMMDD}.csv`
- 從 FileName 提取 CamId：`fileName.LastIndexOf('-')` 後的數字
- 序號數字提取：`ParseGrabIdNum("A00008")` → `8`（Substring(1) parse int）

### InspectionStatsPresenter（tabPageData）

- 7 個 Panel 卡片：BackColor = 綠(≥95%) / 橙(80-95%) / 紅(<80%) / 灰(無資料)
- `listViewStats` 5 欄彙總：相機 / Pass / Fail / Total / 良率（序號模式下分母=唯一序號數）
- `listView1` 逐序號明細：序號 + CAM1~7（Pass/Fail/—），整行紅底 = 任一 CAM Fail
- 控制項命名：`panelStatCam1`~7，`listViewStats`，`listViewGrabDetail`，`cbGrabIdStart`（序號起），`cbGrabIdEnd`（序號迄），`cbStart/EndYear/Month/Day/Hour/Min/Sec`，`btnQueryStats`，`btnSelectDataFolder`

---

## Designer.cs 控制項批次重命名

`Edit` 工具的 `replace_all: true` 可安全批次替換 Designer.cs 中的控制項名稱：

```
old_string: "panel7"  →  new_string: "panelStatCam1"
```

**安全性確認**：`panel7` 不是 `panel70` / `panel17` 等的子字串（因為後跟的是空格、`.`、`)`、`;` 或 `"`），不會誤替換。

**順序**：先替換較長的數字（`comboBox12` 先於 `comboBox1`），避免 `comboBox1` 誤替換 `comboBox12` 中的部分字元。

---

## ListView AutoFit 欄寬

資料填完後呼叫，取 content 與 header 兩者的較大寬度：

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
- **每 500ms 動態更新**（`listViewCameras`）：第一次 Tick 後執行一次（`_telemetryFitDone` flag），之後不重複（避免閃爍）
- **統計資料**（`listViewStats`、`listView1`）：每次 `RefreshStats()` / `UpdateGrabDetailListView()` 後執行

---

## IEC 60073 訊號燈（WinForms）

工廠 IEC 60073 規範顏色語義：

| 顏色 | 含義 | 本專案用途 |
|------|------|-----------|
| 綠 `#388E3C` | 正常運轉中 | 相機抓取中 |
| 灰 `#757575` | 待機/中性 | 待機 |
| 紅 `#C62828` | 危險/故障 | 預留：異常 |
| 黃 `#F9A825` | 警告 | 預留：警告 |

### 架構：Panel（容器）+ Label（訊號燈）

```
panelStatusBar  Dock=Top, Height=32
  └─ lblStatusGrab  Dock=Fill, TextAlign=MiddleRight, Padding=(0,0,12,0)
```

- `Dock=Top` 保證全寬，不需指定 Width
- `Dock=Fill` + `TextAlign=MiddleRight` = 整條著色 + 文字靠右
- 未來新增訊號燈：在 panel 內加新 Label，設 `Dock=Right` + 固定 Width，從右往左排列
- `lblStatusGrab` 改回 `Dock=Fill` 確保填滿剩餘空間

### UpdateGrabButton 模式

```csharp
private void UpdateGrabButton(bool isGrabbing)
{
    btnCameraGrab.Text = isGrabbing ? "停止抓取" : "開始抓取";
    if (isGrabbing)
    {
        lblStatusGrab.Text      = "● 相機抓取中";
        lblStatusGrab.BackColor = Color.FromArgb(56, 142, 60);   // IEC 綠
        lblStatusGrab.ForeColor = Color.White;
    }
    else
    {
        lblStatusGrab.Text      = "● 待機";
        lblStatusGrab.BackColor = Color.FromArgb(117, 117, 117); // IEC 灰
        lblStatusGrab.ForeColor = Color.White;
    }
}
```

### 為何用 Label 而非 Panel

- `Panel` 本身無 Text 屬性（需塞子 Label，結構更深）
- `Label` 天生有 `Text`、`BackColor`、`TextAlign`，是最簡單的著色文字控件
- `PictureBox` 適合圓形 LED（需 Paint 事件），長條形文字不需要

---

## 壓縮存檔格式（JPEG + .bin）

### 捕獲端（AniloxCamera.TrySaveCapture）

- `UseCompressedCapture=true`：GPU resize（`CoreCV_Resize_GPU`）→ `SaveJpegFromPinned`（需 24bpp 轉換）+ `SaveCurveBin`（自描述 .bin）
- `UseCompressedCapture=false`：`MbufExport(.bmp)`（舊行為）
- JPEG 需要 24bpp：GDI+ JPEG encoder 不支援 8bpp indexed；用 `ImageUtils.Create8bppBitmap` 建 8bpp → `Graphics.DrawImage` 至 24bpp `Bitmap` 再 `Save(JpegCodecInfo)`
- `[ThreadStatic] ImageCodecInfo _jpegCodec`：per-thread cache，避免每幀都 `GetImageEncoders()`

### 回顧端（InspectionEngine.ImageProcessing.cs）

- 路徑末尾 `_raw.jpg` → `LoadFromPrecomputedFiles`；否則 BMP+GPU 路徑（向下相容）
- 非處理模式（curves=null）的 ScaleFactor：`ReadScaleFactorFromBin` 只讀 16 bytes 標頭，不載入整個 float[]
- `IsCompressedJpeg` / `ScaleFactor` 統一由 engine 設定，UI 層直接讀取，**不再從 curve/image 比例推斷**

### ImageRepository 混格式掃描

```csharp
// 同時掃兩種格式，讓不同時期的資料共存
Directory.GetFiles(root, "*_raw.jpg", AllDirectories)
    .Concat(Directory.GetFiles(root, "*.bmp", AllDirectories))
    .ToArray()
```
**陷阱**：舊的 either/or 邏輯（先掃 jpg，有就不掃 bmp）會讓混合資料夾丟失 BMP 檔。

---

## 跨倍率 Canvas View 保存（世界座標法）

### 問題

切換時間段（btnLastPeriod/NextPeriod）若前後圖片倍率不同（BMP 1x vs JPEG 5x），pixel zoom/pan 直接還原會造成視窗跳位。

### 解法：mm 世界座標存/取

**Save（切換前，`_imageScaleFactor` 仍為舊圖）**：
```csharp
double pixelLeft  = (0             - pan.X) / zoom * _imageScaleFactor;
double pixelRight = (_canvas.Width - pan.X) / zoom * _imageScaleFactor;
_savedViewLeftMm  = startPosMm + pixelLeft  * opsInMm;
_savedViewRightMm = startPosMm + pixelRight * opsInMm;
_savedYCenterFraction = (canvas.Height/2 - pan.Y) / zoom / image.Height;
```

**Restore（新圖載入後，`_imageScaleFactor` 已更新為新圖）**：
```csharp
double leftPx  = (savedViewLeftMm  - startPosMm) / (opsInMm * _imageScaleFactor);
double rightPx = (savedViewRightMm - startPosMm) / (opsInMm * _imageScaleFactor);
float zoom = (float)(canvas.Width / (rightPx - leftPx));
float panX = (float)(-leftPx * zoom);
float panY = (float)(canvas.Height / 2 - savedYCenterFraction * newImage.Height * zoom);
canvas.SetView(zoom, new PointF(panX, panY));
```

**呼叫時序保證**（`FormInteractionHelper`）：
```
LoadImages() → SaveViewIfNeeded()       ← 舊 scaleFactor
             → RunWorkflowAsync()
                  → OnGallerySelectionChanged()
                       → SetImageScaleFactor(newSf)  ← 更新
                       → UpdateCanvas(newImage)       ← 用新 sf 還原
```

Fallback：settings 不可用時退回 pixel zoom/pan 直接還原（同倍率仍正確）。

---

## /perf-diagnose

效能問題排查流程：

1. 先看 Stopwatch 計時輸出（`[FullRes]`、`[OnSelect]` 等）確認瓶頸在哪一段
2. 區分 IO / GPU / UI 三層，對症下藥
3. MIL 相關卡頓：優先排查是否有多執行緒競爭同一 MIL ID
4. CUDA 相關卡頓：確認是否為冷啟動（首次 `cudaMalloc` / `cudaMallocHost`）
5. UI 卡頓：確認 allocation 是否在 UI 執行緒同步執行，改為 `Task.Run` + `await`
