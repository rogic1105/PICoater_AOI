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

## /perf-diagnose

效能問題排查流程：

1. 先看 Stopwatch 計時輸出（`[FullRes]`、`[OnSelect]` 等）確認瓶頸在哪一段
2. 區分 IO / GPU / UI 三層，對症下藥
3. MIL 相關卡頓：優先排查是否有多執行緒競爭同一 MIL ID
4. CUDA 相關卡頓：確認是否為冷啟動（首次 `cudaMalloc` / `cudaMallocHost`）
5. UI 卡頓：確認 allocation 是否在 UI 執行緒同步執行，改為 `Task.Run` + `await`
