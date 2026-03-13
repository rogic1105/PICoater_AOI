# Envision_MdigGrab — 導讀指南

Matrox MIL 多相機擷取工具，支援 Camera Link + CLProtocol（GenICam）。

---

## 程式進入點

```csharp
// Program.cs
static void Main()
{
    Application.Run(new GrabForm());
}
```

C# WinForms 的慣例：**Main → 建立 Form → 跑起來**。
`Application.Run` 讓視窗開始接收使用者操作，直到關閉為止。

---

## 架構層次

整個專案依職責分層，每層只認識上下一層：

```
GrabForm              ← 視窗（按鈕事件、UI 更新）
    │
    ├─ CameraSession  ← 相機生命週期（Init / Grab / Release）
    │       │
    │       └─ MilCameraUnit × N  ← 每台相機的 MIL 資源
    │               │
    │               └─ MilSystemManager  ← 擷取卡（System）管理
    │
    └─ CameraListViewPresenter  ← ListView 顯示更新
```

**為什麼這樣拆？**
如果全部塞在 `GrabForm` 裡，控制硬體和更新畫面混在一起很難維護。
現在 `GrabForm` 只負責「使用者按了什麼」，硬體細節由 `CameraSession` 和 `MilCameraUnit` 處理，互不干涉。

---

## 從按鈕追蹤一次完整流程

以「Init MIL」按鈕為例，資料如何一層一層往下走：

### GrabForm — button1_Click
```csharp
private void button1_Click(object sender, EventArgs e)
{
    if (_session.HasCameras) return;          // 已初始化就跳過
    _session.Initialize(_configs, checkBoxEnableImageProcessing.Checked);
    _statusTimer.Start();                     // 啟動 500ms 輪詢
}
```
GrabForm 只做兩件事：呼叫 Session、啟動 Timer。

### CameraSession — Initialize()
```csharp
public void Initialize(IList<CameraConfig> configs, bool enableImageProcessing)
{
    MilSystemManager.Initialize();                     // 1. 開啟 MIL Application
    foreach (var cfg in configs)
    {
        sysId = MilSystemManager.AllocateSystem(...);  // 2. 開啟擷取卡
        var cam = new MilCameraUnit(sysId, ...);       // 3. 建立相機物件
        cam.Initialize();                              // 4. 分配相機資源
        cam.SetExposureUs(cfg.ExposureUs);             // 5. 設定曝光
        _cameras.Add(cam);
    }
}
```
Session 的工作是「把設定清單轉換成真實的硬體物件」。

### MilCameraUnit — Initialize()
```csharp
public void Initialize()
{
    MIL.MdigAlloc(..., ref MilDigitizer);    // 開 Digitizer（相機通道）
    MIL.MdispAlloc(..., ref MilDisplay);     // 開顯示器
    MIL.MbufAlloc2d(...);                    // 開影像 Buffer × 多個
    MIL.MdispSelectWindow(..., _panelHandle);// 把影像綁定到 Panel
    StartCLProtocolAsync();                  // 背景啟用 GenICam（不阻塞）
}
```
這層直接跟 MIL 硬體 API 打交道。

---

## Timer 每 500ms 在做什麼

```
Timer Tick
  ├─ _session.UpdatePresence()   ← 問每台相機「你還在線嗎？」
  └─ _listPresenter.Update()     ← 把相機數據刷新到 ListView
```

這個「定時輪詢」模式在工業控制很常見，因為硬體不會主動通知你狀態改變，必須主動去問。

---

## C# 新手重點語法

你在這個 code 裡會常看到以下模式：

**`private readonly` — 只在建構子設定一次，之後不能換**
```csharp
private readonly CameraSession _session = new CameraSession();
```

**`event` + `Action` — 事件通知（發送方不需要知道接收方是誰）**
```csharp
// MilCameraUnit 定義事件
public event Action<int, int, int, int> OnMouseDataChanged;

// CameraSession 轉發給上層
cam.OnMouseDataChanged += (id, x, y, val) => OnMouseDataChanged?.Invoke(id, x, y, val);

// GrabForm 最終接收
_session.OnMouseDataChanged += UpdateGlobalCoordLabel_FromCamera;
```
滑鼠移動時，資料沿著 `MilCameraUnit → CameraSession → GrabForm` 傳遞，每層只認識上下一層。

**`async / await + Task.Run` — 不卡 UI 的背景工作**
```csharp
private async void button3_Click(...)
{
    await _session.ReleaseAsync();  // 背景釋放 MIL 資源，UI 仍可操作
    ResetUI();
}
```

**`volatile bool` — 多執行緒共用的旗標**
```csharp
public volatile bool IsReleasing = false;
```
`volatile` 確保背景執行緒寫入後，UI 執行緒馬上看得到最新值，不會讀到舊的快取。

---

## 建議閱讀順序

### Step 1 — `Config/CameraConfig.cs`
最簡單的檔案，純資料容器，沒有任何邏輯。
先看這裡，認識一台相機需要哪些設定（ID、DCF 路徑、曝光值、顯示 Panel）。

### Step 2 — `UI/GrabForm.cs`
看按鈕事件怎麼呼叫 `CameraSession`，以及 Timer Tick 怎麼更新 UI。
這裡幾乎看不到任何 MIL API，確認「UI 層不碰硬體」的原則。

### Step 3 — `Session/CameraSession.cs`
看 Session 怎麼管理多台相機的生命週期。
重點關注：`Initialize()`、`ToggleGrab()`、`ReleaseAsync()`，以及 `event` 的轉發方式。

### Step 4 — `Hardware/MilSystemManager.cs`
最短的硬體檔案，只有四個 static 方法。
理解 MIL 的初始化順序：`MappAlloc` → `MsysAlloc` → `MsysFree` → `MappFreeDefault`。

### Step 5 — `Hardware/MilCameraUnit.cs`
最複雜的檔案，包含：
- MIL Buffer 分配（Grab Buffer / Display Buffer / Processing Buffer）
- CLProtocol 非同步初始化
- 曝光設定的雙路徑（CLProtocol Feature API vs 傳統 MdigControl）
- MdigProcess Callback（每抓到一幀就觸發）
- GPU 影像處理（CoreCVWrapper）
- Mouse Hook（即時座標回報）

建議先略讀一遍，再配合 Step 6 一起看。

### Step 6 — `UI/CameraListViewPresenter.cs`
看完 `MilCameraUnit` 之後再看這裡。
`Update()` 裡每一個 `cam.GetXxx()` 都對應到 MilCameraUnit 的一個方法，可以對照確認資料來源。
ListView 共 16 欄（索引 0–15），欄位說明見下表。

---

## ListView 欄位速查

| 索引 | 欄位名稱 | 資料來源 |
|------|---------|---------|
| [0] | Camera | CameraConfig.Id |
| [1] | FPS | MdigInquire — M_PROCESS_FRAME_RATE |
| [2] | Target FPS | MdigInquire — M_SELECTED_FRAME_RATE |
| [3] | Line Rate(Hz) | MdigInquireFeature — "AcquisitionLineRate"（CLProtocol）|
| [4] | Exp Set(μs) | _appliedExposureUs（記錄設定值，不回讀硬體）|
| [5] | Exp Meas(μs) | MdigInquireFeature — "ExposureTime"（CLProtocol）|
| [6] | Frames | MdigInquire — M_PROCESS_FRAME_COUNT |
| [7] | Missed | MdigInquire — M_PROCESS_FRAME_MISSED |
| [8] | Grab Miss | MdigInquire — M_GRAB_FRAME_MISSED |
| [9] | Resolution | MdigInquire — M_SIZE_X / M_SIZE_Y |
| [10] | Scan Mode | MdigInquire — M_SCAN_MODE |
| [11] | FPGA(°C) | MsysInquire — M_TEMPERATURE_FPGA（擷取卡）|
| [12] | Cam Temp(°C) | MdigInquireFeature — "DeviceTemperature"（相機）|
| [13] | Mem Free(MB) | MsysInquire — M_MEMORY_FREE |
| [14] | PCIe Lanes | MsysInquire — M_PCIE_NUMBER_OF_LANES |
| [15] | PCIe Speed | MsysInquire — M_PCIE_SPEED |

CLProtocol 初始化完成前（約 1–2 秒），[3][5][12] 欄位會顯示 N/A，屬正常現象。

---

## MIL 資源釋放順序

```
MdigProcess(M_STOP)
  → MdispHookFunction(M_UNHOOK)
  → MbufFree × n
  → MdispFree
  → MdigFree          ← MilCameraUnit.Free()
  → MsysFree × n      ← CameraSession.ReleaseResources()
  → MappFreeDefault   ← MilSystemManager.FreeApplication()
```

順序不能顛倒，否則 MIL 會報錯。
