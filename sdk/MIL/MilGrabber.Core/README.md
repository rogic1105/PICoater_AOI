# MilGrabber.Core

Matrox MIL 取像／顯示封裝 library。**一台相機 = 一個 `MilCamera`**。
純 library：**無 GUI、無 exe、不依賴 `System.Windows.Forms`** — 任何專案（含非 WinForms）都能引用。

> 給 agent：要在新專案接相機，照本檔「最小接線」與「常見流程」抄即可。
> 判斷一段邏輯該不該寫進這裡的準則：**綁不綁 UI 控制項**。綁 TrackBar/Button → 留你的專案；純對相機下指令 → 進 library。

---

## 提供什麼

| 類別 | 職責 |
|------|------|
| `MilCamera` | 單台相機的 MIL 封裝：alloc / grab / display / 參數 / telemetry / CLProtocol / mouse hook |
| `MilCameraParams` | 純函式參數公式（如曝光動態上限），跨專案**單一真相**，避免公式抄多份分歧 |

## 不提供什麼（呼叫端自己做）

- **MApp / MsysAlloc 生命週期** — 由呼叫端管理（library 刻意不碰 MApp，方便多卡 / 多系統自由配置）
- **多相機協調** — N 台 `MilCamera` 的集合管理、佈局、合圖、selection 由呼叫端做
- **所有 UI** — 顯示面板、參數 slider、拖曳互動（放掉才寫）、按鈕 — WinForms 膠水各專案自寫
- **檢測 / 存檔 / 合圖 / 曲線** — 訂閱 `FrameReady` 後在呼叫端做（library 只到 MIL 範圍）

---

## 最小接線（一台相機）

```csharp
using Matrox.MatroxImagingLibrary;
using MilGrabber.Core;

// 1. 呼叫端管理 MApp + System（library 不碰）
MIL_ID app = MIL.M_NULL, sys = MIL.M_NULL;
MIL.MappAlloc(MIL.M_NULL, MIL.M_DEFAULT, ref app);
MIL.MsysAlloc(app, MIL.M_SYSTEM_RADIENTEVCL, 0, MIL.M_DEFAULT, ref sys);

// 2. 一台相機 = 一個 MilCamera（panelHandle 給顯示用，無顯示可傳 IntPtr.Zero）
var cam = new MilCamera(sys, id: 1, devNum: MIL.M_DEV0, dcfPath: @"D:\Anilox\Dcf\Radient_Config.dcf", panelHandle: panel.Handle);
cam.Initialize();

// 3. 取像（可選：訂閱 FrameReady 自己做檢測/存檔；不訂閱則 library 自動把原圖顯示到 panel）
cam.FrameReady += (c, buf) => { /* 你的檢測 / 存檔 / 合圖 */ };
cam.SetUserGrabIntent(true);   // 想要 grab
cam.ApplyGrabState();          // 依 intent 實際啟動/停止 MdigProcess

// 4. 設參數
cam.SetExposureUs(100);
cam.SetLineRateHz(3000);

// 5. 釋放（呼叫端負責 MsysFree / MappFree）
cam.Dispose();
MIL.MsysFree(sys);
MIL.MappFreeDefault(app, MIL.M_NULL, MIL.M_NULL, MIL.M_NULL, MIL.M_NULL);
```

---

## 常見流程

### A. 曝光動態上限（純公式，不需相機在線）
曝光時間 × 線掃速率 ≈ 固定常數 → 線掃越高，曝光上限越低。常數內建於 library，**min/cap 由呼叫端傳**（各專案 UI 範圍不同）：
```csharp
int expMax = MilCameraParams.CalcExposureMaxUs(lineRateHz: 3000, expMin: 1, expMaxCap: 10000); // = 300
```
> UI 膠水（設 slider.Maximum、夾當前值、拖曳放掉才寫）留呼叫端，但**公式本身只在這裡**。

### B. 抓線掃最大速率（需先 grab + 等 CLProtocol 就緒）
線掃上限透過 CLProtocol GenICam feature 取得；**grab 啟動後約 3 秒** CLProtocol 才在背景就緒：
```csharp
cam.SetUserGrabIntent(true); cam.ApplyGrabState();
// 輪詢等就緒（有連線的相機等到 IsClProtocolEnabled；未連線者跳過）
while (cam.IsConnected && !cam.IsClProtocolEnabled)
    await Task.Delay(500);
double maxHz = cam.GetLineRateMaxHz();   // 回傳 0 = 尚未就緒
```
> 多台相機要「同時等」全部就緒，是**多相機協調**，目前由呼叫端 loop（見範例 `btnFetchInfo`）；library 只提供單台的 `IsClProtocolEnabled` / `GetLineRateMaxHz()`。

---

## 主要 API 速查

| 分組 | 成員 |
|------|------|
| 生命週期 | `Initialize()` / `Dispose()` / `CheckPresence()` |
| 取像 | `SetUserGrabIntent(bool)` / `ApplyGrabState()` / `IsLive` / `IsConnected` |
| 顯示 | `SetPrimaryDisplayVisible(bool)` / `SetSecondaryDisplay(IntPtr)` / `CopyToDisplay` / `PutDisplayBytes` / `GetFrameBytes` / `ClearDisplay()` |
| 參數 | `SetExposureUs` / `SetLineRateHz` / `SetGrabHeight` / `GetLineRateMaxHz()`（CLProtocol） |
| telemetry | `CurrentFps` / `GetCameraTemperature` / `GetFrameCount` / `GetFrameMissed` … |
| CLProtocol | `IsClProtocolEnabled` / `IsHwParamsStable` |
| 事件 | `FrameReady` / `OnMouseDataChanged` / `OnCameraClicked` |
| 公式 | `MilCameraParams.CalcExposureMaxUs(lineRateHz, expMin, expMaxCap)` |

---

## 環境

- .NET Framework 4.8 / **x64**（依賴 AMD64 MIL SDK）
- 參考組件：`Matrox.MatroxImagingLibrary`
- 開發機需插 MIL USB dongle（這也是 MIL 元件獨立成 `sdk/MIL/` 的原因：無 dongle 的機器引用會打不開，整區隔離方便換 grabber）

## 範例

完整多相機監控範例：[`sdk/MIL/samples/MilGrabber.Monitor`](../samples/MilGrabber.Monitor)
（`system-settings.json` 配置 N 台相機跨多張卡，示範佈局 / 選相機 / 參數面板 / 抓取相機資訊一鍵流程）。
