# sdk/Bridges/IoBridge — ET-7044 IO 橋接層

> 巢狀 CLAUDE.md（編 IoBridge 下任何檔時載入）。專案總規則見根 `CLAUDE.md`、sdk 分層見 `sdk/CLAUDE.md`。
> 本檔＝IoBridge 的元件地圖 + app 整合接線 + 「設定改完立即生效」流程。

## 元件地圖

```
sdk/Bridges/IoBridge/
├── IoBridge.Core/                      ← 純 library（無 GUI/exe）
│   ├── IModbusTcpClient.cs             ← 介面（供 IoGrabController mock 注入測試）
│   │     ConnectAsync(ip,port) / ReadDiStatuses()=FC02 / ReadDoStatuses()=FC01 / WriteDo(i,v)=FC05
│   ├── IoModuleFactory.cs              ← 型號→client 單一決策點 Create(model)；新型號加 case
│   └── Modules/IcpDasModbusTcpClient.cs ← ICP DAS 標準 Modbus TCP（原生 socket，unit=1，DI/DO 各 8 bit addr0）
└── samples/                            ← 可執行範例（各自 self-contained，輸出 tools/io/）
    ├── ManualControl/                  ← 手動 DI/DO GUI（client，控制真 ET-7044）
    ├── Automation/                     ← FSM 模擬 GUI（client）
    └── IoSimulator/                    ← ★ Modbus TCP server，模擬 ET-7044 連到 app（測試用）
```

## IO mapping（ET-7044，app IoGrabController 約定）

| 腳位 | 方向 | 意義 |
|---|---|---|
| **DI-0** | PLC→PC | PLC ALIVE（保持 high，app 才認為 PLC 在線） |
| **DI-1** | PLC→PC | **START（grab 訊號）**：上升緣→開抓、下降緣→停抓 |
| DO-0 | PC→PLC | PC ALIVE（Form 開啟=High） |
| DO-1 | PC→PLC | MURA（檢測到瑕疵=High；不中斷取像） |
| DO-2 | PC→PLC | PC BUSY（Grab 中=High） |

## app 整合（src/AniloxRoll.Monitor）

- **`Services/IoGrabController.cs`**：背景 Modbus 輪詢 loop（Task.Run，不依賴 message pump）。連線時以 DI-1 START 邊緣控制 grab 啟停；未連線退回 UI 按鈕。注入 `IModbusTcpClient`（可 mock 測試）。
- **`UI/Form/AniloxRollForm.HardwareStatus.cs` `InitIoController()`**：`new IoGrabController(IoModel)` + 接事件（OnStart/Stop/StateChanged/ConnectionChanged/IoUpdated）+ `StartAsync(IoIp, IoPort)` 背景連線。`IoEnabled=false` 時 early-return（不建）。

## ★ IO 設定改完「立即生效」流程（不用重開程式）

改 **IO IP / Port / 型號 / 啟用** 在 PropertyGrid → 走 SSoT：
```
SettingsHub.Changed → AniloxRollForm.OnSettingChanged(c)
  → HandleIoSettingsChanged(c.Name)            // HardwareStatus.cs
      case IoIp / IoPort / IoModel / IoEnabled → RestartIoController()
  → RestartIoController()                       // 停舊(StopAsync+Dispose) → InitIoController()(用新設定重建+背景重連)
```
- **新增「要立即生效」的 IO 設定** → 加進 `HandleIoSettingsChanged` 的 case 即可（別在別處 inline 重啟）。
- `RestartIoController` 是 `async void`（StopAsync 背景跑、不阻塞 dispatcher）；重建期間 `_ioGrabController` 短暫 null → 讀取端一律 `_ioGrabController?.`（null-safe）。
- 改的瞬間先 `UpdateIoConnectionUi(false)`＝顯示斷線/重連中，避免殘留舊 IP 的「已連線」假象。

> 此「設定改完立即生效」是專案 SSoT 慣例（[[feedback_settings_as_single_source]]）：UI 控制項只是入口，副作用（重啟 controller）由 OnSettingChanged 訂閱者跑，不在 PropertyGrid handler inline。光源（HandleLightSettingsChanged）同模式。

## IoSimulator（samples/IoSimulator）— 測試用 Modbus server

模擬 ET-7044 連到 app 做長期循環取像測試（不需真硬體）。Modbus TCP server 回應 client 的 FC01/02/05；GUI 手動切 DI + 自動循環 DI-1 START（拍 N 秒/停 M 秒）+ 顯示 app 寫回的 DO。
- 用法：跑 IoSimulator（**502<1024 需系統管理員**；或用高 port 如 1502 並把 app IO Port 一起改）→ 啟動 server → app 設 IO IP=`127.0.0.1` → 開始循環。
- icon 用官方 `sdk/tools/icon-gen/make_icon.py`（藍＝Bridge 工具）。

## Build

一律 `Release|x64`。`sdk/Bridges/IoBridge/*.sln` 或主 `PICoater_AOI.sln`（已收四個 IoBridge sample 含 IoSimulator）。`.Core` 輸出位置不可改（共用 bin 是刻意設計）。
