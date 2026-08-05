# sdk/Bridges/IoBridge — ET-7044 IO 橋接層

> 編 IoBridge 下任何檔時由 Codex 疊加載入。專案總規則見根 `AGENTS.md`、sdk 分層見 `sdk/AGENTS.md`。
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
- **`UI/Form/AniloxRollForm.IoControl.cs` `InitIoController()`**：`new IoGrabController(IoModel)` + 接事件（OnStart/Stop/StateChanged/ConnectionChanged/IoUpdated）+ `StartAsync(IoIp, IoPort)` 背景連線。`IoEnabled=false` 時 early-return（不建）。

### 重連可用性定義（不是 TCP connected 就算成功）

```
ReconnectTick
  → IcpDasModbusTcpClient.ConnectAsync             // TCP
  → TryAcceptConnectedModule
      → EnterIdle                                  // DO1=0 → DO2=0 → DO0=1（ALIVE 最後發布）
      → ReadDiStatuses                             // 至少一筆合法 Modbus DI 回應
  → OnConnectionChanged(true)                     // 兩關都過才發布綠燈
```
- read/write 逾時：立即關 socket，晚到 NetworkStream task 必由 `ObserveLateFault` 收走；不得進 process-wide
  `UnobservedTaskException`。`.NET Framework` 的 NetworkStream async 無可用 cancellation token，故以 close+observe 收口。
- connect 使用 `SocketAsyncEventArgs`，不得退回 `TcpClient.ConnectAsync` + timeout Close；後者在 .NET Framework
  會讓晚到 `TcpClient.EndConnect` 產生 first-chance `ObjectDisposedException/NullReferenceException`。Connect timeout
  關 socket 後須等 SAEA completion，再解除事件並 Dispose args。
- `Dispose` 會遞增 transport generation；進行中的 connect 即使晚到成功也不得重新發布 socket（切 IP/關程式競態）。
- `IoGrabController.IsConnected` 是安全交握完成後的 accepted gate；抓取/MURA 業務輸出一律讀這個狀態，
  不得直接用底層 `_plc.IsConnected`。底層狀態只供 transport 生命週期（背景 poll/reconnect 與停止清輸出）使用。
- app 先開、IO 後上電必須能自行恢復，不得要求重開 app。共用 log 目錄的 `io-*.log` 於第 1 次及每 10 次失敗留下
  `IO reconnect pending`，成功行帶 attempt；其餘輪次靜默，兼顧可診斷性與全天 log 量。
- `ReconnectIntervalMs` 定義為兩次 connect **起點**的間隔；TCP timeout 已算在週期內，不得 timeout 後再等完整週期
  （否則 UI 顯示 3s、實際最差 6s，會造成重開 app 反而較快的假象）。
- 不用 exponential backoff：單一區網設備恢復後要快速接回；既有 `500ms poll / 3s retry` 保留，可靠性靠生命週期與交握，不靠更密集重試。

## ★ IO 設定改完「立即生效」流程（不用重開程式）

改 **IO IP / Port / 型號 / 啟用** 在 PropertyGrid → 走 SSoT：
```
SettingsHub.Changed → AniloxRollForm.OnSettingChanged(c)
  → HandleIoSettingsChanged(c.Name)            // IoControl.cs
      case IoIp / IoPort / IoModel / IoEnabled
  → RestartAsync@IoConnectionCoordinator        // requested generation 立即使舊 callback 失效
  → lifecycle gate                              // 快速連續設定只保留最後一代
  → StopAsync+Dispose 舊 controller             // 等初次 connect 收口後才釋放
  → 建立 IoGrabController + StartAsync           // 用新設定重建+背景重連
```
- **新增「要立即生效」的 IO 設定** → 加進 `HandleIoSettingsChanged` 的 case 即可（別在別處 inline 重啟）。
- controller lifecycle 的 `SemaphoreSlim + generation` 只由 `IoConnectionCoordinator` 擁有；舊 generation
  的 START/STOP/狀態/LED callback 先在 coordinator 截止，Form marshal 前後再驗 current，不得碰目前 UI 或 Grab。
- 改的瞬間先 `UpdateIoConnectionUi(false)`＝顯示斷線/重連中，避免殘留舊 IP 的「已連線」假象。
- 同一台電腦只能有一個 app process；`Program` 的 named mutex 在 Form 建立前擋掉第二份，避免兩個
  controller 同時輪詢同一台 ET-7044、同一個 DI 邊緣觸發兩次 Grab。

> 此「設定改完立即生效」是專案 SSoT 慣例（[[feedback_settings_as_single_source]]）：UI 控制項只是入口，副作用（重啟 controller）由 OnSettingChanged 訂閱者跑，不在 PropertyGrid handler inline。光源（HandleLightSettingsChanged）同模式。

## IoSimulator（samples/IoSimulator）— 測試用 Modbus server

模擬 ET-7044 連到 app 做長期循環取像測試（不需真硬體）。Modbus TCP server 回應 client 的 FC01/02/05；GUI 手動切 DI + 自動循環 DI-1 START（拍 N 秒/停 M 秒）+ 顯示 app 寫回的 DO。
- 用法：跑 IoSimulator（標準 Modbus TCP Port `502`）→ 啟動 server → app 的 IO Port 固定 `502`、
  IO IP 設 `127.0.0.1` → 開始循環。切回實體 ET-7044 時只把 IO IP 改回 `192.168.255.1`。
  若 502 啟動失敗，先檢查是否被其他 Modbus server 占用；`1502` 僅作衝突排查的備援 Port。
- icon 用官方 `sdk/tools/icon-gen/make_icon.py`（藍＝Bridge 工具）。

### 無人值守 IO 循環

`IoBridge.IoSimulator.exe --auto` 不開 GUI，依參數送出 DI-1 High/Low 後自行結束：

```powershell
IoBridge.IoSimulator.exe --auto --port 502 --cycles 3 `
  --initial-delay-ms 20000 --high-ms 10000 --low-ms 4000 `
  --exit-delay-ms 5000 --result-file D:\Anilox\Logs\io-simulator-dvt.txt
```

DI-0 全程為 High。初始延遲用來等待 app 完成 Modbus 連線；每個 High/Low 的實際時間
與最後 DO 狀態會寫進 result file。此入口由 DVT Runner 的實際取相情境使用，
不得把它當成產品 IO 時序的實作來源。

## Build

一律 `Release|x64`。`sdk/Bridges/IoBridge/*.sln` 或主 `PICoater_AOI.sln`（已收四個 IoBridge sample 含 IoSimulator）。`.Core` 輸出位置不可改（共用 bin 是刻意設計）。

重連變更至少跑：Unit `IoGrabControllerTests`、Integration `IcpDasModbusTcpClientIntegrationTests`
（含 app 先開/server 延後上線）、Stress category `BridgeStress`（真 socket 對端斷線/重連 100 輪）。
