# PICoater DVT Runner

`AniloxRoll.DvtRunner` 是外部 WinForms 操作器。它操作真正的
`AniloxRoll.Monitor` 視窗，依 `verify-flows` DVT contract 等待 Flow 證據，
最後執行 `tools/python/check_all_flows.py`。

它不是單元測試，也不取代壓力或長時間測試。用途是把重複的 smoke/DVT 操作自動化，
讓操作員在旁觀察畫面，並把未覆蓋的流程明確列出。

## 執行

1. 以 `Release|x64` 建置 `AniloxRoll.DvtRunner`。
2. 開啟 `bin/x64/Release/AniloxRoll.DvtRunner.exe`。
3. 確認監控程式與 `D:\Anilox\Logs` 路徑。
4. 選擇情境後按「開始」。Runner 會開啟或接上已開啟的監控程式。
5. 需要停下觀察時按「暫停」；「中止」會嘗試停止 Grab 並還原被修改的 PropertyGrid 設定。
   失敗清理會先等待主程式正常關閉最多 60 秒；仍無法退出才強制結束測試程序，並把正常關閉判為未通過。
6. 情境完成後會先還原設定，再正常關閉監控程式，確認 shutdown Flow 後顯示結果。

初次使用先跑 `Runner 自我檢查（不 Grab）`。它只驗證 UI 連接、設定提交、
Flow tail 與還原，不會啟動相機。

`監控與背景 V1` 若相機或光源尚未就緒，會停在可用性守門持續等待；接妥後自動繼續，
也可隨時按「中止」。這類等待不使用固定秒數，避免不同電腦初始化速度造成假失敗。

## 情境規則

- 情境在 `AniloxRoll.DvtRunner/Scenarios/*.json`。
- 每個操作步驟必須填 `Contract`，指向 `dvt-contract.md` 的 flow。
- `wait-log` 只寫該步需要的最小證據，不重複完整判定規格。
- 跨步驟、禁止行、數量與完整性仍由 `check_all_flows.py` 判定。
- 若 checker 顯示 `NOT COVERED`，代表這次情境沒有操作到該功能，不代表 PASS。

## V1 範圍

`監控與背景 V1` 自動執行：

- 切換流程驗證記錄。
- 暫時停用 IO 自動控制並切到標準去背。
- 取得背景、確認輸出被抑制、預覽與清除。
- 正常 Grab/Stop，確認首組相位、主畫面與 Curve 時序。
- 執行完整 Flow checker。

實體拔線、畫面內容是否符合肉眼預期，以及未納入的回顧、報表、Bridge 流程仍需後續情境。

## 無人值守模式

統一測試入口會以 CLI 執行不需接線的 Runner 自我檢查：

```powershell
AniloxRoll.DvtRunner.exe --scenario runner-self-check --result-file result.txt
```

Runner 會開啟主程式、執行情境、正常關閉主程式、寫入 PASS/FAIL 結果，最後自行結束。
需要相機、光源或 IO 的情境不會被離線測試誤判為通過。

`實體 IO 五分鐘穩定測試（不 Grab）` 只驗證 `192.168.255.1:502` 的安全交握、待機輪詢、
controller 關閉與 Flow checker；不按開始抓取。它仍會依產品協定輸出
MURA/BUSY Low 與 PC ALIVE High。

`儲存電腦五分鐘穩定測試（不 Grab）` 驗證 `\\192.168.10.20\Anilox` 的 SMB 寫入探針、
Storage app heartbeat、UI 綠燈、五分鐘穩定性及正常關閉。產品探針只建立並立刻刪除自己的
`.picoater-write-probe-*`，不讀寫正式影像，也不觸發低磁碟清理。

`SMB 中斷／待傳補送恢復` 需以系統管理員執行。Runner 會找出通往
`192.168.10.20` 的網卡並暫時以軟體停用；情境先把 IO 切到本機模擬器，因此不依賴同張
網卡上的實體 IO。中斷期間完成兩次實際取相，並以 `pending queued` 證明本機 durable marker
已落盤；恢復網卡後必須看到分享寫入驗證、backlog 清空及 heartbeat 恢復。
情境成功、失敗或中止都會嘗試重新啟用該網卡。

儲存電腦上的程式保活另由 Release ZIP 根目錄 `test_storage_restart.bat` 驗證。工具會
強制關閉指定安裝路徑的儲存程式，等待排程工作在 90 秒內以新 PID 拉起，並要求新 PID
發布有效 heartbeat；預設重複三次，報告寫入 `D:\Anilox\Logs\DvtReports`，結束時保持
儲存程式運行。

`IO／光源軟體斷線恢復` 第一次執行會由
`tests/InstallDvtAdminActions.bat` 要求一次 UAC 授權，安裝四個固定的排程動作：
封鎖／解除實體 IO `192.168.255.1:502`，以及停用／啟用光源 `COM17`。之後
`tests/TestRunner.bat` 的選項 11 以一般權限執行，不再詢問 UAC。這不是把整個
TestRunner 或可修改的 repo 永久提升成系統管理員；排程只能執行這四個白名單動作。
Runner 仍會在成功、失敗或中止時解除規則並重新啟用串口。

IO 與光源各跑三輪，必須逐輪看到斷線、`OutputHealth raise`、恢復連線及
`resolve`。這是可重複的軟體故障注入，不能取代最終版本的一次實體拔線／斷電。
IO 每輪使用本機 Loopback `/32` 黑洞路由，並以新 TCP 連線確認端點真的被阻斷。
光源每輪先關閉 controller 釋放
COM17，再停用裝置並重開 controller，避免 Windows 因串口仍被占用而拒絕停用。
需要移除預先授權時，以系統管理員 PowerShell 執行：
`Get-ScheduledTask PICoater-DVT-* | Unregister-ScheduledTask -Confirm:$false`。

`IO＋儲存電腦待機耐久測試（不 Grab）` 同時守住 IO「待機」與儲存電腦綠燈。CLI 的
`--duration-seconds N` 會覆寫情境內的 `soak` 時間；統一測試器每 30 秒記錄主程式
Working Set、Private Bytes、handle、thread、CPU 與 Responding。這些資料用來找資源洩漏與
連線抖動，不等於硬體壽命估算。

`IO 實際取相三循環` 會把主程式暫時切到 `127.0.0.1:502`，由 Runner 啟動
`IoBridge.IoSimulator.exe --auto`，送出三次 10 秒 START High。每輪都必須依序看到：
IO 請求、capture gate、首組相位、影像後 Curve、Low 後尾幀排水、gate 關閉、`.acap`
封裝與遠端待傳。成功或失敗時 Runner 都會關閉模擬器、停止可能仍在進行的 Grab、
還原 PropertyGrid 設定並關閉主程式。

同一個 `PhysicalCapture` 測試入口接著執行 `時間與高度實際取相`。模擬器會在目標
完成前提早送出 START Low：時間模式必須從首幀集合對齊後完整抓取 10 秒；高度模式
必須等所有在線相機共同完成 15,000 列。兩者都不得被提早 Low 截短，並須完成 Curve、
`.acap` 封裝、遠端待傳、設定還原與正常關閉。

情境可用三個通用 helper action 控制同 repo 的外部測試工具：

- `launch-helper`：`Target` 是 repo 相對 exe，`Value` 是命令列參數；以步驟 `Id` 追蹤。
- `wait-helper-exit`：`Target` 指向前述步驟 `Id`，並要求 exit code 0。
- `stop-helper`：提前結束指定 helper；中止／失敗清理也會停止全部 helper。
