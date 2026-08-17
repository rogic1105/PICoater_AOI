# PICoater DVT Runner

`AniloxRoll.DvtRunner` 是外部 WinForms 操作器。它操作真正的
`AniloxRoll.Monitor` 視窗，依 `verify-flows` DVT contract 等待 Flow 證據，
最後執行 `tools/python/check_all_flows.py`。

它不是單元測試，也不取代壓力或長時間測試。用途是把重複的 smoke/DVT 操作自動化，
讓操作員在旁觀察畫面，並把未覆蓋的流程明確列出。

## 測試目錄與真實 Monitor Inspector

Runner 的情境目錄固定分為四個主要責任：`監控`、`回顧`、`報表`、`Bridge`。
跨頁行為以**發出操作的入口**歸類：回顧按鈕影響報表屬回顧；報表按鈕影響回顧屬報表。
一個情境只保留一個主要分類，避免同一段測試在兩處各維護一份。

每個 Scenario JSON 的 `ControlRefs` 只保存 Monitor 的 WinForms 控制項名稱，例如
`btnLiveGrab` 或 `cbReviewId`。Runner 不複製 Monitor Designer，也不再重畫第二套 Monitor
線框。按「選取真實元件」後，Runner 直接從正在執行的 `AniloxRoll.Monitor.exe` 讀取
UI Automation element；滑鼠移到已被 DVT 引用的控制項時會在真實畫面顯示橘框，點一下
只完成篩選並攔截該次點擊，不會觸發 Grab、切換選項或改變產線參數。

PropertyGrid 參數不另外維護平行清單。Runner 從 Scenario 既有的 `set-property` 步驟及
`PropertyGridCoverage.json` 群組推導關聯，再由 UI Automation 辨識真實 PropertyGrid
`DataItem`。平常直接在 Monitor 切換頁籤、捲動 PropertyGrid 或操作畫面；需要查「欄正規值」
等單一參數涵蓋哪些 DVT 時，再開啟一次性選取模式。勾選「跟隨 Monitor 焦點」後，Runner
也會隨真實頁籤、按鈕與參數列焦點自動切換篩選，不攔截正常操作。

因此 Monitor 的排版、縮放或按鈕文字改動不需要同步另一份畫面。控制項若刪除或改名，
Runner 會顯示失效引用，`test_dvt_scenario_catalog.py` 也會失敗，要求 Scenario 明確遷移；
這是 UI Inspector + Object Repository + Test Coverage Map，不是平行開發的第二份 Designer。

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

`監控檢測標準（光源替代刺激）` 會在 Grab 中把光源亮度切為 100 與 255，核對欄／列
Curve、正規值、閾值、欄列曲線判定模式、檢出方向與 O/X 公式。亮暗變化只是穩定的
替代刺激，**不是正式 Mura 模擬，也不能代表真實瑕疵的光學檢出率**。

`監控 IO／布局／檢測功能矩陣` 進一步把 IO 模擬器、真實 Grab、布局與檢測設定放在
同一支情境。它依序測瀑布／即時、上下方向、OPS／Start／Crop、欄／列強化、正規值、
門檻及 O/X；每輪會以實際影像與 Curve peak、主畫面先於 Curve、座標方向和判定公式
做數值驗證。這才是「設定改了以後畫面與 Curve 真的正確」的功能 DVT；
`PropertyGrid 全參數矩陣` 仍只負責輸入、保存和 owner route。
Runner 會先依屬性顯示名稱直接定位列；可寫入欄位以 UI Automation 直接設定，下拉唯讀
欄位再依選項文字選取。若舊版 WinForms accessibility 未暴露下拉選項，才退回
`Home + Down` 鍵盤逐項選取；矩陣報告會列出每種選取方式的實際次數。

`監控 IO 基本一循環` 是上述矩陣的最小前置測試。外部 Runner 啟動
`IoBridge.IoSimulator.exe` 自動模式，只送一次 LOW → HIGH 10 秒 → LOW，將光源設為
255 並開啟存檔，驗證 Grab 開門、首組影像、瀑布首組、Curve、尾幀排水、關門與
`.acap` 封裝；光源底層 `TurnOn` 回傳失敗時直接判 FAIL，不混入布局和檢出參數變化。

`PropertyGrid 全參數矩陣` 由 `PropertyGridCoverage.json` 管理目前 58 個可編輯參數。
Runner 啟動前會用 reflection 對照產品 `InspectionSettings`；產品新增、刪除、改名或調整
同名欄位順序卻未更新目錄時，情境直接 FAIL。一般功能與 Bridge 分開執行：

- `PropertyGrid 全參數矩陣（一般功能）`：角色、OPS／Start／Crop、檢出標準、停止策略、
  顯示報表、儲存與 LOG。每項至少套兩個值，後項會與前項目前值形成累積交叉測試。
- `PropertyGrid 全參數矩陣（Bridge）`：COM／光源及 IO endpoint、啟用狀態和暫停檢出。
  這組會刻意造成短暫斷線與重連，需在設備可被測試時獨立執行。

每次設定必須產生 `ui:設定[internalName]=value` 與緊接的
`setting route internalName ...`。Runner 完成或中止時都會按安全順序還原原值；這一層
證明「全部參數可輸入、可保存、找到唯一 owner」，但需要影像內容的結果仍由監控檢測
標準、背景、回顧、報表與 Bridge 專用情境驗證，不把只有路由成功誤報成功能正確。
`機台角色` 會重建整個 UI，因此放在一般矩陣最後只做一次真實切換；關閉程式後由
Runner 還原測試前備份的 `app-mode.json`，不在已失效的 Accessibility 樹上反向操作。

## 情境規則

- 情境在 `AniloxRoll.DvtRunner/Scenarios/*.json`。
- `Category` 必須為 `monitor`、`review`、`report`、`bridge` 之一。
- `ControlRefs` 必須引用 Monitor Designer 的真實控制項名稱；它同時是真實 UI 選取與反向查詢索引。
- 每個操作步驟必須填 `Contract`，指向 `dvt-contract.md` 的 flow。
- `wait-log` 只寫該步需要的最小證據，不重複完整判定規格。
- `verify-log-absent` 用於證明情境明確禁止的 Flow 行完全未出現，例如純待機 IO 測試不得產生 START／Grab。
- `click-control` 以 WinForms AutomationId 點擊沒有文字的 Chart；目前用來切換監控欄／列強化。
- `verify-monitor-functional-cycle` 對單輪 IO Grab 的主畫面、欄列 Curve、方向及 O/X 做數值驗證。
- 跨步驟、禁止行、數量與完整性仍由 `check_all_flows.py` 判定。
- 若 checker 顯示 `NOT COVERED`，代表這次情境沒有操作到該功能，不代表 PASS。
- `PropertyGridCoverage.json` 的每個群組必須至少被一個情境的
  `exercise-property-group` 使用；漏掉群組時 Runner 連情境清單都不接受。

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

`虛擬 IO 連線與自動重連` 不需要相機、光源或實體 IO。Runner 先讓主程式建立
`127.0.0.1:502` controller，再啟動 `IoBridge.IoSimulator.exe` 的純待機 server；
確認安全交握後讓 server 正常退出，要求主程式判定斷線，最後重開 simulator 並要求
主程式不重啟即可恢復。`--cycles 0` 表示 DI-0=High、DI-1=Low，僅保持
`--initial-delay-ms` 指定的待機時間，不會觸發 Grab。測試結束會還原 IO IP、Port 與啟用狀態。

`儲存電腦五分鐘穩定測試（不 Grab）` 驗證 `\\192.168.10.20\Anilox` 的 SMB 寫入探針、
Storage app heartbeat、UI 綠燈、五分鐘穩定性及正常關閉。產品探針只建立並立刻刪除自己的
`.picoater-write-probe-*`，不讀寫正式影像，也不觸發低磁碟清理。

`SMB 中斷／待傳補送恢復` 第一次執行會安裝固定管理員動作；Runner 之後仍以一般
權限建立 `192.168.10.20/32` 的 Loopback 黑洞路由，只隔離儲存電腦，不停用與 IO 共用的
實體網卡。情境先把 IO 切到本機模擬器。中斷期間完成兩次實際取相，並以 `pending queued` 證明本機 durable marker
已落盤；移除黑洞路由後必須看到分享寫入驗證、backlog 清空及 heartbeat 恢復。
情境成功、失敗或中止都會移除黑洞路由；阻斷與恢復都以新的 TCP `:445` 連線確認。

`低磁碟刪檔與狀態恢復` 不會使用正式 Capture。Runner 固定在
`%TEMP%\PICoater-DVT-Retention` 建立兩天隔離資料，且只有專用 marker 存在時才允許
刪除。門檻會依當下磁碟剩餘量計算；驗收要求只刪最舊完整一天與同日 CSV、保留較新
一天，並逐一完成低磁碟與清理通知的 raise、resolve、ack。無論成功或失敗，最後都會
還原 PropertyGrid 並清除 fixture。

儲存電腦上的程式保活另由 Release ZIP 根目錄 `test_storage_restart.bat` 驗證。工具會
強制關閉指定安裝路徑的儲存程式，等待排程工作在 90 秒內以新 PID 拉起，並要求新 PID
發布有效 heartbeat；預設重複三次，報告寫入 `D:\Anilox\Logs\DvtReports`，結束時保持
儲存程式運行。

`IO／光源軟體斷線恢復` 第一次執行會由
`tests/InstallDvtAdminActions.bat` 要求一次 UAC 授權，安裝六個固定的排程動作：
封鎖／解除實體 IO `192.168.255.1:502`、封鎖／解除儲存電腦
`192.168.10.20:445`，以及停用／啟用光源 `COM17`。之後
`tests/TestRunner.bat` 的選項 10、11 以一般權限執行，不再詢問 UAC。這不是把整個
TestRunner 或可修改的 repo 永久提升成系統管理員；排程只能執行這六個白名單動作。
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

`反覆 Grab 耐久測試` 沿用同一條實體取相流程，以 `High 10 秒 / Low 4 秒` 重複執行。
CLI 的測試分鐘數會換算為完整循環數，最後一定停在 Low，不會為了湊時間截斷一輪。
兩小時的理論值是 514 輪；request、gate、對齊首組、影像後 Curve、gate close、封裝與
遠端待傳都必須逐輪完整。統一測試器每 30 秒另記錄 Working Set、Private Bytes、
handle、GDI/USER、thread、CPU 與 UI Responding。短資格輪通過只證明工具與流程可跑，
不能替代兩小時或八小時耐久結果。

同一個 `PhysicalCapture` 測試入口接著執行 `時間與高度實際取相`。模擬器會在目標
完成前提早送出 START Low：時間模式必須從首幀集合對齊後完整抓取 10 秒；高度模式
必須等所有在線相機共同完成 15,000 列。兩者都不得被提早 Low 截短，並須完成 Curve、
`.acap` 封裝、遠端待傳、設定還原與正常關閉。

情境可用三個通用 helper action 控制同 repo 的外部測試工具：

- `launch-helper`：`Target` 是 repo 相對 exe，`Value` 是命令列參數；以步驟 `Id` 追蹤。
- `wait-helper-exit`：`Target` 指向前述步驟 `Id`，並要求 exit code 0。
- `stop-helper`：提前結束指定 helper；中止／失敗清理也會停止全部 helper。
