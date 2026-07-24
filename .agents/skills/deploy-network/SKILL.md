---
name: deploy-network
description: Configure or troubleshoot inspection-PC and storage-PC networking and deployment. Use for deploy scripts, dual-subnet NIC settings, SMB Guest access, RDP, firewall, and storage connectivity.
---

# deploy-network

處理檢測電腦與儲存電腦的安裝、更新、網路、SMB、RDP 與長時間運轉設定。

## 使用者模型

正式 Release 壓縮包根目錄只有一個操作入口與兩份說明：

- `setup.bat`：依 `AppDir\AniloxRoll.Monitor.exe` 是否存在，自動選擇第一次完整安裝或後續程式更新。
- `manual-install.html`：BAT 不能執行時的 PowerShell 與完全手動備援。
- `VERSION.txt`：封裝來源版本。

統一的是使用者入口，不是兩種行為：第一次安裝可設定 Windows/網路；既有安裝只能更新程式、角色、捷徑與必要排程，不得重設網卡、SMB、RDP 或電源。底層 `install.ps1` 與 `update_app.ps1` 保持分離。安裝參數只存在角色 JSON：
`deploy/storage-pc/storage-config.json` 或
`deploy/inspection-pc/inspection-config.json`。

## 目錄責任

| 路徑 | 責任 |
|---|---|
| `deploy/common/Deploy.Common.ps1` | 讀設定、安裝 app payload、保留 runtime Config、寫角色設定 |
| `deploy/common/setup_nosleep.ps1` | 兩台電腦共用的睡眠/休眠/硬碟停轉設定 |
| `deploy/storage-pc/setup.*` | 儲存電腦統一入口，自動選擇首次安裝或更新 |
| `deploy/storage-pc/install.ps1` | 儲存電腦首次安裝機制，由 setup 或手動救援呼叫 |
| `deploy/storage-pc/update_app.ps1` | 儲存電腦程式更新機制，由 setup 或手動救援呼叫 |
| `deploy/storage-pc/scripts/` | 儲存 NIC/SMB、Guest、RDP、Storage role、排程工作機制 |
| `deploy/inspection-pc/setup.*` | 檢測電腦統一入口，自動選擇首次安裝或更新 |
| `deploy/inspection-pc/install.ps1` | 檢測電腦首次安裝機制，由 setup 或手動救援呼叫 |
| `deploy/inspection-pc/update_app.ps1` | 檢測電腦程式更新機制，由 setup 或手動救援呼叫 |
| `deploy/inspection-pc/scripts/` | IO/儲存網段、Guest client、Inspection role 機制 |
| `deploy/package/package_release.ps1` | 產生乾淨 Storage/Inspection Release zip |
| `deploy/package/rebuild_and_package.bat` | 唯一互動式打包入口；自動判斷正式/測試版，確認後 Rebuild 並產生兩包 |

詳細操作與手動步驟由各角色的 `manual-install.html` 擁有；不要在 skill 複製整份操作手冊。

## 網路不變量

- `192.168.255.x`：PLC、IO、光源等控制網段。
- `192.168.10.x`：檢測電腦與儲存電腦傳輸網段。
- 檢測電腦使用同一 NIC 時，儲存網段必須以 secondary IP 加入，不得刪除既有 PLC IP。
- 儲存電腦共用 `AniloxRoot`，`Captures_pack` 與 `Config` 是其子目錄。
- Storage 角色循環儲存門檻來自 `storage-config.json` 的 `StorageMinFreeGB`，安裝時寫入 `app-mode.json`；不得誤用檢測電腦的 `LocalMinFreeGB`。
- 連線健康必須分開判斷 SMB share 可寫與 Storage app heartbeat；只有兩者皆正常才是綠燈。

## App 更新不變量

- `Config/*.json` 是執行期設定，更新時保留且不放入 Release payload。
- 兩台程式預設都在 `C:\AniloxMonitor`；儲存電腦的大量產出才放 `D:\Anilox`。從舊 `D:\AniloxMonitor` 升級時，只在新 Config 尚未建立時複製舊 Config，不反向覆蓋新設定。
- `Config/Radient_Config.dcf` 是不會由 defaults 產生的 MIL 二進位設定，只放 Inspection package。
- 先驗證 payload 有 `AniloxRoll.Monitor.exe`，再停止目標 `AppDir` 的程式。
- 只刪除前一版 `deploy-manifest.txt` 登記的檔案，不掃除影像、CSV 或未知使用者檔案。
- 兩種角色安裝/更新 payload 後，都必須冪等建立 Public Desktop 的 `PICoater AOI.lnk`；Storage 再重新登記並啟動排程工作，Inspection 另建立儲存資料捷徑且不改網路/系統設定。
- Storage package 不含 `Matrox.MatroxImagingLibrary.dll`；Inspection package 必須包含。
- 每個複製到 `AppDir` 的 payload 檔案必須 `Unblock-File` 並確認無 `Zone.Identifier`；不得把下載封鎖誤判為程式異常。
- ZIP 必須只有一個與包名相同的版本根資料夾；`setup.bat`、`app/` 與 `deploy/` 全部在其下，解壓時不得散落。
- ZIP 驗證成功後刪除 staging 資料夾；`artifacts/deploy/` 完成態只保留 `.zip` 成品。

## Guest SMB

現行設備網段使用 Guest SMB：

- 儲存端：啟用本機 Guest、share ACL、NTFS Modify、
  `SeNetworkLogonRight` 加入 Guest、`SeDenyNetworkLogonRight` 移除 Guest/Guests。
- 檢測端：`AllowInsecureGuestAuth=1` 同時寫一般與 Policies registry 位置，並清除舊 `net use` 快取。

這是封閉設備網段的部署契約。若公司政策禁止 Guest，不得繞過政策；另案改成專用帳號與認證。

## Storage App 自舉

`configure_app_task.ps1` 建立互動式登入排程：

- 任一使用者登入時觸發，執行主體為 `BUILTIN\Administrators`；目前登入的本機系統管理員承接可見 UI。
- 另加每分鐘保活觸發；程式仍執行時由 `MultipleInstances=IgnoreNew` 略過，正常關閉或異常退出後最晚一分鐘重開。
- `RdpUser` / `RdpPassword` 只負責建立遠端桌面帳號，不得作為排程 SSoT 或綁死啟動帳號。
- WorkingDirectory 必須是 `AppDir`。
- 最高權限執行。
- 工作失敗由 RestartCount 處理；正常關閉由每分鐘保活觸發處理，並忽略重複執行個體。

Storage role 每五秒發布 heartbeat；程式沒跑和 SMB 不通是不同故障。
排程安裝後必須立即啟動，並在 10 秒內以完整 EXE 路徑確認真正 process 存活；只看 `Start-ScheduledTask` 沒拋錯不算成功。
`LastTaskResult=267011 (0x00041303)` 表示排程尚未執行；不得誤判為 Smart App Control 或 DLL 封鎖。
檢測機首次安裝必須在 Public Desktop 建立來自 `VerifyPingTarget + StorageShareName` 的 SMB 捷徑。
兩種角色的 `PICoater AOI.lnk` 必須指向實際 `AppDir\AniloxRoll.Monitor.exe`，WorkingDirectory 與圖示也必須使用同一個 `AppDir`；不得寫死 C 或 D 槽。

## PowerShell 5.1 編碼

- `.bat`：ASCII，避免 CMD Big5 解析中文失敗。
- `.ps1`：UTF-8 **with BOM** (`EF BB BF`)；PS 5.1 對無 BOM UTF-8 中文會誤判 ANSI。
- `.json`：以 `[System.IO.File]::ReadAllText(path, Encoding.UTF8)` 讀取。

驗證 BOM：

```powershell
$b = [IO.File]::ReadAllBytes($path)
'{0:X2} {1:X2} {2:X2}' -f $b[0],$b[1],$b[2]
```

## 封裝

UI 操作只保留一個入口：

```text
deploy\package\rebuild_and_package.bat
```

它先檢查 Git 狀態：clean 時顯示 `OFFICIAL RELEASE`，dirty 時顯示
`SMOKE TEST PACKAGE`。使用者確認後才執行 `Release|x64 Rebuild` 與兩個角色包。
Rebuild 只清理 MSBuild 管理的 repository 編譯輸出，封裝白名單仍負責排除共享輸出目錄的其他殘留檔案。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\deploy\package\package_release.ps1 -Role Storage
powershell -NoProfile -ExecutionPolicy Bypass -File .\deploy\package\package_release.ps1 -Role Inspection
```

封裝前必須 `Release|x64` build。輸出在 `artifacts/deploy/`，不得把整個
`bin/x64/Release` 直接壓縮，因為其中可能含測試 DLL、PDB、log、bench、runtime JSON 與原生 `.lib`。
正式包預設拒絕 dirty worktree；只有臨時上機煙測可明確加 `-AllowDirty`，且 `VERSION.txt` 必須標示來源狀態。

## 驗證順序

1. Parse 所有 `.ps1`，確認 BOM，parse 兩份 JSON。
2. 產生 Storage 與 Inspection package，檢查禁止檔案為零。
3. 全新儲存電腦跑 `setup.bat`，確認自動選擇 Install 且 IP/share/Guest/RDP/task/power 正常。
4. 全新檢測電腦跑 `setup.bat`，確認自動選擇 Install 且 IO IP、secondary IP、Guest client 正常。
5. 檢測端測 TCP 445、SMB create/write/flush/delete。
6. 開兩邊 app，確認 share 與 heartbeat 都正常。
7. 再跑一次 `setup.bat`，確認自動選擇 Update，且 Config、captures、網路與 Windows 設定未變。

部署或連線行為變更時，同步更新 `$verify-flows` 的 H/C 契約；純資料夾整理不改 DVT 行為。
