# deploy-network

現場部署：檢測機 + 儲存機網路設定（含匿名 SMB Guest 存取）。

## 使用時機

修改 `deploy/` 下腳本、JSON 設定、或到新現場部署儲存電腦/檢測機網路設定時。

## 架構概念

**雙網段**：
- `192.168.255.x` — PLC 控制（ET-7044 + Nakan）
- `192.168.10.x` — 儲存（檢測機 .10 ↔ 儲存機 .20）

**單 NIC 雙 IP 別名**：檢測機只有一張 NIC，用 `New-NetIPAddress` 加 secondary IP（IO IP 不動）。未來換雙口工業電腦時，把兩個 IP 拆到兩張實體卡即可，**程式不用改**。

**匿名 SMB**：`RemoteCopyService` 走 `\\192.168.10.20\Anilox\Captures`，使用 Guest 帳號不帶憑證。Windows 10/11 預設雙邊都擋，需同時改檢測機（Client 端 `AllowInsecureGuestAuth`）+ 儲存機（Server 端 Guest 帳號啟用 + ACL + SeDenyNetworkLogonRight 清除）。

## 檔案清單

| 位置 | 職責 |
|------|------|
| `deploy/storage-pc/storage-config.json` | 儲存機參數（NicName / IpAddress / StorageFolder / ShareName / RdpUser / RdpPassword） |
| `deploy/storage-pc/setup_storage_pc.ps1` | 儲存機主設定（IP / 資料夾 / NTFS / SMB / 防火牆 / Private profile） |
| `deploy/storage-pc/setup_guest.ps1` | 儲存機 Guest 匿名（Server Guest / 本機帳號 / SMB ACL / NTFS / secedit） |
| `deploy/storage-pc/setup_rdp.ps1` | 儲存機遠端桌面（關閉密碼複雜度 / 建 aroll 帳號 / RDP 服務 / 防火牆） |
| `deploy/storage-pc/setup_nosleep.ps1` | 關閉自動睡眠 / 休眠 / 硬碟停轉（powercfg，AC+DC 都設 0） |
| `deploy/storage-pc/run_setup.bat` | 一次跑完四支 .ps1（自動提權 UAC） |
| `deploy/inspection-pc/inspection-config.json` | 檢測機參數（PlcSubnetPrefix / StorageIp / VerifyPingTarget） |
| `deploy/inspection-pc/setup_inspection_nic.ps1` | 自動找 PLC NIC 加 secondary IP |
| `deploy/inspection-pc/setup_guest.ps1` | 檢測機 Client `AllowInsecureGuestAuth = 1` + GPO 覆寫 + 清除連線快取 |
| `deploy/inspection-pc/setup_nosleep.ps1` | 關閉自動睡眠 / 休眠 / 硬碟停轉（與儲存機同一份） |
| `deploy/inspection-pc/run_setup.bat` | 一次跑完 NIC 設定 + Guest SMB + 關閉休眠 |

## 編碼陷阱（必須嚴守）

PowerShell 5.1 + 中文 Windows（Big5 code page）有三層編碼坑，一個弄錯整腳本掛掉：

| 檔類型 | 必須編碼 | 原因 |
|--------|---------|------|
| `.bat` | 純 ASCII | CMD 讀 .bat 用 Big5，任何中文都會被解為控制字元「不是內部或外部命令」 |
| `.ps1` | UTF-8 **with BOM**（EF BB BF） | PS 5.1 沒 BOM 就假設 ANSI（Big5），中文變亂碼 parse error |
| `.json` | UTF-8（含或不含 BOM 都行） | 讀檔程式碼必須用 `[System.IO.File]::ReadAllText(path, UTF8)`，不可用 `Get-Content -Raw \| ConvertFrom-Json`（後者預設 ANSI） |

**寫 .ps1 檔後驗 BOM**：
```powershell
$bytes = [System.IO.File]::ReadAllBytes($path)
"$($bytes[0].ToString('X2')) $($bytes[1].ToString('X2')) $($bytes[2].ToString('X2'))"  # 應該是 EF BB BF
```

**讀 JSON 的正確寫法**：
```powershell
$json = [System.IO.File]::ReadAllText((Resolve-Path $cfgPath).Path, [System.Text.Encoding]::UTF8)
$cfg  = $json | ConvertFrom-Json
```

## 自動提權模板（.bat）

```batch
@echo off
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Requesting admin privileges...
    powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
    exit /b
)
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_xxx.ps1"
pause >nul
```

- 不可用 `.ps1` 雙擊直接跑（ExecutionPolicy 會擋）
- `-ExecutionPolicy Bypass` 只對子 process 生效，不改系統設定
- `pause` 必須在末尾，否則提權重開的 cmd 跑完瞬間關閉

## 儲存機 Guest 匿名關鍵（setup_guest.ps1 [5/5]）

Windows 預設把 Guest 放在**「拒絕從網路存取此電腦」**（SeDenyNetworkLogonRight）。光啟用 Guest 帳號 + 給 SMB/NTFS ACL **不夠**，還要動安全原則：

1. `secedit /export /cfg <tmp>.cfg /quiet` 匯出目前原則
2. 從 `SeDenyNetworkLogonRight` 移除 `Guest`、`Guests`、`*<GuestSID>`、`*S-1-5-32-546`（Guests 群組 SID）
3. `SeNetworkLogonRight` 加入 `Guest`
4. `secedit /configure /db <tmp>.sdb /cfg <tmp>.cfg /areas USER_RIGHTS /quiet` 套用
5. `gpupdate /force /target:computer`

**關鍵陷阱**：secedit 匯出時 Guest 可能用**名字**（`Guest`）或 **`*SID`** 兩種形式出現，**兩種都要比對**。只比 SID 會漏掉名字形式，反之亦然。

```powershell
$guestSid = (New-Object System.Security.Principal.NTAccount('Guest')).Translate([System.Security.Principal.SecurityIdentifier]).Value
$guestTokens = @('Guest', '*' + $guestSid, 'Guests', '*S-1-5-32-546')
# 然後在 Where-Object { $guestTokens -notcontains $_ } 過濾
```

## 關閉自動休眠（setup_nosleep.ps1）

兩台電腦都加：`powercfg /change` 把 `standby-timeout` / `hibernate-timeout` / `disk-timeout` 的 AC 與 DC 值全設 0。

- 儲存機：SMB 必須隨時可被存取，睡眠會讓檢測機存檔失敗
- 檢測機：長時間取像 / RemoteCopyService 背景複製中不可被系統中斷

**不動的設定**：螢幕 timeout（螢幕仍可關閉省電）、電源/蓋子按鈕動作（使用者主動關機仍有效）、hibernate file 本身（保留，只是超時設 0 永不觸發）。

## 驗證順序

1. 儲存機雙擊 `run_setup.bat`（四步驟自動連跑）
2. 檢測機雙擊 `run_setup.bat`（三步驟自動連跑）
3. 檢測機 PowerShell：
   ```powershell
   Test-NetConnection -Port 445 -ComputerName 192.168.10.20  # TcpTestSucceeded = True
   Out-File -FilePath \\192.168.10.20\Anilox\Captures\test.txt -InputObject "hello"  # 無錯誤
   ```
4. PICoater 屬性 → 儲存設定 → 遠端路徑 = `\\192.168.10.20\Anilox\Captures`
5. 抓圖 → 儲存機 `D:\Anilox\Captures\<yyyy>\<yyyyMM>\<yyyyMMdd>\` 出現檔案

**失敗排查**：看 PICoater Trace log 有無 `[RemoteCopy] Failed after...`。

## 常見錯誤 → 修法

| 錯誤 | 根因 | 修法 |
|------|------|------|
| `.ps1` 雙擊閃退 | ExecutionPolicy 擋 | 改用 `.bat` 啟動器 |
| `.bat` 跑「不是內部或外部命令」 | .bat 中文被 Big5 解 | 重寫為純 ASCII |
| `.ps1` Unexpected token parse error | 無 BOM 被當 Big5 | 存成 UTF-8 with BOM |
| `找不到網卡: 銋云蝬脰楝` | JSON 讀取用 Big5 | 改 `[System.IO.File]::ReadAllText(path, UTF8)` |
| `登入失敗: 未授與使用者這個電腦所要求的登入類型` | Guest 在 SeDenyNetworkLogonRight | 跑 `setup_guest.ps1` [5/5] |
| 儲存機偵測多張 NIC 退出 | Ethernet + Wi-Fi 都在 | 編輯 `storage-config.json` 指定 `NicName`（例 `"乙太網路"`） |
| `EnableGuestAccess` 參數不存在警告 | 新版 Windows 移除該參數 | 非關鍵，try/catch 吞掉（重點在 Guest 帳號 + ACL） |
| `Get-LocalGroup : 找不到群組 S-1-5-32-555` | OEM 改動過的 Windows（例 MSI）缺 Remote Desktop Users 群組 | `setup_rdp.ps1` 已容錯：帳號只加 Administrators 即可；Administrator 不需要 RDU 群組也能 RDP |

## RemoteCopyService（對應程式端）

- `Services/RemoteCopyService.cs` — ConcurrentQueue + 背景執行緒，File.Copy 含 3 次重試、間隔 2 秒
- `CameraFrameSaver.OnFilesSaved` 回呼 → `EnqueueFiles`
- 空 `RemotePath` → 靜默略過（不報錯）
- 執行緒優先級 `BelowNormal`，不影響取像

## 文件同步

修改 `deploy/` 內容後：
