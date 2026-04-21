# PICoater AOI — 現場部署腳本

檢測機（PICoater 主機）+ 儲存機雙網段部署，讓 `RemoteCopyService` 把存檔送到儲存機。

## 架構

```
┌──────────────┐    192.168.255.x    ┌──────────┐
│  檢測機 PC   │ ─────── PLC ──────→ │ ET-7044  │──→ Nakan
│ (PICoater)   │
│              │    192.168.10.x     ┌──────────┐
│              │ ────── SMB ──────→  │ 儲存機    │  C:\AniloxStorage
│  192.168     │                     │192.168    │
│  .255.x +    │                     │.10.20     │
│  .10.10      │                     └──────────┘
└──────────────┘
        ↑
   單張 NIC 掛兩個 IP（secondary alias）
   未來換雙口工業電腦時，把兩個 IP 拆到兩張實體卡即可，程式不用改
```

## 一次部署流程

### ① 儲存機（一次執行 run_setup.bat 搞定）

1. 把整個 `deploy/storage-pc/` 資料夾複製到儲存機
2. 依現場環境編輯 `storage-config.json`：
   ```json
   {
     "NicName": "乙太網路",
     "IpAddress": "192.168.10.20",
     "PrefixLength": 24,
     "Gateway": "",
     "StorageFolder": "C:\\AniloxStorage",
     "ShareName": "AniloxStorage",
     "AllowedUser": "Everyone"
   }
   ```
   - `NicName`：如果只有一張網卡可不填（自動選）；有 Wi-Fi 請明確指定（中文 OK）
   - `StorageFolder`：改成現場儲存機實際可用的磁碟（例 `"D:\\AniloxStorage"`）
3. **雙擊 `run_setup.bat`**（系統會跳 UAC，同意）
4. 看到 `All Done. Press any key to close...` 代表成功
5. 驗證：在儲存機本機跑 `ipconfig` 確認 IP = 192.168.10.20

**run_setup.bat 會自動做這兩件事**：

| Step 1 — 網路 + SMB 共用（setup_storage_pc.ps1） | Step 2 — 匿名 Guest 存取（setup_guest.ps1） |
|---|---|
| 設固定 IP | 啟用 Guest 本機帳號 |
| 建立 `C:\AniloxStorage` + NTFS Everyone Modify | 授予 Guest SMB Full + NTFS Modify |
| 建立 SMB 共用 `AniloxStorage` | **secedit 把 Guest 從「拒絕網路登入」移除** |
| 開放防火牆 File and Printer Sharing | 加入「允許網路登入」 |
| 網路設定檔切 Private | gpupdate /force |

### ② 檢測機（一次執行 run_setup.bat 搞定）

1. 把整個 `deploy/inspection-pc/` 資料夾複製到檢測機
2. 依現場環境編輯 `inspection-config.json`：
   ```json
   {
     "PlcSubnetPrefix": "192.168.255.",
     "StorageIp": "192.168.10.10",
     "StoragePrefixLength": 24,
     "VerifyPingTarget": "192.168.10.20"
   }
   ```
   - `PlcSubnetPrefix`：檢測機現有 PLC 網段前綴（腳本靠這個找對的 NIC）
   - `StorageIp`：要加的第二個 IP（儲存網段）
3. **雙擊 `run_setup.bat`**（跳 UAC，同意）
4. 看 `ping 192.168.10.20 → OK` 代表儲存機可達

**run_setup.bat 會自動做這兩件事**：

| Step 1 — NIC secondary IP（setup_inspection_nic.ps1） | Step 2 — Client 匿名 SMB（setup_guest.ps1） |
|---|---|
| 找到 PLC 那張 NIC | 登錄檔 `AllowInsecureGuestAuth = 1` |
| 新增 IP 別名 192.168.10.10（不動 PLC IP） | GPO 位置覆寫同樣值 |
| ping 儲存機驗證 | `net use * /delete /y` 清除快取 |

### ③ 驗證連線

在**檢測機** PowerShell（**不用管理員**）：

```powershell
# 1. Port 445 通？
Test-NetConnection -Port 445 -ComputerName 192.168.10.20
#   TcpTestSucceeded : True

# 2. 可匿名寫入？
Out-File -FilePath \\192.168.10.20\AniloxStorage\test.txt -InputObject "hello"
#   無錯誤 → OK

# 3. Win+R 打 \\192.168.10.20\AniloxStorage
#   應直接開啟，不要求帳密
```

### ④ 啟用 PICoater 遠端複製

1. 開 PICoater
2. 右側屬性面板 → **儲存設定 → 遠端路徑** 填：
   ```
   \\192.168.10.20\AniloxStorage
   ```
3. **存檔** 打勾
4. 抓一張圖
5. 到儲存機看 `C:\AniloxStorage\<yyyyMMdd>\...` 有檔案 = 完成

背景複製：`RemoteCopyService` 用 ConcurrentQueue + BelowNormal 執行緒，失敗會重試 3 次，**不影響取像效能**。

---

## 檔案清單

```
deploy/
├── storage-pc/
│   ├── run_setup.bat            ← 雙擊入口（一次跑完兩支 .ps1）
│   ├── storage-config.json      ← 參數（NicName / IP / 資料夾 / 共用名）
│   ├── setup_storage_pc.ps1     ← Step 1：網路 + 共用
│   └── setup_guest.ps1          ← Step 2：Guest 匿名 + secedit
└── inspection-pc/
    ├── run_setup.bat            ← 雙擊入口（一次跑完兩支 .ps1）
    ├── inspection-config.json   ← 參數（PLC 前綴 / 儲存 IP）
    ├── setup_inspection_nic.ps1 ← Step 1：NIC secondary IP
    └── setup_guest.ps1          ← Step 2：Client 匿名 SMB
```

---

## 常見錯誤與修法

| 症狀 | 原因 | 修法 |
|------|------|------|
| `.ps1` 雙擊閃退 | ExecutionPolicy 擋 | 用 `.bat` 啟動（已內建 Bypass） |
| `.bat` 跑「不是內部或外部命令」 | .bat 被 Big5 解 | 重寫為純 ASCII |
| `.ps1` Parse error L39/L42 | 檔案不是 UTF-8 BOM | 另存新檔→編碼 UTF-8 with BOM |
| 腳本報「找不到網卡: 銋云蝬脰楝」 | JSON 中文被 Big5 解 | `.ps1` 已改用 `[System.IO.File]::ReadAllText(UTF8)`；若仍有問題確認 JSON 本身是 UTF-8 |
| 儲存機偵測到多張 NIC 退出 | Ethernet + Wi-Fi 都有 | 編輯 `storage-config.json` 的 `NicName`（例 `"乙太網路"`） |
| 檢測機連 SMB 仍要帳密 | Client 端 AllowInsecureGuestAuth 未套用 | `gpupdate /force` 或重開機 |
| 「登入失敗: 未授與使用者這個電腦所要求的登入類型」 | Guest 被鎖在 `SeDenyNetworkLogonRight` | 重跑儲存機 `run_setup.bat`（Step 2 [5/5] 會修） |
| `EnableGuestAccess` 參數不存在警告 | 新版 Windows 移除該參數 | 可忽略（非關鍵，Guest 帳號 + ACL 已足夠） |
| 抓圖後儲存機沒檔案 | 遠端路徑錯 / 網路暫斷 | 看 PICoater Trace log 有無 `[RemoteCopy] Failed after...` |

---

## 安全考量

此配置為 **內網專用**（檢測機 ↔ 儲存機直連 switch，無對外）。因此：

- 允許匿名 Guest SMB → **不可暴露到網際網路或公司內網**
- 儲存機只連那條儲存 switch，不要插公司 LAN
- `AllowInsecureGuestAuth` 僅開在檢測機（會降低該機 Client 安全性，但本機不會成為攻擊標的）

如果未來要上公司網：改用帳密認證（新增 `inspection` 帳號 + 密碼），拿掉 `setup_guest.ps1` 的匿名設定，同時改寫 `RemoteCopyService` 支援 `net use` 帶憑證。

---

## 相關文件

- `docs/user-manual/ui-flow.html` → 「網路部署」章節（流程圖）
- `.claude/skills/deploy-network.md` → 開發者端技術細節（編碼陷阱、secedit、排查步驟）
