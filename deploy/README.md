# PICoater 部署 / Deployment

`deploy/` 負責檢測電腦與儲存電腦的 Windows、網路及程式部署。產品執行邏輯仍屬於 app，可重用的傳輸機制仍屬於 Bridges。

## 一般使用者只需要知道的事

每個 Release ZIP 內只有一個同名版本資料夾，不會在解壓位置散落檔案。版本資料夾根目錄有三個重要檔案：

| 檔案 | 用途 |
|---|---|
| `setup.bat` | 唯一入口：新電腦跑完整安裝，已安裝電腦只更新程式 |
| `manual-install.html` | BAT 無法執行時的 PowerShell 與完全手動備援說明 |
| `VERSION.txt` | 記錄封裝角色、來源 commit、工作區狀態與封裝時間 |

一般使用者不應自行執行 `deploy/*/scripts` 內的個別腳本，除非正在依照手動安裝說明排除問題。

## 保母級安裝教學：先判斷你是哪一台電腦

你會拿到兩種不同的壓縮包，請先看檔名：

| 要安裝的電腦 | 應使用的壓縮包 |
|---|---|
| 儲存影像、CSV，平常沒有相機的電腦 | `PICoater-Storage-版本.zip` |
| 連接相機、IO、光源，產線操作使用的電腦 | `PICoater-Inspection-版本.zip` |

不要把 Storage 包裝到檢測電腦，也不要把 Inspection 包裝到儲存電腦。

### 儲存電腦第一次安裝

1. **把 Storage 壓縮包複製到儲存電腦。**
   建議先放到桌面，不要直接從網路磁碟執行。

2. **在壓縮包上按滑鼠右鍵，選擇「解壓縮全部」。**
   不可以直接雙擊壓縮包後執行裡面的 BAT。解壓縮後會得到一個同名版本資料夾：

   ```text
   PICoater-Storage-版本\
   ├─ setup.bat
   ├─ manual-install.html
   ├─ VERSION.txt
   ├─ app\
   └─ deploy\
   ```

   先進入這個版本資料夾，再執行後續步驟。

3. **先確認儲存電腦的 D 槽存在。**
   預設會把程式放到 `C:\AniloxMonitor`，大量資料放到 `D:\Anilox`。若沒有 D 槽，先不要執行安裝，請修改下一步的設定檔。

4. **用記事本開啟設定檔。**
   進入解壓縮後的：

   ```text
   deploy\storage-pc\storage-config.json
   ```

   一般現場只需要確認以下欄位：

   | 欄位 | 一般預設 | 什麼時候要改 |
   |---|---|---|
   | `NicName` | `乙太網路` | Windows 顯示的網卡名稱不同時 |
   | `IpAddress` | `192.168.10.20` | 現場另有指定儲存電腦 IP 時 |
   | `AniloxRoot` | `D:\Anilox` | 沒有 D 槽或資料要放其他磁碟時 |
   | `StorageMinFreeGB` | `100` | 儲存資料磁碟要保留的最低可用空間；若不小心設成大於等於磁碟總容量，程式只警告、不刪檔 |
   | `AppDir` | `C:\AniloxMonitor` | 程式要放其他位置時 |
   | `PreviousAppDirs` | `D:\AniloxMonitor` | 只供舊版本將 Config 遷移到新路徑，新安裝不需修改 |
   | `RdpUser` / `RdpPassword` | `aroll` / `aroll` | 現場要求不同遠端帳密時；只用於 RDP，不綁定程式排程 |

   修改後按 `Ctrl+S` 儲存，再關閉記事本。不要刪除逗號、雙引號或反斜線。

   舊版儲存程式若已在 `D:\AniloxMonitor`，新安裝器會依 `PreviousAppDirs` 處理：

   | 新 C 路徑 | 舊 D 路徑 | 處理 |
   |---|---|---|
   | 還沒有 Config | 有 Config | 先複製舊 Config 到 C，再安裝與重建排程 |
   | 已有 Config | 任意 | 保留 C 的現行設定，不用舊設定覆蓋 |
   | 還沒有 Config | 也沒有 Config | 視為全新安裝，程式啟動時產生預設設定 |

   舊 `D:\AniloxMonitor` 會先保留作為回退；確認 C 槽程式、Config 與 heartbeat 全部正常後，才可人工刪除舊資料夾。

5. **雙擊解壓縮根目錄的 `setup.bat`。**
   Windows 顯示「是否允許此應用程式變更您的裝置？」時，按 **是**。

6. **等待黑色視窗執行完成。**
   過程會設定網路、共用資料夾、Guest、遠端桌面、自動啟動與電源。不要中途關閉視窗，也不要拔網路線。

7. **看到 `[完成] 儲存電腦已安裝` 後再關閉視窗。**
   如果看到紅色 `[FAIL]`，先不要反覆重跑，直接開啟同一資料夾的 `manual-install.html` 查看失敗項目。

8. **確認資料夾已建立。**
   依照設定檔預設值，檔案總管應看到：

   ```text
   D:\Anilox\Captures_pack
   D:\Anilox\Config
   C:\AniloxMonitor\AniloxRoll.Monitor.exe
   ```

9. **確認程式可以啟動。**
   Public Desktop 會出現 `PICoater AOI` 程式捷徑。安裝器也會為本機 Administrators 群組建立登入自動啟動及每分鐘保活工作、立即啟動，並在 10 秒內確認正確 EXE 仍在執行。目前登入的是 `rogic` 或日後使用 `aroll` 都可以，且只保留一個執行個體。程式正常關閉或異常退出後，最晚一分鐘內會重新開啟。沒有啟動會直接顯示 `[FAIL]`，不再把「排程已建立」誤當成「程式已啟動」。

10. **從檢測電腦測試共用資料夾。**
    在檢測電腦按 `Win+R`，輸入：

    ```text
    \\192.168.10.20\Anilox
    ```

    以上是預設值；若第 4 步改過 `IpAddress` 或 `ShareName`，請改用修改後的值。應直接看到 `Captures_pack` 與 `Config`，而且可以新增並刪除測試文字檔。

### 檢測電腦第一次安裝

1. **把 Inspection 壓縮包複製到檢測電腦並「解壓縮全部」。**
   不要直接在 ZIP 裡面執行。解壓後先進入 `PICoater-Inspection-版本` 資料夾。

2. **先確認相機、IO、光源及儲存網路線的接法。**
   設定 IP 時網路可能短暫斷線，這是正常現象。

3. **用記事本開啟：**

   ```text
   deploy\inspection-pc\inspection-config.json
   ```

4. **確認以下欄位：**

   | 欄位 | 一般預設 | 什麼時候要改 |
   |---|---|---|
   | `AppDir` | `C:\AniloxMonitor` | 程式要安裝到其他位置時 |
   | `IoNicName` | `乙太網路` | IO 所在網卡名稱不同時 |
   | `IoIp` | `192.168.255.10` | 現場控制網段另有規劃時 |
   | `StorageIp` | `192.168.10.10` | 檢測電腦的儲存網段 IP 不同時 |
   | `VerifyPingTarget` | `192.168.10.20` | 儲存電腦 IP 不同時 |

   修改後按 `Ctrl+S` 儲存並關閉記事本。

5. **雙擊根目錄的 `setup.bat`，UAC 詢問時按「是」。**

6. **等待看到 `[完成] 檢測電腦已安裝`。**
   若安裝時 IO 或儲存電腦尚未接電，可能看到黃色連線警告；黃色是當下無法連線，紅色 `[FAIL]` 才是安裝失敗。
   安裝完成後 Public Desktop 會出現 `PICoater AOI` 程式捷徑與 `Anilox 儲存資料` 捷徑；後者雙擊即開啟 `\\192.168.10.20\Anilox`（實際路徑來自設定檔）。

7. **啟動 `C:\AniloxMonitor\AniloxRoll.Monitor.exe`。**
   第一次啟動會自動建立缺少的 `Config/*.json`，這是正常行為。

8. **確認上方狀態列。**
   IO、光源與儲存電腦接線供電後，狀態應能自動從紅色恢復成綠色，不應要求重開主程式。

9. **進行一輪基本功能測試。**
   至少測試相機連線、開始/停止抓取、IO 訊號、光源控制，以及儲存電腦共用與 heartbeat。

### 已安裝電腦更新新版程式

1. 下載與電腦角色相同的新壓縮包。
2. 把新壓縮包完整解壓縮到新的資料夾，不要蓋在舊壓縮包上。
3. 建議先關閉正在執行的 AniloxRoll Monitor；忘記關閉時更新器也會嘗試停止它。
4. **不要自行複製舊的 `Config` 到新壓縮包。**目的電腦安裝目錄中的設定會自動保留。
5. 雙擊新壓縮包根目錄的 `setup.bat`，UAC 詢問時按「是」。它會看到既有 EXE，自動走更新流程。
6. 等待看到綠色完成訊息。
7. 啟動程式並確認原本設定、影像及 CSV 都還存在。

`setup.bat` 偵測到既有程式時，不會重新設定網卡、SMB、RDP 或電源。兩種角色更新都會建立或修正 `PICoater AOI` 程式捷徑；Inspection 還會修正儲存資料捷徑。需要強制重做 Windows 設定時，依 `manual-install.html` 明確指定 Install 模式。

### 安裝時最常見的錯誤

| 現象 | 處理方式 |
|---|---|
| 雙擊 BAT 後視窗立刻消失 | 先確認已完整解壓縮；再開 `manual-install.html` 使用 PowerShell 方式 |
| 顯示找不到 `app\AniloxRoll.Monitor.exe` | 壓縮包未完整解壓，或只複製了 BAT；重新解壓整包 |
| 顯示找不到網卡 | 到「控制台 → 網路連線」查看真正網卡名稱，修改角色 JSON 的 `NicName`/`IoNicName` |
| 顯示存取被拒 | 確認 UAC 有按「是」，且目前帳號允許系統管理員操作 |
| 儲存共用打不開 | 確認兩台電腦 IP、網路線、TCP 445、防火牆及 Guest 設定 |
| Windows 安全性顯示 DLL 發行者不明 | 安裝器會先移除檔案的下載封鎖標記；若仍被 Smart App Control 擋下，代表未簽章程式被系統政策拒絕，需使用可信任數位簽章或由廠內 IT 明確調整 App Control 政策 |
| 儲存安裝顯示排程啟動失敗 | 確認目前登入帳號屬於本機 Administrators，檢查 `StorageTaskName`、`LastTaskResult` 與 `AppDir\AniloxRoll.crash.log`；`267011 (0x00041303)` 表示排程尚未真正執行 |

### 不要做這些事

- 不要直接在 ZIP 壓縮檔內執行 BAT。
- 不要只複製 `setup.bat`。
- 不要刪除已安裝目錄的 `Config`，除非目的是恢復預設值。
- 不要刪除 `D:\Anilox\Captures_pack` 來更新程式。
- 不要自行執行 `scripts` 裡的單一步驟後就假設整台電腦已安裝完成。

---

## 以下內容給工程與維護人員

## 原始碼目錄結構

```text
deploy/
├─ common/
│  ├─ Deploy.Common.ps1        共用設定讀取、程式複製與角色設定
│  └─ setup_nosleep.ps1        兩台電腦共用的電源設定
├─ storage-pc/
│  ├─ setup.bat/.ps1           儲存電腦自動判斷入口
│  ├─ install.ps1              儲存電腦首次安裝機制
│  ├─ update_app.ps1           儲存電腦程式更新機制
│  ├─ storage-config.json      儲存電腦安裝參數唯一來源
│  ├─ manual-install.html      無腳本手動備援
│  └─ scripts/                 儲存電腦專用 Windows 設定機制
├─ inspection-pc/
│  ├─ setup.bat/.ps1           檢測電腦自動判斷入口
│  ├─ install.ps1              檢測電腦首次安裝機制
│  ├─ update_app.ps1           檢測電腦程式更新機制
│  ├─ inspection-config.json   檢測電腦安裝參數唯一來源
│  ├─ manual-install.html      無腳本手動備援
│  └─ scripts/                 檢測電腦專用 Windows 設定機制
└─ package/
   ├─ rebuild_and_package.bat  唯一打包入口；自動判斷正式版或測試版
   └─ package_release.ps1      底層封裝器，供 BAT 與自動化使用
```

## 製作 Release 壓縮包

### 唯一入口：先確認版本類型，再 Rebuild 與打包

直接雙擊：

```text
deploy\package\rebuild_and_package.bat
```

啟動後會自動檢查 Git，並在任何編譯前顯示：

- `OFFICIAL RELEASE`：工作目錄已 commit 且乾淨，可作為正式版。
- `SMOKE TEST PACKAGE`：存在未提交修改，只供實體機測試，不可當正式版。

確認類型後按 `Y` 繼續，按 `N` 就取消，不會開始編譯。繼續後會對整個 solution 執行 `Release|x64 Rebuild`，再一次產生 Storage 與 Inspection 兩包，完成後自動開啟 `artifacts\deploy`。封裝暫存資料夾在 ZIP 驗證成功後會自動刪除，該目錄只保留 ZIP。

`Rebuild` 只清除 repository 的編譯輸出，不會刪除產線 Captures_pack、CSV 或已安裝程式的 Config。測試包的 `VERSION.txt` 會標示 `SourceState=dirty`。ZIP 內會以完整包名建立唯一根資料夾，即使解壓到當前位置也不會散落。

### 由封裝器負責 Build：PowerShell

在 repository 根目錄執行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\deploy\package\package_release.ps1 -Role Storage
powershell -NoProfile -ExecutionPolicy Bypass -File .\deploy\package\package_release.ps1 -Role Inspection
```

腳本預設執行 `Release|x64` Build；加 `-Rebuild` 時改為先 Clean 再 Build。完成後只挑選正式執行所需的檔案，並把壓縮包寫入 `artifacts/deploy/`。

只有已完成可信任 `Release|x64` build 時才可使用 `-SkipBuild`。正式包要求工作區已 commit 且乾淨；臨時上機煙測才可加 `-AllowDirty`，此時 `VERSION.txt` 會記錄 `SourceState=dirty`。

封裝器不會複製執行期產生的 `Config/*.json`。缺少這些 JSON 時，程式會依 code defaults 自動建立。`Config/Radient_Config.dcf` 無法由 defaults 產生，因此只會放進需要 MIL 的 Inspection package。

## 安裝與更新狀態表

| 目前狀態 | 操作 | 下一狀態 | 執行動作 |
|---|---|---|---|
| 壓縮包缺少 `app/AniloxRoll.Monitor.exe` | `setup` | 不變 | 在修改目標電腦前直接失敗 |
| `AppDir` 沒有主程式 | `setup` → Install | 已安裝並完成設定 | 複製 runtime、建立/保留 `Config`、套用角色與 Windows 設定 |
| `AppDir` 已有主程式但未執行 | `setup` → Update | 已更新 | 移除上一版 manifest 登記的檔案、複製新版、保留 runtime JSON |
| `AppDir` 已有主程式且執行中 | `setup` → Update | 已更新並重新啟動 | 只停止設定 `AppDir` 內的程式，更新後由儲存排程重新啟動 |
| 儲存排程不存在或內容過時 | 儲存 Update | 排程已更新 | 重新登記「任一系統管理員登入」自啟及每分鐘保活；正常關閉/異常退出都會重開 |

底層 Install/Update 保持分離；`setup` 只負責選路。工程人員可依 `manual-install.html` 用 `-Mode Install` 或 `-Mode Update` 強制指定。

## 儲存電腦安裝內容

`storage-config.json` 是儲存電腦安裝參數的唯一來源。`install` 會：

1. 把程式複製到 `AppDir`，移除複製檔案的 Windows 下載封鎖標記，並在 Public Desktop 建立 `PICoater AOI` 程式捷徑。
2. 設定儲存網卡、`AniloxRoot`、子目錄、NTFS、SMB 共用、防火牆及私人網路。
3. 啟用目前設備網段使用的 Guest SMB 契約。
4. 建立並設定 RDP 帳號。
5. 寫入 Storage `app-mode.json`，並移除儲存模式不需要的 MIL managed DLL。
6. 為本機 Administrators 群組建立登入自動啟動與每分鐘保活排程，並驗證程式已實際啟動。
7. 關閉自動睡眠、休眠及硬碟停轉。

## 檢測電腦安裝內容

`inspection-config.json` 是檢測電腦安裝參數的唯一來源。`install` 會：

1. 把程式複製到 `AppDir`，並在 Public Desktop 建立 `PICoater AOI` 程式捷徑。
2. 設定 IO IP，並在不刪除 PLC IP 的前提下加入儲存網段 secondary IP。
3. 啟用 Guest SMB client policy。
4. 在 Public Desktop 建立通往 `\\VerifyPingTarget\StorageShareName` 的儲存資料捷徑。
5. 寫入 Inspection `app-mode.json`。
6. 關閉自動睡眠、休眠及硬碟停轉。

## 驗證方式

在檢測電腦讀取目前兩份設定後測試：

```powershell
$inspection = Get-Content .\deploy\inspection-pc\inspection-config.json -Raw | ConvertFrom-Json
$storage = Get-Content .\deploy\storage-pc\storage-config.json -Raw | ConvertFrom-Json
Test-NetConnection -ComputerName $inspection.VerifyPingTarget -Port 445
$p = '\\{0}\{1}\Captures_pack\__deploy_test.txt' -f $inspection.VerifyPingTarget,$storage.ShareName
'ok' | Set-Content $p
Remove-Item $p
```

接著啟動兩台電腦的程式，確認檢測電腦狀態列同時顯示：

- 儲存共用可寫。
- 儲存程式 heartbeat 正常。

## 檔案編碼

- `.bat`：只使用 ASCII，避免 CMD 以 Big5 解讀中文造成失敗。
- `.ps1`：UTF-8 with BOM，避免 Windows PowerShell 5.1 誤判中文編碼。
- `.json`、`.html`、`.md`：UTF-8。

交付前必須解析所有 PowerShell、解析兩份 JSON、驗證 BOM、實際開啟兩份手動安裝 HTML，並檢查兩種 Release 包內容。

## English Summary

- Run `setup.bat` for both first installation and later updates; it selects the safe path automatically.
- Update mode preserves runtime `Config`, captures, CSV files, and Windows/network settings.
- Open `manual-install.html` when BAT/PowerShell execution is unavailable.
- Build clean packages with `deploy/package/package_release.ps1 -Role Storage|Inspection`.
- Official packages require a clean committed worktree; `-AllowDirty` is only for temporary machine smoke tests.
- Storage and Inspection installation parameters live only in their role-specific JSON files.
