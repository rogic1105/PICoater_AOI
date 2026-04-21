# 儲存機：允許 Guest 匿名 SMB 存取（供檢測機 RemoteCopyService 使用）
# 依賴先跑過 setup_storage_pc.ps1（共用資料夾已存在）

function Die([string]$msg) { Write-Host ("[FAIL] " + $msg) -ForegroundColor Red; exit 1 }

$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Die "請以系統管理員身分執行"
}

# 讀取設定檔（只為了取 ShareName）
$cfgPath = Join-Path $PSScriptRoot 'storage-config.json'
if (-not (Test-Path $cfgPath)) { Die ("找不到 " + $cfgPath) }
$json = [System.IO.File]::ReadAllText((Resolve-Path $cfgPath).Path, [System.Text.Encoding]::UTF8)
$cfg  = $json | ConvertFrom-Json
$shareName = $cfg.ShareName

Write-Host ("[Setup] Share = " + $shareName) -ForegroundColor Cyan
Write-Host ""

# ── 1. 啟用 SMB Server Guest ─────────────────
Write-Host "[1/4] 啟用 SMB Server Guest..."
try {
    Set-SmbServerConfiguration -EnableGuestAccess $true -Confirm:$false -Force -ErrorAction Stop
    Write-Host "  -> EnableGuestAccess = True" -ForegroundColor Green
} catch {
    Write-Host ("  (警告) " + $_.Exception.Message) -ForegroundColor Yellow
}

# ── 2. 啟用 Guest 本機帳號 ───────────────────
Write-Host "[2/4] 啟用 Guest 本機帳號..."
$guest = Get-LocalUser -Name 'Guest' -ErrorAction SilentlyContinue
if (-not $guest) {
    Die "找不到 Guest 本機帳號（系統異常）"
}
if (-not $guest.Enabled) {
    Enable-LocalUser -Name 'Guest'
    Write-Host "  -> Guest enabled" -ForegroundColor Green
} else {
    Write-Host "  -> Guest 已啟用" -ForegroundColor Yellow
}

# ── 3. 授予 Guest 共用權限（SMB） ────────────
Write-Host "[3/5] 授予 Guest SMB 共用權限..."
Grant-SmbShareAccess -Name $shareName -AccountName Guest -AccessRight Full -Force | Out-Null
Write-Host ("  -> Guest : Full on " + $shareName) -ForegroundColor Green

# ── 4. NTFS：授予 Guest Modify ───────────────
Write-Host "[4/5] 授予 Guest NTFS 寫入權限..."
$folder = $cfg.StorageFolder
$acl = Get-Acl $folder
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    'Guest', 'Modify', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
$acl.SetAccessRule($rule)
Set-Acl -Path $folder -AclObject $acl
Write-Host ("  -> Guest : Modify on " + $folder) -ForegroundColor Green

# ── 5. 修正 Guest 網路登入權利（關鍵！Windows 預設 Guest 被拒絕網路登入）
Write-Host "[5/5] 修正 Guest 網路登入權利（secedit）..."
$tempCfg = Join-Path $env:TEMP 'aoi_secpol.cfg'
$tempDb  = Join-Path $env:TEMP 'aoi_secpol.sdb'
try {
    secedit /export /cfg $tempCfg /quiet | Out-Null
    $guestSid = (New-Object System.Security.Principal.NTAccount('Guest')).Translate([System.Security.Principal.SecurityIdentifier]).Value
    # secedit 匯出時可能用「名字」或「*SID」表示同一個帳號，兩種都要認得
    $guestTokens = @('Guest', '*' + $guestSid, 'Guests', '*S-1-5-32-546')
    $lines = Get-Content $tempCfg
    $out = @()
    $seenNet = $false
    foreach ($line in $lines) {
        if ($line -match '^SeDenyNetworkLogonRight\s*=\s*(.*)') {
            # 移除 Guest / Guests（名字 + SID 兩種形式）
            $vals = $matches[1] -split ',' | ForEach-Object { $_.Trim() } |
                    Where-Object { $_ -ne '' -and $guestTokens -notcontains $_ }
            if ($vals.Count -gt 0) {
                $out += "SeDenyNetworkLogonRight = " + ($vals -join ',')
            } else {
                $out += "SeDenyNetworkLogonRight ="
            }
        } elseif ($line -match '^SeNetworkLogonRight\s*=\s*(.*)') {
            $seenNet = $true
            $vals = $matches[1] -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
            # 若已有任一 Guest 形式就不重複加
            $hasGuest = $false
            foreach ($v in $vals) { if ($guestTokens -contains $v) { $hasGuest = $true; break } }
            if (-not $hasGuest) { $vals += 'Guest' }
            $out += "SeNetworkLogonRight = " + ($vals -join ',')
        } else {
            $out += $line
        }
    }
    if (-not $seenNet) {
        $out += "SeNetworkLogonRight = Guest"
    }
    Set-Content -Path $tempCfg -Value $out -Encoding Unicode
    secedit /configure /db $tempDb /cfg $tempCfg /areas USER_RIGHTS /quiet | Out-Null
    Write-Host "  -> Guest 加入 SeNetworkLogonRight" -ForegroundColor Green
    Write-Host "  -> Guest 自 SeDenyNetworkLogonRight 移除" -ForegroundColor Green
    gpupdate /force /target:computer 2>&1 | Out-Null
    Write-Host "  -> gpupdate /force 完成" -ForegroundColor Green
} catch {
    Write-Host ("  (警告) secedit 失敗：" + $_.Exception.Message) -ForegroundColor Yellow
} finally {
    Remove-Item $tempCfg, $tempDb -Force -ErrorAction SilentlyContinue
}

# ── 驗證 ─────────────────────────────────────
Write-Host ""
Write-Host "[驗證] SMB Share ACL：" -ForegroundColor Cyan
Get-SmbShareAccess -Name $shareName | Format-Table Name, AccountName, AccessControlType, AccessRight -AutoSize

Write-Host "[驗證] SMB Server 設定：" -ForegroundColor Cyan
Get-SmbServerConfiguration | Select-Object EnableGuestAccess | Format-Table -AutoSize

Write-Host "[驗證] Guest 網路登入權利（secedit）：" -ForegroundColor Cyan
$verifyCfg = Join-Path $env:TEMP 'aoi_secpol_verify.cfg'
try {
    secedit /export /cfg $verifyCfg /quiet | Out-Null
    Get-Content $verifyCfg | Where-Object { $_ -match '^(SeNetworkLogonRight|SeDenyNetworkLogonRight)' } | ForEach-Object {
        Write-Host ("  " + $_)
    }
} finally {
    Remove-Item $verifyCfg -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "[完成] 儲存機 Guest 已開放。" -ForegroundColor Cyan
Write-Host "        接著請在檢測機跑 deploy\inspection-pc\run_setup.bat"
Write-Host "        測試：檢測機 Win+R 輸入 \\192.168.10.20\AniloxStorage 應直接開啟"
