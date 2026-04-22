# 儲存機：啟用遠端桌面（RDP）+ 建立 RDP 帳號
# 依賴先跑過 setup_storage_pc.ps1（讀同一份 storage-config.json）
#
# 內網專用設計：
# - switch 不對外，帳密可以簡單（預設 aroll/aroll）
# - 若要換強密碼，改 storage-config.json 的 RdpUser/RdpPassword 再重跑即可

function Die([string]$msg) { Write-Host ("[FAIL] " + $msg) -ForegroundColor Red; exit 1 }

$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Die "請以系統管理員身分執行"
}

# 讀取設定檔
$cfgPath = Join-Path $PSScriptRoot 'storage-config.json'
if (-not (Test-Path $cfgPath)) { Die ("找不到 " + $cfgPath) }
$json = [System.IO.File]::ReadAllText((Resolve-Path $cfgPath).Path, [System.Text.Encoding]::UTF8)
$cfg  = $json | ConvertFrom-Json

if (-not $cfg.RdpEnabled) {
    Write-Host "[Setup] RdpEnabled = false，略過 RDP 設定" -ForegroundColor Yellow
    exit 0
}

$rdpUser = $cfg.RdpUser
$rdpPass = $cfg.RdpPassword
if ([string]::IsNullOrEmpty($rdpUser) -or [string]::IsNullOrEmpty($rdpPass)) {
    Die "storage-config.json 缺 RdpUser 或 RdpPassword"
}

Write-Host ("[Setup] RDP User = " + $rdpUser) -ForegroundColor Cyan
Write-Host ""

# ── 1. 關閉密碼複雜度規則（允許弱密碼） ──────
Write-Host "[1/5] 關閉密碼複雜度規則（允許弱密碼）..."
$tempCfg = Join-Path $env:TEMP 'aoi_pwpol.cfg'
$tempDb  = Join-Path $env:TEMP 'aoi_pwpol.sdb'
try {
    secedit /export /cfg $tempCfg /quiet | Out-Null
    $lines = Get-Content $tempCfg
    $out = @()
    foreach ($line in $lines) {
        if ($line -match '^PasswordComplexity\s*=') {
            $out += 'PasswordComplexity = 0'
        } elseif ($line -match '^MinimumPasswordLength\s*=') {
            $out += 'MinimumPasswordLength = 0'
        } else {
            $out += $line
        }
    }
    Set-Content -Path $tempCfg -Value $out -Encoding Unicode
    secedit /configure /db $tempDb /cfg $tempCfg /areas SECURITYPOLICY /quiet | Out-Null
    Write-Host "  -> PasswordComplexity=0, MinimumPasswordLength=0" -ForegroundColor Green
} catch {
    Write-Host ("  (警告) " + $_.Exception.Message) -ForegroundColor Yellow
} finally {
    Remove-Item $tempCfg, $tempDb -Force -ErrorAction SilentlyContinue
}

# ── 2. 建立或更新 RDP 帳號 ───────────────────
Write-Host ("[2/5] 建立/更新本機帳號 " + $rdpUser + "...")
$securePw = ConvertTo-SecureString $rdpPass -AsPlainText -Force
$user = Get-LocalUser -Name $rdpUser -ErrorAction SilentlyContinue
if ($user) {
    Set-LocalUser -Name $rdpUser -Password $securePw
    Set-LocalUser -Name $rdpUser -PasswordNeverExpires $true
    Write-Host "  -> 密碼已重設，PasswordNeverExpires=True" -ForegroundColor Green
} else {
    New-LocalUser -Name $rdpUser -Password $securePw `
        -FullName $rdpUser -Description 'PICoater AOI remote access' `
        -PasswordNeverExpires -UserMayNotChangePassword | Out-Null
    Write-Host "  -> 新帳號已建立" -ForegroundColor Green
}

# ── 3. 加入 Administrators + Remote Desktop Users ─
# 中文 Windows 下這些 group 顯示名會翻譯成中文（例「系統管理員」），按英文名字找會失敗。
# 用 well-known SID 查實際名字：Administrators=S-1-5-32-544, Remote Desktop Users=S-1-5-32-555
Write-Host "[3/5] 加入群組..."
$adminGroupName = (Get-LocalGroup -SID 'S-1-5-32-544').Name
$rduGroupName   = (Get-LocalGroup -SID 'S-1-5-32-555').Name
foreach ($gn in @($adminGroupName, $rduGroupName)) {
    try {
        Add-LocalGroupMember -Group $gn -Member $rdpUser -ErrorAction Stop
        Write-Host ("  -> " + $gn + " OK") -ForegroundColor Green
    } catch {
        if ($_.Exception.Message -match 'already a member|已經是') {
            Write-Host ("  -> 已是 " + $gn + " 成員") -ForegroundColor Yellow
        } else { throw }
    }
}

# ── 4. 啟用 RDP 服務（關閉 fDenyTSConnections） ─
Write-Host "[4/5] 啟用遠端桌面服務..."
Set-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Terminal Server' `
    -Name 'fDenyTSConnections' -Value 0 -Type DWord
# NLA：0 = 不要求 NLA（最寬鬆，內網方便）；設 1 會要求客戶端先完成身分驗證
Set-ItemProperty -Path 'HKLM:\System\CurrentControlSet\Control\Terminal Server\WinStations\RDP-Tcp' `
    -Name 'UserAuthentication' -Value 0 -Type DWord
Write-Host "  -> fDenyTSConnections=0, UserAuthentication=0" -ForegroundColor Green

# ── 5. 開放防火牆 Remote Desktop 規則 ────────
Write-Host "[5/5] 開放防火牆..."
Enable-NetFirewallRule -DisplayGroup 'Remote Desktop' -ErrorAction SilentlyContinue
Write-Host "  -> Remote Desktop 規則群組已啟用" -ForegroundColor Green

# ── 驗證 ─────────────────────────────────────
Write-Host ""
Write-Host "[驗證] RDP 設定：" -ForegroundColor Cyan
$r1 = (Get-ItemProperty 'HKLM:\System\CurrentControlSet\Control\Terminal Server').fDenyTSConnections
$r2 = (Get-ItemProperty 'HKLM:\System\CurrentControlSet\Control\Terminal Server\WinStations\RDP-Tcp').UserAuthentication
Write-Host ("  fDenyTSConnections = " + $r1 + "  (0 = 已啟用)")
Write-Host ("  UserAuthentication = " + $r2 + "  (0 = 不要求 NLA)")

Write-Host ""
Write-Host ("[驗證] " + $rduGroupName + " 成員：") -ForegroundColor Cyan
Get-LocalGroupMember -SID 'S-1-5-32-555' | Format-Table Name, ObjectClass -AutoSize

Write-Host ""
Write-Host "[完成] RDP 已開放。" -ForegroundColor Cyan
Write-Host ("        檢測機 Win+R 打 mstsc，電腦=" + $cfg.IpAddress) -ForegroundColor Cyan
Write-Host ("        帳號=" + $rdpUser + "  密碼=" + $rdpPass) -ForegroundColor Cyan
