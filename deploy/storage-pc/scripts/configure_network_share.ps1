# 儲存機一鍵部署：固定 IP + 共用資料夾 + SMB + 防火牆
# Windows 設定：固定 IP + 共用資料夾 + SMB + 防火牆
# 用法（以系統管理員身分執行 PowerShell）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File configure_network_share.ps1
#   (可選) -Config <路徑>
#
# 參數來自同目錄 storage-config.json：
#   NicName         網卡名稱（空=自動選唯一的 Up 網卡）
#   IpAddress       固定 IP（儲存網段，例 192.168.10.20）
#   PrefixLength    子網遮罩長度（24 對應 255.255.255.0）
#   Gateway         預設閘道（空=不設）
#   AniloxRoot      Anilox 根目錄（如 D:\Anilox），對應單一 SMB share
#   ShareName       SMB 共用名稱（如 Anilox）
#   Subdirs         要建立的子目錄陣列（如 ["Captures", "Config"]）
#   AllowedUser     授權使用者（Everyone 或特定帳號）

param(
    [string] $Config = ""
)

function Die([string]$msg) {
    Write-Host ("[FAIL] " + $msg) -ForegroundColor Red
    exit 1
}

# ── 權限檢查 ─────────────────────────────────
$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Die "請以系統管理員身分執行 PowerShell"
}

# ── 讀取設定 ─────────────────────────────────
if (-not $Config) { $Config = Join-Path (Split-Path -Parent $PSScriptRoot) 'storage-config.json' }
if (-not (Test-Path $Config)) { Die ("找不到設定檔: " + $Config) }
# 明確以 UTF-8 讀 JSON（PS 5.1 預設 ANSI 會把 UTF-8 中文當 Big5 解碼）
$json = [System.IO.File]::ReadAllText((Resolve-Path $Config).Path, [System.Text.Encoding]::UTF8)
$cfg  = $json | ConvertFrom-Json

Write-Host ("[Setup] 讀取設定: " + $Config) -ForegroundColor Cyan
Write-Host ("  NicName         = " + $cfg.NicName + "  (空=自動)")
Write-Host ("  IpAddress       = " + $cfg.IpAddress + "/" + $cfg.PrefixLength)
Write-Host ("  Gateway         = " + $cfg.Gateway)
Write-Host ("  AniloxRoot      = " + $cfg.AniloxRoot)
Write-Host ("  ShareName       = " + $cfg.ShareName)
Write-Host ("  Subdirs         = " + ($cfg.Subdirs -join ', '))
Write-Host ("  AllowedUser     = " + $cfg.AllowedUser)
Write-Host ""

# ── 1. 選 NIC ────────────────────────────────
$nicName = $cfg.NicName
if (-not $nicName) {
    $upNics = @(Get-NetAdapter -Physical -ErrorAction SilentlyContinue | Where-Object { $_.Status -eq 'Up' })
    if ($upNics.Count -eq 0) { Die "找不到可用的實體網卡（Up 狀態）" }
    if ($upNics.Count -gt 1) {
        Write-Host "[Setup] 偵測到多張網卡，請在 storage-config.json 明確指定 NicName：" -ForegroundColor Yellow
        $upNics | Format-Table Name, InterfaceDescription, LinkSpeed, Status
        exit 1
    }
    $nicName = $upNics[0].Name
    Write-Host ("[Setup] 自動選擇網卡: " + $nicName) -ForegroundColor Cyan
}
if (-not (Get-NetAdapter -Name $nicName -ErrorAction SilentlyContinue)) {
    Die ("找不到網卡: " + $nicName)
}

# ── 2. 固定 IP ───────────────────────────────
Write-Host "[1/6] 套用固定 IP..."
Get-NetIPAddress -InterfaceAlias $nicName -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Remove-NetIPAddress -Confirm:$false -ErrorAction SilentlyContinue
Get-NetRoute -InterfaceAlias $nicName -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.DestinationPrefix -eq '0.0.0.0/0' } |
    Remove-NetRoute -Confirm:$false -ErrorAction SilentlyContinue
Set-NetIPInterface -InterfaceAlias $nicName -Dhcp Disabled -ErrorAction SilentlyContinue

$newIpArgs = @{
    InterfaceAlias = $nicName
    IPAddress      = $cfg.IpAddress
    PrefixLength   = [int]$cfg.PrefixLength
    AddressFamily  = 'IPv4'
}
if ($cfg.Gateway) { $newIpArgs.DefaultGateway = $cfg.Gateway }
New-NetIPAddress @newIpArgs | Out-Null
Write-Host ("  -> " + $cfg.IpAddress + "/" + $cfg.PrefixLength + " on " + $nicName) -ForegroundColor Green

# ── 3. 建立 Anilox 根目錄 + 子目錄 ──────────────
Write-Host "[2/6] 建立資料夾結構..."
if (-not (Test-Path $cfg.AniloxRoot)) {
    New-Item -ItemType Directory -Path $cfg.AniloxRoot -Force | Out-Null
}
Write-Host ("  -> " + $cfg.AniloxRoot) -ForegroundColor Green
foreach ($sub in $cfg.Subdirs) {
    $subPath = Join-Path $cfg.AniloxRoot $sub
    if (-not (Test-Path $subPath)) {
        New-Item -ItemType Directory -Path $subPath -Force | Out-Null
    }
    Write-Host ("  -> " + $subPath) -ForegroundColor Green
}

# ── 4. NTFS 權限 ─────────────────────────────
Write-Host "[3/6] 設定 NTFS 權限..."
$acl = Get-Acl $cfg.AniloxRoot
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    $cfg.AllowedUser, 'Modify', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
$acl.SetAccessRule($rule)
Set-Acl -Path $cfg.AniloxRoot -AclObject $acl
Write-Host ("  -> " + $cfg.AllowedUser + " : Modify (含子目錄繼承)") -ForegroundColor Green

# ── 5. SMB 共用（單一 share Anilox，內含 Captures/Config 子目錄）─────────────
Write-Host ("[4/6] 設定 SMB 共用 (" + $cfg.ShareName + ")...")
$existing = Get-SmbShare -Name $cfg.ShareName -ErrorAction SilentlyContinue
if ($existing) {
    Remove-SmbShare -Name $cfg.ShareName -Force
}
New-SmbShare -Name $cfg.ShareName -Path $cfg.AniloxRoot -FullAccess $cfg.AllowedUser | Out-Null
Write-Host ("  -> \\" + $env:COMPUTERNAME + "\" + $cfg.ShareName + " → " + $cfg.AniloxRoot) -ForegroundColor Green

# ── 6. 防火牆 + 網路設定檔 ───────────────────
Write-Host "[5/6] 開放 SMB 防火牆規則..."
Enable-NetFirewallRule -DisplayGroup "檔案及印表機共用" -ErrorAction SilentlyContinue
Enable-NetFirewallRule -DisplayGroup "File and Printer Sharing" -ErrorAction SilentlyContinue
Write-Host "  -> File and Printer Sharing 已啟用" -ForegroundColor Green

Write-Host "[6/6] 設定網路設定檔為「私人」..."
try {
    Set-NetConnectionProfile -InterfaceAlias $nicName -NetworkCategory Private -ErrorAction Stop
    Write-Host "  -> Private" -ForegroundColor Green
} catch {
    Write-Host ("  (警告) 無法設為 Private: " + $_.Exception.Message) -ForegroundColor Yellow
}

# ── 驗證 ─────────────────────────────────────
Write-Host ""
Write-Host "[驗證] NIC IP：" -ForegroundColor Cyan
Get-NetIPAddress -InterfaceAlias $nicName -AddressFamily IPv4 |
    Select-Object InterfaceAlias, IPAddress, PrefixLength |
    Format-Table -AutoSize

Write-Host "[驗證] SMB 共用：" -ForegroundColor Cyan
Get-SmbShare -Name $cfg.ShareName | Format-Table Name, Path, ScopeName -AutoSize

$testFile = Join-Path $cfg.AniloxRoot '__write_test.txt'
try {
    "ok $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $testFile -Encoding utf8
    Remove-Item $testFile -Force
    Write-Host "[驗證] 寫入測試: OK" -ForegroundColor Green
} catch {
    Write-Host ("[驗證] 寫入測試: FAIL - " + $_.Exception.Message) -ForegroundColor Red
}

Write-Host ""
Write-Host "[Setup] 完成！" -ForegroundColor Cyan
Write-Host ("         檢測機 PropertyGrid「遠端路徑」請填: \\" + $cfg.IpAddress + "\" + $cfg.ShareName + "\Captures")
Write-Host ("         檢測機 PropertyGrid「遠端 Config 路徑」（如顯示）: \\" + $cfg.IpAddress + "\" + $cfg.ShareName + "\Config")
