# 檢測機新增 secondary IP：讓同一張 NIC 同時在 PLC 網段 + 儲存網段
# 不動現有 PLC IP（只新增別名，Modbus 通訊不中斷）
# 用法（以系統管理員身分執行 PowerShell）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File setup_inspection_nic.ps1
#
# 參數來自同目錄 inspection-config.json：
#   PlcSubnetPrefix       PLC 網段前綴（用來自動找現有 NIC，例 "192.168.255."）
#   StorageIp             要新增的儲存網段 IP（例 192.168.10.10）
#   StoragePrefixLength   子網遮罩長度（24）
#   VerifyPingTarget      驗證用的儲存機 IP（例 192.168.10.20）

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
if (-not $Config) { $Config = Join-Path $PSScriptRoot 'inspection-config.json' }
if (-not (Test-Path $Config)) { Die ("找不到設定檔: " + $Config) }
# 明確以 UTF-8 讀 JSON（PS 5.1 預設 ANSI 會把 UTF-8 中文當 Big5 解碼）
$json = [System.IO.File]::ReadAllText((Resolve-Path $Config).Path, [System.Text.Encoding]::UTF8)
$cfg  = $json | ConvertFrom-Json

Write-Host ("[Setup] 讀取設定: " + $Config) -ForegroundColor Cyan
Write-Host ("  PlcSubnetPrefix      = " + $cfg.PlcSubnetPrefix)
Write-Host ("  StorageIp            = " + $cfg.StorageIp + "/" + $cfg.StoragePrefixLength)
Write-Host ("  VerifyPingTarget     = " + $cfg.VerifyPingTarget)
Write-Host ""

# ── 1. 找 PLC 網卡 ───────────────────────────
$plcIp = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -like ($cfg.PlcSubnetPrefix + '*') } |
    Select-Object -First 1
if (-not $plcIp) {
    Die ("找不到 PLC 網卡（沒有 IP 起始於 " + $cfg.PlcSubnetPrefix + "）")
}
$nicName = $plcIp.InterfaceAlias
Write-Host ("[Setup] 目標網卡: " + $nicName + " (現有 PLC IP " + $plcIp.IPAddress + ")") -ForegroundColor Cyan

# ── 2. 新增 secondary IP ─────────────────────
$existing = Get-NetIPAddress -InterfaceAlias $nicName -IPAddress $cfg.StorageIp -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host ("[Setup] " + $cfg.StorageIp + " 已存在，略過新增") -ForegroundColor Yellow
} else {
    New-NetIPAddress -InterfaceAlias $nicName `
        -IPAddress $cfg.StorageIp `
        -PrefixLength ([int]$cfg.StoragePrefixLength) `
        -AddressFamily IPv4 | Out-Null
    Write-Host ("  -> 已新增 " + $cfg.StorageIp + "/" + $cfg.StoragePrefixLength) -ForegroundColor Green
}

# ── 驗證 ─────────────────────────────────────
Write-Host ""
Write-Host "[驗證] 網卡 IP 列表：" -ForegroundColor Cyan
Get-NetIPAddress -InterfaceAlias $nicName -AddressFamily IPv4 |
    Select-Object InterfaceAlias, IPAddress, PrefixLength |
    Format-Table -AutoSize

if ($cfg.VerifyPingTarget) {
    Write-Host ("[驗證] ping " + $cfg.VerifyPingTarget + " (儲存機)...") -ForegroundColor Cyan
    $ok = Test-Connection -ComputerName $cfg.VerifyPingTarget -Count 2 -Quiet -ErrorAction SilentlyContinue
    if ($ok) {
        Write-Host "  -> OK (儲存機已上線)" -ForegroundColor Green
    } else {
        Write-Host "  -> FAIL (儲存機可能還未設定，或 switch 未接上)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "[Setup] 完成。現在同一張 NIC 同時可達 PLC 網段 + 儲存網段。" -ForegroundColor Cyan
Write-Host "         未來換雙口工業電腦時：把兩個 IP 拆到兩張實體卡即可，程式不用改。"
