# 檢測機：允許匿名 Guest 存取遠端 SMB（Windows 10/11 預設會擋）
# 對應 registry：LanmanWorkstation\Parameters\AllowInsecureGuestAuth

param(
    [string] $Config = (Join-Path (Split-Path -Parent $PSScriptRoot) 'inspection-config.json')
)

function Die([string]$msg) { Write-Host ("[FAIL] " + $msg) -ForegroundColor Red; exit 1 }

$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Die "請以系統管理員身分執行"
}

# ── 1. Registry: 一般策略位置 ────────────────
Write-Host "[1/3] 設定 LanmanWorkstation Parameters..."
$reg1 = 'HKLM:\SYSTEM\CurrentControlSet\Services\LanmanWorkstation\Parameters'
New-ItemProperty -Path $reg1 -Name 'AllowInsecureGuestAuth' -Value 1 -PropertyType DWord -Force | Out-Null
Write-Host "  -> AllowInsecureGuestAuth = 1" -ForegroundColor Green

# ── 2. Registry: GPO 位置（覆寫 group policy） ─
Write-Host "[2/3] 設定 GPO 覆寫..."
$reg2 = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\LanmanWorkstation'
if (-not (Test-Path $reg2)) { New-Item -Path $reg2 -Force | Out-Null }
New-ItemProperty -Path $reg2 -Name 'AllowInsecureGuestAuth' -Value 1 -PropertyType DWord -Force | Out-Null
Write-Host "  -> GPO override OK" -ForegroundColor Green

# ── 3. 清空現有 SMB 連線快取（避免舊的失敗 cache） ──
Write-Host "[3/3] 清除 SMB 連線快取..."
try {
    cmd /c "net use * /delete /y" | Out-Null
    Write-Host "  -> net use cache cleared" -ForegroundColor Green
} catch {
    Write-Host "  (略過) 無舊連線" -ForegroundColor DarkGray
}

# ── 驗證 ─────────────────────────────────────
Write-Host ""
Write-Host "[驗證] Registry 值：" -ForegroundColor Cyan
Get-ItemProperty -Path $reg1 -Name AllowInsecureGuestAuth | Select-Object AllowInsecureGuestAuth | Format-Table -AutoSize

Write-Host ""
Write-Host "[完成] 設定已套用。" -ForegroundColor Cyan
$json = [System.IO.File]::ReadAllText((Resolve-Path $Config).Path, [System.Text.Encoding]::UTF8)
$cfg = $json | ConvertFrom-Json
Write-Host ("        測試：Win+R 輸入 \\" + $cfg.VerifyPingTarget + "\" + $cfg.StorageShareName + " 應直接開啟（不再要求帳密）")
Write-Host "        若仍要求，執行 gpupdate /force 或重開機"
