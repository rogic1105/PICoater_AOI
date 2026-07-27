param(
    [string] $Config = (Join-Path (Split-Path -Parent $PSScriptRoot) 'storage-config.json')
)

$roleRoot = Split-Path -Parent $PSScriptRoot
$deployRoot = Split-Path -Parent $roleRoot
. (Join-Path $deployRoot 'common\Deploy.Common.ps1')

Assert-DeployAdministrator
$cfg = Read-DeployConfig $Config
if (-not $cfg.AppDir) { Stop-Deploy 'storage-config.json 的 AppDir 未設定。' }
if (-not $cfg.StorageMinFreeGB -or [int]$cfg.StorageMinFreeGB -lt 1) {
    Stop-Deploy 'storage-config.json 的 StorageMinFreeGB 必須大於 0。'
}

$storageConfig = Join-Path $cfg.AniloxRoot 'Config'
$storageData = Join-Path $cfg.AniloxRoot 'Captures'
Write-AppRoleConfig `
    -AppDir $cfg.AppDir `
    -Role 'Storage' `
    -StorageConfigFolder $storageConfig `
    -StorageDataPath $storageData `
    -StorageMinFreeGB ([int]$cfg.StorageMinFreeGB)

$milDll = Join-Path $cfg.AppDir 'Matrox.MatroxImagingLibrary.dll'
if (Test-Path -LiteralPath $milDll) {
    Remove-Item -LiteralPath $milDll -Force
}

Write-Host ("[OK] 儲存模式已寫入 " + (Join-Path $cfg.AppDir 'Config\app-mode.json')) -ForegroundColor Green
