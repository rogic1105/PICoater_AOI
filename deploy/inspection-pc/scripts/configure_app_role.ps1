# 檢測機：寫入 app-mode.json (Role = Inspection)

param(
    [string] $Config = (Join-Path (Split-Path -Parent $PSScriptRoot) 'inspection-config.json')
)

$roleRoot = Split-Path -Parent $PSScriptRoot
$deployRoot = Split-Path -Parent $roleRoot
. (Join-Path $deployRoot 'common\Deploy.Common.ps1')

Assert-DeployAdministrator
$cfg = Read-DeployConfig $Config
Write-AppRoleConfig `
    -AppDir $cfg.AppDir `
    -Role 'Inspection' `
    -StorageConfigFolder $cfg.StorageMachineConfigFolder
Write-Host ("[OK] 檢測模式已寫入 " + (Join-Path $cfg.AppDir 'Config\app-mode.json')) -ForegroundColor Green
