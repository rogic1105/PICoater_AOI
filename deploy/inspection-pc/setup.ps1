param(
    [ValidateSet('Auto', 'Install', 'Update')]
    [string] $Mode = 'Auto',
    [string] $Config = (Join-Path $PSScriptRoot 'inspection-config.json'),
    [string] $AppSource = ''
)

$ErrorActionPreference = 'Stop'
$deployRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $deployRoot 'common\Deploy.Common.ps1')

Assert-DeployAdministrator
$cfg = Read-DeployConfig $Config
$installedExe = Join-Path $cfg.AppDir 'AniloxRoll.Monitor.exe'
$selectedMode = if ($Mode -eq 'Auto') {
    if (Test-Path -LiteralPath $installedExe -PathType Leaf) { 'Update' } else { 'Install' }
} else {
    $Mode
}

if ($selectedMode -eq 'Install') {
    Write-Host '[Setup] No installed app detected. Running first-time installation.' -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot 'install.ps1') -Config $Config -AppSource $AppSource
} else {
    Write-Host ('[Setup] Installed app detected: ' + $installedExe) -ForegroundColor Cyan
    Write-Host '[Setup] Running app update; Windows network and power settings stay unchanged.' -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot 'update_app.ps1') -Config $Config -AppSource $AppSource
}
