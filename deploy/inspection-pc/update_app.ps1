param(
    [string] $Config = (Join-Path $PSScriptRoot 'inspection-config.json'),
    [string] $AppSource = ''
)

$deployRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $deployRoot 'common\Deploy.Common.ps1')

Assert-DeployAdministrator
$cfg = Read-DeployConfig $Config
if (-not $AppSource) {
    $AppSource = Join-Path (Get-DeployPackageRoot $PSScriptRoot) 'app'
}
$AppSource = [System.IO.Path]::GetFullPath($AppSource)

Write-Host 'PICoater 檢測電腦：更新程式' -ForegroundColor Cyan
Install-AppPayload -SourceDir $AppSource -AppDir $cfg.AppDir -TaskName ''
Invoke-DeployStep '套用檢測程式模式' (Join-Path $PSScriptRoot 'scripts\configure_app_role.ps1') @{ Config = $Config }
Invoke-DeployStep '更新儲存資料桌面捷徑' (Join-Path $PSScriptRoot 'scripts\configure_storage_shortcut.ps1') @{ Config = $Config }

Write-Host ''
Write-Host '[完成] 程式與儲存資料捷徑已更新；網卡與 Windows 設定沒有重做。' -ForegroundColor Green
