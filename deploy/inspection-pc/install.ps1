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

Write-Host 'PICoater 檢測電腦：第一次完整安裝' -ForegroundColor Cyan
Install-AppPayload -SourceDir $AppSource -AppDir $cfg.AppDir -TaskName ''

$scripts = Join-Path $PSScriptRoot 'scripts'
Invoke-DeployStep '1/6 IO 網路' (Join-Path $scripts 'configure_io_network.ps1') @{ Config = $Config }
Invoke-DeployStep '2/6 儲存網路 secondary IP' (Join-Path $scripts 'configure_storage_network.ps1') @{ Config = $Config }
Invoke-DeployStep '3/6 Guest SMB 用戶端' (Join-Path $scripts 'configure_guest_client.ps1') @{ Config = $Config }
Invoke-DeployStep '4/6 儲存資料桌面捷徑' (Join-Path $scripts 'configure_storage_shortcut.ps1') @{ Config = $Config }
Invoke-DeployStep '5/6 檢測程式模式' (Join-Path $scripts 'configure_app_role.ps1') @{ Config = $Config }
Invoke-DeployStep '6/6 關閉自動睡眠' (Join-Path $deployRoot 'common\setup_nosleep.ps1')

Write-Host ''
Write-Host '[完成] 檢測電腦已安裝。' -ForegroundColor Green
