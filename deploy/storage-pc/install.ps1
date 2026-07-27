param(
    [string] $Config = (Join-Path $PSScriptRoot 'storage-config.json'),
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
$taskName = if ($cfg.StorageTaskName) { $cfg.StorageTaskName } else { 'AniloxRoll Storage Monitor' }

Write-Host 'PICoater 儲存電腦：第一次完整安裝' -ForegroundColor Cyan
Import-PreviousAppConfig -PreviousAppDirs @($cfg.PreviousAppDirs) -AppDir $cfg.AppDir -TaskName $taskName
Install-AppPayload -SourceDir $AppSource -AppDir $cfg.AppDir -TaskName $taskName

$scripts = Join-Path $PSScriptRoot 'scripts'
Invoke-DeployStep '1/6 網路、資料夾與 SMB 共用' (Join-Path $scripts 'configure_network_share.ps1') @{ Config = $Config }
Invoke-DeployStep '2/6 Guest SMB 權限' (Join-Path $scripts 'configure_guest_access.ps1') @{ Config = $Config }
Invoke-DeployStep '3/6 遠端桌面' (Join-Path $scripts 'configure_rdp.ps1') @{ Config = $Config }
Invoke-DeployStep '4/6 儲存程式模式' (Join-Path $scripts 'configure_app_role.ps1') @{ Config = $Config }
Invoke-DeployStep '5/6 自動啟動與異常重啟' (Join-Path $scripts 'configure_app_task.ps1') @{ Config = $Config }
Invoke-DeployStep '6/6 關閉自動睡眠' (Join-Path $deployRoot 'common\setup_nosleep.ps1')

Write-Host ''
Write-Host '[完成] 儲存電腦已安裝。' -ForegroundColor Green
Write-Host ("檢測電腦遠端資料路徑: \\" + $cfg.IpAddress + "\" + $cfg.ShareName + "\Captures")
