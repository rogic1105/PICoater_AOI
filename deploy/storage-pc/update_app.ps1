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

Write-Host 'PICoater 儲存電腦：更新程式' -ForegroundColor Cyan
Import-PreviousAppConfig -PreviousAppDirs @($cfg.PreviousAppDirs) -AppDir $cfg.AppDir -TaskName $taskName
Install-AppPayload -SourceDir $AppSource -AppDir $cfg.AppDir -TaskName $taskName

$scripts = Join-Path $PSScriptRoot 'scripts'
Invoke-DeployStep '套用儲存程式模式' (Join-Path $scripts 'configure_app_role.ps1') @{ Config = $Config }
Invoke-DeployStep '更新自動啟動工作' (Join-Path $scripts 'configure_app_task.ps1') @{ Config = $Config }

Write-Host ''
Write-Host '[完成] 程式已更新；Windows 網路與共用設定沒有重做。' -ForegroundColor Green
