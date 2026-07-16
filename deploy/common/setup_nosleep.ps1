# Disable automatic sleep, hibernation, and disk spin-down on AC and DC power.

$common = Join-Path $PSScriptRoot 'Deploy.Common.ps1'
. $common
Assert-DeployAdministrator

Write-Host '[1/4] 關閉自動睡眠...' -ForegroundColor Cyan
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0

Write-Host '[2/4] 關閉自動休眠...' -ForegroundColor Cyan
powercfg /change hibernate-timeout-ac 0
powercfg /change hibernate-timeout-dc 0

Write-Host '[3/4] 關閉硬碟自動停轉...' -ForegroundColor Cyan
powercfg /change disk-timeout-ac 0
powercfg /change disk-timeout-dc 0

Write-Host '[4/4] 驗證目前電源計畫...' -ForegroundColor Cyan
powercfg /getactivescheme
Write-Host '[OK] 電腦不會自動睡眠、休眠或停止硬碟。' -ForegroundColor Green
