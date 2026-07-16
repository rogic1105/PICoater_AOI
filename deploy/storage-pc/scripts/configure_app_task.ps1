# Install the visible Storage-role WinForms app as an interactive scheduled task.
# Any logged-in local administrator may host the UI; RdpUser is only an RDP credential.
param(
    [string] $Config = ""
)

function Die([string]$msg) {
    Write-Host ("[FAIL] " + $msg) -ForegroundColor Red
    exit 1
}

$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Die "請以系統管理員身分執行 PowerShell"
}

if (-not $Config) { $Config = Join-Path (Split-Path -Parent $PSScriptRoot) 'storage-config.json' }
if (-not (Test-Path $Config)) { Die ("找不到設定檔: " + $Config) }
$json = [System.IO.File]::ReadAllText((Resolve-Path $Config).Path, [System.Text.Encoding]::UTF8)
$cfg = $json | ConvertFrom-Json

if (-not $cfg.AppDir) { Die "storage-config.json 的 AppDir 未設定" }

$appExe = Join-Path $cfg.AppDir 'AniloxRoll.Monitor.exe'
$taskName = if ($cfg.StorageTaskName) { $cfg.StorageTaskName } else { 'AniloxRoll Storage Monitor' }
if (-not (Test-Path $appExe)) {
    Write-Host ("[WARN] 尚未找到程式，跳過排程工作: " + $appExe) -ForegroundColor Yellow
    Write-Host "       部署程式後重新執行 setup.bat 即可補裝。" -ForegroundColor Yellow
    exit 0
}

$action = New-ScheduledTaskAction -Execute $appExe -WorkingDirectory $cfg.AppDir
$logonTrigger = New-ScheduledTaskTrigger -AtLogOn
# RestartCount only handles non-zero task failures. A normal window close returns success, so a
# repeating trigger is also required. While the app is alive the task remains Running and
# MultipleInstances=IgnoreNew makes each watchdog trigger a no-op.
$watchdogTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At ((Get-Date).AddMinutes(1)) `
    -RepetitionInterval (New-TimeSpan -Minutes 1)
$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew
$taskPrincipal = New-ScheduledTaskPrincipal `
    -GroupId 'BUILTIN\Administrators' `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger @($logonTrigger, $watchdogTrigger) `
    -Settings $settings `
    -Principal $taskPrincipal `
    -Force | Out-Null

try {
    Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
} catch {
    Die ("排程工作已安裝，但無法由目前登入的系統管理員立即啟動。`r`n" + $_.Exception.Message)
}

$running = $null
for ($attempt = 0; $attempt -lt 20 -and -not $running; $attempt++) {
    Start-Sleep -Milliseconds 500
    $running = Get-Process -Name 'AniloxRoll.Monitor' -ErrorAction SilentlyContinue |
        Where-Object {
            try { $_.Path -eq $appExe } catch { $false }
        } |
        Select-Object -First 1
}

if (-not $running) {
    $taskInfo = Get-ScheduledTaskInfo -TaskName $taskName -ErrorAction SilentlyContinue
    $lastResult = if ($taskInfo) { $taskInfo.LastTaskResult } else { '無法讀取' }
    $lastResultHex = if ($taskInfo) { ('0x{0:X8}' -f ($taskInfo.LastTaskResult -band 0xffffffffL)) } else { '無法讀取' }
    $crashLog = Join-Path $cfg.AppDir 'AniloxRoll.crash.log'
    Write-Host ("[FAIL] 排程已啟動，但 10 秒內未偵測到儲存程式。LastTaskResult=" + $lastResult + " (" + $lastResultHex + ")") -ForegroundColor Red
    if ($taskInfo -and $taskInfo.LastTaskResult -eq 267011) {
        Write-Host '       0x00041303 表示排程尚未執行；請確認目前登入帳號屬於本機 Administrators。' -ForegroundColor Yellow
    }
    if (Test-Path -LiteralPath $crashLog) {
        Write-Host ("       請檢查 crash log: " + $crashLog) -ForegroundColor Yellow
    }
    Write-Host '       若 Windows 安全性顯示 DLL 發行者不明，請先處理 Smart App Control/程式簽章問題。' -ForegroundColor Yellow
    exit 1
}

$registeredTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
$hasWatchdog = $registeredTask -and @($registeredTask.Triggers | Where-Object {
    $_.Repetition -and $_.Repetition.Interval -eq 'PT1M'
}).Count -gt 0
if (-not $hasWatchdog) {
    Die '排程工作已啟動，但每分鐘保活觸發未成功登記。'
}

Write-Host ("[OK] 排程工作已安裝，儲存程式已啟動: PID=" + $running.Id) -ForegroundColor Green
Write-Host '[OK] 正常關閉或異常退出後，最晚 1 分鐘內自動重新開啟。' -ForegroundColor Green
