param(
    [string] $Config = (Join-Path (Split-Path -Parent $PSScriptRoot) 'storage-config.json'),
    [ValidateRange(1, 100)]
    [int] $Cycles = 3,
    [ValidateRange(30, 300)]
    [int] $RestartTimeoutSeconds = 90,
    [string] $ReportDirectory = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

function Assert-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Run this test as administrator.'
    }
}

function Read-JsonFile([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ('File not found: ' + $Path)
    }
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $json = [IO.File]::ReadAllText($resolved, [Text.Encoding]::UTF8)
    return $json | ConvertFrom-Json
}

function Convert-HeartbeatUtc([object] $Value) {
    if ($Value -is [DateTime]) {
        return ([DateTime]$Value).ToUniversalTime()
    }

    $text = [string]$Value
    if ($text -match '^/Date\((?<milliseconds>-?\d+)(?:[+-]\d{4})?\)/$') {
        $epoch = [DateTime]::SpecifyKind(
            [DateTime]'1970-01-01T00:00:00',
            [DateTimeKind]::Utc)
        return $epoch.AddMilliseconds([double]$Matches.milliseconds)
    }

    return [DateTime]::Parse(
        $text,
        [Globalization.CultureInfo]::InvariantCulture,
        [Globalization.DateTimeStyles]::RoundtripKind).ToUniversalTime()
}

function Get-TargetProcess([string] $ExePath) {
    $items = @(
        Get-Process -Name 'AniloxRoll.Monitor' -ErrorAction SilentlyContinue |
            Where-Object {
                try {
                    $_.Path.Equals(
                        $ExePath,
                        [StringComparison]::OrdinalIgnoreCase)
                }
                catch {
                    $false
                }
            }
    )
    if ($items.Count -gt 1) {
        throw ('Multiple storage app processes found: ' + ($items.Id -join ', '))
    }
    return $items | Select-Object -First 1
}

function Wait-TargetProcess(
    [string] $ExePath,
    [int] $OldProcessId,
    [DateTime] $DeadlineUtc
) {
    while ([DateTime]::UtcNow -lt $DeadlineUtc) {
        $process = Get-TargetProcess $ExePath
        if ($process -and $process.Id -ne $OldProcessId) {
            return $process
        }
        Start-Sleep -Milliseconds 500
    }
    return $null
}

function Wait-Heartbeat(
    [string] $Path,
    [int] $ExpectedProcessId,
    [DateTime] $DeadlineUtc
) {
    while ([DateTime]::UtcNow -lt $DeadlineUtc) {
        try {
            $heartbeat = Read-JsonFile $Path
            $lastSeen = Convert-HeartbeatUtc $heartbeat.LastSeenUtc
            $age = [DateTime]::UtcNow - $lastSeen
            if ([int]$heartbeat.ProcessId -eq $ExpectedProcessId -and
                $age.TotalSeconds -ge 0 -and
                $age.TotalSeconds -le 15) {
                return $heartbeat
            }
        }
        catch {
            # Atomic heartbeat replacement can briefly expose no readable file.
        }
        Start-Sleep -Milliseconds 500
    }
    return $null
}

function Ensure-AppRunning(
    [string] $ExePath,
    [string] $TaskName,
    [int] $TimeoutSeconds
) {
    $process = Get-TargetProcess $ExePath
    if ($process) {
        return $process
    }

    Start-ScheduledTask -TaskName $TaskName
    return Wait-TargetProcess `
        -ExePath $ExePath `
        -OldProcessId 0 `
        -DeadlineUtc ([DateTime]::UtcNow.AddSeconds($TimeoutSeconds))
}

function Write-Reports(
    [string] $Directory,
    [string] $RunId,
    [string] $Overall,
    [object[]] $Rows,
    [string] $Failure,
    [int] $TimeoutSeconds
) {
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
    $csvPath = Join-Path $Directory ($RunId + '-storage-restart.csv')
    $mdPath = Join-Path $Directory ($RunId + '-storage-restart.md')
    $Rows | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding UTF8

    $builder = New-Object Text.StringBuilder
    [void]$builder.AppendLine('# Storage app restart DVT')
    [void]$builder.AppendLine()
    [void]$builder.AppendLine(('- Result: **{0}**' -f $Overall))
    [void]$builder.AppendLine(('- Completed cycles: {0}' -f $Rows.Count))
    [void]$builder.AppendLine(
        ('- Acceptance: new PID and fresh heartbeat within {0} seconds.' -f
            $TimeoutSeconds))
    if ($Failure) {
        [void]$builder.AppendLine(('- Failure: ' + $Failure.Replace('|', '\|')))
    }
    [void]$builder.AppendLine()
    [void]$builder.AppendLine(
        '| Cycle | Old PID | New PID | Process seconds | Heartbeat seconds | Result |')
    [void]$builder.AppendLine('|---:|---:|---:|---:|---:|---:|')
    foreach ($row in $Rows) {
        [void]$builder.AppendLine(
            ('| {0} | {1} | {2} | {3:N3} | {4:N3} | {5} |' -f
                $row.Cycle,
                $row.OldPid,
                $row.NewPid,
                $row.ProcessSeconds,
                $row.HeartbeatSeconds,
                $row.Result))
    }
    [IO.File]::WriteAllText(
        $mdPath,
        $builder.ToString(),
        (New-Object Text.UTF8Encoding($false)))
    return $mdPath
}

Assert-Administrator
$cfg = Read-JsonFile $Config
if (-not $cfg.AppDir) {
    throw 'storage-config.json is missing AppDir.'
}

$appDir = [IO.Path]::GetFullPath([string]$cfg.AppDir)
$appExe = Join-Path $appDir 'AniloxRoll.Monitor.exe'
$taskName = if ($cfg.StorageTaskName) {
    [string]$cfg.StorageTaskName
}
else {
    'AniloxRoll Storage Monitor'
}
$aniloxRoot = if ($cfg.AniloxRoot) {
    [IO.Path]::GetFullPath([string]$cfg.AniloxRoot)
}
else {
    'D:\Anilox'
}
$heartbeatPath = Join-Path $aniloxRoot 'Config\storage-app-heartbeat.json'
if (-not $ReportDirectory) {
    $ReportDirectory = Join-Path $aniloxRoot 'Logs\DvtReports'
}

$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
$rows = New-Object System.Collections.Generic.List[object]
$overall = 'FAIL'
$failure = ''

try {
    if (-not (Test-Path -LiteralPath $appExe -PathType Leaf)) {
        throw ('Storage app was not found: ' + $appExe)
    }

    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
    $hasWatchdog = @(
        $task.Triggers |
            Where-Object {
                $_.Repetition -and $_.Repetition.Interval -eq 'PT1M'
            }
    ).Count -gt 0
    if (-not $hasWatchdog) {
        throw ('Scheduled task has no one-minute watchdog trigger: ' + $taskName)
    }

    $current = Ensure-AppRunning $appExe $taskName $RestartTimeoutSeconds
    if (-not $current) {
        throw 'Storage app did not start before the test.'
    }
    $initialHeartbeat = Wait-Heartbeat `
        -Path $heartbeatPath `
        -ExpectedProcessId $current.Id `
        -DeadlineUtc ([DateTime]::UtcNow.AddSeconds(20))
    if (-not $initialHeartbeat) {
        throw 'Initial storage heartbeat was not healthy.'
    }

    for ($cycle = 1; $cycle -le $Cycles; $cycle++) {
        $oldProcess = Get-TargetProcess $appExe
        if (-not $oldProcess) {
            throw ('Cycle {0}: storage app was not running.' -f $cycle)
        }

        $oldPid = $oldProcess.Id
        $watch = [Diagnostics.Stopwatch]::StartNew()
        Write-Host (
            '[{0}/{1}] Killing PID={2}; waiting for the scheduled task...' -f
                $cycle, $Cycles, $oldPid) -ForegroundColor Cyan
        Stop-Process -Id $oldPid -Force

        $deadline = [DateTime]::UtcNow.AddSeconds($RestartTimeoutSeconds)
        $newProcess = Wait-TargetProcess $appExe $oldPid $deadline
        $processSeconds = $watch.Elapsed.TotalSeconds
        if (-not $newProcess) {
            throw ('Cycle {0}: no new process within {1} seconds.' -f
                $cycle, $RestartTimeoutSeconds)
        }

        $heartbeat = Wait-Heartbeat $heartbeatPath $newProcess.Id $deadline
        $heartbeatSeconds = $watch.Elapsed.TotalSeconds
        if (-not $heartbeat) {
            throw ('Cycle {0}: PID={1} did not publish a fresh heartbeat.' -f
                $cycle, $newProcess.Id)
        }
        $watch.Stop()

        $rows.Add([pscustomobject]@{
            Cycle = $cycle
            OldPid = $oldPid
            NewPid = $newProcess.Id
            ProcessSeconds = [Math]::Round($processSeconds, 3)
            HeartbeatSeconds = [Math]::Round($heartbeatSeconds, 3)
            Result = 'PASS'
        })
        Write-Host (
            '[PASS] PID={0}; process={1:N3}s; heartbeat={2:N3}s' -f
                $newProcess.Id, $processSeconds, $heartbeatSeconds) `
            -ForegroundColor Green
        Start-Sleep -Seconds 5
    }
    $overall = 'PASS'
}
catch {
    $failure = $_.Exception.Message
    Write-Host ('[FAIL] ' + $failure) -ForegroundColor Red
}
finally {
    try {
        $running = Ensure-AppRunning $appExe $taskName $RestartTimeoutSeconds
        if (-not $running) {
            $failure = ($failure + '; cleanup could not leave the storage app running.').Trim('; ')
            $overall = 'FAIL'
        }
    }
    catch {
        $failure = (
            $failure + '; cleanup failed: ' + $_.Exception.Message).Trim('; ')
        $overall = 'FAIL'
    }

    $report = Write-Reports `
        -Directory $ReportDirectory `
        -RunId $runId `
        -Overall $overall `
        -Rows $rows.ToArray() `
        -Failure $failure `
        -TimeoutSeconds $RestartTimeoutSeconds
    Write-Host ('Report: ' + $report)
}

if ($overall -eq 'PASS') {
    exit 0
}
exit 1
