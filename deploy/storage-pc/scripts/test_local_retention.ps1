param(
    [string] $Config = (Join-Path (Split-Path -Parent $PSScriptRoot) 'storage-config.json'),
    [ValidateRange(30, 300)]
    [int] $RestartTimeoutSeconds = 90,
    [ValidateRange(30, 600)]
    [int] $CleanupTimeoutSeconds = 180,
    [string] $ReportDirectory = '',
    [switch] $ConfirmInPlaceDelete
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$GiB = 1024L * 1024L * 1024L
$RequiredCaptureRoot = 'D:\Anilox\Captures'

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
    $raw = [IO.File]::ReadAllText(
        (Resolve-Path -LiteralPath $Path).Path,
        [Text.Encoding]::UTF8)
    return $raw | ConvertFrom-Json
}

function Write-Utf8Atomic([string] $Path, [string] $Content) {
    $directory = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    $temporary = $Path + '.dvt-part-' + [Guid]::NewGuid().ToString('N')
    try {
        [IO.File]::WriteAllText(
            $temporary,
            $Content,
            (New-Object Text.UTF8Encoding($false)))
        Move-Item -LiteralPath $temporary -Destination $Path -Force
    }
    finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
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
            # The app replaces heartbeat atomically, so a transient miss is valid.
        }
        Start-Sleep -Milliseconds 500
    }
    return $null
}

function Restart-StorageApp(
    [string] $ExePath,
    [string] $TaskName,
    [int] $TimeoutSeconds
) {
    $old = Get-TargetProcess $ExePath
    $oldPid = if ($old) { $old.Id } else { 0 }
    if ($old) {
        Stop-Process -Id $old.Id -Force
        try { Wait-Process -Id $old.Id -Timeout 15 -ErrorAction Stop }
        catch { }
    }
    Start-ScheduledTask -TaskName $TaskName
    return Wait-TargetProcess `
        -ExePath $ExePath `
        -OldProcessId $oldPid `
        -DeadlineUtc ([DateTime]::UtcNow.AddSeconds($TimeoutSeconds))
}

function Get-DayRecords([string] $Root) {
    $records = New-Object System.Collections.Generic.List[object]
    foreach ($year in @(Get-ChildItem -LiteralPath $Root -Directory -ErrorAction Stop)) {
        if ($year.Name -notmatch '^\d{4}$') { continue }
        foreach ($month in @(Get-ChildItem -LiteralPath $year.FullName -Directory -ErrorAction Stop)) {
            if ($month.Name -notmatch '^\d{6}$') { continue }
            foreach ($day in @(Get-ChildItem -LiteralPath $month.FullName -Directory -ErrorAction Stop)) {
                if ($day.Name -notmatch '^\d{8}$') { continue }
                $parsed = [DateTime]::MinValue
                if (-not [DateTime]::TryParseExact(
                    $day.Name,
                    'yyyyMMdd',
                    [Globalization.CultureInfo]::InvariantCulture,
                    [Globalization.DateTimeStyles]::None,
                    [ref]$parsed)) {
                    continue
                }
                if ($parsed.Date -ge [DateTime]::Today) { continue }
                $dailyCsv = Join-Path $month.FullName ($day.Name + '.csv')
                $bytes = 0L
                foreach ($file in @(Get-ChildItem -LiteralPath $day.FullName -File -Recurse -ErrorAction Stop)) {
                    $bytes += $file.Length
                }
                if (Test-Path -LiteralPath $dailyCsv -PathType Leaf) {
                    $bytes += (Get-Item -LiteralPath $dailyCsv).Length
                }
                $records.Add([pscustomobject]@{
                    Date = $parsed.Date
                    Path = $day.FullName
                    DailyCsv = $dailyCsv
                    DailyCsvExisted = Test-Path -LiteralPath $dailyCsv -PathType Leaf
                    Bytes = $bytes
                })
            }
        }
    }
    return @($records | Sort-Object Date)
}

function Write-Report(
    [string] $Path,
    [string] $Result,
    [string] $Failure,
    [object] $Before,
    [object] $After,
    [int] $ThresholdGb,
    [object] $Oldest,
    [int] $DeletedDays,
    [bool] $SettingsRestored,
    [bool] $AppHealthy
) {
    $builder = New-Object Text.StringBuilder
    [void]$builder.AppendLine('# Storage local retention DVT')
    [void]$builder.AppendLine()
    [void]$builder.AppendLine(('- Result: **{0}**' -f $Result))
    [void]$builder.AppendLine(('- Root: `{0}`' -f $RequiredCaptureRoot))
    [void]$builder.AppendLine(('- Temporary threshold: {0} GiB' -f $ThresholdGb))
    [void]$builder.AppendLine(('- Oldest day: `{0}` ({1:N3} GiB)' -f
        $Oldest.Path, ($Oldest.Bytes / [double]$GiB)))
    [void]$builder.AppendLine(('- Deleted complete days: {0}' -f $DeletedDays))
    [void]$builder.AppendLine(('- Settings restored: {0}' -f $SettingsRestored))
    [void]$builder.AppendLine(('- Storage app healthy after restore: {0}' -f $AppHealthy))
    if ($Failure) {
        [void]$builder.AppendLine(('- Failure: ' + $Failure.Replace('|', '\|')))
    }
    [void]$builder.AppendLine()
    [void]$builder.AppendLine('| Metric | Theory | Experiment |')
    [void]$builder.AppendLine('|---|---:|---:|')
    [void]$builder.AppendLine(('| Free space before | below {0} GiB | {1:N3} GiB |' -f
        $ThresholdGb, ($Before.FreeBytes / [double]$GiB)))
    [void]$builder.AppendLine(('| Free space after | at least {0} GiB | {1:N3} GiB |' -f
        $ThresholdGb, ($After.FreeBytes / [double]$GiB)))
    [void]$builder.AppendLine(('| Oldest day folder | deleted | {0} |' -f
        (-not (Test-Path -LiteralPath $Oldest.Path))))
    [void]$builder.AppendLine(('| Same-day CSV | deleted if present | {0} |' -f
        (-not $Oldest.DailyCsvExisted -or -not (Test-Path -LiteralPath $Oldest.DailyCsv))))
    [IO.File]::WriteAllText(
        $Path,
        $builder.ToString(),
        (New-Object Text.UTF8Encoding($false)))
}

Assert-Administrator
if (-not $ConfirmInPlaceDelete) {
    throw 'Explicit -ConfirmInPlaceDelete is required.'
}

$cfg = Read-JsonFile $Config
$aniloxRoot = [IO.Path]::GetFullPath([string]$cfg.AniloxRoot).TrimEnd('\')
$captureRoot = [IO.Path]::GetFullPath(
    (Join-Path $aniloxRoot 'Captures')).TrimEnd('\')
if (-not $captureRoot.Equals(
    $RequiredCaptureRoot,
    [StringComparison]::OrdinalIgnoreCase)) {
    throw ('Refusing destructive test outside ' + $RequiredCaptureRoot +
        '. Resolved root: ' + $captureRoot)
}
if (-not (Test-Path -LiteralPath $captureRoot -PathType Container)) {
    throw ('Capture root not found: ' + $captureRoot)
}

$appDir = [IO.Path]::GetFullPath([string]$cfg.AppDir)
$appExe = Join-Path $appDir 'AniloxRoll.Monitor.exe'
$appModePath = Join-Path $appDir 'Config\app-mode.json'
$taskName = if ($cfg.StorageTaskName) {
    [string]$cfg.StorageTaskName
}
else {
    'AniloxRoll Storage Monitor'
}
$heartbeatPath = Join-Path $aniloxRoot 'Config\storage-app-heartbeat.json'
if (-not $ReportDirectory) {
    $ReportDirectory = Join-Path $aniloxRoot 'Logs\DvtReports'
}
New-Item -ItemType Directory -Path $ReportDirectory -Force | Out-Null

$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
$reportPath = Join-Path $ReportDirectory ($runId + '-storage-local-retention.md')
$backupPath = Join-Path $ReportDirectory ($runId + '-app-mode.backup.json')
$originalRaw = ''
$before = $null
$after = $null
$oldest = $null
$thresholdGb = 0
$deletedDays = 0
$settingsRestored = $false
$appHealthy = $false
$result = 'FAIL'
$failure = ''

try {
    if (-not (Test-Path -LiteralPath $appExe -PathType Leaf)) {
        throw ('Storage app not found: ' + $appExe)
    }
    if (-not (Test-Path -LiteralPath $appModePath -PathType Leaf)) {
        throw ('Storage app mode config not found: ' + $appModePath)
    }
    Get-ScheduledTask -TaskName $taskName -ErrorAction Stop | Out-Null

    $dayRecords = @(Get-DayRecords $captureRoot)
    if ($dayRecords.Count -lt 2) {
        throw 'At least two completed day folders are required.'
    }
    $oldest = $dayRecords[0]

    $drive = New-Object IO.DriveInfo([IO.Path]::GetPathRoot($captureRoot))
    $before = [pscustomobject]@{
        FreeBytes = $drive.AvailableFreeSpace
        TotalBytes = $drive.TotalSize
    }
    $thresholdGb = [int][Math]::Floor($before.FreeBytes / [double]$GiB) + 1
    $maximumGb = [int][Math]::Floor($before.TotalBytes / [double]$GiB) - 1
    if ($thresholdGb -gt $maximumGb) { $thresholdGb = $maximumGb }
    $requiredBytes = ($thresholdGb * $GiB) - $before.FreeBytes
    if ($requiredBytes -le 0) {
        throw 'Could not derive a threshold above current free space.'
    }
    if ($oldest.Bytes -lt $requiredBytes) {
        throw ('Oldest day is too small to prove one-day stop behavior. Required={0}, oldest={1}.' -f
            $requiredBytes, $oldest.Bytes)
    }

    $originalRaw = [IO.File]::ReadAllText($appModePath, [Text.Encoding]::UTF8)
    [IO.File]::WriteAllText(
        $backupPath,
        $originalRaw,
        (New-Object Text.UTF8Encoding($false)))
    $appMode = $originalRaw | ConvertFrom-Json
    $appMode.StorageMinFreeGB = $thresholdGb
    Write-Utf8Atomic $appModePath ($appMode | ConvertTo-Json -Depth 10)

    Write-Host ('[TEST] Free={0:N3} GiB; threshold={1} GiB; oldest={2} ({3:N3} GiB)' -f
        ($before.FreeBytes / [double]$GiB),
        $thresholdGb,
        $oldest.Path,
        ($oldest.Bytes / [double]$GiB)) -ForegroundColor Cyan

    $testProcess = Restart-StorageApp $appExe $taskName $RestartTimeoutSeconds
    if (-not $testProcess) {
        throw 'Storage app did not restart with the temporary threshold.'
    }
    $testHeartbeat = Wait-Heartbeat `
        -Path $heartbeatPath `
        -ExpectedProcessId $testProcess.Id `
        -DeadlineUtc ([DateTime]::UtcNow.AddSeconds($RestartTimeoutSeconds))
    if (-not $testHeartbeat) {
        throw 'Storage app did not publish a fresh heartbeat.'
    }

    $deadline = [DateTime]::UtcNow.AddSeconds($CleanupTimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline -and
        (Test-Path -LiteralPath $oldest.Path)) {
        Start-Sleep -Milliseconds 500
    }

    $drive = New-Object IO.DriveInfo([IO.Path]::GetPathRoot($captureRoot))
    $after = [pscustomobject]@{
        FreeBytes = $drive.AvailableFreeSpace
        TotalBytes = $drive.TotalSize
    }
    $remaining = @(Get-DayRecords $captureRoot)
    $deletedDays = @(
        $dayRecords |
            Where-Object {
                -not (Test-Path -LiteralPath $_.Path -PathType Container)
            }
    ).Count

    if (Test-Path -LiteralPath $oldest.Path) {
        throw ('Oldest day was not deleted within {0} seconds.' -f
            $CleanupTimeoutSeconds)
    }
    if ($oldest.DailyCsvExisted -and
        (Test-Path -LiteralPath $oldest.DailyCsv -PathType Leaf)) {
        throw ('Same-day CSV was not deleted: ' + $oldest.DailyCsv)
    }
    if ($deletedDays -ne 1) {
        throw ('Expected exactly one deleted day, observed ' + $deletedDays + '.')
    }
    if ($after.FreeBytes -lt ($thresholdGb * $GiB)) {
        throw ('Free space did not reach the temporary threshold: {0:N3} GiB.' -f
            ($after.FreeBytes / [double]$GiB))
    }
    foreach ($expected in @($dayRecords | Select-Object -Skip 1)) {
        if (-not (Test-Path -LiteralPath $expected.Path -PathType Container)) {
            throw ('A newer day was unexpectedly deleted: ' + $expected.Path)
        }
    }
    $result = 'PASS'
}
catch {
    $failure = $_.Exception.Message
    Write-Host ('[FAIL] ' + $failure) -ForegroundColor Red
}
finally {
    if ($originalRaw) {
        try {
            Write-Utf8Atomic $appModePath $originalRaw
            $settingsRestored = $true
            $restoredProcess =
                Restart-StorageApp $appExe $taskName $RestartTimeoutSeconds
            if ($restoredProcess) {
                $restoredHeartbeat = Wait-Heartbeat `
                    -Path $heartbeatPath `
                    -ExpectedProcessId $restoredProcess.Id `
                    -DeadlineUtc ([DateTime]::UtcNow.AddSeconds($RestartTimeoutSeconds))
                $appHealthy = $null -ne $restoredHeartbeat
            }
        }
        catch {
            $failure = ($failure + '; restore failed: ' +
                $_.Exception.Message).Trim('; ')
        }
    }
    if (-not $settingsRestored -or -not $appHealthy) {
        $result = 'FAIL'
    }
    if (-not $after -and $before) {
        $drive = New-Object IO.DriveInfo([IO.Path]::GetPathRoot($captureRoot))
        $after = [pscustomobject]@{
            FreeBytes = $drive.AvailableFreeSpace
            TotalBytes = $drive.TotalSize
        }
    }
    if ($before -and $after -and $oldest) {
        Write-Report `
            -Path $reportPath `
            -Result $result `
            -Failure $failure `
            -Before $before `
            -After $after `
            -ThresholdGb $thresholdGb `
            -Oldest $oldest `
            -DeletedDays $deletedDays `
            -SettingsRestored $settingsRestored `
            -AppHealthy $appHealthy
        Write-Host ('Report: ' + $reportPath)
    }
}

if ($result -eq 'PASS') {
    Write-Host '[PASS] One oldest complete day was deleted and settings were restored.' `
        -ForegroundColor Green
    exit 0
}
exit 1
