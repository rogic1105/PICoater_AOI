param(
    [ValidateSet("Functional", "Unit", "Integration", "Dvt", "VirtualIo", "ReviewReport30k", "PhysicalCamera", "PhysicalInspectionStandards", "PhysicalCapture", "PhysicalIo", "PhysicalStorage", "PhysicalRecovery", "PhysicalBridgeRecovery", "PhysicalRetention", "PhysicalSoak", "PhysicalCaptureSoak", "Stress", "Soak", "All")]
    [string]$Mode = "All",
    [double]$StressMinutes = 120,
    [double]$SoakMinutes = 120,
    [double]$PhysicalSoakMinutes = 120,
    [double]$PhysicalCaptureSoakMinutes = 120,
    [switch]$RecordLatest,
    [switch]$SkipBuild,
    [string]$ImprovementSummary = "Inspect the tested commit and worktree diff for product changes; the campaign runner does not infer them."
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

if (-not ("TestRunner.NativeMethods" -as [type])) {
    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

namespace TestRunner
{
    public static class NativeMethods
    {
        [DllImport("user32.dll")]
        public static extern int GetGuiResources(IntPtr process, int flags);
    }
}
"@
}

$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "TestRunner.ResourceTrend.ps1")
$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$commit = (& git -C $repoRoot rev-parse --short HEAD).Trim()
$dirty = -not [string]::IsNullOrWhiteSpace(
    (& git -C $repoRoot status --short | Out-String))
$runDirectory = Join-Path $repoRoot ("artifacts\test-reports\" + $runId + "-" + $commit)
$latestReport = Join-Path $repoRoot ".agents\skills\add-test\references\latest-campaign.md"
$results = New-Object System.Collections.Generic.List[object]

New-Item -ItemType Directory -Force -Path $runDirectory | Out-Null

$acceptanceCriteria = @{
    "Release x64 solution build" =
        "Release|x64 build; 0 compiler errors; 0 warnings."
    "Resource trend guard tests" =
        "One-time heap expansion passes; sustained 330 MB/hour growth and post-expansion growth fail."
    "Python flow checker tests" =
        "All discovered checker self-tests pass; 0 failures."
    ".NET unit tests" =
        "All discovered unit tests pass; 0 failures."
    ".NET integration tests" =
        "All discovered integration tests pass; 0 failures."
    "DVT Runner self-check" =
        "Launch the exact app, restore changed settings, close cleanly, and finish the checker with exit code 0."
    "Virtual IO connection recovery" =
        "The app starts before IoSimulator, completes the Modbus safety handshake when the server appears, detects a graceful server exit, reconnects after the simulator restarts, restores settings, and shuts down cleanly without any START/Grab request."
    "Review and report 30,000-record DVT" =
        "Load exactly 30,000 grab IDs; reload jumps to newest; Review rapid/period navigation, enhancement, direction, heatmap, and display crop preserve data contracts; Report single/range curves, Y-axis toggle, fail filter, cross-tab curve reuse, clean shutdown, and the full checker pass."
    "Physical camera/background smoke" =
        "Connected cameras become ready; background capture, preview, Grab/Stop, image-before-curve order, cleanup, and the full checker pass."
    "Physical inspection-standard smoke" =
        "With cameras and light connected, brightness 100 and 255 produce measurable column and row responses; live normalization, mean/max thresholds, metric mode, direction, threshold lines, and O/X formulas match the configured inspection standards. This surrogate stimulus does not validate real Mura detection rate."
    "Physical IO capture cycles" =
        "Three 10-second START High cycles each open the product gate, produce an aligned first set and image-before-curve evidence, drain one tail frame per camera on Low, close cleanly, finalize an archive, and enqueue remote output."
    "Physical fixed-stop capture modes" =
        "Time mode ignores an early START Low and runs 10 seconds from the aligned first set; Height mode ignores an early START Low and stops only after all connected cameras complete 15,000 rows; both finalize archives and remote output."
    "Physical IO five-minute stability" =
        "Physical IO remains connected and Idle for 5 minutes; controller and shutdown flows complete."
    "Physical storage five-minute stability" =
        "SMB probe write and heartbeat remain green for 5 minutes; shutdown is clean."
    "Physical SMB backlog recovery" =
        "A fixed /32 Loopback blackhole isolates only the storage PC while the shared IO NIC stays online; two captures finalize locally and remain pending; after route removal, the share is write-verified, the backlog drains, heartbeat recovers, settings restore, and shutdown is clean."
    "Physical IO and light software recovery" =
        "The physical IO TCP endpoint and light serial device are each isolated and restored three times in software; every cycle raises one disconnect edge and health incident, then reconnects and resolves before clean shutdown."
    "Physical low-disk retention recovery" =
        "A marker-protected TEMP root holds two complete historical days; the threshold is derived from current free space; only the oldest day and its CSV are deleted; the newer day remains; low-space and cleanup incidents complete raise, resolve, and individual acknowledgement; settings and fixture are cleaned up."
    "Physical IO and storage soak" =
        "Fixed hardware topology; IO and storage stay green; UI always responds; Private Bytes sustained growth <=256 MB/hour and total delta <=4 GB; handles/GDI/USER/threads stay within guards; clean shutdown."
    "Physical repeated capture soak" =
        "High 10 seconds / Low 4 seconds for the configured duration; every High produces one request, gate, aligned first set, image-before-curve result, clean gate close, archive, and remote enqueue; storage and light remain green; UI and resource guards pass; clean shutdown."
    "Offline stress tests" =
        "All 9 high-frequency and mock Bridge cases pass for the configured wall-clock budget."
    "Offline endurance tests" =
        "The mixed IO, CSV/CFG, statistics, remote-copy, and cleanup workload runs continuously; queue drains; temp files clean up; Private Bytes <=512 MB, handles <=50, and threads <=15 after warm-up."
}

function Get-AcceptanceCriteria {
    param([string]$Name)
    if ($acceptanceCriteria.ContainsKey($Name)) {
        return $acceptanceCriteria[$Name]
    }
    return "Exit code 0 with no failed cases."
}

function Find-MSBuild {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $path = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild `
            -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
        if ($path) { return $path }
    }

    $command = Get-Command msbuild.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    throw "MSBuild was not found. Install Visual Studio Build Tools."
}

function Add-Result {
    param(
        [string]$Name,
        [string]$Layer,
        [string]$Status,
        [double]$DurationSeconds,
        [string]$LogPath,
        [string]$Detail
    )

    $relativeLog = ""
    if ($LogPath) {
        $relativeLog = $LogPath.Substring($repoRoot.Length).TrimStart("\")
    }
    $results.Add([pscustomobject]@{
        Name = $Name
        Layer = $Layer
        Status = $Status
        DurationSeconds = [Math]::Round($DurationSeconds, 2)
        Log = $relativeLog
        Detail = $Detail
    })
}

function Get-CommandDetail {
    param(
        [string]$Name,
        [string]$LogPath,
        [int]$ExitCode
    )

    if (-not (Test-Path -LiteralPath $LogPath)) {
        return "exit code $ExitCode"
    }

    $trxName = switch ($Name) {
        ".NET unit tests" { "unit.trx" }
        ".NET integration tests" { "integration.trx" }
        "Offline stress tests" { "stress.trx" }
        "Offline endurance tests" { "soak.trx" }
        default { $null }
    }
    if ($trxName) {
        $trxPath = Join-Path $runDirectory $trxName
        if (Test-Path -LiteralPath $trxPath) {
            [xml]$trx = Get-Content -LiteralPath $trxPath -Raw
            $counters = $trx.TestRun.ResultSummary.Counters
            if ($counters) {
                return "total $($counters.total), passed $($counters.passed), failed $($counters.failed), not-executed $($counters.notExecuted)"
            }
        }
    }

    $lines = Get-Content -LiteralPath $LogPath -Encoding UTF8
    if ($Name -eq "Release x64 solution build") {
        $summary = $lines | Where-Object {
            $_ -match "^\s*[0-9]+\s+(warnings?|errors?)"
        } | Select-Object -Last 2
        if ($summary) { return (($summary | ForEach-Object { $_.Trim() }) -join "; ") }
    }

    $testSummary = $lines | Where-Object {
        $_ -match "(Passed!|Failed!|Ran [0-9]+ tests|^OK$)"
    } | Select-Object -Last 2
    if ($testSummary) {
        return (($testSummary | ForEach-Object { $_.Trim() }) -join "; ")
    }
    return "exit code $ExitCode"
}

function Invoke-CommandStep {
    param(
        [string]$Name,
        [string]$Layer,
        [string]$Executable,
        [string[]]$Arguments,
        [hashtable]$Environment = @{}
    )

    $safeName = ($Name -replace "[^A-Za-z0-9_-]", "_").ToLowerInvariant()
    $logPath = Join-Path $runDirectory ($safeName + ".log")
    $previousEnvironment = @{}
    foreach ($key in $Environment.Keys) {
        $previousEnvironment[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable($key, [string]$Environment[$key], "Process")
    }

    Write-Host ""
    Write-Host ("========== {0} ==========" -f $Name) -ForegroundColor Cyan
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $exitCode = 1
    $savedErrorAction = $ErrorActionPreference
    try {
        Push-Location $repoRoot
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments 2>&1 |
            Tee-Object -FilePath $logPath |
            ForEach-Object { Write-Host $_ }
        $exitCode = $LASTEXITCODE
    }
    catch {
        $_ | Out-String | Tee-Object -FilePath $logPath -Append | Write-Host
        $exitCode = 1
    }
    finally {
        $ErrorActionPreference = $savedErrorAction
        Pop-Location
        foreach ($key in $Environment.Keys) {
            [Environment]::SetEnvironmentVariable(
                $key, $previousEnvironment[$key], "Process")
        }
        $watch.Stop()
    }

    $status = if ($exitCode -eq 0) { "PASS" } else { "FAIL" }
    $detail = Get-CommandDetail $Name $logPath $exitCode
    Add-Result $Name $Layer $status $watch.Elapsed.TotalSeconds $logPath `
        $detail
    return $exitCode -eq 0
}

function Stop-DvtRunnerForPath {
    param(
        [string]$RunnerPath,
        [Diagnostics.Process]$KnownProcess = $null
    )

    $fullRunnerPath = [IO.Path]::GetFullPath($RunnerPath)
    $candidates = New-Object System.Collections.Generic.List[Diagnostics.Process]
    if ($KnownProcess) {
        $candidates.Add($KnownProcess)
    }
    foreach ($candidate in @(Get-Process AniloxRoll.DvtRunner `
            -ErrorAction SilentlyContinue)) {
        if (-not $KnownProcess -or $candidate.Id -ne $KnownProcess.Id) {
            $candidates.Add($candidate)
        }
    }

    foreach ($candidate in $candidates) {
        try {
            $candidate.Refresh()
            if ($candidate.HasExited) { continue }
            $candidatePath = $candidate.Path
            if ([string]::IsNullOrWhiteSpace($candidatePath) -or
                -not [string]::Equals(
                    [IO.Path]::GetFullPath($candidatePath),
                    $fullRunnerPath,
                    [StringComparison]::OrdinalIgnoreCase)) {
                continue
            }

            Write-Host (
                "[Cleanup] Closing stale DVT Runner PID={0}" -f
                $candidate.Id) -ForegroundColor DarkYellow
            try { $candidate.CloseMainWindow() | Out-Null } catch { }
            if (-not $candidate.WaitForExit(2000)) {
                Stop-Process -Id $candidate.Id -Force -ErrorAction SilentlyContinue
                $candidate.WaitForExit(5000) | Out-Null
            }
        }
        catch {
            if ($candidate -eq $KnownProcess) {
                Write-Warning (
                    "Failed to close DVT Runner PID={0}: {1}" -f
                    $candidate.Id, $_.Exception.Message)
            }
        }
        finally {
            if ($candidate -ne $KnownProcess) {
                $candidate.Dispose()
            }
        }
    }
}

function Invoke-DvtScenario {
    param(
        [string]$ScenarioId,
        [string]$Name,
        [string]$Layer,
        [int]$SafetyTimeoutSeconds,
        [int]$DurationSeconds = 0,
        [switch]$SampleResources
    )

    $safeScenario = ($ScenarioId -replace "[^A-Za-z0-9_-]", "_").ToLowerInvariant()
    $logPath = Join-Path $runDirectory ($safeScenario + ".log")
    $resultPath = Join-Path $runDirectory ($safeScenario + ".txt")
    $processIdPath = Join-Path $runDirectory ($safeScenario + "-pid.txt")
    $runnerPath = Join-Path $repoRoot "bin\x64\Release\AniloxRoll.DvtRunner.exe"
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $status = "FAIL"
    $detail = ""
    $resourceSamples = New-Object System.Collections.Generic.List[object]
    $resourcePath = Join-Path $runDirectory ($safeScenario + "-resources.csv")

    Write-Host ""
    Write-Host ("========== {0} ==========" -f $Name) -ForegroundColor Cyan
    $process = $null
    try {
        if (-not (Test-Path -LiteralPath $runnerPath)) {
            throw "DVT Runner executable not found: $runnerPath"
        }
        Stop-DvtRunnerForPath $runnerPath

        $runnerArguments = @(
            "--scenario", $ScenarioId,
            "--result-file", ('"' + $resultPath + '"'),
            "--process-id-file", ('"' + $processIdPath + '"')
        )
        if ($DurationSeconds -gt 0) {
            $runnerArguments += @(
                "--duration-seconds", $DurationSeconds.ToString(
                    [Globalization.CultureInfo]::InvariantCulture))
        }

        $process = Start-Process -FilePath $runnerPath -PassThru `
            -ArgumentList $runnerArguments
        $deadline = [DateTime]::UtcNow.AddSeconds($SafetyTimeoutSeconds)
        $nextResourceSample = [DateTime]::UtcNow
        $monitorProcessId = 0
        while (-not $process.HasExited -and [DateTime]::UtcNow -lt $deadline) {
            # Duration belongs to the scenario's soak step. Keep sampling through
            # setup and cleanup so short qualification runs still have a trend.
            $insideResourceWindow =
                $DurationSeconds -le 0 -or
                $watch.Elapsed.TotalSeconds -lt ($DurationSeconds + 300)
            if ($SampleResources -and
                $insideResourceWindow -and
                [DateTime]::UtcNow -ge $nextResourceSample) {
                if ($monitorProcessId -eq 0 -and
                    (Test-Path -LiteralPath $processIdPath)) {
                    $pidText = (Get-Content -LiteralPath $processIdPath `
                        -Raw -ErrorAction SilentlyContinue).Trim()
                    [int]::TryParse(
                        $pidText,
                        [ref]$monitorProcessId) | Out-Null
                }
                $app = if ($monitorProcessId -gt 0) {
                    Get-Process -Id $monitorProcessId `
                        -ErrorAction SilentlyContinue
                } else {
                    $null
                }
                if ($app) {
                    $threadCollection = $null
                    try {
                        $app.Refresh()
                        $threadCollection = $app.Threads
                        $sample = [pscustomobject]@{
                            Utc = [DateTime]::UtcNow.ToString("o")
                            ElapsedSeconds = [Math]::Round(
                                $watch.Elapsed.TotalSeconds, 1)
                            WorkingSetMB = [Math]::Round(
                                $app.WorkingSet64 / 1MB, 1)
                            PrivateMB = [Math]::Round(
                                $app.PrivateMemorySize64 / 1MB, 1)
                            Handles = $app.HandleCount
                            GdiObjects = [TestRunner.NativeMethods]::GetGuiResources(
                                $app.Handle, 0)
                            UserObjects = [TestRunner.NativeMethods]::GetGuiResources(
                                $app.Handle, 1)
                            Threads = $threadCollection.Count
                            CpuSeconds = [Math]::Round(
                                $app.TotalProcessorTime.TotalSeconds, 1)
                            Responding = $app.Responding
                        }
                        $resourceSamples.Add($sample)
                        if ($resourceSamples.Count -eq 1) {
                            $sample | Export-Csv -LiteralPath $resourcePath `
                                -NoTypeInformation -Encoding UTF8
                        }
                        else {
                            $sample | Export-Csv -LiteralPath $resourcePath `
                                -NoTypeInformation -Encoding UTF8 -Append
                        }
                    }
                    catch {
                        # The app may be closing normally between lookup and sampling,
                        # but keep the evidence if instrumentation itself is broken.
                        if (-not $app.HasExited) {
                            Write-Warning (
                                "Resource sample failed: " + $_.Exception.Message)
                        }
                    }
                    finally {
                        if ($threadCollection) {
                            foreach ($thread in $threadCollection) {
                                $thread.Dispose()
                            }
                        }
                        $app.Dispose()
                    }
                }
                $nextResourceSample = [DateTime]::UtcNow.AddSeconds(30)
            }
            Start-Sleep -Milliseconds 500
            $process.Refresh()
        }

        if (-not $process.HasExited) {
            try { $process.CloseMainWindow() | Out-Null } catch { }
            if (-not $process.WaitForExit(5000)) {
                try { $process.Kill() } catch { }
            }
            throw "DVT Runner exceeded its safety timeout."
        }

        if (Test-Path -LiteralPath $resultPath) {
            Copy-Item -LiteralPath $resultPath -Destination $logPath -Force
            $text = Get-Content -LiteralPath $resultPath -Raw -Encoding UTF8
            $status = if ($process.ExitCode -eq 0 -and $text -match "Result:\s+PASS") {
                "PASS"
            } else {
                "FAIL"
            }
            $detail = (($text -split "\r?\n" | Select-Object -First 2) -join "; ")
            if ($ScenarioId -eq "review-report-30000") {
                $comboCount = if ($text -match "DT combo fill count=(\d+)") {
                    $Matches[1]
                } else {
                    "unknown"
                }
                $worstStall = if (
                    $text -match "(?m)^\[PASS\] REVIEW/U\.stall: [^\r\n]*?(\d+)ms"
                ) {
                    $Matches[1] + "ms"
                } else {
                    "unknown"
                }
                $checker = if (
                    $text -match "sessions=\d+ failSessions=\d+ PASS=(\d+) FAIL=(\d+)"
                ) {
                    $Matches[1] + " PASS / " + $Matches[2] + " FAIL"
                } else {
                    "unknown"
                }
                $detail +=
                    "; grabIds=$comboCount; maxUiStall=$worstStall; checker=$checker"
            }
            elseif ($ScenarioId -eq "physical-retention-cleanup") {
                $fixture = if (
                    $text -match
                    "oldest=deleted newer=preserved threshold=(\d+)GiB free=(\d+) fixture=(\d+)"
                ) {
                    "threshold=$($Matches[1])GiB free=$($Matches[2]) " +
                        "fixture=$($Matches[3])B oldest=deleted newer=preserved"
                }
                else {
                    "fixture verification=missing"
                }
                $freed = if (
                    $text -match "Cleanup done: freed (\d+) MB"
                ) {
                    $Matches[1] + "MB"
                }
                else {
                    "unknown"
                }
                $outputHealth = if (
                    $text -match
                    "CAPTURE/C4\.output-health: events=(\d+) states=(\d+) invalid=(\d+)"
                ) {
                    "$($Matches[1]) events/$($Matches[2]) states/" +
                        "$($Matches[3]) invalid"
                }
                else {
                    "unknown"
                }
                $checker = if (
                    $text -match
                    "sessions=\d+ failSessions=\d+ PASS=(\d+) FAIL=(\d+)"
                ) {
                    "$($Matches[1]) PASS/$($Matches[2]) FAIL"
                }
                else {
                    "unknown"
                }
                $detail =
                    "$fixture; freed=$freed; outputHealth=$outputHealth; " +
                    "checker=$checker"
            }
            elseif ($ScenarioId -eq "physical-capture-soak") {
                $countGuards = [regex]::Matches(
                    $text,
                    "count=(\d+) minimum=(\d+)")
                $cycles = if ($countGuards.Count -ge 6) {
                    "$($countGuards[0].Groups[1].Value)/" +
                        "$($countGuards[0].Groups[2].Value)"
                }
                else {
                    "missing"
                }
                $checker = if (
                    $text -match
                    "sessions=\d+ failSessions=\d+ PASS=(\d+) FAIL=(\d+)"
                ) {
                    "$($Matches[1]) PASS/$($Matches[2]) FAIL"
                }
                else {
                    "unknown"
                }
                $detail =
                    "cycles=$cycles; flowCountGuards=$($countGuards.Count)/6; " +
                    "checker=$checker"
                if ($countGuards.Count -lt 6) {
                    $status = "FAIL"
                }
            }
        }
        else {
            "DVT Runner did not produce a result file. ExitCode=$($process.ExitCode)" |
                Set-Content -LiteralPath $logPath -Encoding UTF8
            $detail = "missing result file; exit code $($process.ExitCode)"
        }

        if ($SampleResources) {
            if ($resourceSamples.Count -lt 2) {
                $status = "FAIL"
                $detail += "; resource sampling produced fewer than 2 samples"
            }
            else {
                # Windows PowerShell 5 cannot reliably apply @() directly to a
                # generic List[object]. Materialize first so resource guards
                # cannot fail open with "Argument types do not match".
                $allResourceSamples = $resourceSamples.ToArray()
                $resourceTrend = Get-ResourceTrend $allResourceSamples
                $steady = $resourceTrend.Samples
                $first = $resourceTrend.First
                $last = $resourceTrend.Last
                $maxPrivate = ($steady | Measure-Object PrivateMB -Maximum).Maximum
                $maxHandles = ($steady | Measure-Object Handles -Maximum).Maximum
                $privateDelta = [Math]::Round(
                    $resourceTrend.PrivateDeltaMB, 1)
                $handleDelta = $last.Handles - $first.Handles
                $gdiDelta = $last.GdiObjects - $first.GdiObjects
                $userDelta = $last.UserObjects - $first.UserObjects
                $threadDelta = $last.Threads - $first.Threads
                $steadySeconds = $resourceTrend.SteadySeconds
                $privateRatePerHour = [Math]::Round(
                    $resourceTrend.TotalPrivateRateMBPerHour, 1)
                $medianPrivateRatePerHour = [Math]::Round(
                    $resourceTrend.MedianPrivateRateMBPerHour, 1)
                $postExpansionRatePerHour = [Math]::Round(
                    $resourceTrend.PostExpansionRateMBPerHour, 1)
                $postExpansionSeconds = [Math]::Round(
                    $resourceTrend.PostExpansionSeconds, 1)
                $largestPrivateStep = [Math]::Round(
                    $resourceTrend.LargestPositiveStepMB, 1)
                $cycleTroughDelta = [Math]::Round(
                    $resourceTrend.CycleTroughDeltaMB, 1)
                $cycleTroughRate = [Math]::Round(
                    $resourceTrend.CycleTroughRateMBPerHour, 1)
                $handleRatePerHour = [Math]::Round(
                    $handleDelta * 3600 / $steadySeconds, 1)
                $notResponding = @($steady | Where-Object {
                    -not $_.Responding
                }).Count
                $detail += ((
                    "; resources samples={0} privateMB={1}->{2} max={3} " +
                    "handles={4}->{5} max={6} gdi={7}->{8} user={9}->{10} " +
                    "threads={11}->{12} ratesPerHour=private:{13}MB " +
                    "medianPrivate:{14}MB postExpansionPrivate:{15}MB " +
                    "postExpansionSeconds={16} largestPrivateStepMB={17} " +
                    "cyclic={18} troughDeltaMB={19} troughRateMBPerHour={20} " +
                    "handles:{21} observer=external") -f
                    $resourceSamples.Count,
                    $first.PrivateMB, $last.PrivateMB, $maxPrivate,
                    $first.Handles, $last.Handles, $maxHandles,
                    $first.GdiObjects, $last.GdiObjects,
                    $first.UserObjects, $last.UserObjects,
                    $first.Threads, $last.Threads,
                    $privateRatePerHour, $medianPrivateRatePerHour,
                    $postExpansionRatePerHour, $postExpansionSeconds,
                    $largestPrivateStep, $resourceTrend.CyclicExpansion,
                    $cycleTroughDelta, $cycleTroughRate, $handleRatePerHour)

                $privateLeak = $resourceTrend.PrivateLeak
                $handleRateLeak =
                    $handleDelta -ge 50 -and
                    $handleRatePerHour -gt 200

                if ($notResponding -gt 0 -or
                    $privateLeak -or
                    $handleDelta -gt 200 -or
                    $gdiDelta -gt 100 -or
                    $userDelta -gt 100 -or
                    $threadDelta -gt 25 -or
                    ($steadySeconds -ge 180 -and
                     $handleRateLeak)) {
                    $status = "FAIL"
                    $detail += ((
                        "; resource guard failed privateDeltaMB={0} " +
                        "handleDelta={1} gdiDelta={2} userDelta={3} " +
                        "threadDelta={4} notResponding={5} " +
                        "privateRateMBPerHour={6} " +
                        "medianPrivateRateMBPerHour={7} " +
                        "postExpansionRateMBPerHour={8} " +
                        "postExpansionSeconds={9} cycleTroughDeltaMB={10} " +
                        "cycleTroughRateMBPerHour={11} handleRatePerHour={12}") -f
                        $privateDelta, $handleDelta, $gdiDelta, $userDelta,
                        $threadDelta, $notResponding,
                        $privateRatePerHour, $medianPrivateRatePerHour,
                        $postExpansionRatePerHour, $postExpansionSeconds,
                        $cycleTroughDelta, $cycleTroughRate,
                        $handleRatePerHour)
                }
            }
        }
    }
    catch {
        $_ | Out-String | Set-Content -LiteralPath $logPath -Encoding UTF8
        $status = "FAIL"
        $detail = "campaign analysis failed: " + $_.Exception.Message
    }
    finally {
        if ($process) {
            Stop-DvtRunnerForPath $runnerPath $process
            $process.Dispose()
        }
        $watch.Stop()
    }

    Add-Result $Name $Layer $status $watch.Elapsed.TotalSeconds $logPath $detail
    return $status -eq "PASS"
}

function Test-ModeIncludes {
    param([string]$Item)
    if ($Mode -eq "All") { return $true }
    if ($Mode -eq "Functional") {
        return $Item -in @("Build", "Python", "Unit", "Integration", "Dvt")
    }
    if ($Mode -eq $Item) { return $true }
    return $Item -eq "Build" -and -not $SkipBuild
}

function Write-CampaignReport {
    $finished = Get-Date
    $overall = if ($results.Status -contains "FAIL") { "FAIL" } else { "PASS" }
    $builder = New-Object Text.StringBuilder
    [void]$builder.AppendLine("# Latest automated test campaign")
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("> This file is overwritten by the next recorded campaign. Git history is the durable record.")
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("- Result: **$overall**")
    [void]$builder.AppendLine("- Run: ``$runId``")
    [void]$builder.AppendLine("- Commit: ``$commit``")
    [void]$builder.AppendLine("- Working tree: " + $(if ($dirty) { "dirty" } else { "clean" }))
    [void]$builder.AppendLine("- Mode: ``$Mode``")
    [void]$builder.AppendLine("- Machine: ``$env:COMPUTERNAME``")
    [void]$builder.AppendLine("- Finished: $($finished.ToString("yyyy-MM-dd HH:mm:ss zzz"))")
    [void]$builder.AppendLine("- Raw artifacts: ``artifacts/test-reports/$runId-$commit/`` (local, ignored by Git)")
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("## Results")
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("| Layer | Check | Result | Theory / acceptance | Experimental value / evidence | Seconds |")
    [void]$builder.AppendLine("|---|---|---:|---|---|---:|")
    foreach ($result in $results) {
        $criteria = (Get-AcceptanceCriteria $result.Name) -replace "\|", "\|"
        $detail = $result.Detail -replace "\|", "\|"
        [void]$builder.AppendLine(
            "| $($result.Layer) | $($result.Name) | **$($result.Status)** | $criteria | $detail | $($result.DurationSeconds) |")
    }
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("## Improvement record")
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("- $ImprovementSummary")
    [void]$builder.AppendLine("- The commit STAR body records the exact implementation change and verified result.")
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("## Not covered by this campaign")
    [void]$builder.AppendLine()
    if ($Mode -in @("PhysicalCamera", "PhysicalInspectionStandards", "PhysicalCapture", "PhysicalCaptureSoak", "PhysicalRecovery")) {
        [void]$builder.AppendLine("- Seven-camera full-load acquisition remains untested; this run covered only the connected cameras.")
        if ($Mode -in @("PhysicalCapture", "PhysicalCaptureSoak", "PhysicalRecovery")) {
            [void]$builder.AppendLine("- Background capture and preview are covered by the separate PhysicalCamera scenario, not this run.")
        }
    }
    else {
        [void]$builder.AppendLine("- Physical camera/grabber acquisition, seven-camera frame load, background capture, and live Grab.")
    }
    if ($Mode -eq "PhysicalBridgeRecovery") {
        [void]$builder.AppendLine("- Physical IO/light cable removal and power-cycle recovery remain untested; this run covered repeatable software endpoint and device isolation.")
    }
    else {
        [void]$builder.AppendLine("- Physical IO and light disconnect/reconnect timing.")
    }
    if ($Mode -eq "PhysicalRecovery") {
        [void]$builder.AppendLine("- Physical cable/switch interruption and real-disk/UI low-space status remain untested; this run covered repeatable software SMB isolation and backlog recovery.")
    }
    elseif ($Mode -eq "PhysicalRetention") {
        [void]$builder.AppendLine("- Storage-PC SMB interruption and backlog transfer are covered by their separate recovery campaign, not this run.")
        [void]$builder.AppendLine("- Retention on the storage PC's own local disk remains separate; this run exercised the shared cleanup core and inspection-PC UI state with a marker-protected isolated fixture.")
    }
    else {
        [void]$builder.AppendLine("- Storage-PC SMB interruption, remote backlog transfer, and real-disk/UI low-space status and recovery.")
    }
    [void]$builder.AppendLine("- Shift/24-hour product soak with the IO simulator, cameras, storage transfer, and operator interactions.")
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("These cases remain **NOT COVERED**, not PASS. Run their dedicated DVT or soak campaign before release.")

    $runReport = Join-Path $runDirectory "campaign-report.md"
    [IO.File]::WriteAllText($runReport, $builder.ToString(), (New-Object Text.UTF8Encoding($false)))
    if ($RecordLatest) {
        [IO.File]::WriteAllText($latestReport, $builder.ToString(), (New-Object Text.UTF8Encoding($false)))
    }
    return $overall
}

$allPassed = $true

if ((Test-ModeIncludes "Build") -and -not $SkipBuild) {
    Stop-DvtRunnerForPath (
        Join-Path $repoRoot "bin\x64\Release\AniloxRoll.DvtRunner.exe")
    $msbuild = Find-MSBuild
    $allPassed = (Invoke-CommandStep "Release x64 solution build" "Build" $msbuild @(
        "PICoater_AOI.sln",
        "/t:Build",
        "/p:Configuration=Release",
        "/p:Platform=x64",
        "/m",
        "/nologo"
    )) -and $allPassed
}

if ($Mode -in @("Functional", "Unit", "PhysicalSoak", "All")) {
    $allPassed = (Invoke-CommandStep "Resource trend guard tests" "Unit" `
        "powershell.exe" @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", "tests/TestRunner.ResourceTrend.Tests.ps1"
        )) -and $allPassed
}

if (Test-ModeIncludes "Python") {
    $allPassed = (Invoke-CommandStep "Python flow checker tests" "Functional" "python" @(
        "-m", "unittest", "discover",
        "-s", "tools/python/tests",
        "-p", "test_*.py"
    )) -and $allPassed
}

if (Test-ModeIncludes "Unit") {
    $allPassed = (Invoke-CommandStep ".NET unit tests" "Unit" "dotnet" @(
        "test",
        "tests/AniloxRoll.Monitor.Tests/AniloxRoll.Monitor.Tests.csproj",
        "--configuration", "Release",
        "-p:Platform=x64",
        "--no-restore",
        "--results-directory", $runDirectory,
        "--logger", "trx;LogFileName=unit.trx"
    )) -and $allPassed
}

if (Test-ModeIncludes "Integration") {
    $allPassed = (Invoke-CommandStep ".NET integration tests" "Integration" "dotnet" @(
        "test",
        "tests/AniloxRoll.Monitor.Integration.Tests/AniloxRoll.Monitor.Integration.Tests.csproj",
        "--configuration", "Release",
        "-p:Platform=x64",
        "--no-restore",
        "--results-directory", $runDirectory,
        "--logger", "trx;LogFileName=integration.trx"
    )) -and $allPassed
}

if (Test-ModeIncludes "Dvt") {
    $allPassed = (Invoke-DvtScenario "runner-self-check" `
        "DVT Runner self-check" "DVT functional" 600) -and $allPassed
}

if ($Mode -in @("ReviewReport30k", "All")) {
    $allPassed = (Invoke-DvtScenario "review-report-30000" `
        "Review and report 30,000-record DVT" `
        "Large-data UI DVT" 2400) -and $allPassed
}

if ($Mode -eq "PhysicalCamera") {
    $allPassed = (Invoke-DvtScenario "monitor-background-v1" `
        "Physical camera/background smoke" "Physical camera DVT" 900) -and $allPassed
}

if ($Mode -eq "VirtualIo") {
    $allPassed = (Invoke-DvtScenario "virtual-io-recovery" `
        "Virtual IO connection recovery" "Virtual IO DVT" 600) -and $allPassed
}

if ($Mode -eq "PhysicalInspectionStandards") {
    $allPassed = (Invoke-DvtScenario "monitor-inspection-standards" `
        "Physical inspection-standard smoke" `
        "Physical inspection-standard DVT" 900) -and $allPassed
}

if ($Mode -eq "PhysicalCapture") {
    $allPassed = (Invoke-DvtScenario "physical-io-capture" `
        "Physical IO capture cycles" "Physical acquisition DVT" 900) -and $allPassed
    $allPassed = (Invoke-DvtScenario "physical-fixed-stop-capture" `
        "Physical fixed-stop capture modes" "Physical acquisition DVT" 900) -and $allPassed
}

if ($Mode -eq "PhysicalIo") {
    $allPassed = (Invoke-DvtScenario "physical-io-stability" `
        "Physical IO five-minute stability" "Physical IO DVT" 600) -and $allPassed
}

if ($Mode -eq "PhysicalStorage") {
    $allPassed = (Invoke-DvtScenario "physical-storage-stability" `
        "Physical storage five-minute stability" "Physical storage DVT" 600) -and $allPassed
}

if ($Mode -eq "PhysicalRecovery") {
    $allPassed = (Invoke-DvtScenario "physical-smb-backlog-recovery" `
        "Physical SMB backlog recovery" "Physical storage recovery DVT" 1200) -and $allPassed
}

if ($Mode -eq "PhysicalBridgeRecovery") {
    $allPassed = (Invoke-DvtScenario "physical-bridge-recovery" `
        "Physical IO and light software recovery" `
        "Physical bridge recovery DVT" 1200) -and $allPassed
}

if ($Mode -eq "PhysicalRetention") {
    $allPassed = (Invoke-DvtScenario "physical-retention-cleanup" `
        "Physical low-disk retention recovery" `
        "Physical retention DVT" 1200) -and $allPassed
}

if ($Mode -eq "PhysicalSoak") {
    $physicalSoakSeconds = [Math]::Max(
        1,
        [int][Math]::Round($PhysicalSoakMinutes * 60))
    $allPassed = (Invoke-DvtScenario "physical-io-storage-soak" `
        "Physical IO and storage soak" "Physical soak" `
        ($physicalSoakSeconds + 600) $physicalSoakSeconds `
        -SampleResources) -and $allPassed
}

if ($Mode -eq "PhysicalCaptureSoak") {
    $physicalCaptureSoakSeconds = [Math]::Max(
        14,
        [int][Math]::Round($PhysicalCaptureSoakMinutes * 60))
    $allPassed = (Invoke-DvtScenario "physical-capture-soak" `
        "Physical repeated capture soak" "Physical capture soak" `
        ($physicalCaptureSoakSeconds + 900) $physicalCaptureSoakSeconds `
        -SampleResources) -and $allPassed
}

if (Test-ModeIncludes "Stress") {
    $allPassed = (Invoke-CommandStep "Offline stress tests" "Stress" "dotnet" @(
        "test",
        "tests/AniloxRoll.Monitor.Stress.Tests/AniloxRoll.Monitor.Stress.Tests.csproj",
        "--configuration", "Release",
        "-p:Platform=x64",
        "--no-restore",
        "--filter", "(TestCategory=Stress|TestCategory=BridgeStress)",
        "--results-directory", $runDirectory,
        "--logger", "trx;LogFileName=stress.trx"
    ) @{ STRESS_MINUTES = $StressMinutes.ToString(
            [Globalization.CultureInfo]::InvariantCulture) }) -and $allPassed
}

if (Test-ModeIncludes "Soak") {
    $allPassed = (Invoke-CommandStep "Offline endurance tests" "Soak" "dotnet" @(
        "test",
        "tests/AniloxRoll.Monitor.Stress.Tests/AniloxRoll.Monitor.Stress.Tests.csproj",
        "--configuration", "Release",
        "-p:Platform=x64",
        "--no-restore",
        "--filter", "TestCategory=Soak",
        "--results-directory", $runDirectory,
        "--logger", "trx;LogFileName=soak.trx"
    ) @{ SOAK_MINUTES = $SoakMinutes.ToString(
            [Globalization.CultureInfo]::InvariantCulture) }) -and $allPassed
}

$overall = Write-CampaignReport
Write-Host ""
Write-Host ("Campaign result: {0}" -f $overall) `
    -ForegroundColor $(if ($overall -eq "PASS") { "Green" } else { "Red" })
Write-Host ("Report: {0}" -f (Join-Path $runDirectory "campaign-report.md"))
if ($RecordLatest) {
    Write-Host ("Recorded summary: {0}" -f $latestReport)
}

if (-not $allPassed) { exit 1 }
exit 0
