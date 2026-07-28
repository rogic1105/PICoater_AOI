param(
    [ValidateSet("Functional", "Unit", "Integration", "Dvt", "PhysicalIo", "PhysicalStorage", "Stress", "Soak", "All")]
    [string]$Mode = "All",
    [double]$StressMinutes = 1,
    [double]$SoakMinutes = 10,
    [switch]$RecordLatest,
    [switch]$SkipBuild,
    [string]$ImprovementSummary = "No product behavior change; verification campaign only."
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

$repoRoot = Split-Path -Parent $PSScriptRoot
$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$commit = (& git -C $repoRoot rev-parse --short HEAD).Trim()
$dirty = -not [string]::IsNullOrWhiteSpace(
    (& git -C $repoRoot status --short | Out-String))
$runDirectory = Join-Path $repoRoot ("artifacts\test-reports\" + $runId + "-" + $commit)
$latestReport = Join-Path $repoRoot ".agents\skills\add-test\references\latest-campaign.md"
$results = New-Object System.Collections.Generic.List[object]

New-Item -ItemType Directory -Force -Path $runDirectory | Out-Null

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

function Invoke-DvtScenario {
    param(
        [string]$ScenarioId,
        [string]$Name,
        [string]$Layer,
        [int]$SafetyTimeoutSeconds
    )

    $safeScenario = ($ScenarioId -replace "[^A-Za-z0-9_-]", "_").ToLowerInvariant()
    $logPath = Join-Path $runDirectory ($safeScenario + ".log")
    $resultPath = Join-Path $runDirectory ($safeScenario + ".txt")
    $runnerPath = Join-Path $repoRoot "bin\x64\Release\AniloxRoll.DvtRunner.exe"
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $status = "FAIL"
    $detail = ""

    Write-Host ""
    Write-Host ("========== {0} ==========" -f $Name) -ForegroundColor Cyan
    try {
        if (-not (Test-Path -LiteralPath $runnerPath)) {
            throw "DVT Runner executable not found: $runnerPath"
        }

        $process = Start-Process -FilePath $runnerPath -PassThru -ArgumentList @(
            "--scenario", $ScenarioId,
            "--result-file", ('"' + $resultPath + '"')
        )
        if (-not $process.WaitForExit($SafetyTimeoutSeconds * 1000)) {
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
        }
        else {
            "DVT Runner did not produce a result file. ExitCode=$($process.ExitCode)" |
                Set-Content -LiteralPath $logPath -Encoding UTF8
            $detail = "missing result file; exit code $($process.ExitCode)"
        }
    }
    catch {
        $_ | Out-String | Set-Content -LiteralPath $logPath -Encoding UTF8
        $detail = $_.Exception.Message
    }
    finally {
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
    [void]$builder.AppendLine("| Layer | Check | Result | Seconds | Evidence |")
    [void]$builder.AppendLine("|---|---|---:|---:|---|")
    foreach ($result in $results) {
        $detail = ($result.Detail -replace "\|", "\|")
        [void]$builder.AppendLine(
            "| $($result.Layer) | $($result.Name) | **$($result.Status)** | $($result.DurationSeconds) | $detail |")
    }
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("## Improvement record")
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("- $ImprovementSummary")
    [void]$builder.AppendLine("- The commit STAR body records the exact implementation change and verified result.")
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("## Not covered without wiring")
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("- Physical camera/grabber acquisition, seven-camera frame load, background capture, and live Grab.")
    [void]$builder.AppendLine("- Physical IO and light disconnect/reconnect timing.")
    [void]$builder.AppendLine("- Storage-PC SMB interruption, remote backlog, low-disk deletion, and recovery.")
    [void]$builder.AppendLine("- Shift/24-hour product soak with the IO simulator, cameras, storage transfer, and operator interactions.")
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("These cases remain **NOT COVERED**, not PASS. Run the on-machine DVT and soak campaign when wiring is available.")

    $runReport = Join-Path $runDirectory "campaign-report.md"
    [IO.File]::WriteAllText($runReport, $builder.ToString(), (New-Object Text.UTF8Encoding($false)))
    if ($RecordLatest) {
        [IO.File]::WriteAllText($latestReport, $builder.ToString(), (New-Object Text.UTF8Encoding($false)))
    }
    return $overall
}

$allPassed = $true

if ((Test-ModeIncludes "Build") -and -not $SkipBuild) {
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

if ($Mode -eq "PhysicalIo") {
    $allPassed = (Invoke-DvtScenario "physical-io-stability" `
        "Physical IO five-minute stability" "Physical IO DVT" 600) -and $allPassed
}

if ($Mode -eq "PhysicalStorage") {
    $allPassed = (Invoke-DvtScenario "physical-storage-stability" `
        "Physical storage five-minute stability" "Physical storage DVT" 600) -and $allPassed
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
        "--filter", "(TestCategory=Stress|TestCategory=BridgeStress)",
        "--results-directory", $runDirectory,
        "--logger", "trx;LogFileName=soak.trx"
    ) @{ STRESS_MINUTES = $SoakMinutes.ToString(
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
