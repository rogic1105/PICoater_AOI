param(
    [string]$TestProj,
    [string]$LogFile,
    [string]$Mode,
    [string]$Minutes = "60"
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Set environment variable for StressTests.cs to read
$env:STRESS_MINUTES = $Minutes

function Run-DotnetTest($proj, $log, [string[]]$extraArgs) {
    & dotnet test $proj -p:Configuration=Release -v normal @extraArgs 2>&1 | ForEach-Object {
        Write-Host $_
        $_ | Out-File $log -Append -Encoding utf8
    }
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"=== Started: $timestamp  (Mode=$Mode, STRESS_MINUTES=$Minutes) ===" | Out-File $LogFile -Encoding utf8

if ($Mode -eq "Stress") {
    Run-DotnetTest $TestProj $LogFile "--filter","TestCategory=Stress"
}
elseif ($Mode -eq "All") {
    # Run unit tests first, then stress tests — avoids interleaved output
    "" | Out-File $LogFile -Append -Encoding utf8
    "--- Unit Tests ---" | Out-File $LogFile -Append -Encoding utf8
    Write-Host ""
    Write-Host "--- Unit Tests ---"
    Run-DotnetTest $TestProj $LogFile "--filter","TestCategory!=Stress"

    "" | Out-File $LogFile -Append -Encoding utf8
    "--- Stress Tests ($Minutes min) ---" | Out-File $LogFile -Append -Encoding utf8
    Write-Host ""
    Write-Host "--- Stress Tests ($Minutes min) ---"
    Run-DotnetTest $TestProj $LogFile "--filter","TestCategory=Stress"
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"=== Finished: $timestamp ===" | Out-File $LogFile -Append -Encoding utf8
