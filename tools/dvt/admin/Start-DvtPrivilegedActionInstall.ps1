param(
    [Parameter(Mandatory = $true)]
    [string]$Installer
)

$ErrorActionPreference = "Stop"
$reportDirectory = "D:\Anilox\Logs\DvtReports"
New-Item -ItemType Directory -Force -Path $reportDirectory | Out-Null
$launcherReport = Join-Path $reportDirectory (
    (Get-Date -Format "yyyyMMdd-HHmmss") +
    "-dvt-admin-launch.log")

if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
    Write-Error "Installer file not found: $Installer"
    exit 2
}

$arguments = (
    "-NoProfile -ExecutionPolicy Bypass -File " +
    '"' + $Installer + '"')

try {
    $process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList $arguments `
        -Verb RunAs `
        -PassThru `
        -Wait `
        -ErrorAction Stop
    exit $process.ExitCode
}
catch {
    $message = (
        "[FAIL] UAC elevation did not start the installer.`r`n" +
        $_.Exception.Message)
    $message | Set-Content -LiteralPath $launcherReport -Encoding UTF8
    [Console]::Error.WriteLine($message)
    [Console]::Error.WriteLine("Report: " + $launcherReport)
    exit 1223
}
