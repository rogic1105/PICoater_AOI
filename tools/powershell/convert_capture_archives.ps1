param(
    [string]$Root = 'D:\Anilox\Captures',
    [string]$AppPath = '',
    [switch]$Overwrite,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if ([string]::IsNullOrWhiteSpace($AppPath)) {
    $AppPath = Join-Path $repoRoot 'bin\x64\Release\AniloxRoll.Monitor.exe'
}
if (-not (Test-Path -LiteralPath $AppPath)) {
    throw "Release app not found: $AppPath"
}
if (-not (Test-Path -LiteralPath $Root)) {
    throw "Capture root not found: $Root"
}

$assemblyDir = Split-Path -Parent $AppPath
$oldPath = $env:PATH
$env:PATH = $assemblyDir + ';' + $env:PATH
try {
    [void][Reflection.Assembly]::LoadFrom((Resolve-Path -LiteralPath $AppPath))
    $captureRoot = (Resolve-Path -LiteralPath $Root).Path
    if (-not $ValidateOnly.IsPresent) {
        $summary = [AniloxRoll.Monitor.Core.Services.CaptureArchiveMigration]::ConvertLegacyRoot(
            $captureRoot,
            $Overwrite.IsPresent)
        Write-Host "[ACAP Convert] $summary"
    }
    $validation = [AniloxRoll.Monitor.Core.Services.CaptureArchiveMigration]::ValidateRoot(
        $captureRoot)
    Write-Host "[ACAP Validate] $validation"
    if ($validation -notmatch 'invalidArchives=0;invalidRecords=0;partialFiles=0$') {
        throw "ACAP validation failed: $validation"
    }
}
finally {
    $env:PATH = $oldPath
}
