$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "TestRunner.ResourceTrend.ps1")

function New-Sample {
    param(
        [double]$ElapsedSeconds,
        [double]$PrivateMB
    )

    return [pscustomobject]@{
        ElapsedSeconds = $ElapsedSeconds
        PrivateMB = $PrivateMB
        Handles = 100
        GdiObjects = 10
        UserObjects = 10
        Threads = 10
        Responding = $true
    }
}

function Assert-Condition {
    param(
        [bool]$Condition,
        [string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

$oneTimeExpansion = @(
    for ($index = 0; $index -lt 40; $index++) {
        $privateMB = 1000
        if ($index -ge 20) {
            $privateMB += 200
        }
        New-Sample ($index * 30) $privateMB
    }
)
$expansionTrend = Get-ResourceTrend $oneTimeExpansion
Assert-Condition (-not $expansionTrend.PrivateLeak) `
    "A one-time Server GC-style expansion must not be classified as a leak."
Assert-Condition $expansionTrend.HasExpansion `
    "The expansion test must identify the large Private Bytes step."

$monotonicLeak = @(
    for ($index = 0; $index -lt 40; $index++) {
        New-Sample ($index * 30) (1000 + ($index * 2.75))
    }
)
$leakTrend = Get-ResourceTrend $monotonicLeak
Assert-Condition $leakTrend.PrivateLeak `
    "A sustained 330 MB/hour Private Bytes trend must fail."
Assert-Condition $leakTrend.MedianRateLeak `
    "A monotonic leak must be found by the median interval rate."

$expansionThenLeak = @(
    for ($index = 0; $index -lt 50; $index++) {
        $privateMB = 1000
        if ($index -ge 20) {
            $privateMB += 200 + (($index - 20) * 3)
        }
        New-Sample ($index * 30) $privateMB
    }
)
$postExpansionTrend = Get-ResourceTrend $expansionThenLeak
Assert-Condition $postExpansionTrend.PrivateLeak `
    "Growth that continues after a heap expansion must fail."
Assert-Condition $postExpansionTrend.PostExpansionLeak `
    "Post-expansion retained growth must be reported explicitly."

Write-Output "PASS: resource trend guard tests"
