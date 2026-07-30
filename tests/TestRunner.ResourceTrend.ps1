function Get-MedianValue {
    param([double[]]$Values)

    if (-not $Values -or $Values.Count -eq 0) {
        return 0.0
    }

    $sorted = @($Values | Sort-Object)
    $middle = [int][Math]::Floor($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) {
        return [double]$sorted[$middle]
    }

    return ([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2.0
}

function Get-ResourceTrend {
    param(
        [object[]]$Samples,
        [int]$WarmupSeconds = 300,
        [int]$MinimumRateWindowSeconds = 180,
        [int]$MinimumPostExpansionSeconds = 300,
        [double]$ExpansionStepMB = 64,
        [double]$PrivateRateLimitMBPerHour = 256,
        [double]$PrivateEmergencyDeltaMB = 4096
    )

    if (-not $Samples -or $Samples.Count -lt 2) {
        throw "Resource trend requires at least two samples."
    }

    $steady = @($Samples | Where-Object {
        [double]$_.ElapsedSeconds -ge $WarmupSeconds
    })
    if ($steady.Count -lt 2) {
        $steady = @($Samples)
    }

    $first = $steady[0]
    $last = $steady[$steady.Count - 1]
    $steadySeconds = [Math]::Max(
        1.0,
        [double]$last.ElapsedSeconds - [double]$first.ElapsedSeconds)
    $privateDelta = [double]$last.PrivateMB - [double]$first.PrivateMB
    $totalPrivateRate = $privateDelta * 3600.0 / $steadySeconds

    $intervalRates = New-Object System.Collections.Generic.List[double]
    $largestPositiveStep = 0.0
    $lastExpansionIndex = -1
    for ($index = 1; $index -lt $steady.Count; $index++) {
        $previous = $steady[$index - 1]
        $current = $steady[$index]
        $seconds = [double]$current.ElapsedSeconds -
            [double]$previous.ElapsedSeconds
        if ($seconds -le 0) {
            continue
        }

        $step = [double]$current.PrivateMB - [double]$previous.PrivateMB
        $intervalRates.Add($step * 3600.0 / $seconds)
        if ($step -gt $largestPositiveStep) {
            $largestPositiveStep = $step
        }
        if ($step -ge $ExpansionStepMB) {
            $lastExpansionIndex = $index
        }
    }

    $medianPrivateRate = Get-MedianValue $intervalRates.ToArray()
    $postExpansionSeconds = 0.0
    $postExpansionDelta = 0.0
    $postExpansionRate = 0.0
    if ($lastExpansionIndex -ge 0) {
        $postFirst = $steady[$lastExpansionIndex]
        $postExpansionSeconds = [Math]::Max(
            0.0,
            [double]$last.ElapsedSeconds -
                [double]$postFirst.ElapsedSeconds)
        $postExpansionDelta =
            [double]$last.PrivateMB - [double]$postFirst.PrivateMB
        if ($postExpansionSeconds -gt 0) {
            $postExpansionRate =
                $postExpansionDelta * 3600.0 / $postExpansionSeconds
        }
    }

    # A Server GC heap expansion appears as one large Private Bytes step.
    # Treat it as a new baseline and fail only when growth continues afterward.
    $medianRateLeak =
        $steadySeconds -ge $MinimumRateWindowSeconds -and
        $privateDelta -ge $ExpansionStepMB -and
        $medianPrivateRate -gt $PrivateRateLimitMBPerHour
    $postExpansionLeak =
        $postExpansionSeconds -ge $MinimumPostExpansionSeconds -and
        $postExpansionDelta -ge $ExpansionStepMB -and
        $postExpansionRate -gt $PrivateRateLimitMBPerHour
    $emergencyLeak = $privateDelta -gt $PrivateEmergencyDeltaMB

    return [pscustomobject]@{
        Samples = $steady
        First = $first
        Last = $last
        SteadySeconds = $steadySeconds
        PrivateDeltaMB = $privateDelta
        TotalPrivateRateMBPerHour = $totalPrivateRate
        MedianPrivateRateMBPerHour = $medianPrivateRate
        LargestPositiveStepMB = $largestPositiveStep
        PostExpansionSeconds = $postExpansionSeconds
        PostExpansionDeltaMB = $postExpansionDelta
        PostExpansionRateMBPerHour = $postExpansionRate
        HasExpansion = $lastExpansionIndex -ge 0
        PrivateLeak = $medianRateLeak -or $postExpansionLeak -or
            $emergencyLeak
        MedianRateLeak = $medianRateLeak
        PostExpansionLeak = $postExpansionLeak
        EmergencyLeak = $emergencyLeak
    }
}
