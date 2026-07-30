param(
    [string]$IoAddress = "192.168.255.1",
    [int]$IoPort = 502,
    [string]$ComPort = "COM17",
    [string]$ReportPath = "",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ReportPath)) {
    $reportDirectory = "D:\Anilox\Logs\DvtReports"
    New-Item -ItemType Directory -Force -Path $reportDirectory | Out-Null
    $ReportPath = Join-Path $reportDirectory (
        (Get-Date -Format "yyyyMMdd-HHmmss") +
        "-dvt-admin-install.log")
}
Start-Transcript -Path $ReportPath -Force | Out-Null
Write-Host "[Report] $ReportPath"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $ValidateOnly -and -not $principal.IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this installer as administrator."
}

if (-not $ValidateOnly -and
    (Get-Process -Name "AniloxRoll.Monitor" -ErrorAction SilentlyContinue)) {
    throw "Close AniloxRoll.Monitor before installing DVT privileged actions."
}

$portPattern = [Regex]::Escape("($ComPort)") + '$'
$serialDevice = Get-PnpDevice -Class Ports -ErrorAction Stop |
    Where-Object { $_.FriendlyName -match $portPattern } |
    Select-Object -First 1
if (-not $serialDevice) {
    throw "Serial device $ComPort was not found."
}

$ioRoute = @(Find-NetRoute -RemoteIPAddress $IoAddress -ErrorAction Stop) |
    Where-Object {
        $_.CimClass.CimClassName -eq "MSFT_NetRoute"
    } |
    Select-Object -First 1
if (-not $ioRoute) {
    throw "No route was found for IO endpoint $IoAddress."
}

$blackholeInterfaceIndex = 1
$blackholeNextHop = "0.0.0.0"
$legacyBlackholeNextHop = "192.168.255.254"
$blockedPrefix = "$IoAddress/32"

$taskDefinitions = [ordered]@{
    "PICoater-DVT-Block-IO502" = @"
`$ErrorActionPreference='Stop'
Get-NetRoute -DestinationPrefix '$blockedPrefix' -PolicyStore ActiveStore -ErrorAction SilentlyContinue | Where-Object { (`$_.InterfaceIndex -eq $blackholeInterfaceIndex -and `$_.NextHop -eq '$blackholeNextHop') -or `$_.NextHop -eq '$legacyBlackholeNextHop' } | Remove-NetRoute -Confirm:`$false -ErrorAction Stop
New-NetRoute -DestinationPrefix '$blockedPrefix' -InterfaceIndex $blackholeInterfaceIndex -NextHop '$blackholeNextHop' -RouteMetric 1 -PolicyStore ActiveStore -ErrorAction Stop | Out-Null
"@
    "PICoater-DVT-Unblock-IO502" = @"
`$ErrorActionPreference='Stop'
Get-NetRoute -DestinationPrefix '$blockedPrefix' -PolicyStore ActiveStore -ErrorAction SilentlyContinue | Where-Object { (`$_.InterfaceIndex -eq $blackholeInterfaceIndex -and `$_.NextHop -eq '$blackholeNextHop') -or `$_.NextHop -eq '$legacyBlackholeNextHop' } | Remove-NetRoute -Confirm:`$false -ErrorAction Stop
"@
    "PICoater-DVT-Disable-COM17" = @"
`$ErrorActionPreference='Stop'
`$device=Get-PnpDevice -InstanceId '$($serialDevice.InstanceId.Replace("'", "''"))' -ErrorAction Stop
if (`$device.Status -eq 'OK') {
    `$device | Disable-PnpDevice -Confirm:`$false -ErrorAction Stop
}
"@
    "PICoater-DVT-Enable-COM17" = @"
`$ErrorActionPreference='Stop'
`$device=Get-PnpDevice -InstanceId '$($serialDevice.InstanceId.Replace("'", "''"))' -ErrorAction Stop
if (`$device.Status -ne 'OK') {
    `$device | Enable-PnpDevice -Confirm:`$false -ErrorAction Stop
}
"@
}

$taskPrincipal = New-ScheduledTaskPrincipal `
    -UserId $identity.Name `
    -LogonType Interactive `
    -RunLevel Highest
$taskSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

function Invoke-FixedTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TaskName
    )

    $previousRun = (Get-ScheduledTaskInfo -TaskName $TaskName).LastRunTime
    Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    $deadline = (Get-Date).AddSeconds(20)
    do {
        Start-Sleep -Milliseconds 250
        $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
        $info = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction Stop
        $completed = (
            $info.LastRunTime -gt $previousRun -and
            $task.State -ne "Running")
    } while (-not $completed -and (Get-Date) -lt $deadline)

    if (-not $completed) {
        throw "Scheduled action timed out: $TaskName"
    }
    if ($info.LastTaskResult -ne 0) {
        throw (
            "Scheduled action failed: $TaskName " +
            "LastTaskResult=$($info.LastTaskResult)")
    }
    Write-Host "[TEST] $TaskName"
}

function Test-TcpReachable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Address,
        [Parameter(Mandatory = $true)]
        [int]$Port,
        [Parameter(Mandatory = $true)]
        [int]$TimeoutMilliseconds
    )

    $client = New-Object Net.Sockets.TcpClient
    try {
        $attempt = $client.BeginConnect(
            $Address, $Port, $null, $null)
        if (-not $attempt.AsyncWaitHandle.WaitOne($TimeoutMilliseconds)) {
            return $false
        }
        $client.EndConnect($attempt)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Close()
    }
}

foreach ($entry in $taskDefinitions.GetEnumerator()) {
    $encodedCommand = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($entry.Value))
    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument (
            "-NoProfile -NonInteractive -ExecutionPolicy Bypass " +
            "-EncodedCommand $encodedCommand")
    $task = New-ScheduledTask `
        -Action $action `
        -Principal $taskPrincipal `
        -Settings $taskSettings
    if ($ValidateOnly) {
        Write-Host "[VALID] $($entry.Key)"
    }
    else {
        Register-ScheduledTask `
            -TaskName $entry.Key `
            -InputObject $task `
            -Force |
            Out-Null
        Write-Host "[OK] $($entry.Key)"
    }
}

if ($ValidateOnly) {
    Write-Host
    Write-Host "[PASS] Fixed DVT action definitions are valid."
    Stop-Transcript | Out-Null
    return
}

Get-NetRoute `
    -DestinationPrefix $blockedPrefix `
    -PolicyStore ActiveStore `
    -ErrorAction SilentlyContinue |
    Where-Object {
        ($_.InterfaceIndex -eq $blackholeInterfaceIndex -and
         $_.NextHop -eq $blackholeNextHop) -or
        $_.NextHop -eq $legacyBlackholeNextHop
    } |
    Remove-NetRoute -Confirm:$false -ErrorAction Stop

Get-NetFirewallRule `
    -Name "PICoater-DVT-IO502-Block" `
    -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule -ErrorAction SilentlyContinue

try {
    Invoke-FixedTask "PICoater-DVT-Block-IO502"
    $blockedRoute = Get-NetRoute `
        -DestinationPrefix $blockedPrefix `
        -PolicyStore ActiveStore `
        -ErrorAction SilentlyContinue |
        Where-Object {
            ($_.InterfaceIndex -eq $blackholeInterfaceIndex -and
             $_.NextHop -eq $blackholeNextHop)
        }
    if (-not $blockedRoute) {
        throw "IO blackhole route self-test did not create the route."
    }
    if (Test-TcpReachable `
        -Address $IoAddress `
        -Port $IoPort `
        -TimeoutMilliseconds 1500) {
        throw "IO blackhole route did not block a new TCP connection."
    }

    Invoke-FixedTask "PICoater-DVT-Unblock-IO502"
    $blockedRoute = Get-NetRoute `
        -DestinationPrefix $blockedPrefix `
        -PolicyStore ActiveStore `
        -ErrorAction SilentlyContinue |
        Where-Object {
            ($_.InterfaceIndex -eq $blackholeInterfaceIndex -and
             $_.NextHop -eq $blackholeNextHop)
        }
    if ($blockedRoute) {
        throw "IO blackhole route self-test did not remove the route."
    }
    if (-not (Test-TcpReachable `
        -Address $IoAddress `
        -Port $IoPort `
        -TimeoutMilliseconds 3000)) {
        throw "IO TCP did not recover after removing the blackhole route."
    }

    Invoke-FixedTask "PICoater-DVT-Disable-COM17"
    $serialDevice = Get-PnpDevice `
        -InstanceId $serialDevice.InstanceId `
        -ErrorAction Stop
    if ($serialDevice.Status -eq "OK") {
        throw "COM17 self-test did not disable the device."
    }

    Invoke-FixedTask "PICoater-DVT-Enable-COM17"
    $serialDevice = Get-PnpDevice `
        -InstanceId $serialDevice.InstanceId `
        -ErrorAction Stop
    if ($serialDevice.Status -ne "OK") {
        throw "COM17 self-test did not restore the device."
    }
}
finally {
    Get-NetRoute `
        -DestinationPrefix $blockedPrefix `
        -PolicyStore ActiveStore `
        -ErrorAction SilentlyContinue |
        Where-Object {
            ($_.InterfaceIndex -eq $blackholeInterfaceIndex -and
             $_.NextHop -eq $blackholeNextHop) -or
            $_.NextHop -eq $legacyBlackholeNextHop
        } |
        Remove-NetRoute -Confirm:$false -ErrorAction SilentlyContinue

    $serialDevice = Get-PnpDevice `
        -InstanceId $serialDevice.InstanceId `
        -ErrorAction SilentlyContinue
    if ($serialDevice -and $serialDevice.Status -ne "OK") {
        $serialDevice |
            Enable-PnpDevice -Confirm:$false -ErrorAction SilentlyContinue
    }
}

Write-Host
Write-Host "[PASS] Fixed DVT actions are installed for $($identity.Name)."
Write-Host "       TestRunner option 11 no longer needs a UAC prompt."
Stop-Transcript | Out-Null
