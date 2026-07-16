param(
    [string] $Config = (Join-Path (Split-Path -Parent $PSScriptRoot) 'inspection-config.json')
)

$json = [System.IO.File]::ReadAllText((Resolve-Path $Config).Path, [System.Text.Encoding]::UTF8)
$cfg = $json | ConvertFrom-Json
$nic = $cfg.IoNicName
$ip = $cfg.IoIp
$prefix = [int]$cfg.IoPrefixLength
$verifyTarget = $cfg.IoVerifyPingTarget

$e = Get-NetIPAddress -InterfaceAlias $nic -IPAddress $ip -ErrorAction SilentlyContinue
if ($e) {
    Write-Host ("[OK] " + $ip + " already exists, skip.") -ForegroundColor Yellow
} else {
    New-NetIPAddress -InterfaceAlias $nic -IPAddress $ip -PrefixLength $prefix -AddressFamily IPv4 | Out-Null
    Write-Host ("[OK] " + $ip + "/" + $prefix + " added to " + $nic) -ForegroundColor Green
}

Write-Host ""
Write-Host ("Verifying ping to IO module (" + $verifyTarget + ")...") -ForegroundColor Cyan
$ok = Test-Connection -ComputerName $verifyTarget -Count 2 -Quiet -ErrorAction SilentlyContinue
if ($ok) {
    Write-Host ("[OK] IO reachable at " + $verifyTarget) -ForegroundColor Green
} else {
    Write-Host "[WARN] IO not responding - check cable/power" -ForegroundColor Yellow
}
