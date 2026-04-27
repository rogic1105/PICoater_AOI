$nic = "乙太網路"
$ip  = "192.168.255.10"

$e = Get-NetIPAddress -InterfaceAlias $nic -IPAddress $ip -ErrorAction SilentlyContinue
if ($e) {
    Write-Host ("[OK] " + $ip + " already exists, skip.") -ForegroundColor Yellow
} else {
    New-NetIPAddress -InterfaceAlias $nic -IPAddress $ip -PrefixLength 24 -AddressFamily IPv4 | Out-Null
    Write-Host ("[OK] " + $ip + "/24 added to " + $nic) -ForegroundColor Green
}

Write-Host ""
Write-Host "Verifying ping to IO module (192.168.255.1)..." -ForegroundColor Cyan
$ok = Test-Connection -ComputerName "192.168.255.1" -Count 2 -Quiet -ErrorAction SilentlyContinue
if ($ok) {
    Write-Host "[OK] IO reachable at 192.168.255.1" -ForegroundColor Green
} else {
    Write-Host "[WARN] IO not responding - check cable/power" -ForegroundColor Yellow
}