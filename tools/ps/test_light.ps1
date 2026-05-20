# LTS-3DPA24 光源控制器 RS-232 自動掃描 + 亮度掃描測試
# 協定：8-byte ASCII，9600/8N1，格式 $ + cmd + ch + "0" + hex(2) + xor(2)
# cmd: 1=on, 2=off, 3=set brightness, 4=read brightness
#
# 用法：
#   powershell -NoProfile -ExecutionPolicy Bypass -File tests\ps_tool\test_light.ps1
#   (可選) -Channel 1 -Levels 0,64,128,192,255 -DwellMs 2000

param(
    [int]    $Channel  = 1,
    [int[]]  $Levels   = @(0, 64, 128, 192, 255),
    [int]    $DwellMs  = 2000,
    [string] $ForcePort = ""
)

function Build-Command([int]$cmd, [int]$channel, [int]$value) {
    if ($value -lt 0)   { $value = 0 }
    if ($value -gt 255) { $value = 255 }
    $hex  = '{0:X2}' -f $value
    $data = '0' + $hex
    $cCmd = [char](0x30 + $cmd)        # '0'..'4'
    $cCh  = [char](0x30 + $channel)
    $xor  = [byte][char]'$'
    $xor  = $xor -bxor [byte]$cCmd
    $xor  = $xor -bxor [byte]$cCh
    foreach ($c in $data.ToCharArray()) { $xor = $xor -bxor [byte]$c }
    $cks = '{0:X2}' -f $xor
    return '$' + $cCmd + $cCh + $data + $cks
}

function Open-Port([string]$name, [int]$timeoutMs) {
    $p = New-Object System.IO.Ports.SerialPort $name, 9600, ([System.IO.Ports.Parity]::None), 8, ([System.IO.Ports.StopBits]::One)
    $p.ReadTimeout  = $timeoutMs
    $p.WriteTimeout = $timeoutMs
    $p.Open()
    return $p
}

function Send-Cmd($port, [string]$command, [int]$expectLen = 1, [int]$timeoutMs = 500) {
    try { $port.DiscardInBuffer() } catch {}
    $port.Write($command)
    $buf = New-Object byte[] $expectLen
    $read = 0
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($read -lt $expectLen -and $sw.ElapsedMilliseconds -lt $timeoutMs) {
        if ($port.BytesToRead -gt 0) {
            $n = [Math]::Min($port.BytesToRead, $expectLen - $read)
            $read += $port.Read($buf, $read, $n)
        } else {
            Start-Sleep -Milliseconds 10
        }
    }
    if ($read -le 0) { return $null }
    return [System.Text.Encoding]::ASCII.GetString($buf, 0, $read)
}

function Probe-Port([string]$name, [int]$channel) {
    # 嚴格驗證（PDF §4.1.4 表-4）：
    #   - 8-byte 回應
    #   - [0]='$'，[1]='4'（cmd echo），[2]=channel echo，[3]='0'（data 起始）
    #   - [4][5] 是 hex 亮度
    #   - [6][7] 是 XOR(前 6 bytes) 的 hex
    $probe = Build-Command 4 $channel 0
    $port = $null
    try {
        $port = Open-Port $name 300
        $resp = Send-Cmd $port $probe 8 400
        if (-not $resp -or $resp.Length -lt 8) {
            return [pscustomobject]@{ Ok = $false; Response = $resp; Reason = 'short response' }
        }
        if ($resp[0] -ne '$' -or $resp[1] -ne '4' -or $resp[2] -ne [char](0x30 + $channel) -or $resp[3] -ne '0') {
            return [pscustomobject]@{ Ok = $false; Response = $resp; Reason = 'header/echo mismatch' }
        }
        # checksum
        $xor = 0
        for ($i = 0; $i -lt 6; $i++) { $xor = $xor -bxor [byte][char]$resp[$i] }
        $ck = '{0:X2}' -f $xor
        if ($resp[6] -ne $ck[0] -or $resp[7] -ne $ck[1]) {
            return [pscustomobject]@{ Ok = $false; Response = $resp; Reason = 'checksum mismatch' }
        }
        return [pscustomobject]@{ Ok = $true; Response = $resp; Reason = '' }
    } catch {
        return [pscustomobject]@{ Ok = $false; Response = $_.Exception.Message; Reason = 'exception' }
    } finally {
        if ($port -ne $null -and $port.IsOpen) { $port.Close() }
        if ($port -ne $null) { $port.Dispose() }
    }
}

# ────── 掃描 ──────
Write-Host '[Light] 掃描 COM port...' -ForegroundColor Cyan
$candidates = if ($ForcePort) { @($ForcePort) } else { [System.IO.Ports.SerialPort]::GetPortNames() | Sort-Object }
Write-Host ('  可用 port: ' + ($candidates -join ', '))

$target = $null
foreach ($name in $candidates) {
    Write-Host ('  探測 ' + $name + ' ... ') -NoNewline
    $r = Probe-Port $name $Channel
    if ($r.Ok) {
        Write-Host ('OK  response=' + $r.Response) -ForegroundColor Green
        $target = $name
        break
    } else {
        $reason = if ($r.Reason) { $r.Reason } else { 'no response' }
        Write-Host ('skip (' + $reason + ')') -ForegroundColor DarkGray
    }
}

if (-not $target) {
    Write-Host '[Light] 找不到光源控制器 — 檢查接線、電源、Channel 是否正確' -ForegroundColor Red
    exit 1
}

Write-Host ('[Light] 目標 port: ' + $target + '  Channel=' + $Channel) -ForegroundColor Cyan

# ────── 亮度掃描 (Test A) ──────
$port = Open-Port $target 1000
try {
    Write-Host ('[Light] 亮度掃描: ' + ($Levels -join ' → ') + '   每階 ' + $DwellMs + ' ms')
    foreach ($lvl in $Levels) {
        # 先設亮度 (cmd=3) 再開啟 (cmd=1)
        $c3 = Build-Command 3 $Channel $lvl
        $r3 = Send-Cmd $port $c3 1 800
        $c1 = Build-Command 1 $Channel $lvl
        $r1 = Send-Cmd $port $c1 1 800
        $ok3 = if ($r3 -and $r3[0] -eq '$') { 'OK' } else { 'FAIL' }
        $ok1 = if ($r1 -and $r1[0] -eq '$') { 'OK' } else { 'FAIL' }
        Write-Host ('  level=' + ('{0,3}' -f $lvl) + '  set=' + $ok3 + '  on=' + $ok1)
        Start-Sleep -Milliseconds $DwellMs
    }
    # 結束關閉
    $coff = Build-Command 2 $Channel 0
    $roff = Send-Cmd $port $coff 1 800
    $okOff = if ($roff -and $roff[0] -eq '$') { 'OK' } else { 'FAIL' }
    Write-Host ('[Light] 關閉: ' + $okOff) -ForegroundColor Cyan
} finally {
    if ($port.IsOpen) { $port.Close() }
    $port.Dispose()
}
