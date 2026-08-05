param(
    [ValidateSet('Storage', 'Inspection')]
    [string] $Role,
    [string] $Version = '',
    [switch] $SkipBuild,
    [switch] $Rebuild,
    [switch] $AllowDirty
)

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$outputRoot = Join-Path $repoRoot 'artifacts\deploy'
$buildOutput = Join-Path $repoRoot 'bin\x64\Release'
$dirtyLines = @(& git -C $repoRoot status --porcelain)
$sourceState = if ($dirtyLines.Count -gt 0) { 'dirty' } else { 'clean' }
if ($sourceState -eq 'dirty' -and -not $AllowDirty) {
    throw '工作區有未提交變更。正式 Release 必須從 clean commit 封裝；煙測請明確加 -AllowDirty。'
}
if ($SkipBuild -and $Rebuild) {
    throw '-SkipBuild 與 -Rebuild 不可同時使用。'
}

function Find-MSBuild {
    $command = Get-Command msbuild.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) {
        $path = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -find 'MSBuild\**\Bin\MSBuild.exe' | Select-Object -First 1
        if ($path) { return $path }
    }
    throw '找不到 MSBuild。請安裝 Visual Studio Build Tools，或先自行完成 Release|x64 build 後加 -SkipBuild。'
}

if (-not $SkipBuild) {
    $msbuild = Find-MSBuild
    $target = if ($Rebuild) { 'Rebuild' } else { 'Build' }
    & $msbuild (Join-Path $repoRoot 'PICoater_AOI.sln') ('/t:' + $target) /p:Configuration=Release /p:Platform=x64 /m
    if ($LASTEXITCODE -ne 0) { throw 'Release|x64 build 失敗。' }
}

if (-not (Test-Path -LiteralPath (Join-Path $buildOutput 'AniloxRoll.Monitor.exe'))) {
    throw ("找不到 Release 輸出: " + $buildOutput)
}

if (-not $Version) {
    $Version = Get-Date -Format 'yyyyMMdd-HHmm'
}
$packageName = 'PICoater-' + $Role + '-' + $Version
$stage = Join-Path $outputRoot $packageName
$zipPath = Join-Path $outputRoot ($packageName + '.zip')

$resolvedOutput = [System.IO.Path]::GetFullPath($outputRoot)
$resolvedStage = [System.IO.Path]::GetFullPath($stage)
if (-not $resolvedStage.StartsWith($resolvedOutput + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw '封裝暫存路徑不在 artifacts\deploy 內，停止。'
}

if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
New-Item -ItemType Directory -Path (Join-Path $stage 'app') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stage 'deploy') -Force | Out-Null

$runtimeFiles = @(
    'AniloxRoll.Monitor.exe',
    'AniloxRoll.Monitor.exe.config',
    'IoBridge.Core.dll',
    'LightBridge.Core.dll',
    'StorageBridge.Core.dll',
    'MilGrabber.Core.dll',
    'TanukiCv.Core.dll',
    'TanukiCv.Controls.dll',
    'System.Runtime.CompilerServices.Unsafe.dll',
    'System.Threading.Tasks.Extensions.dll',
    'tanuki_cv_api.dll',
    'tanuki_pipeline_api.dll'
)
if ($Role -eq 'Inspection') {
    $runtimeFiles += 'Matrox.MatroxImagingLibrary.dll'
}

foreach ($name in $runtimeFiles) {
    $source = Join-Path $buildOutput $name
    if (-not (Test-Path -LiteralPath $source)) { throw ("Release 缺少必要檔案: " + $name) }
    Copy-Item -LiteralPath $source -Destination (Join-Path $stage 'app') -Force
}

if ($Role -eq 'Inspection') {
    $dcf = Join-Path $buildOutput 'Config\Radient_Config.dcf'
    if (-not (Test-Path -LiteralPath $dcf)) { throw 'Release 缺少 Config\Radient_Config.dcf。' }
    $dcfDir = Join-Path $stage 'app\Config'
    New-Item -ItemType Directory -Path $dcfDir -Force | Out-Null
    Copy-Item -LiteralPath $dcf -Destination $dcfDir -Force

    $toolFiles = @(
        'tools\io\IoBridge.Automation.exe',
        'tools\io\IoBridge.Automation.exe.config',
        'tools\io\IoBridge.ManualControl.exe',
        'tools\io\IoBridge.ManualControl.exe.config',
        'tools\io\IoBridge.Core.dll',
        'tools\light\LightBridge.Control.exe',
        'tools\light\LightBridge.Control.exe.config',
        'tools\light\LightBridge.Core.dll',
        'tools\storage\StorageBridge.Control.exe',
        'tools\storage\StorageBridge.Control.exe.config',
        'tools\storage\StorageBridge.Core.dll'
    )
    foreach ($relative in $toolFiles) {
        $source = Join-Path $buildOutput $relative
        if (-not (Test-Path -LiteralPath $source)) { throw ("Release 缺少現場工具檔案: " + $relative) }
        $destination = Join-Path (Join-Path $stage 'app') $relative
        $destinationDir = Split-Path -Parent $destination
        if (-not (Test-Path -LiteralPath $destinationDir)) {
            New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null
        }
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
}

Copy-Item -LiteralPath (Join-Path $repoRoot 'deploy\common') -Destination (Join-Path $stage 'deploy\common') -Recurse -Force
$roleFolder = if ($Role -eq 'Storage') { 'storage-pc' } else { 'inspection-pc' }
Copy-Item -LiteralPath (Join-Path $repoRoot ('deploy\' + $roleFolder)) -Destination (Join-Path $stage ('deploy\' + $roleFolder)) -Recurse -Force
Copy-Item -LiteralPath (Join-Path $repoRoot ('deploy\' + $roleFolder + '\manual-install.html')) -Destination (Join-Path $stage 'manual-install.html') -Force

$setupWrapper = @"
@echo off
call "%~dp0deploy\$roleFolder\setup.bat"
exit /b %errorlevel%
"@
[System.IO.File]::WriteAllText((Join-Path $stage 'setup.bat'), $setupWrapper, [System.Text.Encoding]::ASCII)

if ($Role -eq 'Storage') {
    $restartTestWrapper = @"
@echo off
call "%~dp0deploy\storage-pc\test_app_restart.bat"
exit /b %errorlevel%
"@
    [System.IO.File]::WriteAllText(
        (Join-Path $stage 'test_storage_restart.bat'),
        $restartTestWrapper,
        [System.Text.Encoding]::ASCII)

    $retentionTestWrapper = @"
@echo off
call "%~dp0deploy\storage-pc\test_local_retention.bat"
exit /b %errorlevel%
"@
    [System.IO.File]::WriteAllText(
        (Join-Path $stage 'test_storage_retention.bat'),
        $retentionTestWrapper,
        [System.Text.Encoding]::ASCII)
}

$commit = (& git -C $repoRoot rev-parse --short HEAD 2>$null)
$versionText = "Package=$packageName`r`nCommit=$commit`r`nSourceState=$sourceState`r`nBuiltUtc=$([DateTime]::UtcNow.ToString('o'))`r`n"
[System.IO.File]::WriteAllText((Join-Path $stage 'VERSION.txt'), $versionText, [System.Text.Encoding]::ASCII)

$forbidden = Get-ChildItem -LiteralPath (Join-Path $stage 'app') -Recurse -File | Where-Object {
    $_.Extension -in @('.pdb', '.lib', '.exp') -or
    $_.Name -match '(?i)(\.Tests?\.dll$|crash\.log$|\.tmp$)' -or
    $_.DirectoryName -match '(?i)[\\/]logs$' -or
    ($_.Extension -eq '.json' -and $_.DirectoryName -match '(?i)[\\/]Config$')
}
if ($forbidden) {
    throw ("封裝包含禁止檔案: " + (($forbidden | ForEach-Object FullName) -join ', '))
}

Compress-Archive -LiteralPath $stage -DestinationPath $zipPath -CompressionLevel Optimal

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    $rootPrefix = $packageName + '/'
    $invalidEntry = $archive.Entries | Where-Object {
        $normalized = $_.FullName.Replace('\', '/')
        -not $normalized.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)
    } | Select-Object -First 1
    if ($invalidEntry) {
        throw ('ZIP 項目不在版本根資料夾內: ' + $invalidEntry.FullName)
    }

    $expectedSetup = $rootPrefix + 'setup.bat'
    $hasSetup = $archive.Entries | Where-Object {
        $_.FullName.Replace('\', '/').Equals($expectedSetup, [System.StringComparison]::OrdinalIgnoreCase)
    } | Select-Object -First 1
    if (-not $hasSetup) {
        throw ('ZIP 缺少版本根資料夾內的 setup.bat: ' + $expectedSetup)
    }
    if ($Role -eq 'Storage') {
        $expectedRestartTest = $rootPrefix + 'test_storage_restart.bat'
        $hasRestartTest = $archive.Entries | Where-Object {
            $_.FullName.Replace('\', '/').Equals(
                $expectedRestartTest,
                [System.StringComparison]::OrdinalIgnoreCase)
        } | Select-Object -First 1
        if (-not $hasRestartTest) {
            throw ('ZIP 缺少儲存程式重啟測試入口: ' + $expectedRestartTest)
        }
        $expectedRetentionTest = $rootPrefix + 'test_storage_retention.bat'
        $hasRetentionTest = $archive.Entries | Where-Object {
            $_.FullName.Replace('\', '/').Equals(
                $expectedRetentionTest,
                [System.StringComparison]::OrdinalIgnoreCase)
        } | Select-Object -First 1
        if (-not $hasRetentionTest) {
            throw ('ZIP 缺少儲存電腦低磁碟測試入口: ' + $expectedRetentionTest)
        }
    }
}
finally {
    $archive.Dispose()
}
Remove-Item -LiteralPath $stage -Recurse -Force
Write-Host ("[OK] " + $zipPath) -ForegroundColor Green
