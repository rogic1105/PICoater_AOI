param(
    [string] $Config = '',
    [string] $DesktopDirectory = ''
)

$roleDirectory = Split-Path -Parent $PSScriptRoot
$deployRoot = Split-Path -Parent $roleDirectory
. (Join-Path $deployRoot 'common\Deploy.Common.ps1')

if (-not $Config) {
    $Config = Join-Path $roleDirectory 'inspection-config.json'
}
$cfg = Read-DeployConfig $Config
if (-not $cfg.VerifyPingTarget) {
    Stop-Deploy 'inspection-config.json 的 VerifyPingTarget 未設定。'
}
if (-not $cfg.StorageShareName) {
    Stop-Deploy 'inspection-config.json 的 StorageShareName 未設定。'
}

$remotePath = '\\' + $cfg.VerifyPingTarget + '\' + $cfg.StorageShareName
if (-not $DesktopDirectory) {
    $DesktopDirectory = [Environment]::GetFolderPath([Environment+SpecialFolder]::CommonDesktopDirectory)
}
if (-not $DesktopDirectory) {
    $DesktopDirectory = Join-Path $env:PUBLIC 'Desktop'
}
if (-not (Test-Path -LiteralPath $DesktopDirectory)) {
    New-Item -ItemType Directory -Path $DesktopDirectory -Force | Out-Null
}

$shortcutPath = Join-Path $DesktopDirectory 'Anilox 儲存資料.lnk'
$explorer = Join-Path $env:WINDIR 'explorer.exe'
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $explorer
$shortcut.Arguments = '"' + $remotePath + '"'
$shortcut.WorkingDirectory = $DesktopDirectory
$shortcut.IconLocation = $explorer + ',0'
$shortcut.Description = '開啟 PICoater 儲存電腦共用資料夾'
$shortcut.Save()

if (-not (Test-Path -LiteralPath $shortcutPath)) {
    Stop-Deploy ("桌面捷徑建立失敗: " + $shortcutPath)
}
Write-Host ("[OK] 桌面捷徑: " + $shortcutPath + " -> " + $remotePath) -ForegroundColor Green
