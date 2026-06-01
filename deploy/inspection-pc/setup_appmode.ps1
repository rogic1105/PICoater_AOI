# 檢測機：寫入 app-mode.json (Role = Inspection)
# 用法：powershell -NoProfile -ExecutionPolicy Bypass -File setup_appmode.ps1 [-AppDir <路徑>]

param(
    [string] $AppDir = "C:\AniloxMonitor"
)

$cfgDir  = Join-Path $AppDir 'Config'
$cfgFile = Join-Path $cfgDir 'app-mode.json'

if (-not (Test-Path $cfgDir)) {
    New-Item -ItemType Directory -Path $cfgDir -Force | Out-Null
}

$json = @'
{
  "Role": "Inspection",
  "StorageMachineConfigFolder": "D:\\Anilox\\Config"
}
'@

[System.IO.File]::WriteAllText($cfgFile, $json, [System.Text.Encoding]::UTF8)
Write-Host ("[AppMode] Inspection app-mode.json -> " + $cfgFile) -ForegroundColor Green
