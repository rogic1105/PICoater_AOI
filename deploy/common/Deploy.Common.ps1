# Shared deployment helpers. PowerShell 5.1 compatible.

function Stop-Deploy([string] $Message) {
    throw $Message
}

function Assert-DeployAdministrator {
    $principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Stop-Deploy '請以系統管理員身分執行安裝程式。'
    }
}

function Read-DeployConfig([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        Stop-Deploy ("找不到設定檔: " + $Path)
    }

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $json = [System.IO.File]::ReadAllText($resolved, [System.Text.Encoding]::UTF8)
    return $json | ConvertFrom-Json
}

function Invoke-DeployStep([string] $Name, [string] $Script, [hashtable] $Parameters = @{}) {
    if (-not (Test-Path -LiteralPath $Script)) {
        Stop-Deploy ("找不到部署步驟: " + $Script)
    }

    Write-Host ''
    Write-Host ('========== ' + $Name + ' ==========') -ForegroundColor Cyan
    & $Script @Parameters
    if (-not $?) {
        Stop-Deploy ($Name + ' 執行失敗。')
    }
}

function Get-DeployPackageRoot([string] $RoleDirectory) {
    $deployDirectory = Split-Path -Parent $RoleDirectory
    return Split-Path -Parent $deployDirectory
}

function Stop-DeployedApp([string] $AppDir, [string] $TaskName) {
    if ($TaskName) {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    }

    $targetExe = Join-Path $AppDir 'AniloxRoll.Monitor.exe'
    Get-Process -Name 'AniloxRoll.Monitor' -ErrorAction SilentlyContinue |
        Where-Object {
            try { $_.Path -eq $targetExe } catch { $false }
        } |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

function Import-PreviousAppConfig(
    [string[]] $PreviousAppDirs,
    [string] $AppDir,
    [string] $TaskName
) {
    if (-not $PreviousAppDirs -or $PreviousAppDirs.Count -eq 0) {
        return
    }

    $destinationConfig = Join-Path $AppDir 'Config'
    $destinationHasConfig = (Test-Path -LiteralPath $destinationConfig -PathType Container) -and
        ($null -ne (Get-ChildItem -LiteralPath $destinationConfig -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1))
    if ($destinationHasConfig) {
        Write-Host ("[OK] 現行 Config 已存在，不從舊路徑覆蓋: " + $destinationConfig) -ForegroundColor Green
        return
    }

    $resolvedDestination = [System.IO.Path]::GetFullPath($AppDir).TrimEnd('\')
    foreach ($previous in $PreviousAppDirs) {
        if (-not $previous) { continue }
        $resolvedPrevious = [System.IO.Path]::GetFullPath($previous).TrimEnd('\')
        if ($resolvedPrevious.Equals($resolvedDestination, [System.StringComparison]::OrdinalIgnoreCase)) {
            continue
        }

        $sourceConfig = Join-Path $resolvedPrevious 'Config'
        if (-not (Test-Path -LiteralPath $sourceConfig -PathType Container)) {
            continue
        }

        Stop-DeployedApp -AppDir $resolvedPrevious -TaskName $TaskName
        New-Item -ItemType Directory -Path $destinationConfig -Force | Out-Null
        Copy-Item -Path (Join-Path $sourceConfig '*') -Destination $destinationConfig -Recurse -Force
        Write-Host ("[OK] 已將舊 Config 複製到新程式路徑: " + $sourceConfig + " -> " + $destinationConfig) -ForegroundColor Green
        Write-Host ("     舊程式目錄先保留作為回退: " + $resolvedPrevious) -ForegroundColor Yellow
        return
    }
}

function Install-AppDesktopShortcut(
    [string] $AppDir,
    [string] $DesktopDirectory = ''
) {
    $appExe = Join-Path $AppDir 'AniloxRoll.Monitor.exe'
    if (-not (Test-Path -LiteralPath $appExe -PathType Leaf)) {
        Stop-Deploy ("建立程式桌面捷徑前找不到 EXE: " + $appExe)
    }

    if (-not $DesktopDirectory) {
        $DesktopDirectory = [Environment]::GetFolderPath([Environment+SpecialFolder]::CommonDesktopDirectory)
    }
    if (-not $DesktopDirectory) {
        $DesktopDirectory = Join-Path $env:PUBLIC 'Desktop'
    }
    if (-not (Test-Path -LiteralPath $DesktopDirectory)) {
        New-Item -ItemType Directory -Path $DesktopDirectory -Force | Out-Null
    }

    $shortcutPath = Join-Path $DesktopDirectory 'PICoater AOI.lnk'
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $appExe
    $shortcut.WorkingDirectory = $AppDir
    $shortcut.IconLocation = $appExe + ',0'
    $shortcut.Description = '啟動 PICoater AOI'
    $shortcut.Save()

    if (-not (Test-Path -LiteralPath $shortcutPath -PathType Leaf)) {
        Stop-Deploy ("程式桌面捷徑建立失敗: " + $shortcutPath)
    }
    Write-Host ("[OK] 程式桌面捷徑: " + $shortcutPath + " -> " + $appExe) -ForegroundColor Green
}

function Install-AppPayload([string] $SourceDir, [string] $AppDir, [string] $TaskName) {
    $sourceExe = Join-Path $SourceDir 'AniloxRoll.Monitor.exe'
    if (-not (Test-Path -LiteralPath $sourceExe)) {
        Stop-Deploy ("找不到安裝程式內容: " + $sourceExe)
    }

    Stop-DeployedApp -AppDir $AppDir -TaskName $TaskName

    if (-not (Test-Path -LiteralPath $AppDir)) {
        New-Item -ItemType Directory -Path $AppDir -Force | Out-Null
    }

    $configDir = Join-Path $AppDir 'Config'
    if (-not (Test-Path -LiteralPath $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    }

    $manifestName = 'deploy-manifest.txt'
    $oldManifest = Join-Path $AppDir $manifestName
    if (Test-Path -LiteralPath $oldManifest) {
        foreach ($relative in [System.IO.File]::ReadAllLines($oldManifest, [System.Text.Encoding]::UTF8)) {
            if (-not $relative -or [System.IO.Path]::GetExtension($relative) -eq '.json') {
                continue
            }

            $oldPath = Join-Path $AppDir $relative
            if (Test-Path -LiteralPath $oldPath -PathType Leaf) {
                Remove-Item -LiteralPath $oldPath -Force -ErrorAction SilentlyContinue
            }
        }
    }

    $files = Get-ChildItem -LiteralPath $SourceDir -File -Recurse |
        Where-Object {
            $relative = $_.FullName.Substring($SourceDir.Length).TrimStart('\')
            -not ($relative.StartsWith('Config\', [System.StringComparison]::OrdinalIgnoreCase) -and $_.Extension -eq '.json')
        }

    $manifest = New-Object System.Collections.Generic.List[string]
    foreach ($file in $files) {
        $relative = $file.FullName.Substring($SourceDir.Length).TrimStart('\')
        $destination = Join-Path $AppDir $relative
        $destinationDir = Split-Path -Parent $destination
        if (-not (Test-Path -LiteralPath $destinationDir)) {
            New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null
        }
        Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
        try {
            Unblock-File -LiteralPath $destination -ErrorAction Stop
        }
        catch {
            Stop-Deploy ("無法移除下載封鎖標記: " + $destination + "`r`n" + $_.Exception.Message)
        }

        $zone = Get-Item -LiteralPath $destination -Stream 'Zone.Identifier' -ErrorAction SilentlyContinue
        if ($zone) {
            Stop-Deploy ("檔案仍被 Windows 標示為來自網際網路: " + $destination)
        }
        $manifest.Add($relative)
    }

    [System.IO.File]::WriteAllLines($oldManifest, $manifest, [System.Text.Encoding]::UTF8)
    Install-AppDesktopShortcut -AppDir $AppDir
    Write-Host ("[OK] 程式已安裝到 " + $AppDir + "，Config 已保留。") -ForegroundColor Green
}

function Write-AppRoleConfig(
    [string] $AppDir,
    [string] $Role,
    [string] $StorageConfigFolder,
    [string] $StorageDataPath = '',
    [int] $StorageMinFreeGB = 0
) {
    $configDir = Join-Path $AppDir 'Config'
    if (-not (Test-Path -LiteralPath $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    }

    $mode = [ordered]@{
        Role = $Role
        StorageMachineConfigFolder = $StorageConfigFolder
        StorageMachineDataPath = $StorageDataPath
        StorageMinFreeGB = $StorageMinFreeGB
    }
    $json = $mode | ConvertTo-Json
    [System.IO.File]::WriteAllText(
        (Join-Path $configDir 'app-mode.json'),
        $json,
        (New-Object System.Text.UTF8Encoding($false)))
}
