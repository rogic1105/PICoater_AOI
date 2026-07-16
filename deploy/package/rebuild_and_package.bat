@echo off
setlocal
cd /d "%~dp0\..\.."

set SOURCE_STATE=
for /f %%S in ('powershell -NoProfile -Command "$lines = @(git status --porcelain); if ($LASTEXITCODE -ne 0) { exit 2 }; if ($lines.Count -gt 0) { 'dirty' } else { 'clean' }"') do set SOURCE_STATE=%%S
if not defined SOURCE_STATE goto :state_fail

set PACKAGE_MODE=OFFICIAL RELEASE
set MODE_NOTE=Committed source; suitable for production release.
set EXTRA_ARGS=
if /I "%SOURCE_STATE%"=="dirty" (
    set PACKAGE_MODE=SMOKE TEST PACKAGE
    set MODE_NOTE=Uncommitted source; use only for machine testing.
    set EXTRA_ARGS=-AllowDirty
)

for /f %%V in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmm"') do set VERSION=%%V

echo ==========================================
echo  PICoater rebuild and package
echo  Mode: %PACKAGE_MODE%
echo  %MODE_NOTE%
echo  Version: %VERSION%
echo ==========================================
echo.
if not defined PICoaterAutoConfirm (
    choice /C YN /N /M "Continue? [Y/N]: "
    if errorlevel 2 goto :cancel
)
echo.

echo [1/2] Rebuilding Release x64 and packaging Storage...
powershell -NoProfile -ExecutionPolicy Bypass -File ".\deploy\package\package_release.ps1" -Role Storage -Version "%VERSION%" -Rebuild %EXTRA_ARGS%
if errorlevel 1 goto :fail

echo.
echo [2/2] Packaging Inspection from the rebuilt output...
powershell -NoProfile -ExecutionPolicy Bypass -File ".\deploy\package\package_release.ps1" -Role Inspection -Version "%VERSION%" -SkipBuild %EXTRA_ARGS%
if errorlevel 1 goto :fail

echo.
echo [OK] Rebuild completed and both packages are ready:
echo      %CD%\artifacts\deploy
if not defined PICoaterNoOpen start "" "%CD%\artifacts\deploy"
if not defined PICoaterNoPause pause
exit /b 0

:fail
echo.
echo [FAIL] Rebuild or packaging stopped. Read the error above.
if not defined PICoaterNoPause pause
exit /b 1

:state_fail
echo [FAIL] Unable to read Git source state. Packaging was not started.
if not defined PICoaterNoPause pause
exit /b 1

:cancel
echo.
echo [CANCELLED] No build or package was created.
if not defined PICoaterNoPause pause
exit /b 0
