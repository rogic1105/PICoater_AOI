@echo off
setlocal
cd /d "%~dp0"

net session >nul 2>&1
if errorlevel 1 (
  echo Requesting administrator permission...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

echo.
echo WARNING: This test deletes the oldest complete day under:
echo          D:\Anilox\Captures
echo The app setting is restored after the test, but deleted captures are permanent.
echo.
set /p confirm="Type DELETE to continue: "
if /I not "%confirm%"=="DELETE" (
  echo Canceled. No files were changed.
  pause
  exit /b 2
)

powershell -NoProfile -ExecutionPolicy Bypass -File ^
  "%~dp0scripts\test_local_retention.ps1" ^
  -Config "%~dp0storage-config.json" ^
  -ConfirmInPlaceDelete

echo.
if errorlevel 1 (
  echo Result: FAIL. See D:\Anilox\Logs\DvtReports.
) else (
  echo Result: PASS. See D:\Anilox\Logs\DvtReports.
)
pause
endlocal
