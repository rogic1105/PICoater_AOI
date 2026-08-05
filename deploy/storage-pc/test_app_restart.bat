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

set /p cycles="Restart test cycles (default 3): "
if "%cycles%"=="" set "cycles=3"

powershell -NoProfile -ExecutionPolicy Bypass -File ^
  "%~dp0scripts\test_app_restart.ps1" ^
  -Config "%~dp0storage-config.json" ^
  -Cycles %cycles%

echo.
if errorlevel 1 (
  echo Result: FAIL. See D:\Anilox\Logs\DvtReports.
) else (
  echo Result: PASS. See D:\Anilox\Logs\DvtReports.
)
pause
endlocal
