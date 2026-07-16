@echo off
setlocal
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Requesting administrator privileges...
    powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
    exit /b
)

cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
set RESULT=%errorlevel%
echo.
if not "%RESULT%"=="0" echo [FAIL] Setup failed. See manual-install.html.
echo Press any key to close...
pause >nul
exit /b %RESULT%
