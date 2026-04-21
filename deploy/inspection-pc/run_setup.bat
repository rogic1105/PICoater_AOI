@echo off
REM Inspection PC One-Shot Setup - Double-click to run
REM Step 1: NIC secondary IP (PLC + Storage subnets)
REM Step 2: Allow anonymous Guest SMB client

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Requesting admin privileges...
    powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
    exit /b
)

cd /d "%~dp0"
echo ==========================================
echo  Inspection PC Setup  (Step 1/2: NIC Secondary IP)
echo ==========================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_inspection_nic.ps1"

echo.
echo ==========================================
echo  Inspection PC Setup  (Step 2/2: Guest SMB Client)
echo ==========================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_guest.ps1"

echo.
echo ==========================================
echo  All Done. Press any key to close...
echo ==========================================
pause >nul
