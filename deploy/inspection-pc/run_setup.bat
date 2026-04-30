@echo off
REM Inspection PC One-Shot Setup - Double-click to run
REM Step 1: IO NIC IP (192.168.255.10 for IO module)
REM Step 2: NIC secondary IP (Storage subnet 192.168.10.10)
REM Step 3: Allow anonymous Guest SMB client
REM Step 4: Disable auto sleep / hibernate
REM Step 5: Write app-mode.json (Role = Inspection)

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Requesting admin privileges...
    powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
    exit /b
)

cd /d "%~dp0"
echo ==========================================
echo  Inspection PC Setup  (Step 1/4: IO NIC IP)
echo ==========================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_io_ip.ps1"

echo.
echo ==========================================
echo  Inspection PC Setup  (Step 2/4: NIC Secondary IP)
echo ==========================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_inspection_nic.ps1"

echo.
echo ==========================================
echo  Inspection PC Setup  (Step 3/4: Guest SMB Client)
echo ==========================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_guest.ps1"

echo.
echo ==========================================
echo  Inspection PC Setup  (Step 4/4: Disable Sleep)
echo ==========================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_nosleep.ps1"

echo.
echo ==========================================
echo  Inspection PC Setup  (Step 5/5: App Mode)
echo ==========================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_appmode.ps1"

echo.
echo ==========================================
echo  All Done. Press any key to close...
echo ==========================================
pause >nul
