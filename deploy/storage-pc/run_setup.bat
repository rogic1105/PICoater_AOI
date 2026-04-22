@echo off
REM Storage PC One-Shot Setup - Double-click to run
REM Step 1: IP + SMB share + firewall
REM Step 2: Anonymous Guest access
REM Step 3: Remote Desktop (RDP)

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Requesting admin privileges...
    powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
    exit /b
)

cd /d "%~dp0"
echo ==========================================
echo  Storage PC Setup  (Step 1/3: Network + Share)
echo ==========================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_storage_pc.ps1"

echo.
echo ==========================================
echo  Storage PC Setup  (Step 2/3: Guest Access)
echo ==========================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_guest.ps1"

echo.
echo ==========================================
echo  Storage PC Setup  (Step 3/3: Remote Desktop)
echo ==========================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_rdp.ps1"

echo.
echo ==========================================
echo  All Done. Press any key to close...
echo ==========================================
pause >nul
