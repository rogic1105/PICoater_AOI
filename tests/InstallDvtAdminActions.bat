@echo off
setlocal
pushd "%~dp0.."

set "RESULT=1"
set "NO_PAUSE=0"
set "VALIDATE_ONLY=0"
set "INSTALLER=%CD%\tools\dvt\admin\Install-DvtPrivilegedActions.ps1"
set "ELEVATOR=%CD%\tools\dvt\admin\Start-DvtPrivilegedActionInstall.ps1"

if /I "%~1"=="--no-pause" set "NO_PAUSE=1"
if /I "%~1"=="--validate-only" set "VALIDATE_ONLY=1"
if /I "%~1"=="--validate-only" set "NO_PAUSE=1"

if not exist "%INSTALLER%" goto :missing
if not exist "%ELEVATOR%" goto :missing
if "%VALIDATE_ONLY%"=="1" goto :validate

fltmc >nul 2>&1
if errorlevel 1 goto :elevate
goto :install

:elevate
echo First-time Windows administrator authorization is required.
echo After installation, option 11 will not ask for UAC again.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%ELEVATOR%" -Installer "%INSTALLER%"
set "RESULT=%ERRORLEVEL%"
goto :report

:install
powershell -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER%"
set "RESULT=%ERRORLEVEL%"
goto :report

:validate
powershell -NoProfile -ExecutionPolicy Bypass -File "%INSTALLER%" -ValidateOnly
set "RESULT=%ERRORLEVEL%"
goto :report

:missing
echo [FAIL] Installer file not found.
echo        %INSTALLER%
echo        %ELEVATOR%
set "RESULT=2"

:report
echo.
if not "%RESULT%"=="0" goto :failed
echo [PASS] Administrator actions were installed or validated.
echo        Use tests\TestRunner.bat from now on.
goto :finish

:failed
echo [FAIL] Installation failed or UAC was canceled.
echo        ExitCode=%RESULT%

:finish
popd
if "%NO_PAUSE%"=="1" goto :exit
echo.
pause

:exit
endlocal & exit /b %RESULT%
