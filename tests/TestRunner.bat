@echo off
chcp 65001 >nul
pushd "%~dp0.."

echo ============================================
echo  AniloxRoll.Monitor Test Runner
echo ============================================
echo.
echo  1. Unit tests only    (~2 sec)
echo  2. Stress tests only
echo  3. All tests (unit first, then stress)
echo  4. Exit
echo.
set /p choice="Select (1-4): "

set "TEST_PROJ=tests\AniloxRoll.Monitor.Tests\AniloxRoll.Monitor.Tests.csproj"
set "LOG=%~dp0TestRunner.log"

if "%choice%"=="1" goto :unit
if "%choice%"=="2" goto :ask_minutes
if "%choice%"=="3" goto :ask_minutes
if "%choice%"=="4" exit /b
goto :eof

:unit
echo.
echo [%date% %time%] Running unit tests...
dotnet test "%TEST_PROJ%" -p:Configuration=Release -v normal --filter "TestCategory!=Stress"
goto :done

:ask_minutes
echo.
echo  Examples: 6=quick, 60=1hr, 480=8hr, 1440=24hr
set /p minutes="How many minutes? (default=60): "
if "%minutes%"=="" set "minutes=60"
echo.
if "%choice%"=="2" goto :stress
goto :all

:stress
echo [%date% %time%] Running stress tests (%minutes% min)... log: %LOG%
powershell -ExecutionPolicy Bypass -File "%~dp0TestRunner.ps1" "%TEST_PROJ%" "%LOG%" "Stress" "%minutes%"
goto :done

:all
echo [%date% %time%] Running all tests (%minutes% min)... log: %LOG%
powershell -ExecutionPolicy Bypass -File "%~dp0TestRunner.ps1" "%TEST_PROJ%" "%LOG%" "All" "%minutes%"
goto :done

:done
echo.
popd
pause
