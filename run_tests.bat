@echo off
chcp 65001 >nul
echo ============================================
echo  AniloxRoll.Monitor Test Runner
echo ============================================
echo.
echo  1. Unit tests only    (~2 sec)
echo  2. Stress tests only  (~12 hours)
echo  3. All tests          (~12 hours)
echo  4. Exit
echo.
set /p choice="Select (1-4): "

set "TEST_PROJ=tests/dotnet_test/AniloxRoll.Monitor.Tests/AniloxRoll.Monitor.Tests.csproj"
set "LOG=test_overnight.log"

if "%choice%"=="1" (
    echo.
    echo [%date% %time%] Running unit tests...
    dotnet test %TEST_PROJ% -p:Configuration=Release --filter "TestCategory!=Stress"
)
if "%choice%"=="2" (
    echo.
    echo [%date% %time%] Running stress tests...
    echo === Started: %date% %time% === > %LOG%
    powershell -Command "dotnet test %TEST_PROJ% -p:Configuration=Release --filter 'TestCategory=Stress' 2>&1 | Tee-Object -FilePath %LOG% -Append"
    echo === Finished: %date% %time% === >> %LOG%
)
if "%choice%"=="3" (
    echo.
    echo [%date% %time%] Running all tests...
    echo === Started: %date% %time% === > %LOG%
    powershell -Command "dotnet test %TEST_PROJ% -p:Configuration=Release 2>&1 | Tee-Object -FilePath %LOG% -Append"
    echo === Finished: %date% %time% === >> %LOG%
)
if "%choice%"=="4" exit /b

echo.
pause
