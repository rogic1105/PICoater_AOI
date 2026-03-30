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

if "%choice%"=="1" (
    echo.
    echo [%date% %time%] Running unit tests...
    dotnet test tests/dotnet_test/AniloxRoll.Monitor.Tests/AniloxRoll.Monitor.Tests.csproj -p:Configuration=Release --filter "TestCategory!=Stress"
)
if "%choice%"=="2" (
    echo.
    echo [%date% %time%] Running stress tests... log: test_overnight.log
    echo === Started: %date% %time% === > test_overnight.log
    dotnet test tests/dotnet_test/AniloxRoll.Monitor.Tests/AniloxRoll.Monitor.Tests.csproj -p:Configuration=Release --filter "TestCategory=Stress" 2>&1 >> test_overnight.log
    echo === Finished: %date% %time% === >> test_overnight.log
    echo.
    echo Done. Results in test_overnight.log
)
if "%choice%"=="3" (
    echo.
    echo [%date% %time%] Running all tests... log: test_overnight.log
    echo === Started: %date% %time% === > test_overnight.log
    dotnet test tests/dotnet_test/AniloxRoll.Monitor.Tests/AniloxRoll.Monitor.Tests.csproj -p:Configuration=Release 2>&1 >> test_overnight.log
    echo === Finished: %date% %time% === >> test_overnight.log
    echo.
    echo Done. Results in test_overnight.log
)
if "%choice%"=="4" exit /b

echo.
pause
