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
    dotnet test %TEST_PROJ% -p:Configuration=Release -v normal --filter "TestCategory!=Stress"
)
if "%choice%"=="2" (
    echo.
    echo [%date% %time%] Running stress tests...
    powershell -Command "[Console]::OutputEncoding = [Text.Encoding]::UTF8; $log = '%LOG%'; '=== Started: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ===' | Out-File $log -Encoding utf8; dotnet test %TEST_PROJ% -p:Configuration=Release -v normal --filter 'TestCategory=Stress' 2>&1 | ForEach-Object { $_; $_ | Out-File $log -Append -Encoding utf8 }; '=== Finished: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ===' | Out-File $log -Append -Encoding utf8"
)
if "%choice%"=="3" (
    echo.
    echo [%date% %time%] Running all tests...
    powershell -Command "[Console]::OutputEncoding = [Text.Encoding]::UTF8; $log = '%LOG%'; '=== Started: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ===' | Out-File $log -Encoding utf8; dotnet test %TEST_PROJ% -p:Configuration=Release -v normal 2>&1 | ForEach-Object { $_; $_ | Out-File $log -Append -Encoding utf8 }; '=== Finished: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ===' | Out-File $log -Append -Encoding utf8"
)
if "%choice%"=="4" exit /b

echo.
pause
