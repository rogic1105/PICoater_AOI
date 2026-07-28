@echo off
chcp 65001 >nul
pushd "%~dp0.."

echo ==================================================
echo  PICoater AOI 統一測試入口
echo ==================================================
echo.
echo  1. 離線功能測試（Build + Unit + Integration + DVT）
echo  2. 離線壓力測試
echo  3. 離線耐久測試
echo  4. 實體 IO 五分鐘穩定測試（不 Grab）
echo  5. 儲存電腦五分鐘穩定測試（不 Grab）
echo  6. 完整離線測試並記錄最新報告
echo  7. 結束
echo.
set /p choice="請選擇 (1-7): "

if "%choice%"=="1" goto :functional
if "%choice%"=="2" goto :stress
if "%choice%"=="3" goto :soak
if "%choice%"=="4" goto :physical_io
if "%choice%"=="5" goto :physical_storage
if "%choice%"=="6" goto :all
if "%choice%"=="7" goto :done
goto :invalid

:functional
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode Functional
goto :result

:stress
set /p stress_minutes="壓力測試分鐘數（預設 1）: "
if "%stress_minutes%"=="" set "stress_minutes=1"
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode Stress -StressMinutes "%stress_minutes%"
goto :result

:soak
set /p soak_minutes="耐久測試分鐘數（預設 10；正式可填 480 或 1440）: "
if "%soak_minutes%"=="" set "soak_minutes=10"
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode Soak -SoakMinutes "%soak_minutes%"
goto :result

:physical_io
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode PhysicalIo
goto :result

:physical_storage
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode PhysicalStorage
goto :result

:all
set /p stress_minutes="壓力測試分鐘數（預設 1）: "
if "%stress_minutes%"=="" set "stress_minutes=1"
set /p soak_minutes="耐久測試分鐘數（預設 10）: "
if "%soak_minutes%"=="" set "soak_minutes=10"
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode All -StressMinutes "%stress_minutes%" -SoakMinutes "%soak_minutes%" -RecordLatest
goto :result

:invalid
echo 無效選項。
goto :done

:result
echo.
if errorlevel 1 (
  echo 測試結果：FAIL，請查看畫面與 artifacts\test-reports。
) else (
  echo 測試結果：PASS。
)

:done
popd
pause
