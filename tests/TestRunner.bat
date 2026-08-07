@echo off
pushd "%~dp0.."
set "test_exit=0"

echo ==================================================
echo  PICoater AOI Test Runner
echo ==================================================
echo.
echo  1. Offline functional: Build + Unit + Integration + DVT
echo  2. Offline stress
echo  3. Offline soak
echo  4. Physical IO stability, no Grab
echo  5. Storage-PC stability, no Grab
echo  6. Physical IO + Storage-PC soak, no Grab
echo  7. Full offline campaign and latest report
echo  8. Review/report 30,000-record test
echo  9. Physical capture: IO + time + height
echo 10. SMB interruption and backlog recovery, admin required
echo 11. IO/light software disconnect recovery
echo 12. Low-disk retention in isolated TEMP
echo 13. Repeated Grab soak, default 120 minutes
echo 14. Inspection standards with light 100/255 surrogate
echo 15. Virtual IO connect/disconnect/reconnect, no camera required
echo 16. Exit
echo.
set "choice=%~1"
if not defined choice set /p choice="Select (1-16): "

if "%choice%"=="1" goto :functional
if "%choice%"=="2" goto :stress
if "%choice%"=="3" goto :soak
if "%choice%"=="4" goto :physical_io
if "%choice%"=="5" goto :physical_storage
if "%choice%"=="6" goto :physical_soak
if "%choice%"=="7" goto :all
if "%choice%"=="8" goto :review_report_30k
if "%choice%"=="9" goto :physical_capture
if "%choice%"=="10" goto :physical_recovery
if "%choice%"=="11" goto :physical_bridge_recovery
if "%choice%"=="12" goto :physical_retention
if "%choice%"=="13" goto :physical_capture_soak
if "%choice%"=="14" goto :physical_inspection_standards
if "%choice%"=="15" goto :virtual_io
if "%choice%"=="16" goto :done
goto :invalid

:functional
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode Functional
goto :result

:stress
set /p stress_minutes="壓力測試分鐘數（預設一循環 120）: "
if "%stress_minutes%"=="" set "stress_minutes=120"
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode Stress -StressMinutes "%stress_minutes%"
goto :result

:soak
set /p soak_minutes="耐久測試分鐘數（預設一循環 120）: "
if "%soak_minutes%"=="" set "soak_minutes=120"
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode Soak -SoakMinutes "%soak_minutes%"
goto :result

:physical_io
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode PhysicalIo
goto :result

:virtual_io
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode VirtualIo
goto :result

:physical_capture
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode PhysicalCapture
goto :result

:physical_inspection_standards
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode PhysicalInspectionStandards
goto :result

:physical_storage
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode PhysicalStorage
goto :result

:physical_recovery
schtasks /Query /TN "PICoater-DVT-Block-Storage" >nul 2>&1
if errorlevel 1 (
  echo.
  echo 尚未安裝儲存網路固定管理員動作，現在進行一次性安裝。
  call "tests\InstallDvtAdminActions.bat" --no-pause
  if errorlevel 1 goto :result
)
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode PhysicalRecovery
goto :result

:physical_bridge_recovery
schtasks /Query /TN "PICoater-DVT-Block-IO502" >nul 2>&1
if errorlevel 1 (
  echo.
  echo 尚未安裝固定管理員動作，現在進行一次性安裝。
  call "tests\InstallDvtAdminActions.bat" --no-pause
  if errorlevel 1 goto :result
)
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode PhysicalBridgeRecovery
goto :result

:physical_retention
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode PhysicalRetention
goto :result

:physical_soak
set /p physical_soak_minutes="耐久測試分鐘數（預設一循環 120）: "
if "%physical_soak_minutes%"=="" set "physical_soak_minutes=120"
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode PhysicalSoak -PhysicalSoakMinutes "%physical_soak_minutes%"
goto :result

:physical_capture_soak
set /p physical_capture_soak_minutes="反覆 Grab 分鐘數（預設 120）: "
if "%physical_capture_soak_minutes%"=="" set "physical_capture_soak_minutes=120"
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode PhysicalCaptureSoak -PhysicalCaptureSoakMinutes "%physical_capture_soak_minutes%"
goto :result

:review_report_30k
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode ReviewReport30k
goto :result

:all
set /p stress_minutes="壓力測試分鐘數（預設一循環 120）: "
if "%stress_minutes%"=="" set "stress_minutes=120"
set /p soak_minutes="耐久測試分鐘數（預設一循環 120）: "
if "%soak_minutes%"=="" set "soak_minutes=120"
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode All -StressMinutes "%stress_minutes%" -SoakMinutes "%soak_minutes%" -RecordLatest
goto :result

:invalid
echo Invalid selection.
set "test_exit=2"
goto :done

:result
set "test_exit=%errorlevel%"
echo.
if "%test_exit%"=="0" goto :result_pass
echo Test result: FAIL. See artifacts\test-reports.
goto :done

:result_pass
echo Test result: PASS.

:done
popd
if /i not "%~2"=="--no-pause" pause
exit /b %test_exit%
