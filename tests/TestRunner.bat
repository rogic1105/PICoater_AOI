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
echo  6. IO＋儲存電腦待機耐久測試（不 Grab）
echo  7. 完整離線測試並記錄最新報告
echo  8. 回顧／報表 30,000 筆資料測試
echo  9. 實際取相（IO 三循環＋時間／高度）
echo 10. SMB 中斷／待傳補送恢復（需系統管理員）
echo 11. IO／光源軟體斷線恢復（首次安裝固定管理員動作）
echo 12. 低磁碟刪檔與狀態恢復（隔離 TEMP，不碰正式資料）
echo 13. 反覆 Grab 耐久測試（預設 120 分鐘）
echo 14. 結束
echo.
set /p choice="請選擇 (1-14): "

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
if "%choice%"=="14" goto :done
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

:physical_capture
powershell -NoProfile -ExecutionPolicy Bypass -File "tests\TestRunner.ps1" -Mode PhysicalCapture
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
