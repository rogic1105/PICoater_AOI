# 關閉自動睡眠 / 休眠 / 硬碟停轉
#
# 目的：
# - 儲存機：必須隨時可被 SMB 存取，絕對不能進睡眠
# - 檢測機：長時間取像 / 遠端複製中不可被系統中斷
#
# 作法：用 powercfg 把目前啟用電源計畫的 timeout 全部設 0（= 永不觸發）
# 不禁用休眠機制本身（hibernate file 保留），只是把時間設到 0

function Die([string]$msg) { Write-Host ("[FAIL] " + $msg) -ForegroundColor Red; exit 1 }

$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Die "請以系統管理員身分執行"
}

Write-Host "[1/4] 關閉自動睡眠（standby）..." -ForegroundColor Cyan
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
Write-Host "  -> AC / DC standby = 0 (永不睡眠)" -ForegroundColor Green

Write-Host "[2/4] 關閉自動休眠（hibernate）..." -ForegroundColor Cyan
powercfg /change hibernate-timeout-ac 0
powercfg /change hibernate-timeout-dc 0
Write-Host "  -> AC / DC hibernate = 0 (永不休眠)" -ForegroundColor Green

Write-Host "[3/4] 關閉硬碟自動停轉..." -ForegroundColor Cyan
powercfg /change disk-timeout-ac 0
powercfg /change disk-timeout-dc 0
Write-Host "  -> AC / DC disk = 0 (硬碟永不停轉)" -ForegroundColor Green

# 蓋子 / 電源按鈕動作不動（保留原廠預設，使用者主動關機仍有效）
# 螢幕 timeout 不動（螢幕仍可關閉省電，系統不睡就好）

Write-Host "[4/4] 驗證目前電源計畫..." -ForegroundColor Cyan
$activeScheme = (powercfg /getactivescheme) -replace '.*: ', ''
Write-Host ("  目前計畫：" + $activeScheme) -ForegroundColor Gray

Write-Host ""
Write-Host "[完成] 自動睡眠 / 休眠 / 硬碟停轉 已全部關閉。" -ForegroundColor Cyan
Write-Host "       可用 powercfg /query 檢視詳細設定。" -ForegroundColor Gray
