# 壓力測試規劃 — PICoater AOI

> **目的**：在正式現場運行前驗證系統在持續高負載下的穩定性、效能、資源管理、失效恢復能力。
> **狀態**：規劃中（2026-05-15）。實際執行請填上日期 + 結果於每項勾選。

---

## 0. 預設環境

- **硬體**：Inspection PC（含 CUDA GPU、MIL 板卡、7 台 line-scan 相機、PLC ET-7044、光源 LTS-3DPA24）+ Storage PC（SMB share）
- **網路**：1Gbps 跨機（Inspection ↔ Storage）
- **資料盤**：SSD ≥ 500GB 可用空間
- **每秒 grab 量**：7 張 × ~1Hz = 7 張/秒（依 PLC trigger）
- **典型一張 .bmp 約 50MB（全解析度 16384×3001 8bpp）**

---

## Phase 0：準備（30 分鐘）

| 項目 | 內容 | Pass 條件 |
|---|---|---|
| P0-1 | Clean build：`Release|x64`、unit tests（41 個）全綠 | ✅ |
| P0-2 | 清空 `D:\Anilox\Captures` 確保 baseline | 資料夾大小 < 100MB |
| P0-3 | 確認 `bin\x64\Release\` 內 native DLL 都是新版（picoater_api.dll、Module_GetPICoaterBackground.lib）| 時間戳比 commit `15b7902` 新 |
| P0-4 | Storage PC 也是同版本程式 + 同 settings JSON | git commit hash 一致 |
| P0-5 | PLC / 光源連線 OK（lblPlcConn / lblLightConn 綠色）| 連線狀態列綠燈 |
| P0-6 | 開 Performance Monitor 監控：CPU、RAM、GPU、磁碟 IO、網路、Handle 數 | 啟動 baseline 記下 |

---

## Phase 1：單元壓測（既有 `StressTests.cs`，2~3 小時）

擴充既有測試規模到「正式現場 1 小時量」：

| 項目 | Test | 規模 | 環境變數 | Pass |
|---|---|---|---|---|
| P1-1 | `PlcStressTests` Modbus 100 萬循環 | 既定 | `STRESS_MINUTES=60` | < 60 分完成且無 ConnLost |
| P1-2 | `CsvStressTests` 50 萬筆 append + parse | 既定 | 同上 | 無 IOException、ContentKey 命中 |
| P1-3 | `SettingsStressTests` 14.5 萬讀寫 | 既定 | 同上 | round-trip 無誤差 |
| P1-4 | **新增**：`HessianRescaleHelper` 對 16384 點 × 100 萬次 | 新測 | — | < 5 秒、結果穩定 |
| P1-5 | **新增**：`InspectionStatisticsService.OpenCsvShared` 多 thread 並發讀 | 新測 | — | 無 deadlock、無 IOException |

執行：`TestRunner\TestRunner.bat`（雙擊）→ 選 `Stress`。

---

## Phase 2：整合壓測 — Live Grab + 全鏈路（4~6 小時）

啟動程式、實際取像，跑滿一個典型生產 shift 模擬。

### P2-1：短測 30 分鐘（功能煙霧測試）

| 子項 | 觸發 | 預期 |
|---|---|---|
| Live grab 連續取像 | 按【開始抓取】| 每秒 7 張、無 frame drop |
| Mura 觸發 DO_MURA | 故意放標準 Mura 樣本 | DO1 觸發 + lblIoDoMura 亮黃 + CSV maxExceed=1 |
| 暫停 Mura 偵測 | 點 lblIoDoMura | 顯示 ⏸ + DO1 不觸發 |
| CSV 寫入正常 | grep 觀察 `D:\Anilox\Captures\{date}.csv` | 每秒新增 7 行、無錯誤行 |
| 切到 Data tab | 取像中切 tab | 不卡 UI、btnSelectDataFolder 載入順暢 |

**Pass**：30 分鐘內無 crash、無 trace 中出現 Exception。

### P2-2：拖 PropertyGrid（H3 debounce 驗證）

| 子項 | 操作 | 預期 |
|---|---|---|
| 拖「垂直正規值」slider 連續變動 0.1 → 1.0 → 0.1 | 5 秒內拖 50 次以上 | 1) 拖動順暢、2) 停止後 300ms 內 chart 更新一次、3) 無 5+ 次 RefreshStats 連環 |
| 拖完立即關程式 | 關閉 form | 無 NRE、`_statsRefreshDebouncer` Dispose 正確 |

**Pass**：CPU 拖動期間 < 30%；停止後一波刷新 < 1 秒。

### P2-3：跨 Tab 高頻切換（H1 + cleanup 驗證）

| 子項 | 操作 | 預期 |
|---|---|---|
| Live ↔ Review ↔ Data 連續切換 100 次 | 5 秒/次 | 無 memory 累積（< 50MB）、無 CUDA leak（vRAM 穩定）|
| 切到 Review tab 載歷史 grab | btnSelectFolder | 載入 < 3 秒、chartOverview 對齊 |
| Data tab 點選 listViewGrabDetail row 50 次 | 不同 row | 每次 chartMuraProfile 立即重畫、cbDataGrabId 對齊 |

**Pass**：handle 數 < 1000、RAM 不持續上漲。

### P2-4：GroupBox 模式快速切換

| 子項 | 操作 | 預期 |
|---|---|---|
| 點【單片】→【序號範圍】→【時序範圍】→【單片】循環 30 次 | 1 秒/次 | 1) 切到 GrabIdRange 自動攤 [oldest..newest]、2) 切到 TimeRange 自動攤 [min..max]、3) 切回 SingleSheet 顯示 cbDataGrabId 當前 |
| 每次切換 chartMuraProfile 內容 | 觀察 | 模式對應的視圖正確（單片 stitch / aggregate）|

**Pass**：每次切換 chart 更新 < 500ms、listViewGrabDetail 不爆量。

---

## Phase 3：長時間 Soak Test（24 小時，最關鍵）

連續取像 24 小時，模擬一個生產日全程。**這是壓力測試的最終目標**。

### 監控指標（每 10 分鐘記錄一次）

| 指標 | Baseline | Pass 條件 |
|---|---|---|
| RAM（私有工作集） | 啟動 ~800MB | 24h 後 < baseline + 200MB |
| Handle 數 | 啟動 ~500 | < 1500 持平 |
| GPU vRAM | 啟動 ~1GB | 持平（± 100MB）|
| CUDA pinned memory（透過 `GetMemoryFreeMB`）| baseline | 持平 |
| 磁碟可用空間 | 預留 100GB | 不低於 LocalMinFreeGB（觸發 retention）|
| RemoteCopy queue 長度 | < 100 | 持平、不單調上升 |
| CSV append 平均延遲 | < 5ms | < 10ms |
| Mura DO 觸發到 PLC 收到 | < 50ms | < 100ms |
| Frame drop | 0 | < 0.01%（一天 < 60 frames）|

### 故意觸發的事件（24h 內穿插）

| 時段 | 事件 | 預期 |
|---|---|---|
| 2h | 改正規值 V 0.3 → 0.5 | 立即套用、chart 坡度減半、不影響 grab |
| 4h | 改閾值 ErrorValueMaxV 0.4 → 0.5 | 立即套用、Pass/Fail 立即重算（debounce 後 300ms）|
| 6h | 觸發 IO Grab Start/Stop 30 次 | 每次都成功啟停、無 FSM 異常 |
| 8h | 切到 Data tab 跑 RefreshStats（資料量已累積 ~20 萬筆 CSV row）| < 5 秒完成 + UI 不凍結 > 1 秒 |
| 12h | 切【序號範圍群組】掃整天資料 | < 10 秒，statistics 正確 |
| 16h | 開啟 chartYearly 看連續切換 | 不卡、period charts 同步 |
| 20h | 暫停 5 分鐘觀察 idle 行為 | watchdog 不誤觸、storage retention 不誤刪 |

**Pass**：所有監控指標達標、無 unhandled exception、無 user-visible bug。

---

## Phase 4：失效注入（半日，每項獨立）

### P4-1：網路斷線（SMB share 不可達）

| 步驟 | 預期 |
|---|---|
| 取像中拔網路線 30 秒 | RemoteCopy queue 累積、本地寫不受影響 |
| 接回網路 | RemoteCopy 自動恢復、queue 排空（不阻塞 main loop）|

**Pass**：本地 CSV 完整、無 frame drop、RemoteCopy 重試成功率 > 95%。

### P4-2：PLC 斷線

| 步驟 | 預期 |
|---|---|
| 取像中拔 PLC 網線 | lblIoConn 變紅、watchdog 不誤觸停止取像 |
| 接回 PLC | 自動重連、繼續觸發 grab |

**Pass**：PLC reconnect time < 30 秒。

### P4-3：光源斷線

| 步驟 | 預期 |
|---|---|
| 取像中拔光源 USB | lblLightConn 變紅、影像變暗 |
| 接回光源 | 自動重連、亮度恢復 |

**Pass**：reconnect 成功。

### P4-4：磁碟空間不足

| 步驟 | 預期 |
|---|---|
| 寫滿磁碟至剩 50GB | StorageRetentionService 開始刪最舊資料夾 |
| 繼續寫到剩 30GB | 仍持續刪舊資料夾、Live grab 不中斷 |

**Pass**：retention 正確刪、CSV 保留、grab 不中斷。

### P4-5：相機斷線（1/7）

| 步驟 | 預期 |
|---|---|
| 取像中拔 1 台相機 | lblCamCount 6/7、合圖該位置空白 |
| 接回相機 | 自動偵測、合圖恢復 |

**Pass**：其他 6 台不受影響、reconnect 流程順暢。

### P4-6：Storage PC 同時運行 cleanup

| 步驟 | 預期 |
|---|---|
| Inspection PC 寫 CSV 的同時，Storage PC 點 cleanup-request.flag | B-H1 修正後不應有 IOException |
| 監控雙方 Trace.WriteLine | 無 `IOException: 程序無法存取檔案` |

**Pass**：B-H1 fix 驗證通過。

---

## Phase 5：邊界 case（1 小時）

| 子項 | 操作 | 預期 |
|---|---|---|
| 正規值 V = 0.0001 | PropertyGrid 改 | chart 顯示峰值極大、不崩 |
| 正規值 V = 10.0 | PropertyGrid 改 | chart 顯示峰值極小、threshold 線位置正確 |
| 空 CaptureRoot 資料夾 | btnSelectDataFolder | listViewGrabDetail 空、chart 清空、無 NRE |
| 1M+ CSV rows（手動生成）| btnSelectDataFolder | RefreshStats 完成（即使慢）、無 OOM |
| CSV 內 #CFG 損壞 | 手動編輯 | TryParse 跳過、其他 row 正常 |
| HM_V_capture = HM_V_current（ratio=1）| 預設情況 | 不做 rescale（noOp），效能與舊版相同 |

---

## Phase 6：Regression 檢查（針對最近 12 個 commit）

| Commit | 驗證項 | 預期 |
|---|---|---|
| `15b7902` `.bin` 中性化 | 取像、檢查 `.bin` 內值是否有 > 255（峰值未截斷）| `check_bin_neutral.py` 報告新檔 |
| `10f0b6d` V/H 分離 | 改 HM_V 影響 V chart、HM_H 影響 H chart | 互不影響 |
| `97f69a9` chartMuraProfile 對齊 | btnSelectDataFolder 立即顯示單 grab stitch | 不需切 Review tab |
| `9f9ee47` GroupBox 切模式 | 點三個 GroupBox 切換 | 行為對齊 P2-4 |
| `0e01f95` listView 點選同步 | 點 row | cbDataGrabId 對齊 |
| `cf29600` legacy fallback 移除 | 舊 CSV（無 V/H）讀取 | 用 InspectionDefaults |
| `2294621` H/M/L 18 個 | 各項目見 round-2 report | round-2 已驗證 |
| `5f5a020` round-2 9 個 | B-H1 跨 process race、debouncer Dispose | P4-6 + P2-2 |

---

## 監控腳本建議

### A. RAM / Handle 數抓取

```powershell
while ($true) {
    $p = Get-Process AniloxRoll.Monitor -ErrorAction SilentlyContinue
    if ($p) {
        "$(Get-Date -Format 'HH:mm:ss'),$($p.WorkingSet64 / 1MB),$($p.HandleCount),$($p.Threads.Count)" |
            Out-File -Append D:\stress-monitor.csv
    }
    Start-Sleep -Seconds 60
}
```

### B. CSV append 延遲

CSV 路徑 `D:\Anilox\Captures\{yyyy}\{yyyyMM}\{yyyyMMdd}.csv`，比較相鄰 row 的 timestamp 差距：

```python
import csv, re
from pathlib import Path

p = Path(r"D:\Anilox\Captures\2026\202605\20260515.csv")
times = []
with p.open(encoding="utf-8-sig") as f:
    for row in csv.reader(f):
        if not row or row[0].startswith("#") or row[0] == "Id": continue
        # FileName: 20260515_HHmmss.fff-camId
        m = re.search(r"_(\d{6})\.(\d{3})-", row[1])
        if m:
            hms, ms = m.group(1), m.group(2)
            t = int(hms[:2])*3600000 + int(hms[2:4])*60000 + int(hms[4:6])*1000 + int(ms)
            times.append(t)

# 相鄰 row 時間差
diffs = [times[i+1]-times[i] for i in range(len(times)-1)]
print(f"avg={sum(diffs)/len(diffs):.1f}ms max={max(diffs)}ms")
```

### C. RemoteCopy queue 長度

加 `Trace.WriteLine($"[RemoteCopy] queue={_queue.Count}")` 到 `RemoteCopyService.cs` 的迴圈內。

---

## Pass / Fail 總判定

| 嚴重度 | 條件 | 行動 |
|---|---|---|
| **Critical Fail** | 24h 內 unhandled crash / 資料損壞 / RAM 持續上漲 | 必修、不可上線 |
| **High Fail** | Frame drop > 0.1%、Mura latency > 200ms、reconnect 失敗 | 評估後修 |
| **Medium Fail** | UI 凍結 > 2 秒、queue 累積 > 1000 | 優化後再測 |
| **Low Fail** | 邊界 case 行為奇怪但不影響主流程 | 補文件 + 監控 |
| **Pass** | 所有 Phase 通過、監控指標達標 | 可上線 |

---

## 預估時程

| Phase | 時長 | 累計 |
|---|---|---|
| Phase 0 準備 | 0.5h | 0.5h |
| Phase 1 單元 | 3h | 3.5h |
| Phase 2 整合短測 | 6h（含 P2-1~4）| 9.5h |
| **Phase 3 Soak** | **24h（隔夜）** | **33.5h** |
| Phase 4 失效注入 | 4h | 37.5h |
| Phase 5 邊界 | 1h | 38.5h |
| Phase 6 Regression | 2h | 40.5h |

**建議分 2~3 天執行**：Day 1 = Phase 0/1/2、Day 2 = Phase 3 Soak（隔夜）、Day 3 = Phase 4/5/6 + 報告。
