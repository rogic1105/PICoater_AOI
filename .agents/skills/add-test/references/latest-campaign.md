# 最新自動化測試紀錄

> 本檔只保留最近一次彙總；下次正式測試會覆寫。Git commit history 才是長期紀錄。

- 結果：**PASS**
- 日期：2026-07-28
- 基準：`6ef23b9` 加上本次測試工具變更
- 電腦：`DESKTOP-C1MN5KD`
- 產品行為變更：無

## 測試結果

| 層級 | 測試 | 結果 | 實際證據 |
|---|---|---:|---|
| Build | `Release|x64` 全方案建置 | **PASS** | 0 warnings / 0 errors |
| Flow checker | Python DVT checker 自我測試 | **PASS** | 115 / 115 |
| Unit | .NET 單元測試 | **PASS** | 143 / 143 |
| Integration | .NET 整合測試 | **PASS** | 113 / 113 |
| DVT functional | Runner 自我檢查、開啟與正常關閉主程式 | **PASS** | 1 scenario |
| Stress | 離線壓力測試，含 IO / Storage bridge | **PASS** | 8 / 8，87.47 秒 |
| Soak | 離線耐久測試 | **PASS** | 8 / 8，693.69 秒 |
| Physical IO DVT | ET-7044 待機輪詢五分鐘，不 Grab | **PASS** | 305.59 秒；15 flow PASS / 0 FAIL |

實體 IO 測試證明：

- `192.168.255.1:502` controller 正常啟動。
- 主程式完成交握後進入「待機」，五分鐘後仍維持待機。
- 測試期間沒有 Grab、沒有啟動光源。
- Runner 還原設定後正常關閉主程式。
- `IO controller stop reason=shutdown` 與 `shutdown resources released` 均有 log 證據。

本機原始輸出位於 `artifacts/test-reports/`，該目錄不進 Git：

- `20260728-211907-6ef23b9/`：Build、Flow checker、Unit、Integration、DVT self-check。
- `20260728-205359-6ef23b9/`：Stress。
- `20260728-205545-6ef23b9/`：10 分鐘離線 Soak。
- `20260728-211306-6ef23b9/`：實體 IO 五分鐘 DVT。

## 測試中改善

1. 將原本分散的測試入口統一為 `tests/TestRunner.bat` / `tests/TestRunner.ps1`。
2. DVT Runner 新增命令列 scenario、自動輸出結果與自動關閉，能由測試腳本無人值守執行。
3. DVT log monitor 會保留前一步期間已到達但尚未使用的證據，避免硬體事件早到被誤判 timeout。
4. 壓力測試 fixture 補上現行 IO 啟動交握資料；產品要求沒有降低。
5. Stress / Soak 納入 `BridgeStress`，不再漏掉 IO 與 Storage bridge。
6. 每次測試產生原始 artifact；只有這份最新彙總進 Git，避免測試報告無限累積。

## 尚未覆蓋

以下不是失敗，但目前沒有接線或環境，因此明確記為 **NOT COVERED**：

- 實體相機／grabber、七台相機負載、背景取得與正式 Grab。
- 實體光源連線、斷線與恢復。
- 實體 IO 斷線／恢復與 IO START 觸發 Grab；本次只測穩定待機。
- 儲存電腦 SMB 中斷、待傳 backlog、低磁碟刪除與恢復。
- 全硬體接線下的一個班次或 24 小時產品耐久測試。

下一階段應在接線恢復後，先跑完整功能 DVT，再決定壓力時間，最後才跑長時間 Soak。
