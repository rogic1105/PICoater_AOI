# 驗證平台口令

本專案將 Unit、Integration、DVT Runner、Flow 探針、Checker、Stress、Soak 與測試報告統稱為
**驗證平台**。不要再用「外掛／內掛」描述測試工具，避免和程式注入或修改記憶體混淆。

## 簡短口令

| 使用者口令 | 固定範圍 |
|---|---|
| `進行功能測試` | Release x64 Build、Unit、Integration、受影響功能的 DVT／Flow checker，以及必要的短版實機 smoke。 |
| `進行完整驗證` | 依序執行功能測試、壓力測試、耐久測試與故障恢復測試，最後產生總報告。 |
| `進行壓力測試` | 大量資料、快速操作、重複連線／Grab、佇列與競爭條件；驗證負載邊界，不代表長時間穩定。 |
| `進行耐久測試` | 指定時數的 soak；檢查 crash、RAM、Handle、VRAM、Queue、存檔、遠傳和重連趨勢。 |
| `檢查全天 Log` | 使用 `verify-flows` 與 `check_all_flows.py` 檢查指定 session 或整日 trace。 |

## 完整驗證的執行規則

1. 順序固定為 `Build -> Unit -> Integration -> DVT/Smoke -> Stress -> Soak -> Failure recovery`。
2. 先完成短測且通過，才進入耗時或具破壞性的階段。
3. 需要相機、IO、光源、儲存電腦或人工斷線時，先跑不需要接線的部分；缺少的項目列為
   `NOT COVERED`，不得記為 `PASS`。
4. 低磁碟、刪檔、斷網等具副作用測試必須明確列出目標路徑與恢復方式。
5. 報告至少記錄 commit、設定、硬體配置、理論判準、實測值、PASS／FAIL／NOT COVERED 與證據路徑。

## 元件正式名稱

| 元件 | 正式名稱 |
|---|---|
| `AniloxRoll.DvtRunner.exe` | 外部黑箱／端對端測試工具 |
| 主程式 `[Flow]`、座標與耗時紀錄 | Flow 測試探針／可觀測性儀器 |
| 可控制設定、Mock Bridge、故障注入點 | Test Hook／Test Seam |
| 純公式或狀態轉換測試 | Unit Test |
| JSON、CSV、bin、CFG 與 Mock Bridge 測試 | Integration Test |
| 長迴圈、負載及耐久測試 | Stress／Soak Test |

「進行功能測試」不包含數小時的 Stress／Soak；需要整套時必須使用「進行完整驗證」。
