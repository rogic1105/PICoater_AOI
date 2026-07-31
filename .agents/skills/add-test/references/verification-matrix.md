# PICoater AOI verification matrix

## Latest offline baseline (2026-07-30)

Tested dirty worktree based on commit `458942f`. Raw reports remain local under
`artifacts/test-reports/`; `latest-campaign.md` is the durable summary.

| Layer | Theory / acceptance | Experimental result | Status | Evidence |
|---|---|---|---:|---|
| Build | `Release|x64`, 0 compiler errors and 0 warnings | Full solution built successfully | **PASS** | `20260730-172525-458942f` |
| Flow checker | All checker self-tests pass | 126 / 126 | **PASS** | on-machine Python unittest at 2026-07-30 17:25 |
| Unit | All unit tests pass | 147 / 147 | **PASS** | `20260730-172525-458942f` |
| Integration | All file/JSON/CSV/mock Bridge tests pass | 115 / 115 | **PASS** | `20260730-172544-458942f` |
| DVT runner | Launch exact app, restore settings, close cleanly, checker exit 0 | 1 / 1 scenario | **PASS** | `20260730-114957-3c30df3` |
| 30,000-record UI DVT | 30,000 IDs; Review/Report navigation and charts; no contract failure | 44 PASS / 0 FAIL; max UI Stall 1000 ms; `transitionDrift=0` | **PASS** | `20260730-021810-9307c6c` |
| Offline stress | All nine high-frequency/mock Bridge cases pass | 9 / 9 | **PASS** | `20260730-021810-9307c6c` |
| Short offline soak | Queue drains, temp files clean up, resource guards pass | 1 / 1 for 6 seconds | **PASS** | `20260730-021810-9307c6c` |
| Two-hour offline stress | Same nine cases under a two-hour budget | 9 / 9 | **PASS** | `20260729-201511-9307c6c` |
| Two-hour offline soak | Mixed IO/CSV/CFG/statistics/copy/cleanup remains bounded | 7200.2 s; 222,434 cycles; Private +277.1 MB; handles -89; threads -2 | **PASS** | `20260729-221921-9307c6c` |

Physical camera/background, five actual capture cycles, and repeatable IO/light software fault
injection are now covered with the connected hardware. Seven-camera full load, physical cable/power
disconnect injection, SMB interruption, real-disk low-space UI transition, and an uninterrupted
eight-hour on-machine soak remain separate.

本表是測試狀態的單一總覽。每一列同時記錄：

- **理論／驗收值**：測試開始前定義的 PASS 標準。
- **最近實測值**：實驗真正量到的數據，不用「看起來正常」代替。
- **證據**：本機原始資料位於 `artifacts/test-reports/<run>/`；正式摘要可由 Git 歷史追溯。

`artifacts/test-reports/` 不進 Git，保存單次測試的 log、TRX、CSV 與 campaign report。
本表與 `latest-campaign.md` 進 Git，保存可長期追溯的結論。

## 已完成或已有基準

| 層級 | 測試項目 | 理論／驗收值 | 最近實測值 | 狀態 | 證據 |
|---|---|---|---|---:|---|
| Build | `Release\|x64` 全方案 | 0 errors、0 warnings | 0 errors、0 warnings | **PASS** | `20260730-172525-458942f` |
| Flow checker | Python checker 自我測試 | 全部通過、0 fail | 126 / 126 | **PASS** | 2026-07-30 17:25 on-machine unittest |
| Unit | .NET 單元測試 | 全部通過、0 fail | 147 / 147 | **PASS** | `20260730-172525-458942f` |
| Integration | JSON、CSV、檔案與 Mock Bridge | 全部通過、0 fail | 115 / 115 | **PASS** | `20260730-172544-458942f` |
| DVT | Runner 自我檢查 | 開啟正確程式、還原設定、正常關閉、checker exit 0 | 1 scenario；7.12 秒；結束後 Runner／Monitor process 均不存在 | **PASS** | `20260730-015108-9307c6c` |
| DVT | Runner 失敗清理 | 情境失敗仍還原設定；先正常關閉最多 60 秒；不得留下 Runner／Monitor 孤兒程序 | 關閉硬體後故意讓 Light 守門逾時；設定全還原；Monitor 正常關閉；process=none | **PASS** | `cleanup-failure-smoke-20260729.txt` |
| Stress | 離線設定／統計／PLC／IO／Storage 壓力 | 9 case 全部通過且不逾時；六項可調工作各持續 20 分鐘 | 9 / 9，7213.59 秒；1000 筆待傳檔案 662 ms 完成恢復與排空 | **PASS** | `20260729-201511-9307c6c` |
| Soak | 離線混合耐久 | IO 狀態、CSV／CFG、統計、遠端複製與清理持續 120 分鐘；queue=0；Private 增量 <=512 MB；Handles <=+50；Threads <=+15 | 7200.2 秒；222,434 cycles；2,785 copied；56 statistics；Private +277.1 MB；Handles -89；Threads -2 | **PASS** | `20260729-221921-9307c6c` |
| Physical IO DVT | ET-7044 待機 5 分鐘 | IO 全程連線且 Idle；正常釋放 | 305.59 秒；Flow 15 PASS / 0 FAIL | **PASS** | `20260728-211306-6ef23b9` |
| Failure injection | IO＋光源軟體斷線／恢復 | IO 端點與 COM17 各隔離三輪；每輪依序 raise→resolve；最後正常關閉且不留下路由、停用裝置或孤兒程序 | 55.84 秒；IO 斷線／恢復／raise／resolve 各 3 次；光源各 3 次；checker 17 PASS / 0 FAIL | **PASS** | `20260730-171929-458942f` |
| Physical Storage DVT | SMB 與 heartbeat 5 分鐘 | 探針可寫、heartbeat 綠燈、正常釋放 | 306.67 秒；Flow 15 PASS / 0 FAIL | **PASS** | `20260728-223305-8a74e41` |
| Failure injection | Storage app 關閉／自動重啟 | 測試工具只終止程式，不代替排程啟動；90 秒內取得新 PID；15 秒 freshness 內恢復 heartbeat；連跑三輪 | 3 / 3；新 PID 與 heartbeat 分別在 8.824、54.410、55.025 秒恢復 | **PASS** | commit `458942f` STAR result；storage restart DVT report |
| Failure injection | SMB 網路中斷與 backlog | 只隔離儲存端點；本機持續保存至少兩輪；恢復後補傳；本機／遠端內容一致；無幽靈 marker；正常關閉 | 95.94 秒；2 輪 capture；pending queue 最大 3 筆／17,218,956 bytes；恢復後 copied=4／17,224,269 bytes；2 個 `.acap` 本機／遠端長度與 SHA-256 全相同；CSV 同尺寸同時間；pending marker=0；checker 32 PASS / 0 FAIL | **PASS** | `20260731-081640-eec7b84` |
| Retention DVT | 實際磁碟低空間＋UI 狀態 | marker 保護的隔離根目錄建立兩個完整日期；只刪最舊日與同日 CSV；保留較新日；空間恢復；`LocalLowSpace` 與 `RetentionCleanup` 皆完成 raise→resolve→ack；還原設定並刪除 fixture | 10.75 秒；門檻 1554 GiB；fixture 450,621,440 bytes；實際釋放 429 MB；oldest=deleted、newer=preserved；OutputHealth 6 events／5 states／0 invalid；checker 16 PASS / 0 FAIL；fixture=0 | **PASS** | `20260731-082930-eec7b84` |
| Physical combined | IO＋Storage 10 分鐘資格測試 | 固定硬體；UI 全回應；狀態全綠；資源守門通過 | 609.66 秒；Private 2440.4→2456.1 MB；Handles 1290→1296；GDI 134→134；USER 284→284；Threads 104→103 | **PASS** | `20260729-052442-9307c6c` |
| Physical camera DVT | 兩台相機、光源、背景與 Grab 短煙測 | 相機 Ready；光源 COM 探測；背景取得／預覽；Grab/Stop；圖片先於 Curve；正常關閉；checker 0 fail | 23.13 秒；背景採樣 3061 ms；CAM1/2 各 24 幀；完整 checker 28 PASS / 0 FAIL | **PASS** | `20260729-103519-9307c6c` |
| Physical camera DVT | 最新背景與 Grab 回歸 | 相機 Ready；背景取得／預覽；Grab/Stop；圖片先於 Curve；設定還原；正常關閉 | 22.49 秒；兩台在線相機；情境與完整 checker 通過 | **PASS** | `20260730-114918-3c30df3` |
| Physical acquisition DVT | IO 停止三循環 | 三次 High 10 秒皆開 gate、首組對齊、主圖先於 Curve；Low 後每台一筆尾幀；封裝與遠端待傳 | 84.42 秒；3/3 輪完成；每輪 `aligned=True`、tail complete、`.acap`、`remoteFiles=2` | **PASS** | `20260730-114606-3c30df3` |
| Physical acquisition DVT | 時間／高度停止 | Low 提早到不得截短；時間從首組起滿 10 秒；高度等所有在線相機共同完成 15,000 列 | 75.80 秒；Time=10.012 秒；Height=15,005 列；兩輪皆完成 Curve、封裝、遠端待傳；checker 32 PASS / 0 FAIL | **PASS** | `20260730-114606-3c30df3` |
| Physical storage output | 實際遠端落檔 | 五輪 `.acap` 必須在儲存電腦存在，大小與本機一致 | `260730-114640`、`114653`、`114708`、`114800`、`114819` 全部存在於 `\\192.168.10.20\Anilox\Captures\2026\202607\20260730`，5/5 大小一致 | **PASS** | `20260730-114606-3c30df3` |
| Physical combined | IO＋Storage 1 小時歷史基準 | IO／Storage 全綠、正常關閉 | 3608.08 秒；舊版資源判準通過 | **PASS（已被新守門取代）** | `20260729-005833-9307c6c` |
| Physical combined | 8 小時耐久校準輪 | 測試期間硬體拓撲固定 | 4 小時內相機由 0→2 台，資源基線改變；UI 全程有回應；正常中止 | **無效基準，不是產品 FAIL** | `20260729-053546-9307c6c` |
| Physical soak | 固定拓撲 2 小時 IO＋Storage＋Light | IO／Storage／Light 全程綠；UI 0 次無回應；Private 持續成長 <=256 MB/h、總增量 <=4 GB；Handles 增量 <=200；GDI／USER <=100；Threads <=25；正常關閉 | 7210.64 秒；IO 13982 / 13982 成功；Private 2751.8→3185.8 MB；Handles 1283→1368；GDI 135→135；USER 282→284；Threads 109→128；checker 17 PASS / 0 FAIL；正常關閉 | **PASS** | `20260729-141939-9307c6c` |
| Physical soak | 8 小時嘗試（外部斷電中止） | 固定硬體拓撲連續 8 小時 | 3133.78 秒前全健康；IO 6017 / 6017 成功。17:14:18 起相機 2→1，接著 Light、IO 同時斷線，符合人工關閉硬體電源；Private 2752.0→2902.8 MB；Handles 1264→1292；GDI 135→136；USER 280→276；Threads 109→105 | **INTERRUPTED，不是產品 FAIL** | `20260729-162221-9307c6c` |
| UI load | 30,000 筆回顧／報表操作 | 不 crash；重新讀取跳最新；快滾 latest-only；回顧強化不重讀 Curve；方向／熱力圖／顯示裁切符合 S3/S5/S6；報表 Y 軸與異常篩選正確；最大 UI Stall < 1 秒 | 30,000 grab、210,000 影像索引；38.26 秒；最大 Stall 625 ms；完整 checker 42 PASS / 0 FAIL；正常關閉 | **PASS** | `20260730-015505-9307c6c` |
| Integration | 低磁碟刪除邊界 | 只刪最舊一整天；同日 `.acap`、CSV、ticks、summary 與 pending 一起處理；空間恢復即停止 | 兩日隔離資料；最舊日 32 MB；刪除 1 日；新日 4 類產出全保留；pending 3→1 | **PASS** | `20260730-012346-9307c6c` |

## 待完成

| 層級 | 測試項目 | 理論／驗收值 | 目前狀態 | 下一步 |
|---|---|---|---:|---|
| Physical soak | 連續 8 小時最終耐久 | 前述守門連續 8 小時成立，中途不重啟且硬體拓撲不變 | **PENDING** | 2 小時資格輪已通過；待硬體可連續供電時，重新跑完整 8 小時。中止輪不可冒充本項 |
| Failure injection | IO／光源實體拔線或斷電恢復 | 每輪 raise→resolve→ack 正確；不產生孤兒 Grab；可恢復 | **PENDING** | 軟體隔離已通過；仍需最終版本各做一次實體線材／電源故障 |
| Camera load | 七台相機滿負載 | 七台首組完整；無持續掉幀；存檔、Curve、畫面與停止邊界一致 | **BLOCKED** | 目前沒有七台相機 |

## 執行順序

1. 相機／光源／背景短 DVT 已全綠；固定目前硬體配置。
2. 2 小時 IO＋Storage＋Light 資格輪已通過。
3. 每輪都獨立輸出理論值、實測值與正常關閉結果；外部斷電或拓撲變更記為 `INTERRUPTED`，不得冒充 PASS 或產品 FAIL。
4. 硬體可連續供電時，另跑一次不中斷的完整 8 小時測試。
5. 耐久通過後再個別做斷線、重啟、SMB 與低磁碟故障注入。
6. 七台相機滿載等硬體齊全後補證據，不以兩台結果代替。
