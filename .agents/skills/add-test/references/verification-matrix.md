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

Physical camera/background, actual capture cycles, IO/light software and physical power fault
injection, SMB interruption/backlog recovery, and real-disk low-space retention are now covered
with the connected hardware. Seven-camera full load and an uninterrupted eight-hour on-machine
soak remain separate.

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
| Failure injection | IO＋光源軟體斷線／恢復 | IO 端點與 COM17 各隔離三輪；每輪依序 raise→resolve；最後正常關閉且不留下路由、停用裝置或孤兒程序 | 55.59 秒；IO 斷線／恢復／raise／resolve 各 3 次，恢復 3.528／4.002／4.009 秒；光源斷線／恢復／raise／resolve 各 3 次，恢復 6.501／6.499／6.511 秒；checker 17 PASS / 0 FAIL；IO TCP、COM17 與程序收尾皆正常 | **PASS** | `20260731-164731-51235fb` |
| Failure injection | IO＋光源實體斷電／恢復 | IO 與光源各斷電／上電三輪；每輪必須各自完成 disconnect→raise→reconnect→resolve；UI stall <1 秒；最後正常關閉 | IO 三輪恢復 9.494／6.494／6.492 秒；光源三輪恢復 14.110／16.025／12.111 秒；兩次斷電探測 UI stall 421／438 ms；OutputHealth invalid=0、hardware duplicate=0；checker 17 PASS / 0 FAIL；正常關閉 | **PASS** | `trace-20260731_165130` on `51235fb` |
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
| Physical soak | 固定拓撲 2 小時 IO＋Storage＋Light | IO／Storage／Light 全程綠；UI 0 次無回應；Private 持續成長 <=256 MB/h、總增量 <=4 GB；Handles 增量 <=200；GDI／USER <=100；Threads <=25；正常關閉 | 7209.9 秒；120 / 120 健康快照；IO 14,041 / 14,041；Private 2751.6→3213.1 MB，總體 241.1 MB/h、中位 154.5 MB/h、擴張後 19.9 MB/h；Handles 1256→1365；GDI 134→135；USER 281→283；Threads 106→129；UI 無回應 0 次；待機相位三次越過 5 ms 均自動重同步至 0.192～0.320 ms；checker 17 PASS / 0 FAIL；正常關閉 | **PASS** | `20260731-083512-8f184b4` |
| Physical capture soak | 5 分鐘反覆 Grab 資格輪 | High 10 秒／Low 4 秒；21 輪 request、gate、首組、Curve、close、finalize 全完整；遠端實際落檔；UI 與資源守門通過 | 352.89 秒；21 / 21 輪；六個聚合守門全通過；Curve 206 筆；checker 31 PASS / 0 FAIL；Private 2,760～34,383 MB，結束前回落至 7,391 MB；UI 無回應 0 次；遠端 21 個 `.acap` | **PASS（資格輪）** | `20260731-110010-96035a3` |
| Physical capture soak | 2 小時產品流程輪（Runner 收尾缺陷） | High 10 秒／Low 4 秒共 514 輪；取相、Curve、封裝與遠端落檔完整；Runner 正常產生最終報告 | 514 / 514 request、gate open、aligned first set、gate close、finalize；Curve 4,936 筆；checker 31 PASS / 0 FAIL；主程式正常關閉；Runner 將約 40,000 行舊 evidence 重播到 UI，外層 8,105 秒 safety timeout | **產品流程 PASS／Runner FAIL（已修復並於下列正式輪重驗）** | `20260731-110727-96035a3` |
| Physical capture soak | 2 小時反覆 Grab 正式輪 | High 10 秒／Low 4 秒共 514 輪；六條取相／存檔鏈完整；遠端持續落檔；UI 與資源趨勢有界；Runner 自行完成報告 | 7,248.6 秒；514 / 514 輪；六個聚合守門 6 / 6；checker 31 PASS / 0 FAIL；239 資源樣本；UI 無回應 0；Private 最大 42,366.9 MB，循環低谷後半比前半下降 801.7 MB（-834.3 MB/hour）；Handles +148（77/hour）；GDI +1；USER +3；Threads +1；正常關閉 | **PASS** | `20260731-133218-96035a3` |
| Physical soak | 8 小時嘗試（外部斷電中止） | 固定硬體拓撲連續 8 小時 | 3133.78 秒前全健康；IO 6017 / 6017 成功。17:14:18 起相機 2→1，接著 Light、IO 同時斷線，符合人工關閉硬體電源；Private 2752.0→2902.8 MB；Handles 1264→1292；GDI 135→136；USER 280→276；Threads 109→105 | **INTERRUPTED，不是產品 FAIL** | `20260729-162221-9307c6c` |
| UI load | 30,000 筆回顧／報表操作 | 不 crash；重新讀取跳最新；快滾 latest-only；回顧強化不重讀 Curve；方向／熱力圖／顯示裁切符合 S3/S5/S6；報表 Y 軸與異常篩選正確；最大 UI Stall < 1 秒 | 30,000 grab、210,000 影像索引；38.26 秒；最大 Stall 625 ms；完整 checker 42 PASS / 0 FAIL；正常關閉 | **PASS** | `20260730-015505-9307c6c` |
| Integration | 低磁碟刪除邊界 | 只刪最舊一整天；同日 `.acap`、CSV、ticks、summary 與 pending 一起處理；空間恢復即停止 | 兩日隔離資料；最舊日 32 MB；刪除 1 日；新日 4 類產出全保留；pending 3→1 | **PASS** | `20260730-012346-9307c6c` |

## 待完成

| 層級 | 測試項目 | 理論／驗收值 | 目前狀態 | 下一步 |
|---|---|---|---:|---|
| Physical soak | 連續 8 小時最終耐久 | 前述守門連續 8 小時成立，中途不重啟且硬體拓撲不變 | **PENDING** | 2 小時資格輪已通過；待硬體可連續供電時，重新跑完整 8 小時。中止輪不可冒充本項 |
| Physical capture soak | 連續 2 小時反覆 Grab | High 10 秒／Low 4 秒共 514 輪；六條取相／存檔鏈逐輪完整；遠端持續落檔；資源趨勢有界；Runner 自行完成報告 | **PASS** | 正式重跑 514 輪、checker 31/0、資源守門與正常關閉全通過；前一輪的 Runner UI evidence 重播缺陷已修復 |
| Storage retention | 儲存電腦本機 `D:\Anilox\Captures` 低磁碟 | 低於門檻時每次只刪最舊完整一天及同日 CSV；空間達標即停；較新日保留；heartbeat 記錄釋放量 | **PASS** | 連續兩輪 2 小時遠傳讓空間兩次跨門檻；依序刪 `20260103`、`20260104`，兩日各 8.398 GiB 且同日 CSV 同步消失；`20260105` 後全保留；最新 heartbeat freed=9,017,128,089 bytes、free=28.087/99.999 GiB，15 秒後未再刪。另保留本機一鍵工具供日後可控重驗 |
| Failure injection | IO／光源實體拔線或斷電恢復 | 每輪 raise→resolve 正確；不產生孤兒 Grab；可恢復；正常關閉 | **PASS** | IO 與光源同時斷電／上電三輪；各自完成 disconnect→raise→reconnect→resolve；checker 17 PASS / 0 FAIL；實測值見上表 |
| Camera load | 七台相機滿負載 | 七台首組完整；無持續掉幀；存檔、Curve、畫面與停止邊界一致 | **BLOCKED** | 目前沒有七台相機 |

## 執行順序

1. 相機／光源／背景短 DVT 已全綠；固定目前硬體配置。
2. 2 小時 IO＋Storage＋Light 資格輪已通過。
3. 每輪都獨立輸出理論值、實測值與正常關閉結果；外部斷電或拓撲變更記為 `INTERRUPTED`，不得冒充 PASS 或產品 FAIL。
4. 硬體可連續供電時，另跑一次不中斷的完整 8 小時測試。
5. 耐久通過後再個別做斷線、重啟、SMB 與低磁碟故障注入。
6. 七台相機滿載等硬體齊全後補證據，不以兩台結果代替。
