# verify-flows — UI 行為契約與驗證（DVT）

> DVT＝設計驗證測試（Design Verification Test）：契約＝設計規格、跑一輪 log 對數＝測試。
> 與硬體階段性 DVT 不同——本檔是**每次改接線都重跑的持續迴歸契約**，非一次性階段 gate。

app ＝【監控/回顧/報表】三個 tab。每條 flow 有兩面，**驗證時兩面都要對**：
- **log-flow**（執行期腳印）：`[Flow]` 行序列＝行為判準——治非同步接線病（事件/訂閱/時序）。
- **code-flow**（靜態地圖）＝兩種工件，各有完備性要求：
  ① **責任鏈（hop chain）**：每條 flow 一份＝「Ctrl+點擊追蹤」的完整跳點序列（`函式名@檔名`，
     一跳一行、幾十跳記完整；**不記行號**——行號腐化最快，函式名可 grep 重定位）。
     「少量穩定載重」只決定**哪些跳加注解**（⚠地雷/不變量/單一決策點/轉換點#），不拿來刪跳。
  ② **值鏈盤點表**：每個「值維度」（如垂直方向值）一張＝該值**每一個**產生/轉換/消費點，
     必須完備＋附 grep pattern，且列到**實作點**層級（概念點 6 個可對應實作點 15 個——狗糧實測）。
     **防包層的偵測機制＝grep 對清單**：命中 − 已登記實作點清單 ＝ 應為空集合；多出來的
     ＝新包的層，當場現形（稀疏節點防不了包層，完備性才防得了；「對數字」會因佈線誤報）。
  F1 為責任鏈範本、`$row-chart-coordinates` 6 點表為值鏈範本；其餘契約逐步補。
改任何「使用者動作 → 接線 → 顯示/資料連動」流程後**必跑本 skill**：
先「模擬測試」（順 code 推演對契約）再（必要時）真機比對 log。
顯示接線核心檔（改到必跑）：LiveCameraManager / LiveDisplayCoordinator / ImageDisplayView /
WaterfallView / AniloxRollForm.Live|Background|Review。

## 三層驗證（同一份契約，責任不同——契約永遠只寫一份）

1. **code-flow 稽核（不開 UI）**：順著 code 把受影響 flow 的責任鏈追一遍，對照本契約；
   漏訂閱、漏 teardown、狀態 owner 分岔在這一步處理。
2. **自動測試**：NUnit 驗純邏輯、IO boundary 與 coordinator 機制；`tools/python/tests/` 驗
   `check_all_flows.py` 的判讀規則。它們保證零件與裁判正確，**不假裝已操作真實 WinForms／硬體**。
3. **log-smoke（真機比對）**：使用者操作一輪後，執行 `python tools/python/check_all_flows.py --latest`。
   `FAIL` 是已操作但違反契約；`NOT COVERED` 是這輪沒有操作到。每次 smoke 結束必直接回報
   `NOT COVERED` 清單，讓使用者決定補測哪些，不靠人工一邊盯畫面一邊猜流程。

log 格式為 `[Flow] HH:mm:ss.fff T{執行緒} 訊息`，唯一出口是 `Services/FlowTrace.cs`。
未建立「自動駕駛整個 WinForms UI」的 headless harness；除非未來實機 smoke 成本成為瓶頸，否則不以它取代
真實 MIL、GDI、硬體與視覺驗收。

## Log 記錄範圍（PropertyGrid 可直接找到）

設定位置固定為 `5. Log 設定（記錄／除錯） > 記錄範圍`，每個 session 啟動或切換模式時記
`log mode=Operational|FlowVerification|FullDiagnostic`：

| UI 選項 | 適用時機 | 額外證據 |
|---|---|---|
| 日常運行 | 產線常駐／預設 | 操作、連線、錯誤、存檔、開始／停止、異常觸發的 UiStall/UiPing/UiStack/UiSlow/UiPaint |
| 流程驗證 | smoke/DVT 驗收 | 再加 rowChart、IC/WF state、viewEdges、prefit、mainRange、chartRange |
| 完整診斷 | 短時間深度除錯 | 再加 IC/WF stats 與 `Logs\fsm\` 原始 UI action JSONL；檔案較大 |

`check_all_flows.py` 必讀 session 的 mode；日常模式缺少 DVT-only 證據時回 `NOT COVERED`，不得誤判
`FAIL`。需要完整驗收時先切到「流程驗證」再操作；測完可切回日常運行。舊 trace 無 mode 行，視為
舊版全探針開啟。

## 檔案組織規則

- **按領域拆、永不按方法拆**（按方法拆＝同一條契約抄三份，違反唯一來源）。
- 現在一個檔就好；哪個 tab 的 F 系列長大到佔半個檔 → 按 tab 拆
  （`flows-live.md`/`flows-review.md`/`flows-data.md`），本檔留方法論當入口。
- **本檔＝前端（UI 接線）專用**：flow 契約治「事件/訂閱/時序」這種非同步病。
  後端 native（同步函式，驗輸入輸出）歸 `tests/` NUnit + benchmark，**不用 flow 契約**。

## 契約寫什麼：標準預期 flowtrace（log 級），不寫函式流程

契約驗「行為」不驗「實作」：函式呼叫鏈重構就變（契約跟著重寫＝白費）；log 級契約
（使用者做 X → 觀察到哪些 `[Flow]` 行）重構後依然成立，只有行為真的變了才改。
函式流程屬於 `AGENTS.md`／程式註解，不進契約。每條契約用三種型態組合：

| 型態 | 寫法 | 抓什麼 |
|---|---|---|
| ①正序 | 做 X 必依序出現 A→B→C | 缺步驟、順序錯 |
| ②禁止 | 階段 P 不得出現 Y 行 | **多出來的步驟**（穩態 churn、縮放中被 fit） |
| ③完整性 | 每台在線相機首幀必恰一行 | 靜默失敗（訂閱斷、幀沒到） |

**穩態靜默通則（②的總綱）**：穩態（無使用者互動、無相機增減、無設定變更）下，
**不得出現任何顯示狀態變更行**（clearFrame / lodRebind / autoFit / Teardown / create / SwitchMainDisplay）。
出現＝有東西在空轉重設（例：2026-07-07 離線台被狀態 timer 每 500ms 重清 → lodRebind → fit reset
→ 縮放被拉回；靠本通則+`clearFrame`/`lodRebind` 儀器一輪 log 定罪）。

**孤兒判讀規則（多 flow 並行時的歸屬）**：每一行顯示變更，往上找最近的 `ui:...` intent 行——
找得到＝合法（歸該動作的契約管）；找不到＝孤兒＝違規（系統自己在動）。兩條 flow 同時跑
（如 grab 中按 Review【讀取資料】→ 出現 DisableGlobalMerge）＝各自歸各自的 intent 驗，交錯本身不驗。
**intent 行清單**（使用者動作入口各記一行 `ui:...`）：tab 切換、【開始抓取】【取得背景】【預覽背景】
【讀取資料】(Review/Data)【單片序號】【時段導航】【暫停Mura檢測】【IO暫停】
【明細列表】【報表序號】【序號範圍】【期間-年/月/日/全局】【良率導航】【篩選異常】+
S0 通用（所有 PropertyGrid 設定自動記 `ui:設定[名]=值`）。新增會動到顯示的入口 → intent 行一併加。

## 本檔的效力位階（先讀這個再用本檔）

1. **契約＝「當下定案的行為描述」，不是真理**。log 與契約不符時有兩種可能：
   改壞了（修 code）、或行為該進化（修契約）——**判定權在使用者/設計意圖，不在文件**。
2. **架構演化的正確姿勢**：先「有意識地改契約」（commit 訊息聲明哪些條款作廢＋為何），
   再改 code。契約擋的是「無意識偏離」，不是擋演化——被契約否決的架構改動，
   正確反應是質疑契約，不是放棄改動。
3. **分級**：🔒使用者定版鐵則（列圖表排版嚴禁動／拖曳連動不可節流／IsReversed 禁用——
   只有使用者能解鎖）vs 一般契約（agent 可提案修訂，說明理由）。
4. **文件會過時**：與 code 衝突時，code 是「事實」、文件是「意圖」——先 git 考古判斷
   哪邊該改，勿直接信任任一邊（蓋章/驗證紀錄跟語意版本綁定，版本變了要重驗）。

## 兩類病、兩類工具（2026-07-08 座標戰役 30+ 輪的教義修正）

| 病類 | 特徵 | 工具 |
|---|---|---|
| **非同步接線病**（事件/訂閱/時序） | 缺步驟/多步驟/靜默失敗 | log 契約（本檔 F/R/D/S/M/P/H 系列）——函式鏈重構就變，故不記呼叫鏈 |
| **同步語意病**（座標/方向/單位/縮放） | **log 能定罪（數字對不上）卻不能定位（哪層轉錯）**；每個 handler 各自「看起來對」 | **轉換點盤點表**（函式級）＋正向拆解先行——加再多儀器也只是繞圈 |

**判別訊號**：儀器加了好幾輪、每輪都「排除一個嫌疑」但症狀不動 → 大概率是語意鏈病 →
**停止加儀器，改做正向拆解**。

**正向拆解 SOP（動任何 方向/座標/單位 鏈之前必做）**：
1. `grep -rn "Flip|Reverse|Invert|IsReversed|total.*-|n - 1 -|ToLogical"` 相關鏈全檔
2. 列「轉換點盤點表」：位置｜何時作用｜性質（物理必需/語意必需/數學必需/**抵銷層**）
3. 奇偶配對分析（每條路徑翻轉總次數）→ 才准動刀
4. 盤點表放對應領域 skill（例：`$row-chart-coordinates` 的 6 點表）並隨改動同步

**明文禁止「就地包一層」**：看到方向/正負/顛倒症狀，在出錯處新包一層轉換＝本次事故成因
（每層局部合理、疊加後奇偶失控、且下一個 agent 看不見）。新增任何轉換層的前提：
①盤點表已更新 ②證明不是抵銷層（抵銷層＝去改參數化根源，不准包） ③符合單一決策點原則。
**稽核機制（包層必被抓）**：值鏈盤點表附 grep pattern＋已登記實作點清單，commit 前跑 grep——
命中 − 清單 ＝ 應為空集合，多出者退回。（清單完備性是前提；「旗標佈線」不入清單、新旗標名才登記。）

**函式級記錄的界線（修正版）**：一般呼叫鏈不記（重構即腐化）；**轉換點/單一決策點/不變量**
（少量、穩定、載重）必須記——它們是「值在哪裡被改變」的地圖，語意病的 audit 靠它不靠 log。

## 偏序驗證規則（多執行緒鐵則）

- **同一執行緒（同 T）內驗全序**：順序必須完全符合契約。T1=UI 執行緒（按鈕/設定/接線），Tn=各相機 MIL 回呼。
- **跨執行緒只驗「因果 + 完整性」**：如 `StartGrab 必早於所有 firstFrame`、`每台在線相機首幀必出現`。
  **不驗**非決定性交錯（cam1/cam2 誰先本來就不定，驗了必誤報）。

## 不變量（任何 flow 都不得違反；見 app 巢狀 `AGENTS.md` 顯示鐵則）

- app 內零 MIL 原生顯示視窗/滑鼠 hook（headless；MilCamera panelHandle=Zero）。
- 主畫面永遠合圖（即時=ImageDisplayView、瀑布=WaterfallView）；縮圖兩模式一律即時 ThumbStrip（橘框選中）。
- `SwitchMainDisplay` 的 `center=True` **只**允許出現在明確點縮圖/狀態字之後（拖曳/程式化路徑=False，否則回彈）。
- **UI 執行緒零耗時 MIL/序列埠同步呼叫**（2026-07-07 [UiStack] 全清單）：MdigInquire/MdigControl/
  MdigProcess/CLProtocol feature 讀寫/SerialPort.Write 一律背景或限於已證明零阻塞的旗標切換。
  `StopGrab` 只關產品 gate、不碰 MdigProcess；實體 Pause／Release 在背景執行。已修：
  CamStatusTick、SyncCameraParamsFromHardware、StopGrab、LightTurnOn/Off。
  開機配置只允許 acquisition 階段的 MIL System／Digitizer／grab-buffer 建立留在 UI 執行緒；
  AOI pipeline、managed buffer、CUDA pinned slab 必須由單一背景 worker 依相機順序建立。
  新增 native 呼叫點一律先確認執行緒與釋放競態。
- **拖曳/hover 重繪限流 ~120fps**（ImageCanvas）：高輪詢滑鼠每 move Invalidate＝paint 風暴
  （386/s）餓死全體 WM_TIMER。pan 值照每 move 累積、只限「畫」；MouseUp 尾緣補繪。

## 效能卡頓儀器（U 系列——常駐，判讀決策樹）

儀器（`Services/FlowTrace.cs` + ImageCanvas）：
```
[UiStall] {gap}ms（GC0+a GC1+b GC2+c） ← 33ms UI timer 遲到 ≥100ms（含 GC 世代增量）
[UiPing] {rtt}ms                        ← 背景 BeginInvoke 往返 ≥100ms
[UiStack] {top frames}                  ← ping 200ms 無回應當下的 UI 執行緒堆疊（直接點名）
[UiSlow] {name} {ms}ms                  ← 7 個 handler 計時 >50ms
[UiPaint] {control} {ms}ms              ← chart WM_PAINT >50ms；IC/RV paint …=canvas OnPaint
IC|WF stats paints=N/s paintMs=M statusEv=K/s ← 完整診斷；canvas 每秒重繪組成（>5 次/秒才記）
```
耐久資源量測由外部 `tests/TestRunner.ps1` 每 30 秒取樣，不在產品程序內列舉
`Process.Threads` 或建立診斷同步物件。量測者與被測程式分離，避免觀測動作本身造成
handle 成長。外部取樣包含 Private Bytes、Working Set、handles、GDI、USER、
threads、CPU 與 UI responsiveness；一次跳升後持平不算洩漏，持續成長才依耐久門檻失敗。
`UiStallDetector` 在 Form ctor 建立，但必須等 `Shown` 的全 tab 預熱完成後才
`BeginInteractiveMeasurement`；建構期使用者尚不能互動，不得把 ctor／預熱時間算成第一筆 stall。
**判讀決策樹（2026-07-07 十輪教訓的結晶）**：
1. UiStall 有 GC 增量 → GC/LOH 問題；全零 → 往下。
2. UiStall 大 + UiPing 也大 → **阻塞型**（單件慢）→ 看 UiStack 點名。⚠ 按時間窗切開判讀
   （開機時段的大 ping 會污染整體結論）。
3. UiStall 大 + UiPing 靜默 → **飽和型**（件多不慢）→ 看 IC stats：paints > ~150/s＝paint 風暴回歸
   （限流後正常 ≤ ~130/s）。**飽和型用計數器抓、阻塞型用計時器抓——只裝計時器抓不到飽和**。
4. UiStack 點到的都是真 bug 但不一定是你要的 bug——修掉後重測，別急收工。
自動 checker 的 `U.stall` 依同一原則成對判讀：`UiStall >1000ms` 只有在該 gap 時間窗內同時出現
`UiStack`，或 `UiPing >= max(200ms, min(1000ms, gap/2))` 才判 **FAIL／真阻塞**；沒有佐證的
大 `UiStall` 記為「計時器飢餓」並保留次數，但不得單獨判 FAIL。這避免高速滾輪持續產生 input/paint 時，
低優先 `WM_TIMER` 晚執行卻被誤報為 UI 執行緒停止；GC 增量只作歸因線索，也不能單獨定罪。
契約：拖曳中 `IC stats paints` 不得 >150/s（風暴回歸紅旗）；`[UiSlow] CamStatusTick/TelemetryTick`
出現＝MIL 查詢又回到 UI 執行緒（背景化被回退）。
- view 訂閱 `cam.OnDisplayFrame` 且 Enable* 冪等 → **相機批次換新（Allocate/Free）前後必有對稱 teardown**。

## 狀態快照儀器（方向/座標「機器可判」——2026-07-09 故障注入盲測 4 例定版）

**啟用條件**：記錄範圍＝「流程驗證」或「完整診斷」。

**目的**：改 A 壞 B 當下從 log 抓到，不靠肉眼盯畫面。每幀狀態自動記錄（免滑鼠——「要滑鼠移動才更新」
曾是 log 誤判源）、每秒節流。**快照行＝儀器輸出、非狀態變更行——穩態靜默通則對其豁免。**

**行清單（產地）**：
```
LC|RV row rowChart dir=D n=N total=Tmm view a~b dataPhys c~dmm dataChart e~f
      ← RowCurveDisplayAdapter.FlowApply（chart 更新後記）；dataPhys=映射前資料非零物理值域、
        dataChart=helper「實際畫上 chart」的值域（LastDataOccLo/Hi，量實際非意圖——
        adapter 自算預期值＝假綠，第二輪盲測抓到的量測學錯誤）
WF state 占用=0~w/H 最新內容畫面端={頂|底}   ← WaterfallView.FlowState（band 寫入後）
IC state viewX a~b viewY c~d                 ← ImageDisplayView.FlowViewState（RefreshMain 上畫後）
IC|WF viewEdges X …｜Y …                     ← 拖曳放開時畫面四邊（滑鼠驅動，與 IC state 同源不同路）
```

**方向判讀基準（關係跑掉＝哪層壞，直接定罪）**：
| 量 | 由上而下（TopToBottom） | 由下而上（BottomToTop） |
|---|---|---|
| dataPhys↔dataChart | **鏡射**（dataChart=total−dataPhys） | **直通**（同值） |
| WF state 畫面端 | 頂 | 底 |
| viewY / view（chart 視窗） | 上小下大 | 上大下小 |

**判讀規則**：
- **雙快照對數**：`viewEdges` vs `IC state` 同秒同源——矛盾＝兩條換算路分岔（B 類故障一行定罪）。
- **WF 自我矛盾**：`畫面端` label 與方向設定不符＝翻轉接線反（C 類故障）。
- **映射層試金石＝瀑布漸進填充**：即時模式每幀滿幅→鏡射=自身、值域不可判（已知限制）；
  瀑布 dataPhys 逐秒增長，dataChart 關係一眼可判（D 類故障）。
- **盲測法**（使用者提議、4 例驗證）：故意反接一處鏡像 → 跑 2×2 流程 → 只讀 log 定罪到層。
  改方向/座標/翻轉鏈後的迴歸驗證＝跑一輪對上表，不用肉眼。

## Flow 契約

### F1 開機配置（AutoAllocateCameras）

**初始化狀態與事件表（相機配置只有這一條路）**

| State | Event | Next State | Action |
|---|---|---|---|
| Unallocated | EnsureAllocated | AllocatingAcquisition | UI 執行緒依序建立 MIL System／Digitizer／grab buffers |
| AllocatingAcquisition | MIL 全部完成 | AllocatingProcessing | 單一背景 worker 依序建立 AOI pipeline、managed buffers、pinned slabs |
| AllocatingProcessing | processing 全部完成 | Parameterizing | 回 UI 執行緒啟動 CLProtocol、建立顯示 view、發布實際在線數 |
| Parameterizing | 每台 CLProtocol 工作與曝光／線掃寫入實際返回 | Warming | 發布 `acquisition parameters ready`；各在線相機才可啟動 hot standby。10 秒 timeout 只告警，不解除 gate |
| Warming | 各在線相機觀測到第一個 raw frame | ReadyIdle | 發布 `OnHwReady`、解鎖 Grab；產品收幀 gate 仍關閉 |
| ReadyIdle | Start intent | Synchronizing | 保持產品 gate 關閉；平行 drain 全部在線相機 |
| Synchronizing | 全部 drain 完成 | Warming | 在相機停止時重套各台現行 Line Rate，再 back-to-back resume |
| Warming | 各台新 raw frame 到齊且同板首幀 spread ≤ 5ms | Armed | 清顯示世代、設使用者意圖、發布 `StartGrab`；產品 gate 仍關閉 |
| Warming | 相位超限且 attempt < 3 | Synchronizing | 重新 drain／timing-reset／resume，不用固定等待 |
| Warming | 第 3 次仍超限或 warm timeout | ReadyIdle | 保持 gate 關閉，不建立 capture plan |
| Armed | capture plan 與抓取上限已就緒 | Capturing | 一次開啟全域收幀 gate |
| Capturing | StopGrab | ReadyIdle | 先關全域收幀 gate，再清使用者意圖；MIL 保持 hot standby |
| ReadyIdle | 參數重配置 | ReadyIdle | 背景執行參數寫入；產品 gate 原本即關閉 |
| Capturing | 單台／All 曝光調整 | Capturing | 背景寫曝光；gate 保持 open，不 stop/start、不重設顯示世代 |
| Capturing | 線掃速度／擷取高度修改 | Capturing | UI 停用且 Form／Manager 拒絕，不寫設定或硬體、不關 gate |
| AllocatingAcquisition／AllocatingProcessing | 任一步失敗 | Unallocated | 釋放本輪已建立資源、保留 Grab gate、回報錯誤 |
| 任意配置中狀態 | Release | Releasing → Unallocated | 等目前 native call 返回後釋放；晚到結果不得發布 Ready |
| ReadyIdle／Capturing | EnsureAllocated | 原狀態 | 冪等，不重複配置 |

禁止：多台相機平行呼叫 `TanukiCv_AllocPinned`；MIL 與 processing 不得各自建立第二條配置入口。

**產品收幀 gate 子狀態（每次 OpenCaptureGate 重新建立）**

| State | Event | Next State | Action |
|---|---|---|---|
| Closed | Arm/Open | AwaitingHeadProbe | 每台第一個跨邊界 callback 只收時間戳並丟棄，不進產品流程 |
| AwaitingHeadProbe | 全台 probe 到齊且同板 spread ≤ 5ms | AwaitingFirstSet | 發布 `capture head guard ... aligned=True`，下一組才可進產品流程 |
| AwaitingHeadProbe | probe 相位超限 | Rejected | 立即關產品 gate、完成 waiter=False、停止本輪，不得出現 firstFrame／CSV／存檔 |
| AwaitingFirstSet | 全台首組產品幀到齊且相位通過 | Active | 完成 waiter=True；Time 從此刻起算 |
| AwaitingFirstSet | 首組相位超限 | Rejected | 關產品 gate、停止本輪並標記 phase invalid |
| 任意 | Stop/rearm | Closed／新 AwaitingHeadProbe | 舊 waiter=False，舊 callback 不得進入新一輪 |

**log-flow（執行期腳印＝判準）**
```
T1: AllocateCameras begin（expect N）
T1: camera init cam=N phase=acquisition ms=X size=WxH thread=U       × 配置成功台數；U 為 UI thread
T1: camera init phase=acquisition done cams=M ms=X
T1: camera init phase=processing begin cams=M
Tbg: camera init cam=N phase=processing ms=X pinnedMB=P allocCalls=2 thread=K
                                                                    × M；K 必須固定且不得為 T1
T1: camera init phase=processing done cams=M ms=X
T1: （前次 view 存在才有）TeardownImageDisplay / TeardownWaterfall
T1: ApplyMainDisplayMode → {ImageCanvas|Waterfall}
T1: {EnsureImageDisplay|EnableWaterfall} create + subscribe M cams
T1: SwitchMainDisplay cam=1 center=False
T1: AllocateCameras done（配置 M、在線 P/N）   ← P=CheckPresence 實際在線（配置≠在線：quad 卡空通道
                                                  也配得起來；報配置數＝幽靈相機數，2026-07-07 修正）
T1: camera init summary cams=M totalMs=X acquisitionMs=A processingMs=B
Tbg: acquisition parameters ready camN cl={True|False} lineRate=R × 在線相機；必早於同台 standby start
Tbg: acquisition standby start camN                          × 在線相機（只在本次實體 M_START 時出現）
Tbg: acquisition standby ready camN tick=T                  × 在線相機（raw callback 實測，不用固定等待）
T1: （全部在線相機 warm 後）EnableGlobalMerge（slots=7）
```

效能判準：完整配置期間單筆 UI stall 不得超過 1000ms；processing 期間 UI heartbeat 不得出現
由 `TanukiCv_AllocPinned` 引起的 `[UiStall]`／`[UiStack]`。
每台 processing 必須 `allocCalls=2`（AOI pool slab 1 次＋存檔縮圖 slab 1 次），同一台不得退回 11 次小配置。

**code-flow（靜態地圖＝責任鏈＋載重點；audit 時兩者都要對）**
```
AutoAllocateCameras(Form)                    顯示基線 set:[顯示基線] 一行
 └ await LiveCameraManager.AllocateCamerasAsync
    ├ SemaphoreSlim allocation gate（Ensure／Release 共用；配置只有一條入口）
    ├ CameraSystemManager.Initialize / per-cfg AllocateSystem（板=SystemNum 共用）
    ├ T1 per-cam AniloxCamera.InitializeAcquisition
    │   └ MilCamera.Initialize（System／Digitizer／grab buffers；保持既有 MIL 配置順序）
    ├ Task.Run（單一 worker、per-cam 依序）→ AniloxCamera.InitializeProcessingResources
    │   ├ managed input/output buffers＋AoiService.Initialize
    │   ├ NativeBufferPool：8 區共用 1 個 64-byte aligned pinned slab
    │   └ resize raw/proc_c/proc_r：3 區共用 1 個 64-byte aligned pinned slab
    ├ per-cam CheckPresence → BeginCLProtocolInit（只對在線台；空通道 enable 會卡 MIL 鎖）
    ├ TeardownImageDisplay/Waterfall → ApplyMainDisplayMode   ← 先拆後建＝訂閱綁「這批」相機
    │    └ EnsureImageDisplay：FlipVertical=方向、VerticalZeroAtBottom=方向（座標約定，轉換點#1/#3）
    ├ SwitchMainDisplay(Selected)            center=False（程式化不置中）
    └ 發布「在線數」（非配置數）→ OnCameraCountChanged
（背景）CLProtocol 工作全數實際返回 → CameraStatusTimer_Tick
 ├ per-cam `acquisition parameters ready`（含 CL 啟用結果與套用線掃率）
 ├ per-cam EnableHotStandby@AniloxCamera.cs → EnableHotStandby@MilCamera.cs → MdigProcess(M_START)
 ├ ProcessingFunction raw callback → HasObservedFrameSinceAcquisitionStart=true
 └ 全部在線相機 warm → OnHwReady → 解鎖 grab 鈕 + EnableGlobalMerge（佈局=MergeLayout 唯一來源）
```
單一決策點：顯示狀態=f(he_MainDisplay, 背景預覽靜音鍵)——ApplyMainDisplayMode 唯一計算點（F8）；方向=ShouldFlipVertical。
設定契約：新生成設定或 JSON 缺少 `MainDisplay` 時預設 `Waterfall`；既有 JSON 的明確值優先，不做遷移覆寫。
瀑布空畫面契約：`FeedWaterfallLayout → SetLayout` 必須在首幀前用 PropertyGrid 的
`OPS + START + WaterfallTotalHeight` 建立 7 槽黑底 LOD，Y 軸使用速度／線掃率算出的 row pitch，
並立即發布 `ViewRangeMmChanged`、四邊座標、倍率與游標 mm。`Reset` 只清像素及寫入狀態，
不得關閉 LOD、改變 zoom/pan 或清除座標；實際幀寬等於配置寬時，第一個 band 只填內容，不得再次 fit。
不變量：view 建立前必 teardown（防空訂閱家族）；MdispSelectWindow 必帶 panelHandle 守門。

### F2 開始抓取（btnLiveGrab，已配置）

#### MIL/IO boundary state table (experiment: edge coverage)

The machine `DI START` high interval is the requested capture window. MIL remains physically
armed while the product gate is closed; synchronization work must finish before the rising edge.

| State | Event | Next state | Required action |
|---|---|---|---|
| `PhaseInvalid` | idle status tick, cameras ready | `PhaseSynchronizing` | Close product gate; pause/drain all connected cameras, reapply timing, resume back-to-back. |
| `PhaseSynchronizing` | all cameras warm and phase spread within tolerance | `ReadyIdle` | Mark phase verified; keep MIL armed; product gate stays closed. |
| `PhaseSynchronizing` | timeout, phase mismatch, disconnect, release | `PhaseInvalid` | Keep gate closed; log reason; retry only from a later idle tick. |
| `ReadyIdle` | line-rate/height/presence generation changes or standby becomes stale | `PhaseInvalid` | Invalidate readiness before another capture may start. |
| `ReadyIdle` | manual start | `HeadGuard` | Create grab owners and open the product gate without restarting MIL. |
| `ReadyIdle` | IO HIGH (rising, startup-held, or retry-held) | `HeadGuard` | After the existing form-level light command/warm-up, open without a physical MIL restart. A temporarily busy request returns to `Idle` and is retried only while DI remains HIGH. |
| `PhaseInvalid` / `PhaseSynchronizing` | manual start | `HeadGuard` or `PhaseInvalid` | Manual flow may wait for one full synchronization; open only after verification succeeds, then reject one boundary callback per camera. |
| `PhaseInvalid` / `PhaseSynchronizing` | IO HIGH | same state | Reject the attempt with `capture-not-ready`; BUSY remains off and the same held HIGH may retry after readiness completes. |
| `HeadGuard` | first callback from each connected camera | `HeadGuard` | Drop it before processing/display/persistence because a hot-standby line-scan frame can cross the light-off interval. |
| `HeadGuard` | next complete frame from every connected camera | `Capturing` | Log per-camera hardware ticks and validate circular phase spread. A mismatch invalidates the next start but does not truncate the current HIGH window. |
| `Capturing` | IO falling edge and stop condition=`IO` | `TailDrain` | Snapshot each camera's last accepted tick; accept exactly one newer complete frame per camera. |
| `Capturing` | IO falling edge and stop condition=`Time`/`Height` | `Capturing` | Record the edge but keep the product gate open; the selected fixed target owns stop timing. |

待機時的完整相位重算最多每 5 秒一次，避免 500ms 狀態輪詢持續配置 LINQ／快照物件。
任何相位失效事件會立即解除此節流；每次真正開始 Grab 仍在開 gate 前同步重驗，
因此節流只降低無操作時的背景成本，不放寬開始條件。
| `Capturing` | IO communication/PLC-alive loss and stop condition=`IO` | `ReadyIdle` | Stop immediately without waiting for a tail frame; the IO boundary is no longer trustworthy. |
| `Capturing` | IO communication/PLC-alive loss and stop condition=`Time`/`Height` | `Capturing` | Keep the product gate open and finish the selected fixed target; report the hardware fault independently. |
| `Capturing` | fixed time or common completed-row target reached | `ReadyIdle` or `AwaitingStartLow` | Close the gate immediately. If START is still High, wait for Low before rearming; otherwise return Idle. |
| `TailDrain` | IO HIGH returns before drain completes | `TailDrain` | Reject with `capture-not-ready:tail-drain`; BUSY stays Low. Controller returns to `Idle` and retries the held HIGH after drain/readiness completes. Never accept this as `already-grabbing`. |
| `TailDrain` | every camera completes its newer frame | `ReadyIdle` or `PhaseInvalid` | Close gate, stop product grab, keep MIL armed; readiness follows the last phase verdict. |
| `TailDrain` | timeout or disconnect | `PhaseInvalid` | Close gate and stop; log the missing cameras. Never wait indefinitely. |
| `Capturing` | manual stop or safety limit | `ReadyIdle` or `PhaseInvalid` | Close gate immediately; manual/safety stop does not extend the requested window. |

Boundary invariants:
- The MIL segment after form-level illumination readiness contains no `PauseAcquisition`,
  `ResumeAcquisition`, MIL fixed delay, or target-next-frame wait. DI polling and light warm-up
  remain separately measurable upstream costs and are not hidden as MIL synchronization time.
- `IO fall` closes only after one post-edge full frame per connected camera completes, or after
  the bounded tail timeout.
- First/tail frame decisions use hardware frame-start ticks. Wall-clock time is used only for
  freshness and timeout guards.
- One callback per connected camera is rejected after every new gate-open. It must produce
  `capture head frame dropped ... reason=cross-boundary` and must not reach image processing,
  display, Curve, CSV, or image persistence. These callbacks form the head phase probe; all
  cameras must then produce `capture head guard ... aligned=True` before any first accepted set.
- `capture head guard ... aligned=False` must close the product gate and stop the capture without
  any `capture first-set ready`, `firstFrame`, CSV record, or persisted image for that grab.
- IO start is level-sensitive only while the controller is `Idle`. `Running` and
  `AwaitingStartLow` consume that HIGH, so a fixed-target or safety stop cannot reopen repeatedly
  until DI first returns LOW.
- Stop condition is snapshotted when an IO request starts. `IO` stops on the falling edge and keeps
  the configured time as a safety ceiling; `Time` ignores an early falling edge and stops at
  `GrabLimitSeconds`; `Height` ignores an early falling edge and stops when the minimum accepted
  completed rows across all connected cameras reaches `WaterfallTotalHeight`.
- `IO grab accepted busy=on state=already-grabbing` is forbidden between `capture tail begin` and
  `StopGrab`; that would consume the next HIGH while the previous grab is still closing.
- A phase fault may reject the next IO pulse, but must not silently shorten an already accepted
  machine pulse.

IO edge log-flow (this supersedes the legacy `reason=start` sequence below for IO starts):
```
Tbg: acquisition phase synchronizing reason=idle previous=...
Tbg: acquisition idle prepare begin reason=... cams=P
Tbg: acquisition sync begin/paused/timing-reset/resumed/ready/phase/complete reason=idle ...
Tbg: acquisition phase verified reason=idle-sync
Tbg: acquisition idle prepare ready cams=P

T1: acquisition start path=verified-standby cams=P
T1: StartGrab...
T1: capture plan grab=...
T1: IO grab request stopCondition=IoSignal|Time|Height stopOnLow=True|False
T1: grab stop armed condition=IoSignal limit=Ns configured=10s grace=Gs source=io grab=...
    | grab stop waiting condition=Time configured=Ns source=io grab=...
    | grab stop armed condition=height limit=Hpx source=io grab=...
T1: capture gate open cams=P warm=True path=verified-standby
Tn: capture head frame dropped camN tick=T reason=cross-boundary × P
Tn: capture head phase system=S cams=... periodMs=... periodMismatchMs=...
    spreadTicks=D spreadMs=X limitMs=5.000 measurable=True aligned=True
Tn: capture head guard path=verified-standby cams=... aligned=True
Tn: capture first-set phase system=S cams=... periodMs=... periodMismatchMs=...
    spreadTicks=D spreadMs=X limitMs=5.000 measurable=True aligned=True
Tn: capture first-set ready path=verified-standby cams=... aligned=True
Tn: grab stop armed condition=Time limit=Ns configured=Ns grace=0s source=io
    start=first-set grab=...                                              ← 僅 Time 模式

T1: capture tail begin cams=... timeoutMs=N
Tn: capture tail frame accepted camN tick=T
Tn: capture tail frame complete camN tick=T
T1: capture tail complete pending=
T1: StopGrab
T1: capture gate closed standby=on
```

IO code-flow:
```
CameraStatusTimer_Tick@LiveCameraManager.Telemetry.cs
 -> PrepareIdleCaptureStandbyAsync@LiveCameraManager.CaptureBoundary.cs
    -> SynchronizeAcquisitionAsync(reason=idle) -> MarkCapturePhaseVerified

IoStartGrabAsync@AniloxRollForm.IoControl.cs
 -> snapshot CaptureStopCondition and set IoGrabController.StopCaptureOnStartLow
 -> TryGetCaptureStandbyReady
 -> ToggleLiveGrabAsync(ioControlled=true)
    -> StartGrabAsync(requireVerifiedStandby=true)
       -> verified standby: no Pause/Resume/fixed delay/next-frame target
    -> capture plan + selected duration/height guard
    -> Arm@CaptureStopCoordinator.cs
       -> IO: ArmedIo + GrabDurationCoordinator safety timer
       -> Time: WaitingForFirstSet (no timer)
       -> Height: ArmedHeight (common-row threshold)
    -> OpenCaptureGate -> ArmCaptureBoundary
    -> Time first-set ready: ActivateTimeAfterFirstSet -> ArmedTime + fixed timer

ProcessingFunction@MilCamera.cs
 -> updates LastFrameStartTicks + LastFrameObservedTimestamp for every standby callback
 -> OnMilFrameReady@AniloxCamera.cs
    -> CaptureFrameAccepted(cam,tick)
    -> processing/display/save
    -> CaptureFrameCompleted(cam,tick)
       -> NotifyCaptureFrameCompleted accumulates accepted rows per camera
       -> OnCaptureCommonRowsCompleted(min rows across connected cameras)
          -> ObserveCommonRows@CaptureStopCoordinator.cs
             -> terminal request only in ArmedHeight at the snapshotted threshold

IoStopGrabAsync@AniloxRollForm.IoControl.cs
 -> TryRequestIoStop@CaptureStopCoordinator.cs evaluates the snapshotted state and IoStopRequestReason
 -> IO condition + StartLow: ToggleLiveGrabAsync(drainIoTail=true)
    -> DrainIoTailAsync: exactly one newer completed frame per connected camera, or timeout
    -> StopGrab
 -> IO condition + PlcAliveLost/CommunicationLost: ToggleLiveGrabAsync(drainIoTail=false) -> StopGrab immediately
 -> Time/Height condition + any IO stop reason: log ignored and keep the product capture gate open

HandleCaptureStopRequested@AniloxRollForm.Live.cs
 -> ToggleLiveGrabAsync
 -> NotifyFixedGrabCompleted@IoGrabController.cs
    -> START Low: Idle
    -> START High: AwaitingStartLow; the following falling edge returns Idle
```

Capture stop state table (the coordinator is the only owner of these transitions):

| Current state | Event | Next state | Action |
|---|---|---|---|
| Idle | Arm(IO) | ArmedIo | Snapshot condition/limit/grab; arm configured time + boundary grace safety timer |
| Idle | Arm(Time) | WaitingForFirstSet | Snapshot condition/limit/grab; do not start timer |
| Idle | Arm(Height) | ArmedHeight | Snapshot height/grab; watch common accepted rows |
| WaitingForFirstSet | FirstSetReady | ArmedTime | Arm configured fixed timer |
| WaitingForFirstSet | FirstSetFailed | StopPending | Cancel timer; Form rolls back the start |
| ArmedIo | START Low | StopPending | One terminal request with tail drain |
| ArmedIo | PLC alive lost / communication lost | StopPending | One terminal request without tail drain |
| WaitingForFirstSet / ArmedTime / ArmedHeight | Any IO stop request | unchanged | Ignore IO stop; fixed target remains authoritative |
| ArmedIo / ArmedTime | TimerElapsed | StopPending | One terminal request |
| ArmedHeight | CommonRows >= snapshotted height | StopPending | One terminal request |
| StopPending | Any terminal trigger | StopPending | Ignore duplicate |
| Any active state | Complete / Cancel | Idle | Disarm generation timer and clear snapshot |

**log-flow（執行期腳印＝判準）**
```
T1: acquisition sync begin reason=start attempt=A gate=closed cams=P
T1: acquisition sync paused reason=start attempt=A cams=P
Tbg: acquisition sync timing-reset reason=start lineRates=cam1=R,cam2=R,...
T1: acquisition sync resumed reason=start attempt=A cams=P
T1: acquisition sync ready reason=start attempt=A camN system=S tick=T freq=F       × P
T1: acquisition sync phase reason=start attempt=A system=S cams=... spreadTicks=D
    spreadMs=X limitMs=5.000 measurable=True aligned=True sampleSource=warm-snapshot
                                                                                   × 有在線相機的板
T1: acquisition sync complete reason=start attempts=A cams=P phase=True
T1: capture charts reset reason=start-grab
T1: StartGrab（cams=M）
T1: ApplyMainDisplayMode → 同模式    ← 冪等：不得出現 create/teardown 行
T1: WF reset generation=G pendingDropped=N queuedDropped=N writerActive=B clearTile=True
T1: viewRange refire reason=capture-start mode=WF|IC
    ← 清除上一輪 Curve 視野後，用主畫面既有幾何主動重發；不得等滑鼠或首幀才補
T1: capture output begin grab=… date=yyyyMMdd
T1: capture plan grab=… root=… imageDir=… csv=… archive=….acap assets=… preview=1920x1080x3 scale=…
T1: grab stop armed condition=IoSignal limit=Ns configured=Ns grace=Gs source=io grab=…
    | grab stop waiting condition=Time configured=Ns source=io|manual grab=…
    | grab stop armed condition=height limit=Hpx source=io grab=…
    ← 本輪開始時依停止條件只建立一種 owner；Time 此時只等待，不得先啟動 timer
T1: capture gate open cams=P warm=True   ← P=在線數；必晚於同步完成與 plan/limit，早於所有 firstFrame
Tn: capture first-set ready ... aligned=True
Tn: grab stop armed condition=Time limit=Ns configured=Ns grace=0s source=io|manual
    start=first-set grab=…                  ← Time 真正起算點；前面的跨邊界丟幀不算入擷取時間
Tn: WF band first generation=G seq=S ticks=A~B startRow=0 height=H
Tn: firstFrame camX WxH → {ImageDisplayView|Waterfall}   ← 每台「在線」相機恰一行，順序不定
（首幀齊後進入穩態 → 適用「穩態靜默通則」：無互動下不得再有顯示狀態**變更**行。
  狀態**快照**行〔rowChart/WF state/IC state/stats，見§狀態快照儀器〕＝儀器輸出，穩態每秒出現正常）
```

**code-flow（靜態地圖＝責任鏈＋載重點；audit 時兩者都要對）**
```
 btnLiveGrab_Click@AniloxRollForm.Live.cs → ToggleLiveGrabAsync
                                                    intent 行 ui:【開始抓取】鈕
 ├（IsBgPreviewActive）ClearBackgroundPreview@AniloxRollForm.Background.cs
 │   └＝ExitBackgroundPreview（清幀＋回設定模式；共用顯示路後不再 FreeCameras）
 ├ AreCamerasHwReady@LiveCameraManager.cs 未就緒 → return   ← 守門：擋 IO 觸發路徑
 │                                                （IoStartGrab 直呼本方法繞過按鈕灰色）
 ├（未抓取→啟動）await Task.Run(LightTurnOn@AniloxRollForm.HardwareStatus.cs)
 │                                                  ⚠ 命令完成即續行；無固定暖機延遲
 │                                                  ⚠ 序列埠寫入一律背景（UI 執行緒零 MIL/序列埠鐵則）
 ├（未抓取）_muraExceedLatch 歸零
 │   ＋ UpdateMuraLed(false) ＋ ClearMura@IoGrabController.cs   ← MURA 閂鎖歸零（latch 非脈衝，M1）
 ├（未配置）await EnsureAllocatedAndToggleGrabAsync(deferCaptureGate:true)@LiveCameraManager.cs
 │   → AllocateCamerasAsync（=F1 全序）→ ToggleGrabAsync
 │   └（回 form）LoadBackgroundBins@AniloxRollForm.Background.cs ＋ EnableGlobalMerge@LiveCameraManager.Merge.cs
 ├（已配置）await ToggleGrabAsync(deferCaptureGate:true)@LiveCameraManager.cs
 │   └ StartGrabAsync@LiveCameraManager.cs
 │      ├ AreCamerasHwReady（CLProtocol ready＋每台在線相機已觀測 raw frame）未滿足 → return
 │      ├ _captureGateOpen=false                     ← 組態調整期間不接受任何 callback
 │      ├ SynchronizeAcquisitionAsync(reason=start)
 │      │  ├ Parallel PauseAcquisition：全部在線相機 M_STOP+M_WAIT＋M_GRAB_ABORT
 │      │  ├ ReapplyLineRatesForSynchronization：停止狀態重套各台現行 Line Rate
 │      │  ├ back-to-back ResumeAcquisition
 │      │  ├ raw callback 到齊後讀 Data Latch 首幀 tick
 │      │  └ 同板 spread≤5ms 才成功；超限最多重試 3 次，失敗不進產品狀態
 │      ├ ResetFlowFirstFrame@LiveDisplayCoordinator.cs（每輪 grab 重驗「幀有流到 view」）
 │      ├ ApplyMainDisplayMode@LiveDisplayCoordinator.cs   ← 冪等（view 已存在早退）＝本 flow 不得出現 create/teardown 行
 │      ├ ResetWaterfallIfActive@LiveDisplayCoordinator.cs → Reset@WaterfallView.cs（清舊圖＋重置 tick 對齊，防新幀接舊網格錯位）
 │      ├ OnCaptureSequenceReset → ResetLiveChartsForDisplayTransition@AniloxRollForm.Live.cs
 │      │  → 清列曲線累積位置、待上畫資料及欄／列相機快取
 │      │  → `capture charts reset reason=start-grab`
 │      │  ← 一般 Grab、IO Grab、取得背景共用；Form 不得另走專用重置
 │      ├ IsLiveGrabbing = true
 │      ├ RefireMainViewRange(reason=capture-start)@LiveDisplayCoordinator.cs
 │      │  └ RefireViewRange@WaterfallView.cs｜ImageDisplayView.cs → ApplyLiveViewRange
 │      └ per-cam SetUserGrabIntent(true)@AniloxCamera.cs
 │         └ SetUserGrabIntent@MilCamera.cs（同步 M_START 已完成；此處只開產品意圖）
 ├（啟動成功）NextGrabId@InspectionLogService.cs → _currentGrabId
 │  ├ BeginCaptureOutput@LiveCameraManager.cs → 每台 CaptureGrabId/CaptureDate 快照
 │  ├ capture plan 行（C1）
 │  └ Arm@CaptureStopCoordinator.cs
 │     ├（IO）ArmedIo＋GrabDurationCoordinator(limit＋boundary grace) → grab stop armed 行
 │     ├（時間）WaitingForFirstSet，只記 `grab stop waiting`，尚不啟動 timer
 │     └（高度）ArmedHeight，等 OnCaptureCommonRowsCompleted
 │        ← 停止條件與數值在本輪開始時拍快照；PropertyGrid 中途改值從下一輪生效
 ├ OpenCaptureGate@LiveCameraManager.cs
 │  ├ _captureGateOpen=true                         ← 單一全域寫入點；資料 owner 已準備後，7 台 callback 才一起取得資格
 │  ├ IsCaptureFrameAccepted@LiveCameraManager.CaptureBoundary.cs
 │  │  → AwaitingHeadProbe：每台第一 callback 收 tick 後丟棄
 │  │  → ValidateHeadBoundaryProbeSet：同板 spread≤5ms 才轉 AwaitingFirstSet；失敗立即關 gate
 │  │  → 全部在線相機首組產品幀到齊且相位驗證通過才轉 Active
 │  └ WaitForCaptureFirstSetReadyAsync@LiveCameraManager.CaptureBoundary.cs
 │     → IO／Time／Height 三種停止條件都等待同一個首組結果；失敗一律取消本輪
 │     →（僅時間）ActivateTimeAfterFirstSet@CaptureStopCoordinator.cs
 │        → ArmedTime＋Arm(configured seconds)@GrabDurationCoordinator.cs
 └ UpdateGrabButton@AniloxRollForm.Live.cs
（每幀幀流，MIL 回呼執行緒 Tn）
ProcessingFunction@MilCamera.cs（MdigProcess hook，static）
 └ FrameReady 事件 → OnMilFrameReady@AniloxCamera.cs
    ├ UserWantsGrab && CaptureFrameAccepted(cam,tick) 才繼續；未准入時不進 GPU／顯示／CSV／存檔
    ├ TryApplyPicoaterRidge@AniloxCamera.cs（GPU 檢測，一律跑）  ⚠ _picoaterLock＋尺寸守門（高度變更瞬間跳過幀防 AV）
    │  ├ ProcessImage@AoiService.cs（P/Invoke TanukiPipeline_Process；fused 存檔縮圖 wantResize＝grab-level 決策）
    │  ├ OnLiveCurveData 事件 → OnLiveCurveData@AniloxRollForm.Live.cs → CheckLiveMura("v")（M1）＋ _liveOverviewDirty=true
    │  └ OnLiveRowCurveData 事件 → OnLiveRowCurveData@AniloxRollForm.Live.cs → CheckLiveMura("h")
    │     ＋ pending latest-only cache（不得直接更新列 chart）
    ├ PutDisplayBytes@MilCamera.Display.cs（強化）｜CopyToDisplay@MilCamera.Display.cs（原圖）
    ├ OnDisplayFrame 事件（目前選定影像）→ OnCameraDisplayFrame@LiveDisplayCoordinator.cs
    │  ├ 模式錯掛自檢（⚠ 契約違規 行）＋ FlowFirstFrame（firstFrame 行，每台恰一）
    │  └（即時）PushFrame@ImageDisplayView.cs（存快照＋餵 ThumbStrip＋_mainDirty）
    ├ OnWaterfallFrame 事件（同幀 raw＋column＋row）→ OnCameraWaterfallFrame@LiveDisplayCoordinator.cs
    │  ├ 模式錯掛自檢（⚠ 契約違規 行）＋ FlowFirstFrame（firstFrame 行，每台恰一）
    │  └ PushFrameVariants@WaterfallView.cs → PlaceFrame（tick 網格只對齊一次）→ TryFlush → ComposeJob
    │      （佈局=MergeLayout.Compute 唯一來源）→ KickWriter → Task.Run WriteBand（三層背景 memcpy，不卡 UI）
    │      ＋ PushFrame@ThumbStrip.cs（縮圖顯示目前選定層）
    ├（hook 返回 MilCamera 後）CopyDisplayToMergeTarget@MilCamera.cs   ← 合圖貼圖在 grab hook（display buffer 更新後）
    └ TrySaveCapture@AniloxCamera.cs（→ CameraFrameSaver 背景存檔 → C1/C2）
（顯示重繪，UI 執行緒 T1）
RefreshMain@ImageDisplayView.cs（33ms _timer）
 ├ UpdateReverseThumbSync@ImageDisplayView.cs（快拖補刷）
 ├（LOD）lodRebind 留痕 → EnableLod/RefreshLod@ImageCanvas.cs
 │   → LodTileApplied(generation) → ContentPresented
 ├（非 LOD）BuildMerge｜BuildSingle@ImageDisplayView.cs → autoFit(firstFrame) 留痕 → FitToScreen@ImageCanvas.cs
 │   → RefireViewRange@ImageDisplayView.cs → ContentPresented
 └ FlowViewState@ImageDisplayView.cs（上畫後，1s 節流）→ IC state 快照行（免滑鼠）
瀑布顯示：_flushTimer(30ms)@WaterfallView.cs → TryFlush ＋ PushLodRefresh ＋ UpdateCenterCam
 → LodTileApplied(generation) → ContentPresented
 ＋ FlowState@WaterfallView.cs（band 寫入後，1s 節流）→ WF state 快照行
ContentPresented → MainContentPresented@LiveDisplayCoordinator/LiveCameraManager
 → PresentPendingLiveRowCurves@AniloxRollForm.Live.cs
 → OnLiveRowCurveDataUi → RowCurveDisplayAdapter.FlowApply（rowChart 快照行）
```

瀑布每輪 Reset 前由 `LiveDisplayCoordinator.SeedWaterfallFramePeriod` 使用相機已套用的
`FrameHeight / AppliedLineRateHz` 與 Data Latch clock 預載分組週期：
`WF bootstrap period camN periodMs=P source=applied-hardware`。第一批已對齊相機幀到齊後即可組成 band，
不得為了重新量測週期而額外等待同一台相機的第二幀；硬體週期不可用時才允許
`WF bootstrap period unavailable; learn from runtime frames`，後續仍由實際 tick delta 校正。

列曲線顯示不變量：GPU 完成只能更新 pending cache；必須等同一條主畫面 API 發出
`ContentPresented` 才可更新 chart。LOD 以 content generation 對應實際安裝的 tile，不能在
`RefreshLod` 僅提出請求時提前發布。`rowCurve present after=mainImage cams=N mode=IC|WF`
只能由 `RowCurveSyncCoordinator.DataAccepted` 在確認 chart 已具備完整資料與視野、即將同步更新時發布；資料仍因缺視野停在
pending 時不得先記成 present。快速累積時只保留每台最新一筆，禁止
用固定延遲猜影像完成時間。

同步相位不變量：`acquisition sync ready` 取得的 `tick/freq/height/lineRate` 是同一批不可變
warm snapshot；`acquisition sync phase ... sampleSource=warm-snapshot` 必須只用這批數值計算。
禁止驗證時重新讀取仍在前進的 `LastFrameStartTicks`，否則低 Line Rate 會把不同幀世代誤判成
約一個 frame period 的相位差，讓 manager 永久停在 busy。

### F3 停止抓取

**log-flow（執行期腳印＝判準）**
```
T1: StopGrab
T1: capture gate closed standby=on
T1: rowCurve present after=mainImage cams=N mode=IC|WF
    （可選且最多一行；僅限前面已有 `capture tail complete pending=` 的最後一組合法尾幀）
Tn: drop drainedFrame after StopGrab camN（可選；每台最多一行）
（之後不得再出現 firstFrame / 新的 CSV、影像或 Curve 更新，直到下一個動作；
  最後一組 row Curve 是 Stop 前已接受並完成的 IO tail 幀，只是 UI 合併呈現晚於 StopGrab；
  drop 行是 gate 關閉競態的觀測儀器，不是顯示更新）
```

**code-flow（靜態地圖＝責任鏈＋載重點；audit 時兩者都要對）**
```
手動：btnLiveGrab_Click@AniloxRollForm.Live.cs（wasGrabbing=true）→ ToggleLiveGrabAsync
                                                          intent 行 ui:【開始抓取】鈕
時間／IO 安全逾時：Elapsed@GrabDurationCoordinator.cs
 → HandleTimerElapsed@CaptureStopCoordinator.cs → StopPending → SafeBeginInvoke
 → HandleCaptureStopRequested@AniloxRollForm.Live.cs
                                                          intent 行 auto:抓取停止 condition=... limit=Ns grab=… → 停止
高度：OnCaptureCommonRowsCompleted@LiveCameraManager.cs
 → ObserveCommonRows@CaptureStopCoordinator.cs → StopPending → SafeBeginInvoke
 → HandleCaptureStopRequested@AniloxRollForm.Live.cs
                                                          intent 行 auto:抓取停止 condition=Height rows=R limit=Hpx grab=… → 停止
 ├ ToggleLiveGrabAsync@AniloxRollForm.Live.cs
 │  └ ToggleGrab@LiveCameraManager.cs
    └ StopGrab@LiveCameraManager.cs
       ├ FlowTrace "StopGrab"
       ├ _captureGateOpen=false               ← 第一個動作；所有相機同一個原子 gate，晚到 callback 一律跳過
       ├ IsLiveGrabbing=false
       ├ per-cam SetUserGrabIntent(false)      ← 清產品意圖；KeepAcquiringWhenIdle=true，故不做 M_STOP
       └ FlowTrace "capture gate closed standby=on"
          └ callback 若已跨過 gate 讀取邊界：最多一個 drop 行，不進 Hessian/row chart/CSV/存檔
             ← 防停止尾幀「有效影像 + 黑尾」被 Hessian 當水平脊線（黑白硬邊界）寫到最後 row
（form 收尾，T1）
 ├ CompleteStop@CaptureStopCoordinator.cs → Disarm@GrabDurationCoordinator.cs
 │                                                            ← 回 Idle 並作廢 generation，舊 callback 不得停掉下一輪
 ├ Task.Run(LightTurnOff@AniloxRollForm.HardwareStatus.cs)   ⚠ [UiStack] 曾定罪停止時卡 SerialStream.Write → 一律背景
 ├ TriggerRetentionAndFlagAsync@AniloxRollForm.StorageStatus.cs
 ├ UpdateMuraLed(false) ＋ ClearMura@IoGrabController.cs   ← MURA latch 清除時機＝檢測結束（M1；手動流程不經 FSM 必須自清 DO）
 ├ UpdateGrabButton@AniloxRollForm.Live.cs
 └（IO 安全逾時）NotifyGrabStopped@IoGrabController.cs       ← 先把 DO_PC_INSPECT 拉低；FSM 在 DI START
                                                            維持 High 時不重啟，等下降後回 Idle
   （時間／高度完成）NotifyFixedGrabCompleted@IoGrabController.cs
                                                            ← START 已 Low 直接 Idle；仍 High 則進
                                                            AwaitingStartLow，下降後才可接受下一輪
實體停止邊界只有兩種：
 - Start 同步／停止狀態的高度重配置：PauseAcquisition → M_STOP+M_WAIT＋M_GRAB_ABORT → 修改 → ResumeAcquisition
 - Release：先關 capture gate，再平行 PauseAcquisition，完成後才能釋放 merge／camera buffers
```

**StopGrab 校稿工具**
```
python tools/python/check_stopgrab_flow.py [trace.log]
```
- PASS 判準：每個 `StopGrab` 必接 `capture gate closed standby=on`；其後到下一個
  `ui:`/`StartGrab`/`AllocateCameras begin` 前只允許：
  ①已有 `capture tail complete pending=` 時的一組 `LC row` 快照與最多一行
  `rowCurve present after=mainImage`；②`drop drainedFrame after StopGrab camN`。
  不得再出現 `firstFrame`、`capture csv`、新的 IC/WF display 更新或第二組 Curve 呈現。

### F4 切「主畫面顯示」設定（即時↔瀑布，即時生效）

**log-flow（執行期腳印＝判準）**
```
T1: ApplyMainDisplayMode → 新模式
T1: Teardown{舊 view}（unsubscribe M）
T1: {新 view} create + subscribe M
（grab 中切換：接著每台在線相機 firstFrame → 新 view）
座標發布順序：`ImageDisplayView.RefireViewRange` / `OnCanvasStatus` 必須先確認 `ContentW/H > 0` 才發布 view range；column/row chart 初始化不得吃空畫面的暫態 range。
```

**code-flow（靜態地圖＝責任鏈＋載重點；audit 時兩者都要對）**
```
propertyGridSettings.PropertyValueChanged → _propertyGrid_PropertyValueChanged@AniloxRollForm.cs
 └ NotifyExternalChange@SettingsHub.cs            ← SSoT：所有變更走 Hub（Source=PropertyGrid）
    └ Changed 事件 → OnSettingChanged@AniloxRollForm.cs（唯一 dispatcher，semaphore 序列化；勿拆）
       └ SettingImpactClassifier → owner=LiveLayout
          └ HandleLiveLayoutSettingsChanged@AniloxRollForm.Live.cs（name==he_MainDisplay）
          ├ ResetLiveChartsForDisplayTransition@AniloxRollForm.Live.cs（column/row chart、row cache、waterfall row buffer、live view range 歸零）
          └ ApplyMainDisplayMode@LiveCameraManager.Display.cs（forwarder）
             └ ApplyMainDisplayMode@LiveDisplayCoordinator.cs   ← 模式單一決策點（he_MainDisplay 唯一入口）＋錯掛自檢旗標歸零
                ├ →瀑布：TeardownImageDisplay@LiveDisplayCoordinator.cs（unsubscribe OnDisplayFrame＋Dispose＋GPU LOD Release）
                │        → EnableWaterfallDisplay@LiveDisplayCoordinator.cs
                │           （new WaterfallView＋new ThumbStrip〔縮圖一律即時，鐵則1〕＋FeedWaterfallLayout
                │             ＋subscribe 各 cam.OnDisplayFrame）
                └ →即時：DisableWaterfallDisplay@LiveDisplayCoordinator.cs（unsubscribe＋Dispose view+thumbs）
                         → EnsureImageDisplay@LiveDisplayCoordinator.cs
                            （new ImageDisplayView＋ApplyImageDisplayOptions〔FlipVertical/VerticalZeroAtBottom＝
                              轉換點#1/#3〕＋SetLayout＋subscribe＋ClearMissingCameraFrames＋SetLodMode）
（grab 中切換）幀流不中斷 → 新 view FlowFirstFrame → 每台在線相機 firstFrame → 新 view
不變量：Enable*/Ensure* 冪等（view!=null 早退）→ 建新 view 前必先 teardown 舊 view（否則殘留舊訂閱；與 F1/F7 對稱）
```

### F5 點縮圖/狀態字（縮圖→主畫面連動）

**log-flow（執行期腳印＝判準）**
```
即時模式點縮圖：T1: SwitchMainDisplay cam=N center=False
  ← 置中由 ImageDisplayView「內部」完成（thumb click → CenterOnCamera），外部呼叫只同步選中 → False 是對的
瀑布模式點縮圖／任一模式點狀態字：T1: SwitchMainDisplay cam=N center=True
  ← 置中由 coordinator 做（WaterfallView/ImageDisplayView.CenterOnCamera）
（兩者皆：主畫面置中到 cam N、橘框=N）
```

**code-flow（靜態地圖＝責任鏈＋載重點；audit 時兩者都要對）**
```
即時模式點縮圖（置中在 view 內部完成）：
ThumbView.MouseClick@ThumbStrip.cs → SelectRequested 事件
 └ ThumbStrip.SelectRequested handler（ctor 內）@ImageDisplayView.cs
    ├（合圖）CenterOnCamera@ImageDisplayView.cs   ← 內部置中：pan 定位該相機槽中心（保 zoom；用 _mergePlacements 同份佈局）
    ├ SetSelected@ImageDisplayView.cs（橘框）
    └ SelectRequested 事件 → ImageSelectCamera@LiveDisplayCoordinator.cs
       → SwitchMainDisplay(camId)＝center=False    ← False 是對的（置中已在 view 內部做完，再置中＝重複/回彈）
瀑布模式點縮圖：
ThumbView.MouseClick@ThumbStrip.cs → _waterfallThumbs.SelectRequested handler
（EnableWaterfallDisplay 內）@LiveDisplayCoordinator.cs → SwitchMainDisplay(idx+1, centerView:true)
任一模式點狀態字/縮圖底 panel（浮動 label）：
displayPanel/status.MouseClick（SetupLivePanel 內）@LiveDisplayCoordinator.cs → SwitchMainDisplay(cameraIndex, centerView:true)
共同下游：SwitchMainDisplay(cameraIndex, centerView)@LiveDisplayCoordinator.cs
 ├（InvokeRequired）BeginInvoke 自轉 UI 執行緒
 ├ Flow "SwitchMainDisplay cam=N center=…"
 ├ SetSelected@ImageDisplayView.cs ／ SetSelected@ThumbStrip.cs（橘框＝選中框唯一視覺來源）
 └（centerView=true）CenterOnCamera@ImageDisplayView.cs｜CenterOnCamera@WaterfallView.cs   ← coordinator 置中（契約 True 分支）
另一入口（瀑布主畫面點擊選台，center=False）：
OnCanvasMouseClick@WaterfallView.cs → SelectRequested 事件 → OnWaterfallSelectRequested@LiveDisplayCoordinator.cs
 → SwitchMainDisplay(camId)（畫布點擊選取＝不置中，否則蓋掉使用者拖出的視野）
```

### F6 拖動主畫面（主畫面→縮圖反向連動）

**log-flow（執行期腳印＝判準）**
```
（不得出現任何 center=True 行——出現即回彈 bug）
T1: centerCam → camX（IC|WF）   ← 中心相機每跨一台一行（快拖連續數行=正常；跳號=補刷失效）
（即時=ImageDisplayView.UpdateReverseThumbSync、瀑布=WaterfallView.CenterCamChanged；
  兩者皆有 30/33ms timer 補刷，快拖不跳格）
```

**code-flow（靜態地圖＝責任鏈＋載重點；audit 時兩者都要對）**
```
OnMouseMove@ImageCanvas.cs（拖曳中；UI 執行緒 T1）
 ├ pan 每 move 累積＋重繪限流 ~120fps（只限「畫」，MouseUp 尾緣補繪）   ← 不變量：paint 風暴防護（IC stats paints ≤~130/s）
 └ 首個位移立即 TriggerStatusChange，其後 ~30fps 限流 → StatusChanged 事件
    └ `IC|WF|RV drag(view-published)` 在同步 subscriber 全返回後才留痕
       （圖片跟手但 curve 只在 MouseUp 動＝首發被 hover 共用節流窗吃掉；2026-07-13 修）
即時分支：OnCanvasStatus@ImageDisplayView.cs
 ├ PixelMmMapper 換算＋VerticalZeroAtBottom 鏡射    ← 轉換點#3（各邊映自己的值，勿交叉——2026-07-08 邊界方向錯根因）
 ├ SetRangeOverlay＋SetCursorMm@ImageCanvas.cs → DrawOverlays
 │  → 滑鼠旁顯示位置 mm＋目前顯示像素亮度；四邊顯示視野範圍、右下顯示實體倍率
 │    （游標固定 `(x, y)mm | 亮度:N`；LOD 從目前 tile 取亮度；`lblInfo` 不接游標事件；文字以實際尺寸失效舊/新區域，不得留下移動殘影）
 ├ ViewRangeMmChanged 事件 → OnImageViewRange@LiveDisplayCoordinator.cs → OnLiveViewRange 事件
 │  → ApplyLiveViewRange@AniloxRollForm.Live.cs      ⚠ 勿節流此連動（三次教訓：拖曳中曲線必須逐事件跟隨）
 │     ├ SetViewRange@RowCurveSyncCoordinator.cs → RowCurveDisplayAdapter → RowCurveChartHelper（列 chart Y zoom＝轉換點#4/#5）
 │     └ UpdateViewRange@ColumnCurveChartHelper（欄全覽 X zoom；首次就緒→LiveOverviewTimer_Tick 原子畫一次不閃）
 └ UpdateReverseThumbSync@ImageDisplayView.cs → SelectedCamChanged 事件
    → handler（EnsureImageDisplay 內）@LiveDisplayCoordinator.cs → Flow "centerCam → camX（IC）"
瀑布分支：OnCanvasStatus@WaterfallView.cs
 ├ TryComputeViewRange@WaterfallView.cs → ViewRangeMmChanged 事件 → OnImageViewRange@LiveDisplayCoordinator.cs
 │  →（同上）ApplyLiveViewRange@AniloxRollForm.Live.cs
 ├ UpdateCenterCam@WaterfallView.cs → CenterCamChanged 事件 → OnWaterfallCenterCam@LiveDisplayCoordinator.cs
 │  → SetSelected@ThumbStrip.cs＋Flow "centerCam → camX（WF）"（程式化來源，不回頭置中防遞迴）
 └ SetRangeOverlay＋SetCursorMm@ImageCanvas.cs → DrawOverlays（同上，含 LOD tile 亮度）
補刷保險（快拖事件合併不跳格）：
 即時：_timer(33ms) → RefreshMain@ImageDisplayView.cs 開頭 UpdateReverseThumbSync
 瀑布：_flushTimer(30ms)@WaterfallView.cs → UpdateCenterCam
拖曳尾緣：OnMouseUp@ImageCanvas.cs → Invalidate＋TriggerStatusChange 補發＋FlowLog "viewEdges …" 一行
```

### F6a 畫布資訊模式／主工作區全寬

**log-flow**
```
T1: IC|WF|RV overlay mode=Coordinates|CoordinateFrames|CoordinateFramesParameters|Hidden
T1: ui:canvas overlay mode={mode} sync=live+review persisted=true
T1: canvas overlay restore mode={mode} sync=live+review
T1: ui:monitor tab five-click rightPanel=hidden|visible workspaceW=N rightPanelW=N
T1: workspace restore rightPanel=hidden|visible workspaceW=N rightPanelW=N
```

- 主畫面右鍵循環順序固定為：全空 → 座標數值 → 座標數值＋七台相機影像框線 →
  座標數值＋七台相機影像框線＋參數表格 → 全空；後兩態為累加顯示。
- 監控即時、監控瀑布與回顧共用一份 `CanvasOverlayMode` session 狀態；任一主畫面右鍵切換後，
  其他畫面下一次顯示時必為同一模式。狀態寫入 `Config/session-state.json`，重開程式後還原。
- 監控參數資訊讀目前 PropertyGrid；回顧／報表共用的回顧畫布讀該序號拍攝時 `#CFG`。
  SDK 只負責畫字，不得引用 `InspectionSettings` 或其他 app policy。
- 相機框線使用與合圖相同的 `MergeLayout` placements；每台畫實際影像區域，七台相鄰時必可見外框與六條分隔線，
  不得以 control 外框冒充影像邊界。
- `tabMain` 的「監控」頁籤標籤區連續左鍵五下只切換一次右側 `tabControlRight`；點頁面內容或其他
  頁籤不計數。一般模式由 `MainWorkspaceLayoutController` 以可用寬度分配：右側約 1/5、`tabMain`
  約 4/5，視窗改變大小時重新計算，再由 `ProportionalScaler` 等比例縮放頁內元件。隱藏時
  `tabMain` 使用原右側邊界；還原時必回到 4/5＋1/5，Designer 亦使用相同比例作為設計基準。
  狀態寫入 `Config/session-state.json`，下次啟動在
  `PrewarmAllTabs` 後還原。這是 UI session 狀態，不屬於檢測設定或 CSV `#CFG`。

**code-flow**
```
OnMouseDown@ImageCanvas.cs（右鍵）
 → CanvasOverlayMode 累加四態
   ├ Hidden：全空
   ├ Coordinates：座標數值
   ├ CoordinateFrames：座標數值＋CameraFrameRegionsProvider
   │  └ ImageDisplayView|WaterfallView 以現行 MergeLayout placements 提供七台影像區域
    └ CoordinateFramesParameters：座標數值＋框線＋InformationTextProvider
       ├ Live：CanvasParameterTextBuilder.FromCurrentSettings
       └ Review：CanvasParameterTextBuilder.FromCaptureConfig(CurrentGrabConfig)
 → OverlayModeChanged@ImageCanvas.cs
   → LiveDisplayCoordinator|ReviewDisplayManager → AniloxRollForm.ApplyCanvasOverlayMode
      ├ 同步 Live 即時／瀑布與 Review 畫布
      └ UserSessionState.SaveCanvasOverlayMode

tabMain.MouseDown（hit-test 必須落在 tabPageLiveView 標籤區）
 → MainWorkspaceLayoutController 連續按下計數（相鄰按下間隔 ≤1200ms）
   └ 第五下 → MainWorkspaceLayoutController.ApplyLayout
      ├ 一般模式：可用寬度按 tabMain 4/5＋tabControlRight 1/5 重算
      ├ 全寬模式：tabControlRight.Visible=false＋tabMain 使用原右側邊界
      ├ ProportionalScaler.RescaleActiveTabs
      └ UserSessionState.SaveMainWorkspaceFullWidth
啟動 PrewarmAllTabs 完成
 → MainWorkspaceLayoutController.ApplyPersistedLayout
```

### F6b 滾輪縮放主畫面

**log-flow（執行期腳印＝判準）**
```
T1: IC|WF wheelZoom in|out → zoom=Z（fit=F min=M content=WxH）   ← 每手勢至少一行（100ms 節流）
T1: IC|WF|RV fit(double-click) / physical1x(triple-click)   ← 使用者 fit/1x 手勢（合法的視野重設主人）
（縮放/互動期間**不得出現 `autoFit(...)`/`lodRebind(...)` 行**——出現＝系統 fit 跟使用者縮放打架。
  `FitRelativeZoom=false` 時 M 必須等於 `max(1/W, 1/H)`（最低保護值 `0.000001`）；
  不得再出現固定 `0.01` 縮小牆。Z 不得低於 M，且超寬合圖必須能縮到 `0.01` 以下。
  zoom 突然回 fit 而無 fit(double-click) 行＝有東西在暗中重設（孤兒判讀）。
  `autoFit(firstFrame ...)` 只允許在 view 建立後首幀；`autoFit(sizeChanged@fitView ...)` 只允許在
  使用者「未動過視野」時的尺寸變更。centerCam 行在縮放中出現＝正常（中心相機隨視野變）。）
```

**code-flow（靜態地圖＝責任鏈＋載重點；audit 時兩者都要對）**
```
OnMouseWheel@ImageCanvas.cs   ← 滾輪一律 canvas 自理（app 無全域訊息濾鏡）
    ├ `FitRelativeZoom=false`（監控即時／瀑布與回顧一致；fit 不是最小值，允許再縮小總覽）
    ├ `MinimumUsefulZoom=max(1/ContentW, 1/ContentH, 0.000001)`（較短邊至少 1px；
    │   內容尺寸自適應，禁止固定 `0.01` 下限）
    ├ zoom ×1.1^(e.Delta/120)     ← 正比實際轉動量（事件合併時大 e.Delta 也按比例；修卡頓漏算）
    ├ FlowLog "wheelZoom in|out"（100ms 節流＝每手勢至少一行）
    ├ pan 錨定游標點＋Invalidate（zoom 防抖：滾動中拉伸舊 cache/tile，_zoomSettleTimer 停 150ms 才重建）
    ├ TriggerStatusChange@ImageCanvas.cs → StatusChanged 事件 →（下游同 F6：OnCanvasStatus →
    │   ViewRangeMmChanged/CursorStatusChanged/centerCam → chart 連動）
    └ RestartLodSettle@ImageCanvas.cs（LOD 停住才重算 crisp tile，互動中先用舊 tile 拉伸）
fit/1x 手勢（合法視野重設主人）：OnMouseDown@ImageCanvas.cs → MultiClickDetector.RegisterClick
 ├ 雙擊（非 fit 時才動作）：FitToScreen@ImageCanvas.cs → FitPerformed 事件 → FlowLog "fit(double-click)"
 │  （ImageDisplayView ctor 接線）
 └ 三擊：ZoomToOneToOne@ImageCanvas.cs → Physical1xPerformed 事件 → FlowLog "physical1x(triple-click)"
違規源頭定位（縮放中不得出現的兩行，只有這些產地）：
 autoFit 只在 RefreshMain@ImageDisplayView.cs（firstFrame / IsAtFitView 下的尺寸變更）；
 lodRebind 只在 RefreshMain 的 LOD 綁定處（EnableLod 內建 FitToScreen）——縮放中出現＝
 ClearFrame/尺寸 churn 在暗中重設（孤兒判讀；ClearFrame@ImageDisplayView.cs 已冪等守門：空幀不動顯示狀態）
```

### F7 重配置（FreeCameras → 再配置）

**視窗關閉狀態與事件表**

| State | Event | Next State | Action |
|---|---|---|---|
| Running | FormClosing | ClosingAsync | `e.Cancel=true`；停止 UI timers／grab；await IO stop＋camera Release＋service Dispose |
| ClosingAsync | FormClosing | ClosingAsync | 忽略重複關閉，不啟動第二條釋放鏈 |
| ClosingAsync | cleanup 完成或失敗 | ReadyToClose | 記 `shutdown resources released`；設完成旗標後再次 `Close()` |
| ReadyToClose | FormClosing | Closed | 不再 cancel，交還 WinForms 正常結束 |

禁止把必要釋放放在 `async FormClosed`：WinForms 不等待 async event handler，程序可能在第一個 `await`
後直接退出，造成 `FreeCameras` 未執行。

**log-flow（執行期腳印＝判準）**
```
T1: ui:關閉程式
Tbg: FreeCameras（cams=M）
Tbg: TeardownImageDisplay / TeardownWaterfall（有哪個拆哪個）
T1: shutdown resources released
T1: （再配置時）F1 全序重跑——view 必須重建+重訂閱新相機批次
```

**code-flow（靜態地圖＝責任鏈＋載重點；audit 時兩者都要對）**
```
ReleaseAsync@LiveCameraManager.cs
 ├ 呼叫端先 _cameraStatusTimer.Stop ＋ IsReleasing=true
 ├ await allocation gate（配置中的 native call 返回前不得釋放）
 └ Task.Run → FreeCamerasCore@LiveCameraManager.cs
 ├ IsReleasing=true ＋ _cameraStatusTimer.Stop ＋ _captureGateOpen=false ＋ IsLiveGrabbing=false
 ├ Parallel.ForEach cams → PauseAcquisition@MilCamera.cs
 │  └ DrainGrab：M_STOP+M_WAIT → M_GRAB_ABORT
 │     ← 不變量：所有 digitizer 完成實體 drain 後才可釋放任何 MIL buffer（防 UAF）
 ├ DisableGlobalMerge@LiveCameraManager.Merge.cs   ← 順序鎖死：必在 cam.Free 之前
 │   （先清各台 merge target 再由工頭 MbufFree 合併 buffer，防 grab hook 把幀複製進已釋放 buffer）
 ├ TeardownImageDisplay@LiveDisplayCoordinator.cs ＋ TeardownWaterfallDisplay@LiveDisplayCoordinator.cs
 │   ← Enable*/Ensure* 冪等（view!=null 早退）→ 不 teardown 就不會重訂閱新相機批次
 │     （「預覽背景→開始抓取」瀑布空白的根因）
 ├ per-cam Free@AniloxCamera.cs → Dispose（MIL digitizer/buffer 釋放）
 ├ FreeSystem@CameraSystemManager.cs ×板 ＋ FreeApplication@CameraSystemManager.cs
 └ IsAllocated=false
Timer.Tick 跑在 UI 執行緒，不先 Stop 則 Tick 可能在背景 cam.Free() 期間存取同一台相機。
`IsReleasing` 使仍在 processing 迴圈的 worker 於相機邊界提早離開；allocation gate 保證同一 native call
不會同時 Allocate／Free。
（再配置）AllocateCamerasAsync@LiveCameraManager.cs＝F1 全序
   （開頭 TeardownImageDisplay/Waterfall → ApplyMainDisplayMode 先拆後建，與本 flow 對稱＝訂閱一定綁「這批」相機）
```

### F8 取得背景 / 預覽背景
取得背景=借用現有 grab 採集（啟停包夾）；預覽=走 grab 同一個 ImageDisplayView 共用路（顯示鐵則0：
預覽主畫面＝7 台背景合圖）。預覽狀態＝coordinator 靜音鍵（不動 he_MainDisplay 設定），
**建拆唯一權威＝ApplyMainDisplayMode 閘門**（靜音鍵是閘門輸入之一，非平行通道——2026-07-09 對齊）。

**code-flow（靜態地圖＝責任鏈＋載重點；audit 時兩者都要對）**
```
取得背景：
btnLiveGetBackground_Click@AniloxRollForm.Background.cs      intent 行 ui:【取得背景】鈕
 ├ IsStandardBgSubEnabled 守門（非標準去背 → MessageBox return）
 ├（舊預覽）ClearBackgroundPreview@AniloxRollForm.Background.cs
 ├ SetCaptureSuppressed(true)＋`background capture begin output=disabled`
 │   ← 背景採樣借用 grab／演算法／顯示，但不是產品擷取；不得寫圖片、CSV 或產生 CaptureWriteFailure
 ├（未配置）await EnsureAllocatedAndToggleGrabAsync@LiveCameraManager.cs（=F1＋F2 借道）
 ├（未抓取）LightTurnOn@AniloxRollForm.HardwareStatus.cs（命令完成即續行；無固定暖機延遲）
 │   → ToggleGrab@LiveCameraManager.cs ＋ UpdateGrabButton(true)   ← 借用現有 grab（啟停包夾）
 ├ CaptureAndActivateAsync@BackgroundCaptureCoordinator.cs
 │   ├ 產生 version=`yyyyMMdd-HHmmssfff`
 │   ├ WaitForCaptureFirstSetReadyAsync@LiveCameraManager.CaptureBoundary.cs
 │   │   ← 全部在線相機首組完整幀到齊且相位通過後，背景採樣秒數才開始計算
 │   ├ 採集迴圈（await Task.Delay(100) × BackgroundSampleSeconds，UI 執行緒非阻塞、按鈕倒數）
 │   │   └ per-cam TryComputeColumnMean@AniloxCamera.cs → accum 累加
 │   ├ per-cam 平均 → SaveCameraProfile@BackgroundProfileRepository.cs
 │   │   → `bg_{width}_{cam}_{version}.bin`（MCBF v2；CreateNew＋WriteThrough＋Flush）
 │   ├ 全部在線相機成功 → ActivateVersion@BackgroundProfileRepository.cs
 │   │   原子替換 `active-background.json`
 │   └ 任一相機失敗 → DeleteVersion@BackgroundProfileRepository.cs 刪本次 version 檔，
 │       manifest 不動、上一組背景繼續生效
 ├ LoadBackgroundBins@AniloxRollForm.Background.cs
 │   → ReadManifest＋ResolveCameraProfilePath＋LoadProfile@BackgroundProfileRepository.cs
 │   → manifest 指向的同一版 bin → 驗證長度/有限值
 │   → TanukiCv_AllocPinned → ReplacePrecomputedColumnMean@AniloxCamera.cs）
 │   ← 只要求目前在線相機具備同版背景；離線相機清除舊綁定／舊警示並記 `status=skipped reason=offline`
 │   ← pinned 生命週期：相機的 `_picoaterLock` 內原子換新，離鎖後 FreePinned 舊 buffer；
 │      切回單張去背也走 ClearPrecomputedColumnMean，不得遺失舊指標造成漏記憶體
 │   ← 新檔載入失敗保留上一份已綁定背景並回報 OutputHealth；完全沒有可用背景時正式 Grab 必須阻擋，
 │      不得在「標準去背」設定下靜默退回每幀自算
 ├ 每次正式 Grab 第一個成功處理幀：TryApplyPicoaterRidge@AniloxCamera.cs
 │   → AoiService.ProcessImage → TanukiPipeline_Process(precomputed_col_mean)
 │   → `background apply` 留下 native 呼叫實際收到的來源（非 UI 設定意圖）
 ├ 任一相機失敗 → Form catch
 │   → OutputHealth `BackgroundCaptureFailure` 深橘提示
 ├ finally：ToggleGrab 停止（=F3）＋ LightTurnOff ＋ SetCaptureSuppressed(false)
 │   ＋ `background capture end output=disabled result=ok|failed` ＋ UpdateStandardBgSubLockState
 ├（_autoStartGrabAfterBg）await ReleaseAsync → btnLiveGrab_Click（IO 觸發自動回抓）→ return
 └ 尾端自動預覽：btnLiveViewBackground_Click（直呼）
按鈕可用性：
UpdateStandardBgSubLockState@AniloxRollForm.Background.cs
 ├ 輸入＝相機已就緒＋光源已就緒＋目前未 Grab
 ├ 由 OnCamerasHwReady 與光源狀態轉變共同觸發，不依賴兩者誰先完成
 └ 狀態轉變記 `background capture ready=...`
時間設定不變量：`BackgroundSampleSeconds` 只管本段背景採樣；`GrabLimitSeconds` 只在 F2 正式監控啟動成功後，
依 `CaptureStopCondition` 作為時間模式停止值或 IO 模式安全上限，兩者不得互相中止。高度模式不武裝此 timer。
PropertyGrid 分別顯示為檢測設定的 `背景採樣(秒)`，以及「畫布設定」下的
`停止條件`／`總時間(秒)`／`總高度`。
預覽背景：
btnLiveViewBackground_Click@AniloxRollForm.Background.cs     intent 行 ui:【預覽背景】鈕
 ├（IsBgPreviewActive）ClearBackgroundPreview → return       ← 再按一次＝清除（toggle）
 ├ EnterBackgroundPreview@LiveDisplayCoordinator.cs（LCM forwarder 經過）
 │   └＝靜音鍵 _bgPreviewOverride=true → ApplyMainDisplayMode()   ⚠ 只改狀態→呼閘門，不自建/拆 view
 │       閘門 BgPreview 分支：DisableWaterfall＋EnsureImageDisplay＋ApplyBgPreviewLayout
 │                            （合圖未啟用→用設定 start/ops 餵佈局）
 ├ ClearLiveRowChartForBackgroundPreview@AniloxRollForm.Live.cs
 │   → 清列圖表、待上畫資料及列曲線快取；預覽期間 MainContentPresented 不得重新上畫列曲線
 ├ ReadManifest → per-cam ResolvePreviewProfilePath＋LoadProfile@BackgroundProfileRepository.cs
 │   （同一 active version；manifest 損壞不得 fallback 到 legacy bin）
 │   → ExpandColMeanToGray@AniloxRollForm.Live.cs
 │   → PushStaticFrame@LiveDisplayCoordinator.cs（與 grab 幀同一條 PushFrame 路＝合圖/縮圖/縮放/overlay 全免費）
 └（pushed==0）ExitBackgroundPreview＋MessageBox
清除：ClearBackgroundPreview@AniloxRollForm.Background.cs＝ExitBackgroundPreview
 └ Exit＝靜音鍵 off → ClearFrame×N → ApplyMainDisplayMode()（回設定模式；WF 設定則重建瀑布）
不變量（S 系列同源）：
 - 顯示狀態＝f(he_MainDisplay 設定, 靜音鍵)，唯一計算點＝ApplyMainDisplayMode 閘門；
   閘門以外任何路徑自建/拆 view＝退件（「平行通道」是 2026-07-09 前東漏西漏的病根）
 - 「存活」policy：預覽中改設定/相機數變化/合圖重建 → 呼閘門仍得預覽畫面（設定記著，Exit 才生效）
 - 預覽狀態唯一真相＝coordinator `IsBgPreviewActive`（form 唯讀轉發，不自存旗標）
 - 相機殘幀 gate：OnCameraDisplayFrame 首行 `_bgPreviewOverride → return`（取得背景自動停後晚到幀不進 view）
```

**log-flow（預覽背景）**
```
T1: ui:【預覽背景】鈕
T1: ApplyMainDisplayMode → BgPreview（靜音鍵，設定不動）    ← 閘門分支（WF 設定時前面多一行 TeardownWaterfall）
T1: EnterBackgroundPreview（view=True merge=… mode=…）
T1: background preview rowChart clear
T1: bgPreview push camN WxH（view=True）× 有 bin 的台數
再按/開始抓取：ExitBackgroundPreview → ApplyMainDisplayMode → {Waterfall|ImageCanvas}（回設定模式）
```
預覽期間列圖表必須保持空白；`EnterBackgroundPreview` 後不得出現
`rowCurve present after=mainImage`，直到 `ExitBackgroundPreview`。

**log-flow（去背演算法實際生效）**
```
設定／啟動載入：
background bind camN mode=standard source=bg_...bin status=ready width=W samples=W min=A max=B mean=C
或
background bind camN mode=single source=per-frame status=ready
離線相機：
background bind camN mode=standard source=none status=skipped reason=offline

取得背景非產品採樣：
background capture ready=True camReady=True lightReady=True grabbing=False
background capture begin output=disabled
background capture waiting first-set timeoutMs=N
capture first-set ready ... aligned=True
background capture sampling start duration=Ns
...
background capture sampling complete durationMs=M frames=cam1:A,cam2:B,...
background capture end output=disabled result=ok|failed

每次正式 Grab 第一個成功 native 幀（每台恰一行）：
background apply camN grab=G mode=standard source=precomputed width=W
或
background apply camN grab=G mode=single source=per-frame width=W
```
- `bind` 證明背景檔成功解析並綁定；`apply` 證明 `TanukiPipeline_Process` 成功返回且實際收到該來源，兩者都要有。
- 標準模式出現 `source=per-frame`、單張模式出現 `source=precomputed`，或已完成 Grab 的 `apply` 台數少於
  `capture gate open cams=N`，均為契約違規。
- 載入失敗行為：
  `background bind camN mode=standard ... status=failed reason=... retained=True|False`。
  `retained=True` 表示保留上一份已驗證背景；`False` 時正式 Grab 必須出現
  `capture start blocked reason=standard-background-not-ready`，不得開 capture gate。
- `background capture begin` 到 `end` 之間不得出現 `CaptureWriteFailure`，且 begin/end 必須成對。
- 成功採樣必須有 `sampling start/complete`，且 `durationMs >= duration*1000`；Stop 後仍在途的 callback
  以進入 callback 當下的 `captureSuppressed` 快照為準，不得在旗標恢復後補寫產品檔。
- `tools/python/check_all_flows.py` 的 `LIVE/F8.background-subtraction` 與
  `LIVE/F8.background-capture`，以及 `LIVE/F2.time-origin` 自動檢查上述關係。

## 硬體連線契約（H 系列）——邊緣觸發（同 MURA 模式：轉變才記，不洗版）

### H1 IO / 光源 / 儲存電腦 連線轉變

啟動時 `啟用 IO=否` 不建立空的 `IoConnectionCoordinator`；之後第一次啟用才建立
乾淨的 controller generation。耐久測試每 2 秒讀取實際 label 文字，不能只驗
handle 存活；IO、儲存電腦與光源任一文字離開綠燈即失敗。

`virtual-io-recovery` 使用 `IoBridge.IoSimulator --cycles 0` 單獨驗證 H1：主程式先啟動、
server 後上線仍須完成安全交握；server 正常退出須發布一次斷線；server 重啟後不重開
主程式即恢復。DI-1 全程 Low，因此此情境不得出現 IO START 或 Grab 請求。
情境必須先明確停用 IO、清除舊證據，再設定 endpoint 並重新啟用；不可假設執行前 JSON
與測試值不同，否則相同值不會觸發 controller restart。
```
Tn: ⚠ IO 斷線 ／ IO 恢復連線            ← 光源/儲存分享 同格式
Tn: ⚠ IO 未連線（開機基線）             ← 首次觀測就不在線（拔線開機/初始化未完，恢復行會跟著出現）
Tn: 儲存程式 heartbeat 恢復 pid=N age=Ns
Tn: ⚠ 儲存程式 heartbeat 未回報 reason=…
T1: IO controller start generation=N endpoint=IP:Port
IO disabled by settings: wait for the old controller generation to stop, then show the disabled state
and recompute the manual Grab/background gates. The Grab button must not remain disabled as
`IO controlling` after the controller has been removed.
穩態（DVT 每 30 秒）: IO poll state attempts=N successes=N snapshots=N connected=True state=Idle
設定變更：IO controller stop generation=N reason=settings
        → IO controller start generation=N+1 endpoint=IP:Port
快速連改：IO controller restart coalesced generation=N（可有；該代不得再 start）
關閉：IO controller stop generation=N reason=shutdown
DI START：io:DI START 上升緣 → 抓取請求
       → （接受時）io:DI START 上升緣 → 開始抓取
       → IO grab accepted busy=on …｜IO grab rejected busy=off reason=…（每個請求恰一）
啟動途中失效：capture start cancelled before gate reason=io-request-invalid
             → StopGrab → IO grab rejected busy=off reason=…
```
**IO code-flow（連線生命週期與產品擷取分層）**
```
InitIoController／HandleIoSettingsChanged@AniloxRollForm.IoControl.cs
 → StartAsync／RestartAsync@IoConnectionCoordinator.cs
    ├ requested generation 立即使舊 callback 失效
    ├ lifecycle gate 序列化 StopAsync／Dispose 舊 controller
    ├ 快速連改只建立最後 generation
    └ 建立 IoGrabController → StartAsync(IP, Port)
IoGrabController events
 → IoConnectionCoordinator.IsCurrent（舊 generation 截止）
 → OnIoController*Requested／DispatchCurrentIoController@AniloxRollForm.IoControl.cs
     ├ Start／Stop → Form 的 request generation＋transition gate＋既有 Grab FSM
     ├ State／Connection → UI 呈現
     └ IoUpdated → 原子替換最新快照
        → TelemetryTimer_Tick@AniloxRollForm.Telemetry.cs
        → ApplyPendingIoSnapshot（只套用最新狀態，不為每次 500ms poll 建立 BeginInvoke）
關閉程式
 → ShutdownIoControllerAsync@AniloxRollForm.IoControl.cs
 → ShutdownAsync@IoConnectionCoordinator.cs
 → generation 失效 → StopAsync／Dispose
```
- `IoConnectionCoordinator` 只擁有 controller 建立、替換、關閉、active generation 與 lifecycle gate；
  不得擁有 GrabId、相機準備、停止條件、BUSY／MURA 等產品政策。
- 耐久測試不能只看綠燈；`IO poll state` 的 attempts／successes／snapshots 必須持續增加且相等，
  才能證明背景 polling 與 UI snapshot 鏈仍在運作。
- Modbus 穩態讀寫保留 `Task` API，但 `IcpDasModbusTcpClient.SendAndReceive` 必須在 worker task
  使用同步 `Socket.Send/Receive` 與 socket timeout；禁止回到每次 poll 建立
  `NetworkStream.ReadAsync/WriteAsync`，否則 .NET Framework 會隨輪詢累積 Event／Thread handle。
- 光源定期健康探測由 `LightConnectionCoordinator` 的單一 `LightProbe` worker 串行執行；
  不得每 2 秒新建 ThreadPool `Task`，否則大型 MIL 程序長時間未 GC 時會累積 Event handle。
- Form 只保留 IO 擷取 request generation 與 transition gate；controller lifecycle 欄位不得在 Form
  另存第二份。

**儲存 code-flow（觀測與呈現分層）**
```
TelemetryTimer_Tick
 → UpdateConnectionStatusLabels@AniloxRollForm.HardwareStatus.cs
 → Tick@StorageHealthCoordinator.cs
    ├ RefreshLocalCapacity → DriveInfo（本機容量快照）
    └（Inspection，每 2 秒）ProbeStorageTransportReachable(TCP 445)
       → RemoteCopyService.ProbeRemoteWritable（分享實際寫入探針）
       → StorageAppHeartbeatService.TryRead（程式存活＋遠端容量）
       → StorageHealthSnapshot
 → UpdateStorageConnLabel／RefreshCapacityInfoLabel@AniloxRollForm.StorageStatus.cs
 → FlowHardwareEdges@AniloxRollForm.HardwareStatus.cs
```
- `StorageHealthCoordinator` 只擁有觀測狀態與探測節拍；循環刪檔仍由 `StorageRetentionService`、
  遠端傳輸仍由 `RemoteCopyService` 負責。Form 不得另存第二份容量、分享或 heartbeat 狀態。

- **IO Start 交界狀態表**（Form 與 `IoGrabController` 之間的產品 gate；完整轉移不得拆成零散 guard）：

| 目前狀態 | 事件／條件 | 下一狀態 | 動作 |
|---|---|---|---|
| Idle | START 上升緣，controller 為目前代且已連線 | Starting | 建立 request generation，取得 IO Grab transition gate |
| Starting | START 下降、IO 斷線或 controller 換代 | CancelPending | 立即使 request generation 失效；Stop 等待同一 transition gate |
| Starting | 相機準備完成，且開產品 gate 前 request 仍有效、IO 仍為 Running | Capturing | 建 GrabId／capture plan → 開 capture gate → `NotifyGrabStarted` |
| Starting／CancelPending | 開產品 gate 前發現 request 已失效 | Idle／CommLost | capture gate 維持關閉；若相機已進 StartGrab 則 rollback StopGrab；記一筆 rejected |
| Capturing（IO） | START 下降 | Idle | 取得同一 transition gate → drain／StopGrab → `NotifyGrabStopped` |
| Capturing（IO） | IO 斷線／PLC Alive 遺失 | CommLost／Faulted | 取得同一 transition gate → 不等尾幀直接 StopGrab；BUSY Low |
| Capturing（時間／高度） | START 下降或 IO 斷線／PLC Alive 遺失 | Capturing | IO FSM 可進 Idle／CommLost／Faulted，但產品 capture gate 保持開啟，直到時間／高度目標完成 |
| Capturing（時間／高度） | START 下降 | Capturing | 不停止；固定時間／共同完成列數是本輪唯一停止 owner |
| Capturing（時間／高度） | 固定目標完成且 START High | AwaitingStartLow | StopGrab → `NotifyFixedGrabCompleted`；BUSY Low，禁止同一段 High 重啟 |
| Capturing（時間／高度） | 固定目標完成且 START Low | Idle | StopGrab → `NotifyFixedGrabCompleted` |
| AwaitingStartLow | START 下降 | Idle | 解除本次 High 的消耗狀態，下一個上升緣才可開始 |
| AwaitingStartLow | IO 斷線 | CommLost | 沒有產品 capture；BUSY Low，等待連線安全交握恢復 |

- **單 process／單 controller**：`Program` named mutex 必須在 Form 建立前擋掉同機第二份程式；同一 session
  任一時刻只能有一個 active generation。restart 以 lifecycle gate 序列化，舊 generation callback 不得進 UI/Grab。
- **Start／Stop 不得交錯 Toggle**：IO Start、下降緣 Stop、斷線 Stop 共用一個 transition gate；下降緣、斷線與
  controller 換代必須先讓在途 request generation 失效。`ToggleLiveGrabAsync` 在建立 GrabId／capture plan／開產品
  gate 前必須再次驗證 request；禁止 Stop 先看見 `IsLiveGrabbing=false` 返回、Start 稍後才留下 gate-closed 半啟動；
  也禁止下降緣 Stop 與斷線 Stop 同時進入共用 Toggle，造成第一個關閉、第二個反向重開。
- **BUSY 代表事實，不代表請求**：DI 上升緣只提出 intent；必須等共用 `ToggleLiveGrabAsync` 成功且 capture gate
  已開啟，才能 `NotifyGrabStarted` 拉高 PC BUSY。CLProtocol 未就緒／相位同步失敗時回
  `IO grab rejected busy=off`，FSM 回 Idle；禁止「沒抓到但 PLC 看見 BUSY」。
- IO 重連倒數以 `IoGrabController.NextReconnectAtUtc` 為唯一來源，顯示到 `0s` 代表正在嘗試連線；
  不得在 `1s` 後退回沒有秒數的空白狀態。`ReconnectIntervalMs` 是 connect 嘗試起點間隔，
  TCP timeout 必須包含在週期內，不得 timeout 後再重複等待完整週期。
- **IO 恢復連線的定義＝TCP + 安全交握全過**：`ReconnectTick → ConnectAsync → TryAcceptConnectedModule`
  必須完成 `EnterIdle`（DO1=0 → DO2=0 → DO0=1，ALIVE 最後發布）及一次合法 `ReadDiStatuses`，之後才可
  `OnConnectionChanged(true)`；任一步失敗須 Dispose 並維持 Disconnected/CommLost，禁止假綠。
- **連線狀態 SSoT**：業務層一律讀 `IoGrabController.IsConnected`（accepted gate）；TCP 已接上但交握未過時，
  `NotifyGrabStarted/Stopped` 與 `NotifyMuraDetected/ClearMura` 必須靜默拒絕，不得用 `_plc.IsConnected` 繞過。
- **逾時收口**：`SendAndReceive` 以 socket `SendTimeout/ReceiveTimeout` 將
  `TimedOut/WouldBlock` 轉為 `TimeoutException`，外層關閉 transport；Connect timeout
  關 socket 後等待 SAEA completion 再釋放 args。全天 crash log 不得新增來源為 ConnectAsync/Socket 的
  `UnobservedTaskException`。
  Connect 必須走 `SocketAsyncEventArgs`；debugger 不得再出現晚到 `TcpClient.EndConnect` 的
  `ObjectDisposedException/NullReferenceException` first-chance 例外。
- **冷開機恢復**：app 先開、IO 後上電不得要求重開 app；Bridge log 第 1 次及每 10 次失敗記
  `IO reconnect pending: attempt N, {TCP unavailable|handshake rejected}`，成功行必帶 attempt。
- **儲存電腦綠燈＝兩層都通**：第一層為 TCP 445 + `RemotePath` 建立/寫入/flush/刪除唯一探針檔；
  第二層為 `RemoteConfigPath\storage-app-heartbeat.json` 的 `LastSeenUtc` 不超過 15 秒。分享不可用顯示紅色；
  分享可寫但 heartbeat 缺少/過期顯示黃色；兩層都通才綠。曾成功讀取且最後有效 `LastSeenUtc`
  尚未超過 15 秒時，SMB 原子替換窗口的一次性讀檔失敗不得讓綠燈閃黃；超過 15 秒仍讀不到才算未回報。
  TCP 445 timeout 使用 non-blocking socket + `Socket.Select`，不得為每次 2 秒探測建立
  `AsyncWaitHandle`；長時間 DVT 的總 handle 不得隨探測次數單調增加。
  斷線時每 2 秒重試，app 先開、儲存機後上電
  不得要求重開 app。
- **光源關閉收尾**：關閉程式前，Light coordinator 必須等待已排隊的 COM 探測工作收尾；
  不得讓 `SerialStream` 在程序 finalizer 階段才釋放或產生背景 fatal。
- **儲存程式自舉**：Release 根目錄 `setup.bat` 自動選擇完整安裝或更新；兩路都必須移除 payload 下載封鎖標記，並以
  `BUILTIN\Administrators` 群組主體安裝「任一使用者登入」＋「每分鐘保活」雙觸發排程；
  `RdpUser` 只供遠端登入，不得綁死排程。程式存活時 `MultipleInstances=IgnoreNew` 必須讓保活觸發靜默略過；
  正常關閉或異常退出後，最晚一分鐘內必須重新啟動。
  安裝當下必須由目前登入的系統管理員立即啟動，並在 10 秒內確認正確 EXE process 存活；
  `LastTaskResult=267011 (0x00041303)` 代表尚未執行，必須判定失敗而非歸因 DLL 封鎖。Storage role 每 5 秒原子發布
  heartbeat（PID、啟動/回報時間、磁碟與最後清理結果）。SMB 可寫只證明分享，不得當成程式存活。
  Release 根目錄 `test_storage_restart.bat` 是保活 DVT 唯一入口：在儲存電腦上強制結束指定
  `AppDir` 的程序後，禁止測試工具自行啟動 app；每輪必須由排程在 90 秒內產生不同 PID，且該 PID
  在 15 秒 freshness 規則內發布 heartbeat。工具必須接受產品 `JavaScriptSerializer` 寫出的
  `\/Date(milliseconds)\/` UTC 日期格式，也可相容 ISO 8601。預設三輪，報告寫入
  `D:\Anilox\Logs\DvtReports`，無論成功或失敗，測試收尾都必須讓儲存程式維持運行。
- **儲存資料捷徑**：檢測電腦 `setup.bat` 必須冪等在 Public Desktop 建立
  `Anilox 儲存資料.lnk`，目標由 `\\VerifyPingTarget\StorageShareName` 計算，不得額外寫死 `192.168.10.20`。
- **程式桌面捷徑**：Storage 與 Inspection 的安裝及更新都必須冪等建立 Public Desktop
  `PICoater AOI.lnk`；TargetPath＝`AppDir\AniloxRoll.Monitor.exe`、WorkingDirectory＝`AppDir`、圖示＝同一 EXE，
  不得寫死磁碟位置。捷徑存在但目標過時必須覆寫修正。
- **遠端複製不丟資料**：`EnqueueFiles` 必須先在本機 `.remote-copy-pending` 持久化標記才進 worker；
  複製失敗保持 pending 並退避重試，程式重開從標記復原。禁止恢復「重試固定次數後丟棄」語意。
- **發布原子性**：遠端先寫同目錄 `.part-*`，確認來源前後長度穩定且遠端長度一致後，再原子
  move/replace 成正式檔名；正式發布且 pending 標記成功刪除後才算完成。
- **封裝完成才遠傳**：同一 Grab 的相機 callback 與背景 append 全部 drain，三張 preview atlas 寫入成功後，
  才把固定不再變動的 `.acap` 與每日 CSV 排入 durable remote-copy。
- **Retention 以可持續寫入優先**：空間低於預留值時，檢測與儲存電腦都刪除「最舊的完整一天」；
  今天的資料不得刪。刪除集合＝日期資料夾內全部 `.acap`/legacy assets/`_curve_summary`
  ＋月份層同日期 `yyyyMMdd.csv`；任一低空間清理成功後，非 active 的舊背景版本也列入候選。若該日仍有 pending 遠端檔案，必須先取消並移除
  對應 durable marker，再刪檔，禁止 worker 對已刪來源永久重試。刪到未送達資料須留下深橘狀態與 log。
- **光源釋放**：SerialPort ownership 必須先在 lock 內從 `_port` 移除，再對 detached instance 單次 Dispose；
  全天 crash log 不得新增 `SerialStream.Finalize → ObjectDisposedException（已關閉安全控制代碼）`。
- **光源重連防呆**：開機首次偵測 `AutoDetect` 全 port；離線後每 2 秒先 `TryConnect` 設定 COM，
  每 5 次失敗（約 10 秒）必須 `AutoDetect` 全 port 一次。找到不同 COM 要回寫 SSoT；禁止移除全掃描造成
  工廠現場無法自救，也禁止每輪全掃描造成 SerialPort handle churn。
- **光源 code-flow**：
  `InitLightController@AniloxRollForm.HardwareStatus.cs`
  → `Start@LightConnectionCoordinator.cs`
  → 背景 `AutoDetect@LightController.cs`；
  `TelemetryTimer_Tick@AniloxRollForm.Telemetry.cs`
  → `UpdateConnectionStatusLabels`
  → `Tick@LightConnectionCoordinator.cs`
  → 每 2 秒背景 `Probe`／`TryConnect`／第 5 次 `AutoDetect`。
  coordinator 的 `StateChanged` 只回 Form 執行 `UpdateLightConnLabel`，
  `ActivePortChanged` 只經 `SettingsHub.SetBatch` 回寫 COM；
  `LightTurnOn/LightTurnOff` 只轉送 coordinator，Form 不得再持有或替換 `LightController`。
- 光源停用（LightEnabled=false）/ 遠端路徑空 → 該項不觀測（靜默合法）。
- 開機常見「未連線（開機基線）→ 恢復連線」＝平行初始化的正常時序，非異常。

**儲存電腦 trace 判讀**：
```
[RemoteCopy] remote share unavailable: ...             ← TCP/路徑/寫入任一層未通
[RemoteCopy] remote share accepted (write verified)    ← 實際寫入交握通過
[RemoteCopy] pending queued added=N queue=N bytes=N     ← 本機 durable marker 已落盤；斷線期間待傳正式成立
[RemoteCopy] retry pending attempt=N queue=N file=...  ← 保留待傳；第 1 次及每 10 次留痕
[RemoteCopy] restored pending queue count=N            ← 程式重開復原
[RemoteCopy] backlog drained: copied=N bytes=N          ← 斷線積壓清空
```
狀態轉換：`未排程 --持久標記成功--> 待傳 --複製失敗/重開--> 待傳
--長度驗證+原子發布+刪標記--> 完成`。任何失敗不得進完成態。

**H1/C3 軟體斷線 DVT**：`physical-smb-backlog-recovery` 以固定 `/32` Loopback
黑洞路由只隔離儲存電腦 `192.168.10.20`。IO 與儲存共用實體 NIC，禁止停用整張網卡；
情境仍把 IO 切到本機模擬器，使 START 循環不依賴外部控制器。
阻斷期間必須完成至少兩輪本機封裝，
看見 `remote share unavailable` 與 `pending queued`；移除黑洞路由後必須依序出現
`remote share accepted`、`backlog drained`，且 heartbeat 恢復。Windows UNC 呼叫可能在
網路中斷期間阻塞，恢復後直接成功，因此 `retry pending` 是診斷證據而非必要成功條件。`check_all_flows.py`
的 `H1.remote-copy-recovery` 驗證上述順序。安裝器與 Runner 都必須以新的 TCP 連線量測
SMB `:445` 確實由可達→不可達→可達，不能以路由存在冒充故障已生效。Runner 在成功、
失敗與中止時都必須移除自己建立的黑洞路由。此測試適合反覆執行；最終版本仍需另做一次實體拔線，以涵蓋網卡、
交換器與線材，不得把軟體阻斷冒充實體接線證據。

**H1/H.Light 軟體故障注入 DVT**：`physical-bridge-recovery` 以暫時 `/32` Loopback 黑洞路由
阻斷實體 IO `192.168.255.1`，並以 PnP 暫停光源 `COM17`；兩者各三輪。檢測電腦的
IO `192.168.255.x` 與儲存 `192.168.10.x` 共用實體 NIC，不得停用整張網卡。Windows
Firewall profile 也可能被停用，不能以 Firewall rule 存在冒充已阻斷。黑洞路由固定送至
本機 Loopback（ifIndex 1／next hop `0.0.0.0`），不得假設區網內某個未用 IP 永遠不可達。
每輪加入黑洞路由後必須在 PropertyGrid 將 `啟用 IO` 切成 `否 → 是`，強迫新的
controller generation 在阻斷狀態下重新連線；安裝器與 Runner 都必須量測新的 TCP
確實失敗，解除路由後再量測恢復。
光源因 Windows 不允許停用仍由主程式持有的串口，每輪須先在 PropertyGrid 將
`啟用光源` 切成 `否` 釋放 COM17，再停用 PnP 裝置，接著切回 `是`，讓新
controller 在裝置停用狀態下探測失敗；恢復 COM17 後才驗證自動重連。
每輪必須嚴格完成 `⚠ {IO|光源} 斷線 → OutputHealth raise →
{IO|光源} 恢復連線 → OutputHealth resolve`，最後 IO 回待機且正常關閉。
第一次由 `tests/InstallDvtAdminActions.bat` 經 UAC 安裝六個固定白名單排程動作；
後續 Runner 本身仍以一般權限執行，只能要求封鎖／解除固定 IO、儲存端點及停用／啟用
固定 COM17，不得建立可執行任意 repo script 的永久提升入口。
Runner 在成功、失敗及中止時，都必須移除自己建立的黑洞路由並重新啟用 PnP
裝置。此情境驗證產品與 Windows 驅動之間的失聯恢復，不代表線材、電源或硬體重新上電
已覆蓋；正式交付仍各需一次實體拔線／斷電證據。

### H2 相機在線數轉變
```
T1: ⚠ 相機離線 4→3/7 ／ 相機在線 0→4/7   ← 數量變化才記（開機 0→N＝配置完成）
```
- 由來 2026-07-07：使用者拔 IO+相機測試，flow log 完全靜默＝硬體事件盲區——現場排障最需要的
  「什麼時候斷的、斷了多久、有沒有回來」從此有記錄。
- 判讀：斷線行之後的顯示異常（黑縮圖/幀停）歸硬體事件管，不是接線 bug。

## 相機參數契約（P 系列）

### P1 滑桿/數字框調參（停止時三種皆可；Grab 中只開放曝光）
```
T1: ui:【相機參數】camN {param}={v}｜All {param}={v}    ← 帶參數名+值單行自足（Exp/LineRate/Height…）
（之後的 HtRealloc/合圖佈局重算等程式化行歸此 intent 管；滑桿拖曳 vs 數字框輸入同一路徑，log 不區分）
停止 Grab 時可調曝光、線掃速度與擷取高度；Grab 中只開放曝光。線掃速度與擷取高度的單台／All
控制項必須停用，Form command 與 LiveCameraManager setter 也必須拒絕繞過 UI 的寫入。
拒絕時只留 `parameter change blocked scope={scope} param={param} reason=GrabActive`，不得寫硬體、
不得產生使用者 intent 或進入 parameter reconfigure。
開機初始化控制項期間 `_cameraParameterControlsReady=false`，不得排程硬體寫入、不得出現上述 `ui:` 行，
也不得建立 `paramchange-*.csv`。初始硬體值的權威路徑是 Allocate/Initialize 套用 settings，及 CLProtocol
就緒後重套線掃；`paramchange` 只記使用者完成的實際調整。
禁止：調參數不得出現任何 MIL 視窗——headless 鐵則：**每一個 MdispSelectWindow 呼叫點都必須帶
`_panelHandle != IntPtr.Zero` 守門**（MIL 對 Zero handle 會自開獨立浮動視窗；2026-07-07 實例：
改高度 realloc 路徑漏守門 → 4 台各跳一個視窗）。新增 MdispSelectWindow 呼叫點＝必帶守門。
```

**自動校稿（`flow_checks/parameter.py`）**：
- `P1.startup`：從配置開始至 `AllocateCameras done` 後 1 秒為開機靜默窗口，不得出現相機參數 intent。
  多留 1 秒是為了抓「初始化時誤排 debounce、配置完成後才發作」的歷史病。
- `P1.intent`：只接受 `cam1~7 Exp|LineRate|Height=N` 或
  `All ExpAll|LineRateAll|HeightAll=N`，scope 與參數尾綴必須一致。
- `P1.live-policy`：capture gate 開啟期間只允許 `Exp|ExpAll` intent；任何
  `LineRate|LineRateAll|Height|HeightAll` intent 直接 FAIL。後端正確拒絕的 `parameter change blocked`
  算已覆蓋且不算違規。
- `P1.responsiveness`：調參後 5 秒內 `UiStall > 1000ms` 判 FAIL；沒有使用者調參則回
  `NOT COVERED`，開機靜默仍可獨立判 PASS/FAIL。
- `P1.synchronization`：只驗 Grab 中曝光調整。每個 intent 必須完成以下同 scope 快速套用；
  不得關 capture gate、不得 stop/start digitizer、不得重設顯示世代或重跑相位同步。套用失敗、
  超過 5 秒或出現任何 `parameter reconfigure`／`acquisition sync reason=parameter` 都判 FAIL。
  調參途中若使用者 StopGrab，曝光寫入可完成，但 terminal 必須如實記 `gate=closed`。

**Grab 中曝光 log-flow（單台與 All 共用）**：
```
T1: ui:【相機參數】{scope} {param}={v}
T1: exposure live apply begin scope={scope} gate=open
Tbg: （實際曝光寫入）
T1: exposure live apply complete scope={scope} gate=open elapsedMs=N
（禁止 capture gate closed／parameter reconfigure／parameter sequence reset／firstFrame）
```

**調參途中 StopGrab log-flow（抓取上限與手動停止共用）**：
```
T1: exposure live apply begin scope={scope} gate=open
T1: StopGrab
T1: capture gate closed standby=on
T1: exposure live apply complete scope={scope} gate=closed elapsedMs=N
```

**code-flow 與不變量**：
```
TrackBar/NUD settle
 → ApplyCamParamAsync｜ApplyAllCamParamAsync@AniloxRollForm.SettingsTabs.cs
   ├（Grab 中非曝光）parameter change blocked → return
   ├（Grab 中曝光）SetParamControlsLocked(true)＋SetCaptureSuppressed(true)
   └（Grab 中曝光）await ApplyExposureFastAsync@LiveCameraManager.cs
       ├ _allocationGate 序列化實際硬體寫入
       ├ Task.Run：寫曝光；不碰 line timing
       └ finally：恢復存檔、立即解鎖控制項
UpdateGrabButton@AniloxRollForm.Live.cs
 └ RefreshCameraParameterControlState
    ├ 停止：曝光／線掃／高度皆 Enabled
    └ Grab：只有曝光 Enabled；線掃／高度（單台＋All）Disabled
```
- **曝光不重啟原則**：曝光只改 integration time，不改 Line Rate／幀高／資料節奏；因此沿用目前
  acquisition generation 才是最小副作用。為曝光去 stop/start 反而會製造多秒空窗與相位重建風險。
- **存檔保護**：`SuppressCapture` 只在硬體曝光寫入期間暫停落盤；GPU、主畫面與曲線仍持續，
  capture gate 全程保持 open。若使用者同時 StopGrab，Stop 的 gate 狀態優先。
- **UI 執行緒零 MIL**：實際曝光寫入在背景執行，UI 不固定睡秒數、不輪詢 `MdigInquire`。

## Mura 警告契約（M 系列）

### M1 曲線超過門檻（grab 中）
```
Tn: ⚠ MURA 超標（v|h）mean=…/max=…（thr …/…，IO已連線|未連線→僅畫面警告）   ← 邊緣觸發（進入超標一行）
Tn: MURA 恢復（v|h）                                                        ← 離開超標一行
暫停：ui:【暫停Mura檢測】鈕 → set:[MuraDetectPaused]=True → MURA 暫停 → 清除 DO1
```
- **畫面警告與 IO 解耦**：lblIoDoMura 超標一律亮（無 IO 硬體也要看得到）；
  DO 輸出（給 Nakan）才看 IO 連線。暫停 Mura 檢測（MuraDetectPaused）期間兩者皆不動。
- **亮燈時序＝閂鎖（latch）非脈衝**（既有 DO_MURA 規範，io_diagrams+FSM 唯一來源）：
  亮到「該次檢測結束」——清除時機＝grab 停止（=FSM 回 Idle 的無 IO 等價）/ ClearMura / 新一輪 grab 啟動歸零。
  ⚠ 勿發明固定秒數（2026-07-07 曾誤做 3 秒被使用者抓包＝違反既有時序規範的實例）。
- **硬體 DO 閂鎖必須同步清**：手動 grab 停止/啟動時呼 ClearMura（手動流程不經 FSM，不清則
  DO 永遠掛著 → Nakan 誤報 + IO 暫停→恢復後 snapshot 讀回殘留 latch、燈「自己亮」——盲測輪3實例）。
- **IO 暫停＝視同離線**：暫停中超標不發 DO（僅畫面警告），log 標「IO暫停中→僅畫面警告」三態之一。
- 超標期間不洗版（狀態轉變才記）；每輪 grab 啟動重置邊緣狀態。
- **唯一輸出路**：`OnLiveCurveData|OnLiveRowCurveData → CheckLiveMura → WarnMuraVisual + NotifyMuraDetected`；
  `OnCameraInspectionResult` 只負責 CSV/遠端存放，不得直接寫 Mura DO。暫停切換由
  `HandleMuraPauseSettingsChanged` 重設兩方向 edge latch 並立即 `ClearMura`，恢復後只接受新的超標事件。
- 欄方向 `CheckLiveMura("v")` 必須和報表 O/X 共用 `ThresholdContext.EvaluateColumnFailureCause`；
  `欄曲線判定=顯示平均|顯示最大|顯示兩者` 分別只啟用 Mean、只啟用 Max、或任一超標。
  列方向維持 Mean/Max 任一超標，不受欄選項影響。
- 違規樣本：chart 明顯超標卻無「MURA 超標」行＝判定鏈斷（2026-07-07 盲測抓到：舊版被
  IO 未連線 early-return 整段跳過＝操作員零警告）。

**自動校稿（`flow_checks/mura.py`）**：
- `M1.edges`：v/h 各自必須超標→恢復交替；Start/Stop grab 或暫停切換會重置 edge latch。
- `M1.health`：每筆超標/恢復後 1 秒內，必有同方向且未被前一事件使用的
  `OutputHealth raise/resolve`；整份 session 完全沒有此儀器的舊版 log 回 `NOT COVERED`。
- `M1.pause`：每次按暫停鈕 3 秒內必有 `set:[MuraDetectPaused]`；切到 True 後 3 秒內
  必有 `MURA 暫停 → 清除 DO1`。未操作的子流程回 `NOT COVERED`。

## 資料存放與檢測契約（C 系列；capture/storage）

### C1 抓取存放計畫（開始 grab 後一次）
```
T1: capture plan grab={yyMMdd-HHmmss} root={CaptureRootPath}
    imageDir={root}\yyyy\yyyyMM\yyyyMMdd
    csv={root}\yyyy\yyyyMM\yyyyMMdd.csv
    archive={grabId}.acap
    assets=raw|proc_c|proc_r|hessian_c|hessian_r|mean_c|max_c|mean_r|max_r
    preview=1920x1080x3
    scale={DefaultSaveResizeScale}
    hessianScale={DefaultHessianStandardMapScale}
```
- `proc_c/proc_r` 是相容舊資料與快速預覽的 U8 JPEG；`hessian_c/hessian_r` 是正規化前的
  binary16 標準圖（HSM1，無損壓縮）。兩者不得互相冒充。
- 新資料的標準圖固定取相機輸入 `/25`，縮小採區塊最大值以保留細線尖峰；改欄／列正規值時只從 HSM1 重新映射顯示與熱力圖，
  不得重新執行 Hessian。舊資料缺 HSM1 時才回退到 `proc_c/proc_r`。
- `CaptureRootPath` 預設為 `D:\Anilox\Captures`；遠端預設為
  `\\192.168.10.20\Anilox\Captures`。舊 `Captures_pack` 預設在 JSON 載入時自動升級。
- `imageDir`、`csv` 與 archive path 必須由 `CaptureStoragePaths` 推導。
- 新寫端只產生每 Grab 一個 `.acap`；九種獨立 asset 與 frame tick 都在 record 內。舊散檔名稱只保留讀取相容。
- 這行是每輪 grab 的「存放方式/位置」摘要；逐幀大小與資源量仍歸 `resource-monitor-*.csv`，不得用 `[Flow]` 洗版。

### C2 檢測 CSV 寫入（每個 grab 首筆 + CFG 變更）
```
Tn: capture csv open path=… cfg=yes|no              ← 新檔或換日首次開啟
Tn: capture csv cfg path=… speed=N lr=N HM=V/H ridge=N thrV=mean/max thrH=mean/max
Tn: capture csv firstRecord grab=… path=… file=… verdict=max0|1/mean0|1 peak=…/… rowPeak=…/… maxCMean=… thrV=…/…
Tn: capture csv curveSummary grab=… cams=N hm=… source=merged-saved-frames path=…
Tn: capture layout final grab=… ops=… start=… speed=N head=H tail=T path=…
    ← Stop 後每個 grab 恰一行；機台布局以停止前最後值為準
```
- `firstRecord` 每個 grab 只出一行，用來確認檢測結果有落到哪一份 CSV；逐相機逐幀細節看 CSV 本體。
- `cfg` 行出現代表 `#CFG` 已寫入同一 CSV；`ridge` 是捕捉時的細線濾除值。
- `#CFG` 的機台佈局必須完整保存 `OPS + START(CamN_Pos) + CROP(TrimHead/TrimTail)`；
  列實體尺度必須保存 `AniloxRollSpeedMPerMin + CamN_Lr`（總高由 row bin 點數推導，不另存衍生尺寸）；
  檢測設定必須包含欄／列正規值、細線濾除與欄／列門檻。非布局設定變更後，下一筆資料前必須出現新版 `#CFG`。
- Grab 開始時凍結該 grab 的初始布局；Grab 中修改 OPS／START／A輪速度／Crop 只更新設定，
  逐幀 `#CFG` 仍使用初始布局。Stop 時追加唯一的 `#LAYOUT_FINAL`，回顧／報表用它覆蓋該
  grab 全部資料的布局語意，因此同一序號不會同時存在多套座標。
- `#LAYOUT_FINAL` 只改布局解讀，不改圖片、Curve bin、檢測結果或 `.acap` 內容。
- `#CFG` 刻意與每日資料列同檔，不拆成平行設定檔：每筆資料以上方最近一行 `#CFG` 為設定版本，
  避免斷電或跨檔寫入失敗造成資料與設定失配。
- `verdict` 使用寫入 CSV 同一組 V 閾值，與 `AppendRecord@InspectionLogService.cs` 的 `MaxExceed/MeanExceed` 同源。
- CSV 資料列格式＝`Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs,MaxCMean,MeanRPeak,MaxRPeak`；
- 報表欄 O/X 不以單一 frame 一票否決作最終答案。擷取完成後追加
  `#CURVE-C,1,grabId,camId,HM_V,meanPeak,maxPeak`：`meanPeak` 是該序號顯示用
  `CurveMean` 的實際峰值，僅比較欄平均閾值；`maxPeak` 是顯示用 `CurveMax` 的實際峰值，
  僅比較欄最大閾值。此摘要覆蓋同 grab/cam 的逐幀暫定結果；監控即時告警仍維持逐幀判定。
  `MaxCMean`＝該幀 `MaxC`（column curve）全點平均後除以 255（0~1），是報表範圍 `CurveMax` 候選排序值，**不是 MaxPeak**。
  `MeanRPeak/MaxRPeak`＝同幀 Row curve 峰值除以 255，供報表用當前列正規值／門檻重判 O/X。
- CSV 讀取唯一格式入口＝`InspectionCsvReader.TryParseRecord`（統計／回顧影像查詢／curve 候選共用）；
  舊 4/9/10 欄 CSV 合法，缺少的 `MaxCMean` 視為 unknown；缺少 Row peaks 時列判定顯示 `—`；
  範圍內找不到任何有效分數時，`CurveMax` 回退均勻 50 筆。

**code-flow（曲線統計值寫入）**
```
TrySaveCapture@AniloxCamera.cs
 → CaptureContext.MaxC/MeanR/MaxR → SaveCapture@CameraFrameSaver.cs
   → ComputeCurveMeanNormalized（sum(MaxC)/length/255）
   ＋ ComputeCurvePeakNormalized（Row peak/255）
   → OnInspectionResult(grabId,camId,file,meanPeak,maxPeak,maxCMean,meanRPeak,maxRPeak)
     → LiveCameraManager forwarder → OnCameraInspectionResult@AniloxRollForm.Live.cs
       → AppendRecord@InspectionLogService.cs → CSV 第 10~12 欄 MaxCMean/MeanRPeak/MaxRPeak
```

### C3 遠端交付與循環儲存
```
CameraFrameSaver.SaveCapture
 → AppendFrame（七種 asset + camera id + frame tick）→ `{grabId}.acap`
StopGrab
 → FinalizeCaptureOutputsAsync
   → WaitForCaptureSavesAsync（同 grab callback + background append 全部歸零）
   → AddPreviewAtlasesToArchive（raw/column/row，1920x1080）
   → RemoteCopyService.EnqueueFiles（`.acap` + daily CSV）
      → 本機 `.remote-copy-pending` durable marker
      → remote `{destination}.part-{guid}`
      → 來源穩定/長度一致 → Move|Replace 正式檔 → 刪 marker

StorageRetentionService.RunCleanup
 → Storage role 使用 app-mode `StorageMinFreeGB`；Inspection role 使用 `LocalMinFreeGB`
 → Inspection PropertyGrid 輸入 `LocalMinFreeGB >= volumeTotal` → 自動調整為 `floor(totalGB)-1`，
   深橘提示保留到使用者確認，不跳 MessageBox
 → Storage role 的 app-mode `minFree >= volumeTotal` → `Cleanup skipped... No files were deleted`（狀態邊緣單發）
 → free < minFree < volumeTotal → 最舊日期資料夾優先
 → 今天跳過，只處理已結束日期
 → 取消該日 pending marker/worker 任務
 → 刪除整個日期資料夾＋月份層 `yyyyMMdd.csv`
 → 空月份/年份資料夾一併移除

容量狀態列（`lblInfo`，不顯示游標座標）
 → Storage role：TelemetryTimer → `StorageHealthCoordinator` → DriveInfo(`StorageMachineDataPath`) → `儲存電腦：剩餘/總容量`
 → Inspection role 本機：TelemetryTimer → `StorageHealthCoordinator` → DriveInfo(`CaptureRootPath`) → `檢測電腦：剩餘/總容量`
 → Inspection role 遠端：`StorageHealthCoordinator` probe → heartbeat FreeBytes/TotalBytes → `儲存電腦：剩餘/總容量`
 → Inspection role 待傳：`RemoteCopyService.PendingBytes/QueueCount` → `待傳：N GB（M 檔）`
 → Inspection role 成功時間：`OnFilesSaved`／`RemoteCopyService.LastSuccessfulCopyUtc`
   → `最近存檔 HH:mm:ss`／`最近遠傳 HH:mm:ss`
 → heartbeat/磁碟不可讀 → 對應電腦顯示`無法讀取`
```
低磁碟整合測試原則上使用隔離 volume/root，將門檻設為高於該測試磁碟目前可用空間、但低於磁碟總容量即可直接觸發，
不必真的填滿磁碟。只有使用者明確確認目前沒有正式資料時，才可直接使用實際 Captures；執行前仍須記錄來源、目的地與檔案量，
複製只能合併、不得 `/MIR` 或預先刪除目的資料。

**C3/C4 檢測電腦低磁碟 DVT**：`physical-retention-cleanup` 只可使用
`%TEMP%\PICoater-DVT-Retention`，且刪除前必須核對 Runner 專用 marker，禁止指向正式
`D:\Anilox\Captures`。Runner 建立前天與昨天兩個完整日期資料；依當下 volume free space
選擇 `floor(freeGiB)` 為門檻，並只在最舊日配置足以讓 free 暫時低於門檻的檔案。
清理後必須同時成立：

- 最舊日期資料夾及月份層同日 CSV 消失；
- 較新日期資料夾、`.acap` 與 CSV 全部保留；
- free space 回到門檻以上，`LocalLowSpace` 與 `RetentionCleanup` 各自完成
  `raise → resolve → ack`；
- Runner 還原 `Anilox 根目錄`、`預留空間 (GB)`，再刪除 marker 保護的 fixture；
- 成功、失敗或中止都不得碰正式 Capture，且需正常關閉與通過完整 checker。

**C3 儲存電腦本機最終驗收**：只有使用者明確確認實際資料可刪時，才可在儲存電腦本機執行
`test_storage_retention.bat`。工具必須硬性核對根目錄恰為 `D:\Anilox\Captures`、要求人工輸入
`DELETE`、備份 `C:\AniloxMonitor\Config\app-mode.json`，並把暫時門檻設為
`floor(currentFreeGiB)+1`（且小於 volume total）。最舊完整日的容量必須足以單獨達標，否則拒絕執行。
驗收必須證明只刪最舊完整一天與同日 CSV、較新日期全保留、free 達標；最後還原原始 JSON、重啟
儲存程式並取得新鮮 heartbeat。報告與 JSON 備份寫入 `D:\Anilox\Logs\DvtReports`。

長時間聚合驗證使用 `verify-log-min-count` 直接統計 evidence；已納入聚合統計的高密度行不得留在
`wait-log` 的逐行 UI 輸出佇列。否則 2 小時約 4 萬行會在關閉階段重播，產品雖已完成取相與關閉，
Runner 仍會因 UI 佇列塞住而觸發外層 safety timeout。

反覆 Grab 的 Private Bytes 是大幅鋸齒波；同一輪配置影像／封裝 buffer，之後由 Server GC 回收。
資源守門偵測到至少三次擴張與三次回收後，必須改比較穩態前半／後半的最低 retained trough，
不可用任意最後樣本與第一樣本相減。trough 持續成長超過 256 MB/hour 或增加超過 4 GB 才判定
Private leak；UI 無回應、handle、GDI、USER、thread 仍各自獨立判定。

### C4 產出健康度與底部狀態列

`OutputHealthService` 是產出健康度唯一狀態機；writer、remote-copy、retention、設定與背景流程只回報事件，
不得各自在 UI 判斷顏色。`lblInfo` 使用 StatusStrip 預設背景，只顯示容量、待傳量與最近成功時間；每個未確認問題各自
對應一個 `ToolStripStatusLabel`，顯示 service 的完整 incident 清單，禁止只顯示最高嚴重度而藏掉其他問題。
`OutputHealthPresenter` 是唯一 UI owner：訂閱 service snapshot、建立／排序／移除獨立 label，並把使用者確認回送
`AcknowledgeResolved(code)`；Form 不得另存 incident label 字典或自行套色。

| Current state | Event | Next state | Action |
|---|---|---|---|
| 任意 | 檢測／硬體／網路連線異常 | 紅色 Critical | 繼續可執行流程；顯示原因 |
| Normal/Notice | 本機寫檔失敗、資料被捨棄、設定自動重建、背景取得失敗 | 深橘 OutputFault | 保留可用舊資料或預設值並繼續；顯示原因 |
| Normal | 待傳超過 20 GB、接近預留空間、正在清理 | 黃色 Notice | 繼續抓取／重試／清理 |
| 任意 | 新問題或既有問題變更 | 各問題維持自身狀態 | 每個 code 一個獨立 label；嚴重度高者排前 |
| 異常 active | 使用者點該問題 label | 不變 | 未解決問題不得被確認清掉 |
| 異常 resolved | 使用者點該問題 label | 只移除該 code | 其他 active／resolved label 不受影響 |
| 任意 | 問題恢復 | 黃色 resolved 待確認 | 保留原嚴重度供 log／排序，UI 改黃底黑字，直到使用者看到並點選 |

顏色語意固定：預設背景＝資訊區、黃＝容量／積壓警示或已恢復待確認、深橘＝產出失敗或資料捨棄、
紅＝檢測異常或硬體／網路連線異常。已恢復狀態除黃色外，必須同時顯示「已恢復，點擊關閉」，
不得只用顏色傳達可操作性。
產出問題不停止抓取、不觸發 IO 異常；歷史細節交給 log。

**log-flow（只記狀態邊緣，不洗版）**
```
[OutputHealth] raise code=C severity=Notice|OutputFault|Critical message=...
[OutputHealth] resolve code=C message=...
[OutputHealth] state Normal -> Notice code=C active=True
ui:【產出狀態】確認 code=C
[OutputHealth] ack codes=C
[OutputHealth] state Notice -> Normal code=none active=False
```
- 同 code、同 severity、同 message 重複回報不得重記。
- 非最高嚴重度問題 raise／resolve 也必須刷新 incident labels；`resolve` 後未確認仍保留該 label，
  並統一顯示黃色待確認，不沿用 active 時的深橘／紅色。
- active 問題即使點擊也不得被移除；確認 resolved code 只能移除同 code，不得一次清掉其他已恢復問題。
- `CAPTURE/C4.output-health` validator 必須檢查：同 code 未 resolve 前不得重複 raise、resolve 必須有 active
  來源、每筆 ack 恰好一個且只能移除 resolved code；沒有操作到健康度轉變時回 `NOT COVERED`，不得假綠。
- DVT 必驗：每次 raise/resolve 僅一行、穩態無重複、恢復後確認才轉黑、低空間只刪最舊完整一天、
  刪除日的日期資料夾與月份 CSV 同時消失、pending marker 不得留下幽靈重試。

### C5 設定、背景與診斷資料保存

- 設定 JSON 缺少時由 `*Defaults.cs` 重建並寫回；JSON 損壞時同樣重建，但必留下深橘 resolved 事件，
  等使用者點狀態列確認。寫入採同目錄 temp + `Flush(true)` + replace/move；執行中寫入失敗必即時進
  `OutputHealth`，不可只留到下次啟動才看到。
- `app-mode.json` 與 `system-settings.json` 也必須走同一個 `SettingsStoreHelper` 原子寫入／損壞回報；
  Storage role 的 `StorageMinFreeGB` 是部署 bootstrap，啟動後與 PropertyGrid `LocalMinFreeGB` 同值，
  使用者修改時同步寫回 app-mode，禁止顯示值與實際清理門檻分歧。
- 背景檔案 I/O 的唯一 owner 是 `BackgroundProfileRepository`；Form 只編排取得／載入／預覽 intent，
  不得自行掃描、解析、寫入或刪除背景檔。
- 背景版本只有在所有在線相機都完成 `CreateNew + WriteThrough + Flush` 後，才原子替換
  `active-background.json`。manifest 存在但無法解析時不得 fallback 到 legacy bin 混用；保留目前已載入背景，
  並回報 `BackgroundManifestInvalid`。
- 診斷檔只由 `LogFileCatalog` 納管：`trace`、`resource-monitor`、`dropdiag`、`phaselog`、
  `paramchange`、`ui-actions`、`io`、`crash`。`LogRetentionHours` 預設 168 小時，啟動後 5 秒及每小時清理；
  目前 process 建立的 log 與未知檔案不得刪。PropertyGrid 改保存時間後立即補跑一次清理。
- Log 模式與保留時間共用 PropertyGrid `5. Log 設定（記錄／除錯）` 類別；JSON 寫入 `Logging.Mode`
  與 `Logging.RetentionHours`。舊 `Storage.LogRetentionHours` 與 `DebugUiActionLog` 只在載入時遷移，禁止再寫回。
- `Program.InitializeRuntimeLogging` 決定本次可寫目錄（正常為 `D:\Anilox\Logs`，不可寫才退到 `%TEMP%`），
  trace、`io-yyyyMMdd.log` 與 `AniloxRoll-crash.log` 必須共用這個目錄。app 注入 IoLogger 的目錄與 `io`
  前綴；Bridge sample 保持自己的 exe-relative 預設，禁止 SDK 反向硬編產品路徑。
- CSV 與影像寫入是兩條獨立咽喉：`InspectionLogService.WriteFailed/WriteSucceeded` 與
  `AniloxCamera.OnCaptureSaveFailed/OnFilesSaved` 都必須進 `OutputHealth`；任一條失敗都不得被另一條成功假綠。
  影像事件碼使用 `CaptureWriteFailure.CAMn`，成功只能 resolve 同一台相機，禁止 CAM2 成功解除 CAM1 失敗。
- pending marker 無法解析時移到 `.remote-copy-pending\quarantine`，保留檔案供工程追查並留下深橘提示；
  不得每次開機反覆讀同一損壞 marker。pending marker 建立失敗必即時顯示，下一次成功建立才 resolve。
- 遠端 `.part-*` 是未完成發布檔；worker 每次首次使用一個目的資料夾時清除超過 24 小時的 stale part，
  低空間日期清理也會連同當日資料夾移除。讀取端永遠不得把 `.part-*` 當正式產出。

## 全天 Flow DVT 自動校稿架構

```
python tools/python/check_all_flows.py --latest
python tools/python/check_all_flows.py --date 2026-07-13
python tools/python/check_all_flows.py trace-a.log trace-b.log
```

- `check_all_flows.py`＝總入口：依 trace 檔切分 app session，逐一呼叫 registry 內的 validator，再輸出全天摘要。
- `tools/python/flow_checks/core.py`＝唯一 log parser、session model、結果三態（PASS／FAIL／NOT COVERED）。
- `flow_checks/registry.py`＝validator 掛載點；每個 domain 獨立模組，不得把所有規則堆回總入口。
- `NOT COVERED`＝該 session 沒操作到該 flow，**不得算 PASS**；validator 尚未實作則列在 `尚待自動化`，
  總結必標 `PARTIAL`，不得宣稱整份 DVT 全綠。
- session 若為 `Operational`，需 DVT 探針的規則同樣回 `NOT COVERED` 並提示切換「流程驗證」；
  一般操作／錯誤／生命週期規則仍照常判定。
- 現況（2026-07-20）：已掛 `GLOBAL`（任何 `契約違規` 行即 FAIL）＋`LIVE/F`＋`REVIEW/R`＋
  `DATA/D`＋`CAPTURE/C`＋`HARDWARE/H`＋`SETTINGS/S`＋`MURA/M`＋`PARAM/P`，registry 無待接 domain。
  `FULL` 只表示每個已登記 domain 都有 validator；單一 session 未操作到的 flow 仍必須回
  `NOT COVERED`，不可把「有檢查器」誤寫成「所有功能已實際測過」。
- domain 專用舊指令保留為薄 wrapper（例如 `check_review_flows.py`），規則實作只能存在
  `flow_checks/{domain}.py` 一份，避免 wrapper／總入口兩份判準分歧。

### 實際取相自動 DVT

`physical-io-capture` 不是模擬產品影像；它使用實體相機與光源，只把 PLC START
替換成 Runner 管理的本機 Modbus 模擬器。固定執行三次 `High 10s / Low 4s`，每輪必須具備：

```
IO grab request stopCondition=IoSignal stopOnLow=True
capture gate open cams=P
capture head frame dropped ... × P
capture first-set ready ... aligned=True
rowCurve present after=mainImage
capture tail begin cams=...
capture tail complete pending=
StopGrab
capture gate closed standby=on
capture finalize grab=... archive=...acap ... remoteFiles=N
```

- 三輪須各自成對，不得以第一輪的 gate 搭配第二輪的 finalize。
- `P` 為當次在線相機數；七台未接齊時可驗既有相機，但報告必明列未覆蓋七台滿載。
- `remoteFiles>0` 只證明完成封裝並加入遠端待傳；實際 SMB 可寫與 heartbeat 由
  `physical-storage-stability` 驗證，兩者不得互相冒充。
- Runner 中止或失敗時，必先停止可能仍在進行的 Grab，再終止所有 helper、
  還原 PropertyGrid 並關閉主程式。helper exit code 非 0 直接判 FAIL。

`physical-capture-soak` 將同一契約延伸為耐久驗證，不另造產品判準：

- IO 模擬器固定 `High 10s / Low 4s`，測試時間只換算完整循環數；結束必落在 Low。
- 兩小時理論值為 `floor(7200 / 14) = 514` 輪。
- 每輪必須各有一筆 request、gate open、`aligned=True` first-set、gate close 與
  `capture finalize ... remoteFiles>0`；每輪至少一筆 `rowCurve present after=mainImage`。
- 六個聚合計數守門只負責及早指出缺輪；最終仍由 `check_all_flows.py` 逐輪檢查順序，
  不能以總數相同掩蓋跨輪錯配。
- 外部資源探針每 30 秒量測 UI 回應、Private Bytes、handle、GDI/USER 與 thread。
  影像封裝造成的暫時尖峰可接受，但暖機後基線不可持續墊高。
- 儲存與光源狀態必須全程健康；七台未接齊時，報告必明列七台滿載仍為未覆蓋。

`physical-fixed-stop-capture` 延續同一條實體影像鏈，另外驗證停止 owner：

```
Time:
IO grab request stopCondition=Time stopOnLow=False
grab stop waiting condition=Time configured=10s source=io
capture first-set ready ... aligned=True
grab stop armed condition=Time limit=10s ... start=first-set
IO START edge=Low stopOnLow=False action=continue-fixed-target
auto:抓取停止 condition=Time limit=10s

Height:
IO grab request stopCondition=Height stopOnLow=False
grab stop armed condition=height limit=15000px source=io
IO START edge=Low stopOnLow=False action=continue-fixed-target
auto:抓取停止 condition=Height rows=N limit=15000px
```

- Time 的十秒從 `first-set` 起算，不含跨邊界丟幀與等待相機成組時間。
- Height 的 `rows` 是所有在線相機已完成列數的最小值，不得用最快單台提前停止。
- 兩種模式皆須另外具備主圖先於 Curve、gate 關閉、封裝、遠端待傳與正常清理證據。

## 回顧 tab 契約（R 系列；儀器前綴 RV）

**回顧鏈自動校稿工具**（`flow_checks/review.py` 的相容入口；改回顧/跨 tab 同步後必跑）：
```
python tools/python/check_review_flows.py [trace.log]    # 預設抓最新 log；exit 0=全 PASS
```
判準：①R2 快路跟隨（最後選取的 grabId 必有成功 `RV curves`，全 drop=曲線沒跟上）②R2 token+begin/done 配對
③卡頓紅線（回顧互動期間 `UiStall >1000ms` 且同窗有高 `UiPing`／`UiStack` 才算真阻塞）
④讀取資料跳最新（第 2 次起不得停在舊序號）
⑤時段導航去重（同時點不得重複載入）⑥曲線 single-flight（兩個 paths 間必有 done/stale）
⑦方向對數（dataPhys↔dataChart 鏡射/直通，見§狀態快照儀器）⑧切入回顧必有
`RV tabVisible repaint view=True|False`；若先前已有回顧內容或 Data 預載選取，必須再有
`RV visiblePaint ready=True lod=… size=WxH` 證明內容真正可畫。程式啟動後尚未讀取任何資料時，
`view=False` 是合法空狀態，不得誤判為上畫失敗；已有內容時只有 repaint intent 仍不算通過。
2026-07-10 基線：①③④⑤ 皆紅＝回顧戰役待修清單（兇手=每格序號同步觸發 Data 統計全重算於 UI 執行緒
〔SyncDataGrabIdFromReview→RefreshStats→掃目錄+CSV 全解析〕＋時段 date/time 串聯重複觸發）。

### R1 讀取資料（btnReviewSelectFolder）
```
T1: ui:【讀取資料】鈕（Review）
T1: RV folder selected root=…
T1: RV repo scan begin root=…
Tn: RV repo scan root=… files=N csvRecords=C csvArchives=A archiveFallback=F legacy=L ms=M
    ← 索引工作在背景執行，UI 不得出現同時窗 `UiStall`；有每日 CSV 的 `.acap` 直接以 CSV
      FileName 建時序索引，不逐包掃 payload。只有沒有 CSV 對應的封裝才計入 `archiveFallback`。
    ← `python tools/python/measure_display_performance.py --latest` 分別統計清單、縮圖換圖與完整圖載入，
      不得把清單掃描耗時誤判成圖片解碼耗時；`--strict` 用於完整效能驗收。
T1: （首次）RV EnsureImageDisplay create（thumbs=7）
T1: RV loadGrab begin {grabId}（proc=…）
Tn: RV loadGrab paths {grabId} root=… images=N cams=P cfg=yes|no align=tick|filename source=acap|legacy
    ← CSV 已選到的單序號必須 images>0 且 cams>0；`R2.assets` 自動判斷，避免索引存在但實體輸出不可讀
Tn: RV hessian standard {grabId} dir=C|R gain=G scale=S sampleMin=A sampleMax=B sampleMean=M
    ← G 是目前 PropertyGrid 正規值的線性顯示增益；A/B/M 是送進主畫面的實際灰階抽樣值，
      不是由設定值反推的預期值。同一 grabId、同一 dir 下 G 增加時，sampleMean/sampleMax 不得下降，
      且同方向 Curve 必須一起增加；任一關係相反即代表設定有到但顯示資料未生效。
    ← 新資料選強化圖時，完整圖從正規化前 HSM1 依目前欄／列正規值 M 映射；S 是相機原圖到
      本次顯示資料的總縮小倍率（HSM1 目前為 25）。同一 grab 改 M 只准重載顯示圖，不重算 Curve/Hessian。
      prefit `/5` 像素尺寸與 HSM1 `/25` 上畫尺寸可以不同，但換算後的物理 X/Y 範圍必須相同；
      `feedScale` 已代表整數總倍率，row pitch 只可再乘 `exactScale/feedScale` 的小數修正，不得重複乘 `/5`。
      舊資料沒有 HSM1 時不出此行，合法回退到 `_proc_c/_proc_r.jpg`。
T1: RV prefit {grabId} content=WxH viewport=WxH viewX=L~R viewY=T~B
    ← 只讀 JPEG 表頭＋CFG，完整解碼前先由主畫面同源公式算好欄／列視野
T1: RV prefitPaint {grabId} chart=col|row after=Nms axis=A~B/view=L~R
T1: RV prefitApply {grabId} after=Nms visible=True col=axis=…/view=… row=axis=…/view=…
    ← `prefitPaint` 是 MSChart 的實際 PostPaint，不是呼叫意圖；回顧頁可見時兩張圖都必須早於
      該筆第一個 `lodRebind` 或 `pushFrames`。只改 Axis 後呼叫 `Update()` 不算完成：prepared 路必須
      `Invalidate() → Update()`，否則沒有 pending paint 時會等到圖片／Curve 上畫才重繪，肉眼仍看到座標跳位。
      第一次可見 paint 才能取得 MSChart 真實 `InnerPlotPosition`；若本次 paint 才完成 plot 凍結，prepared 路
      必須在同一 UI 動作內以同一主畫面範圍再套一次補償後 zoom，不能把第二次修正留給 BeginInvoke／資料上畫。
T1: RV mainRange {grabId} viewX=L~R viewY=T~B
T1: RV chartRange {grabId} chart=col|row axis=A~B/view=L~R
    ← 真實狀態邊緣：主畫面 ViewRangeMmChanged 與 MSChart PostPaint 只有在座標值改變時才記；不是 intent。
      WinForms 可在 ComboBox 原生 selection message 尚未返回時同步觸發 PostPaint，使同 grabId 的一個
      `DT chartRange` 比 managed intent 早數毫秒；checker 以「上一個不同 grabId intent 後至切換模式前」
      的同 ID selection burst 關聯，不以嚴格行序誤判。
      圖片／Curve 上畫後若又出現不同的 chart axis 或 view，代表座標仍有二次跳位，不能因首個
      prefitPaint 很快就判綠。主畫面視野同時擁有 Axis 與 ScaleView；Curve 資料長度不得擴張座標軸。
T1: RV pushFrames P/7（merge=True, feedScale=…, chartView=publish）   ← P=該 grab 有影像的相機數；缺台=黑占位
T1: RV loadGrab done {grabId}（…ms）
（若由 Data 頁隱藏預載：切到回顧後延後一個 UI message，出現 `RV tabVisible repaint view=True` →
 `RV visiblePaint ready=True lod=… size=WxH`；以可見尺寸補 LOD tile + paint，不重讀檔、不重設視野）
（grab 中按：另會出現 DisableGlobalMerge 等監控行——歸本 intent 管，見孤兒判讀規則）
不變量：手按【讀取資料】＝刷新+跳最新（loadGrab 的 grabId=該次 `DT list reload range` 最新值；
切換 Captures/Captures_pack 等資料根目錄時不得拿前一根目錄的序號比較；2026-07-10 修「停在舊選取」）；
已退場的同層 `Captures_pack` 或 Anilox 根目錄會先解析成設定的 `CaptureRootPath`，並寫回 session；
log 留 `RV|DT data root upgraded from=… to=…`。其他外部封存資料夾仍保留使用者選擇。
開機自動恢復上次位置不在此限。
大量資料不變量：`ImageRepository.LoadDirectory` 與 `InspectionStatisticsService.LoadSnapshot` 必須在
背景執行；四個 30,000 筆序號 ComboBox 仍以 `AddRange` 批次填入，但每個 ComboBox 完成後必須讓出
一次 UI message（`DT combo fill count=N yieldMs=50`）。總載入時間可超過一秒，期間不得形成連續
`UiStall >1000ms`。
載入 busy 視覺唯一 owner＝`BusyUiBinder`；`AniloxRollPresenter.BusyStateChanged` 與
`ReviewStitchCoordinator.LoadGrabStitchedViewAsync` 共用同一實例。圖片 latest token 與 busy lease 由
`ReviewImageLoadGate` 同時管理：新序號 intent 作廢舊圖片時，若舊圖片仍持有 lease，必須立即出現
`RV loadGrab busy off reason=invalidated` 並恢復游標；新 loader 已開始時，舊 loader 的 finally 不得解除新 lease。
回顧 CFG 與螢幕校正 runtime state 唯一 owner＝`ReviewRuntimeState`；單片曲線快路、完整圖片載入與時段載入
都只更新/讀取此實例，不得在 Form、Presenter 或 Coordinator 另存第二份 CFG。
```

### R2 單片序號切換（cbReviewId）——分層載入（2026-07-07 定版）
```
T1: ui:【單片序號】→ {grabId}
T0: RV curve load policy latest-only minCycleMs=80
Tn: RV curves paths {grabId} root=… images=N cams=P cfg=yes|no align=tick|filename|summary source=bins|summary|memory-bins|memory-summary coalesced=N
T1: RV prefit {grabId} …
T1: RV layout intent {grabId} images=N cams=P align=tick|filename before=curves
T1: RV curves {grabId}（…ms） presentation=progressive|latest
    ← 快路：先套新序號主畫面幾何，再讓欄+列曲線跳號跟隨；停下後最後一筆必為 latest
（快速滾動：曲線 single-flight/latest-only；第一筆立即讀，之後每 80ms 最多啟動一筆，期間只保留最新 intent。
 中間 intent 可無 paths；已開始的舊筆可 progressive 上畫，冷卻結束再讀等待中的最新筆；
 切時序／重讀資料等明確離開單序號情境才 stale-drop）
Tn: RV thumbnail begin {grabId}
Tn: RV thumbnail coalesced {grabId} skipped=N minCycleMs=33
T1: RV thumbnail done {grabId} total=Nms decode=Nms images=P ratio=R source=atlas|frames cache=cold|join|hit atlas=WxH|none
    或 RV thumbnail unavailable {grabId}（Nms）／RV thumbnail stale-drop {grabId}（Nms）
Tn: RV plan prepare begin {grabId}｜RV plan prepare reuse-inflight {grabId}
    ← Curve／縮圖／完整圖共用同一份 layout+CFG 準備工作；同 grabId 不得各自重讀同一份日 CSV
    ← `.acap` 有內嵌預覽時，快滾可先上低解析圖片；已開始的圖片准依序完成上畫，
      尚未開始的中間 intent 合併為最新一筆，不顯示 busy cursor。
      `atlas`＝所選 raw／proc C／proc R 只讀一筆 1080p 預覽合圖；`frames`＝舊逐幀縮圖 fallback
（影像 debounce 250ms：滾動中不發完整載入；停下才同步日期/時間、載「最後選取」完整圖；session 也只在 settle 落盤一次）
T1/Tn: RV loadGrab begin {grabId} → RV loadGrab paths … → RV prefit …
  → （尺寸改變）RV lodRebind merge …（fit reset）
  → RV fit(record-change) → RV pushFrames → RV loadGrab done
Tn: RV prefetch begin center={grabId} neighbors={next}|{previous}
 → RV prefetch ready center={grabId} neighbor={neighborId} thumbnail=cold|join|hit total=Nms
 或 RV prefetch unavailable center={grabId} neighbor={neighborId} error=no-preview
```
- **分層**：單步時曲線立即載；快速滾動時曲線最多「執行中 1 筆＋等待中最新 1 筆」，中間序號不讀檔；
  已開始的 Curve 完成後以 `presentation=progressive` 跳號上畫，停下後最後一個 intent 必有
  `presentation=latest`。若 `coalesced` 總數大於零卻完全沒有 progressive 上畫，代表只有讀取節流、
  沒有視覺跟隨，判定失敗。圖片只載 settle 後的最後一張；
  **Data tab 同步（統計/Mura 圖重算）也只在 settle 後做一次、排在影像之後**——唯一觸發點
  `SyncDataTabFromReviewSettled@AniloxRollForm.Data.cs`，不得回到逐格 inline
  （2026-07-10 定罪：逐格全量重算＝快撥 UiStall 5.7s＋曲線快路全餓死）。
- **預覽是可重建快取，不是真實來源**：新格式每個 grab 內嵌 raw／proc C／proc R 各一筆
  `PreviewAtlas`；每筆是先依 tick 時間軸將各相機垂直拼接，再橫向裝入單張
  `<=1920×1080` JPEG，並保存 camera rectangle 與原始寬高。快滾只讀／解碼所選模式一筆，
  再於記憶體切回各相機，最後沿用 OPS／START／CROP 合併。舊逐幀 thumbnail 只作 fallback；
  兩者都缺時記 `unavailable` 並等待既有 250ms 完整圖路徑，不得退回快滾中解碼完整 JPEG。
  縮圖與完整圖使用不同 latest-only token；完整 `RV loadGrab begin` 必須立即作廢正在讀的縮圖，
  因此其後只准 `stale-drop`，不得讓舊縮圖覆蓋完整圖。縮圖沿用完整圖的 OPS／START／CROP
  幾何與物理座標，只降低像素解析度，不得使欄／列 Curve 或視野跳位。
  預覽上畫有 33ms 最短週期，滑輪輸入更快時 running 依序完成、pending 只保留最新序號；
  `RV thumbnail coalesced skipped=N` 必須留下被合併數量，不得用每個 intent 塞滿 UI。
  效能門檻只判 `thumbnail done` 當下仍為目前選中 grabId 的結果；正在完成的中途預覽另列吞吐資訊，
  因其完成後再合併到最新是既定 progressive policy，不得誤判為最後序號卡頓。
  只有切時序／模式、離開單序號或完整圖開始載入時才准 invalidate thumbnail；
  一般序號變更只准作廢 250ms settle 的完整圖，不得清除 thumbnail pending。
- **相鄰預載只在穩定選取後開始**：最後序號的完整圖成功上畫後，依最近滾動方向依序準備前後各一筆的
  layout/CFG、Curve 匯總與低解析縮圖；快速滾動期間不得另外開預載工作。新 intent 立即增加 prefetch generation，
  已開始的單筆可完成並供前景 `join`，但第二個鄰居不得再開始。完整圖不預載。
  plan cache 上限 32 筆；thumbnail cache 上限 24 筆／96 MB；Curve 沿用 512 筆／256 MB bounded cache。
  前景命中已完成預載記 `cache=hit`，接手進行中工作記 `cache=join`，兩者都不得再次讀取相同縮圖。
  沒有可用縮圖（無影像或縮放比 `<=1`）必須記 `error=no-preview`，不得進快取或伪裝成 `ready/hit`。
- `check_all_flows.py` 的 `REVIEW/R2.prefetch` 會對帳 `begin` 宣告的相鄰清單、`ready/unavailable`
  終態及後續 `thumbnail ... cache=hit`；未宣告的鄰居不得被記成預載成功。
- **日期/session 分層**：滾動中不得逐格呼叫 `SetPeriodToCombo`；其 `Items.Contains/SelectedItem`
  會線性搜尋時間 Combo，大量資料時可單獨阻塞 UI。日期/時間同步與 `SaveCurrentSelection`
  都只在 250ms settle 對最後序號執行一次；不得走 `NavigateTo` 的完整 Initialize/Save。
- **換序號＝重設視野（fit）＝預期**：完整新序號上畫必有 `RV fit(record-change)`。
  尺寸不同時 `lodRebind` 也會先 reset；尺寸相同時 LOD 不重綁，因此仍須由
  `ReviewDisplayManager.PushFrames(preserveChartView:false)` 明確 fit，否則會沿用上一筆的 pan/zoom，
  造成 prefit 先畫全幅、圖片完成後又跳回舊視野。同序號切強化與快速縮圖
  `preserveChartView:true`，不得觸發此重設。
- **prefit 到完整圖片之間禁止舊視野回灌**：`StitchedLayoutReady` 必先呼叫
  `ReviewDisplayManager.BeginRecordTransition(grabId)`。直到完整新序號完成
  `fit(record-change)` 前，上一筆主畫面的延遲 `ViewRangeMmChanged` 必須被丟棄（首筆記
  `RV staleView drop {grabId}`），不得把欄／列圖表從 prefit 範圍拉回舊 pan/zoom。
  完整圖片 fit 後只發布一次最終視野；同序號快速縮圖與強化切換仍保持目前視野。
- **報表／回顧初始 fit SSoT**：`ReviewImageDataLoader.Prepare` 只讀 JPEG 表頭與 `#CFG`，不解碼像素；
  `ImageDisplayView.TryComputeMergeFitViewRange` 再以實際主畫面的 `MergeLayout + AspectFitCalculator + PixelMmMapper`
  預算四邊界。實際上畫後的 `ViewRangeMmChanged` 也走同一座標換算核心，不得由 bin 長度另推一份公式，
  也不另存 CSV 總長寬。`RV prefit` 必早於同 grab 的 `RV lodRebind merge`，且兩者 `content=WxH` 必相等；
  完整回顧圖載入後仍須以實際 `ViewRangeMmChanged` 覆核並同步報表欄／列視野。
- **Curve 快路也必須先取得同源 fit**：每筆成功 `RV curves` 前必有同 grabId 的
  `RV curves paths → RV prefit → RV layout intent → RV curves`；圖片像素仍維持 250ms settle 後才解碼。
  這不是提前載圖，而是先讀表頭／CFG 並發布座標。欄、列 helper 在同步模式下，Axis 與 ScaleView 都只由
  主畫面 viewport 決定；新 bin 的資料長度不得把 Axis 先撐大，再等圖片出現後縮回。
- **settle 後的列 Curve 首次上畫必須原子帶入新影像 fit 範圍**：
  `SuspendUntilNextData` 先讓上一筆 view range 失效；`StitchedLayoutReady → SetPreparedViewRange` 在完整解碼前
  以 `UpdateViewRangeImmediate` 定位並實際重繪座標軸，新 row bin 仍只可 pending，直到 Resume 才整筆換入。
  一般拖曳／實際 view range 覆核仍走原本 `UpdateViewRange`，不得把 prepared 強制 invalidation 擴散到連續互動路。
  實際圖片的
  `PushFrames → RefreshNow／RefireViewRange → ViewRangeMmChanged` 負責覆核；相同像素尺寸不會重綁 LOD，
  因此 `RefireViewRange` 不可省略。禁止新資料先套舊範圍，或圖片出現後座標才跳位。
- **列實體尺度 SSoT**：回顧主畫面、回顧列 Curve、報表列 Curve 皆由
  `RowCurvePhysicalScaleResolver` 取同一筆 `#CFG` 的 A 輪速度＋CAM1 線掃速率；舊 CFG 缺值才回退目前設定。
  總長＝row bin 點數 × mm/row，不另存 CSV 總長／總高欄位。
- **token 分治**：曲線與圖片各自最後贏，兩者不得共用 token（圖片開始載入不得讓同序號曲線 stale）；
  每個序號 intent 立即 invalidate 舊圖片，settle 回呼另以 selection token 守住 Data 同步。
- 最後一個非 stale 的 `curves`/`loadGrab done` 的 grabId 必須＝最後一個 intent 的 grabId——不符＝token 破了。
- begin 無對應 done/stale-drop＝載入中斷；pushFrames P 與 CSV 台數不符＝掉圖。

**code-flow（曲線快路與 settle 圖片路分治）**
```
OnReviewGrabIdSelected@AniloxRollForm.Data.cs
 ├ InvalidateSettledImageLoad@ReviewStitchCoordinator.cs
 │  └ Invalidate@ReviewImageLoadGate.cs（只讓舊完整圖失效；同時釋放該圖片的 busy lease）
 ├ LoadGrabCurvesOnlyAsync@ReviewStitchCoordinator.cs
 │  └ Enqueue@LatestGrabLoadCoordinator.cs
 │     ├ pending 僅保留最新一筆；running 恆單工；首筆立即執行，之後維持 80ms 最短週期
 │     └ LoadGrabCurvesCoreAsync@ReviewStitchCoordinator.cs
 │        ├ Prepare@ReviewImageDataLoader.cs（只讀 CFG／JPEG 表頭，與曲線 IO 同在背景執行）
 │        ├ Load@SingleGrabCurveDataLoader.cs（回顧／報表共用、無 WinForms 的 IO／合併 service）
 │        │  ├ SingleGrabCurveCache（512 筆／256 MB 上限；回顧／報表各自持有 bounded cache）
 │        │  ├ TryLoad@SingleGrabCurveSummaryStore.cs（與報表共用 `_curve_summary` materialized view）
 │        │  └ miss → LoadForGrabId＋ResolveAlignment＋MergeCurves／MergeRowCurves
 │        │       └ QueueSave summary（下次回顧／報表直接讀匯總；原始 bins 仍是 SSoT）
 │        ├ IsCurrent@LatestGrabLoadCoordinator.cs（只標記 presentation=latest|progressive）
 │        └ CanApplyStarted@LatestGrabLoadCoordinator.cs
 │           ├ false（明確 Invalidate）→ RV curves stale-drop
 │           └ true → PublishPreparedLayout＋StitchedLayoutReady（先發布 fit）
 │                    → ReviewRuntimeState＋SetCurves@ReviewDisplayContent.cs
 │                    → UpdateStitchedOverviewChart＋UpdateGlobalRowChart@ReviewChartPresenter.cs
 ├ LoadGrabThumbnailAsync@ReviewStitchCoordinator.cs
 │  ├ LatestGrabLoadCoordinator（與完整圖 token 分離；running 單工、pending 只留最新、minCycle=33ms）
 │  │  └ CanApplyStarted（新 intent 不作廢 running；明確切模式／完整圖載入才 stale-drop）
 │  ├ ReviewAsyncLruCache（24 筆／96 MB；single-flight，前景可 join 相鄰預載）
 │  ├ Load(useThumbnail=true)@ReviewImageDataLoader.cs（cache miss 才執行）
 │  │  ├ TryLoad@CapturePreviewAtlasCodec.cs
 │  │  │  └ ReadAsset(PreviewAtlas raw|C|R) 一筆 → JPEG 解碼一次 → camera rectangle 記憶體切片
 │  │  └ atlas miss → StitchCamera(useThumbnail=true)@GrabImageStitcher.cs（舊逐幀縮圖 fallback）
 │  └ StitchedImagesReady(chartView=preserve)（只換低解析圖片，不重讀／重畫 Curve）
 └ 250ms settle
    ├ SetPeriodToCombo＋UpdatePeriodNavigationState＋SaveCurrentSelection（只套最後一筆）
    ├ LoadGrabStitchedViewGuardRowRangeAsync@AniloxRollForm.Review.cs
    ├ SuspendUntilNextData@RowCurveSyncCoordinator.cs（舊 view range 失效；新 row data 等新 fit）
    └ LoadGrabStitchedViewAsync@ReviewStitchCoordinator.cs
       ├ Invalidate thumbnail token（完整圖開始後，舊縮圖不得再上畫）
       ├ TryGetPreparedPlan；快路已有同 grab plan 時直接重用，否則 Prepare
       │  ├ LoadForGrabId@InspectionImagePathRepository.cs
       │  │  └ InspectionCsvReader.OpenShared＋TryParseRecord＋TryExtractCameraId（影像依 cam 分組）
       │  ├ LoadForGrabId@InspectionConfigRepository.cs
       │  │  └ InspectionCsvReader.OpenShared＋TryParseRecord（取 grab 上方最近 #CFG）
       │  ├ ResolveAlignment@FrameTickIndex.cs（與曲線快路共用，不得另寫 fallback）
       │  └ TryGetStitchedSize@GrabImageStitcher.cs（只取尺寸，不解碼全圖）
       ├ StitchedLayoutReady → TryComputeMergeFitViewRange@ImageDisplayView.cs
       │  └ ApplyReviewViewRangeToCharts(prepared=true)
       │     └ UpdateViewRangeImmediate（Invalidate→Update；先畫回顧＋報表欄／列座標）→ RV prefitPaint
       ├ Load(plan)@ReviewImageDataLoader.cs
       │  └ StitchCamera＋MergeCurves／MergeRowCurves＋BitmapGrayConverter
       └ IsCurrent@ReviewImageLoadGate.cs
          ├ false → DisposeImages＋RV loadGrab stale-drop（不得套 UI）
           └ true → ReplaceImages@ReviewDisplayContent.cs＋ReviewRuntimeState
                    → ApplyRowPhysicalScale@ReviewChartPresenter.cs
                   → StitchedImagesReady（先餵同一 mm/row）
                   → PushFrames.RefreshNow＋RefireViewRange → 欄／列 chart
    └ BeginAdjacentPrefetch@ReviewStitchCoordinator.cs（完整圖成功且 selection token 仍為最後一筆）
       ├ ReviewAdjacentPrefetchPolicy（依滾動方向排序前後各一筆）
       ├ ReviewAsyncLruCache<ReviewImageLoadPlan>（32 筆，layout/CFG single-flight）
       ├ PrefetchAsync@SingleGrabCurveDataLoader.cs（只填 bounded Curve cache）
       └ ReviewAsyncLruCache<ReviewThumbnailSnapshot>（只保存低解析 gray；不保存 Bitmap／完整圖）
```

`LoadGrabStitchedViewAsync` 的載入模式是行為邊界，不得以布林特例分散回 Form：

| 模式 | 使用時機 | Curve／視野行為 |
|---|---|---|
| `Full` | 新序號 settle 後載圖 | 發布 prefit、載入 bin，並以實際圖片視野覆核 Curve |
| `ReuseSharedCurves` | 報表已載 Curve 後切到回顧 | 套用報表共用快照，不重讀 bin；仍為新序號建立 fit |
| `ImageVariantOnly` | 同序號切原圖／欄強化／列強化 | 只解碼並換圖；不 Suspend、不 prefit、不讀 bin、不發布 view range、不重畫 Curve；以物理 mm 保存並還原主畫面視野 |

快速滾動的 latest-only 機制只適用「新序號」；同序號切圖片版本必須保留現有 Curve 與視野，
不得用完整載入來間接重畫。

### R3 時段導航（cbReviewDate/cbReviewTime 手動）
```
T1: ui:【時段導航】（cbReviewDate/Time）
T1: RV period begin {yyyy-MM-dd HH:mm:ss.fff}
T1: RV period load {yyyy-MM-dd HH:mm:ss.fff} images=P/7 proc=True|False cfg=yes|no
T1: RV pushFrames P/7（merge=True, feedScale=…, chartView=publish）
T1: RV row … / RV state …（chart/狀態快照視資料而定）
T1: RV period done {yyyy-MM-dd HH:mm:ss.fff}
```
- 時段模式不進 `RV loadGrab begin/done`；它走 request 的 immutable period → `ApplyGlobalMergeForPeriod` → `StitchedImagesReady` → `ReviewDisplayManager.PushFrames`。
- **時序選擇 policy（2026-07-30 大量資料修訂）**：單次有效時點 intent 仍立即刷新三項＝圖片＋欄曲線＋列曲線；
  loader 不加固定 debounce，但採「執行中 1 筆＋等待中最新 1 筆」single-flight。前一筆執行時若持續快滾，
  中間 pending 可被較新的時點取代；執行中的舊時點若已存在較新 pending，只能完成讀取後記
  `RV period stale-drop`，不得上畫或同步序號 ComboBox。最後停下的時點才同步序號並完整刷新三項。
  各 request 持有 immutable period；
  不得並行後在 await 尾端重讀 ComboBox（會讓多筆都套用最新時點）。
  切回 R2 序號時 `Invalidate` 清 pending，running request 只能記 stale-drop、不得上畫面。
- **同一時點只載一次**：日期 combo 串聯改時間 combo 必掛 `_updating`（DateTimeNavigator）；
  同時點連發 `RV period load`＝串聯去重失效（2026-07-10 修 ×2~×6 重複載入）。
- `RV period load` 中 `cfg=yes` 代表 `RefreshReviewConfigForCurrentPeriod` 已從該日 CSV 找到 #CFG；`cfg=no` 時座標/閾值 fallback 當前 settings。

**code-flow**
```
PeriodSelectionChanged@DateTimeNavigator.cs
 → OnPeriodComboChanged@AniloxRollForm.Review.cs（擷取 immutable DateTime）
   ├ InvalidateImageLoad@ReviewStitchCoordinator.cs（R2 舊圖片不得覆蓋 R3）
   └ Enqueue@ReviewPeriodLoadCoordinator.cs
      ├ 同 period+mode running/pending → 去重
      ├ 不同 period → running 單工、pending 只保留最新（不得平行）
      └ LoadReviewPeriodRequestAsync@AniloxRollForm.Review.cs
         ├ RunWorkflowForPeriodAsync@AniloxRollPresenter.cs → GetImages(DateTime)@ImageRepository.cs
         ├ generation 失效 → stale-drop（不得 apply）
         └ ApplyGlobalMergeForPeriod@ReviewStitchCoordinator.cs
            → Apply@ReviewPeriodImagePresenter.cs
              ├ GetImages(DateTime)@ImageRepository.cs
              ├ LoadFrames@ReviewPeriodDataLoader.cs
              └ StitchedImagesReady（ReviewStitchCoordinator 只轉發，不再自行找檔／解碼）
         └ ApplyPostLoadDisplay(period)
             ├ ApplyGlobalMergeForPeriod → LoadFrames@ReviewPeriodDataLoader.cs → PushFrames（圖片/LOD）
             ├ UpdateOverviewChartForPeriod → LoadColumnCurves@ReviewPeriodDataLoader.cs（欄 curve）
             └ UpdateRowChartForPeriod → LoadMergedRowCurves@ReviewPeriodDataLoader.cs（列 curve）
R2 入口：LoadGrabStitchedViewGuardRowRangeAsync → ReviewPeriodLoadCoordinator.Invalidate
```

### R4 回顧主畫面互動（點縮圖/拖曳/縮放）
與 F5/F6/F6b 同款（同一個 ImageDisplayView），行前綴=RV：
點縮圖＝內部置中；拖曳＝橘框跟隨；縮放中不得出現 `RV autoFit`/`RV lodRebind`；
**回顧靜態看圖＝永久穩態**（無新幀）→ 穩態靜默通則全程適用（比監控更嚴）。

## 模擬測試的極限（誠實邊界）

模擬蓋得住：步驟順序/訂閱生命週期/置中來源/事件接線。蓋不住：硬體時序（stall/掉幀）、native 行為、
視覺呈現（配色/佈局/重繪）。這三類改動仍需真機 spot check——但頻率遠低於逐項手測。

## 設定變更契約（S 系列）——PropertyGrid 的多維度流程

**多維度的寫法判準**：維度＝「模式 × 設定」，但**不列組合爆炸**——
① 每個設定寫「一條」契約（正序：該出現什麼反應 + 禁止：不相關的東西不得動）；
② 只有行為真的隨模式分歧時才寫變體（如 F4 的 即時/瀑布 分支）；
③ 模式互斥用「不變量」寫一次，所有設定共用（不用每條重複）。

### S0 通用（單一掛點，蓋所有設定）
```
T1: ui:設定[{屬性名}]={新值}   ← 使用者從 PropertyGrid 改（孤兒判讀規則的主人；值截 40 字）
T1: set:[{屬性名}]={新值}      ← 程式化來源（自動掃描寫回等）；有主人但非使用者動作
T1: setting route {屬性名} owner={owner|None} effects={effect+effect|None}
    ← 每個 setting intent 的下一行；分類一次後才執行副作用
（之後的反應行歸此 intent 管）
```
PropertyGrid「檢出標準」的數值列可用滑鼠滾輪調整：欄／列正規值與欄／列平均、最大閾值
每格 `0.1`，細線濾除每格 `1`，最小值各為一格。第一次點選數值列才啟用滾輪，
再點同一列即取消；切到其他列時原列必須解除，避免一般捲動誤改設定。狀態行為：
`property wheel {armed|disarmed} setting={name}`。滾輪入口不得另建設定旁路：
`WM_MOUSEWHEEL → PropertyGridNumericWheelInterceptor → PropertyDescriptor.SetValue
→ SettingsHub.NotifyExternalChange`，之後必須產生與鍵盤輸入完全相同的 `ui:設定 → setting route`。
非上述數值列不得攔截滾輪，仍由 PropertyGrid 正常捲動畫面。

**⚠ 非 PropertyGrid 的使用者入口（點 label/chart 走 Hub.Set）會被記成 set:（程式來源）**——
這類入口必須自帶 `ui:` intent 行（如【暫停Mura檢測】【IO暫停】），否則盲測會認錯兇手身份
（2026-07-07 盲測實例：點 lblIoDoMura 被誤判為程式動作、點 lblIoConn 完全無痕漏抓）。

**設定副作用路由不變量**：
- `OnSettingChanged` 是唯一序列化 dispatcher；不得新增平行 `SettingsHub.Changed` 副作用訂閱者。
- `SettingImpactClassifier` 是「setting → feature owner + cross-feature impacts」唯一決策表；
  owner 內部可以依同一 setting 選動作，但不得在其他 feature 再複製跨域判斷。
- `CapturePolicy` 只准出現在 `AniloxRootPath`、`EnableAutoCapture`、`SaveOriginalBmp`、
  `dc_HessianMaxFactorV`、`de_RidgeSigma`、`eb_RidgeDir`。IO、光源、顯示、報表等設定
  不得重送相機曝光／線掃速率／擷取高度。
- `owner=None effects=None` 代表設定已保存、但本次不需立即 runtime 副作用，不是漏接。

**code-flow（設定變更）**：
```
SettingsHub.Changed
 → OnSettingChanged@AniloxRollForm.cs（SemaphoreSlim 單線序列化）
    ├ S0 intent → SettingImpactClassifier.Classify → setting route
    ├ ApplyCrossFeatureSettingImpacts（只套 route 指定的跨功能影響）
    ├ grab 中 ForceWriteConfig（ContentKey 去重，僅 #CFG 真值變更才落盤）
    └ DispatchSettingOwner（只呼叫一個 feature handler）
CapturePolicy
 → RefreshCapturePolicy@LiveCameraManager.cs
    └ 只更新存檔／演算法政策；不得呼叫 SetExposureUs／SetLineRateHz 或改 CameraGrabHeight
```

### S 系列不變量：view 互斥
**任一時刻主畫面 view 唯一**：設定[主畫面顯示]=即時 期間，不得出現任何 WF 前綴行/EnableWaterfall；
=瀑布 期間反之（不得出現 IC 主畫面 view 建立行）。切換瞬間走 F4（teardown 舊→create 新）。
**第三態＝背景預覽**（靜音鍵，F8）：預覽期間主畫面恆 IC view（顯示背景合圖）、瀑布讓位；
預覽中改設定→閘門仍出預覽畫面（「存活」policy），**不得出現 F4 的 teardown/create 序列**，Exit 才生效。
**執行期自檢**：幀流進不屬於當前模式的路徑時 code 會當下自報
`⚠ 契約違規：瀑布模式下幀流入 IC 路徑` / `⚠ 契約違規：即時模式下幀流入瀑布路徑`
（每 view 週期一次）——log 出現此行＝訂閱錯掛/殘留，不用比對即定罪。

### S1 檢測標準（監控／回顧／報表共用設定）

涵蓋 `dc_`／`dd_` 欄列正規值、`eb_` 檢出方向、`eca_` 欄曲線判定及
`ec_`～`ef_` 欄列平均／最大閾值。這些設定不是 Data tab 專用；監控 Live 必須同時生效。

**log-flow**：
```
T1: ui:設定[{name}]={value}
T1: setting route {name} owner=DataStats effects=…+LiveInspectionCurves
T1: live inspection apply setting={name} hm=C/R thresholdC=Mean/Max
    thresholdR=Mean/Max mode={Mean|Max|Both} direction={Vertical|Horizontal|Both}
    action={normalization-reset|refresh}

（dc_／dd_ 改變後，下一筆監控列 Curve）
Tn: live row normalize captureHm=C rowHm=R ratio=R/C

（報表單序號已載入時）
Tn: DT curve refresh {grabId} reason=setting-{name}
    column={True|False} row={True|False} source=memory preserveRange=True rangeDelta=Dmm
```

`rangeDelta` 是重算前後實讀欄圖 `AxisX`／列圖 `AxisY` 的最大座標差；設定重算必須
`<= 0.01 mm`。超過即判 FAIL，不能只憑 `preserveRange=True` 宣告成功。

**code-flow 與數值契約**：
```
SettingsHub.Changed
 → OnSettingChanged@AniloxRollForm.cs
 → SettingImpactClassifier：S1 全部帶 LiveInspectionCurves
 → ApplyLiveInspectionSettings@AniloxRollForm.Live.cs
    ├ 閾值／欄曲線模式／方向：保留資料，立即更新線與下一幀 O/X
    └ 欄／列正規值：清除舊尺度 Live Curve 緩衝，禁止同圖混用兩種尺度

ProcessImage@AniloxCamera.cs
 → OnLiveRowCurveData(cam, rawMean, rawMax, frameHmC)
 → Live 顯示列值 = raw × currentHmR / frameHmC
 → CheckLiveMura（使用換算後值）→ pending row → chart

Report 單序號設定重畫
 → RefreshForSettingsChange(name)@MuraProfileChartPresenter.cs
 → 使用已呈現的 raw Curve＋既有 prefit view 原地重算，不得重新讀 bin／CFG 或重新計算 prefit
    ├ 欄正規值只重畫欄；列正規值只重畫列
    └ 閾值／顯示模式只更新其所屬圖表
```

正規值是顯示增益：目前值加倍，Curve 高度也加倍；不得再出現反比
`HM_capture / HM_current`。報表重畫時欄 X 與列 Y 的物理座標範圍必須保持不變。

欄與列的 O/X 都遵守同一公式：啟用平均時 `mean > meanThreshold` 才因平均失敗；
啟用最大時 `max > maxThreshold` 才因最大失敗。不得以平均閾值判斷最大曲線。

**光源替代刺激 Smoke（S1.LightSurrogate）**：流程驗證模式可在 Grab 中切亮度
`100 ↔ 255`，等待 500 ms 後各取一筆欄／列 peak、當下閾值與 O/X：
```
live inspection stimulus brightness=B direction={col|row} mean=M max=X
threshold=TM/TX mode={Mean|Max|Both} verdict={O|X} source=light-surrogate-not-mura
```
checker 必須驗證兩個亮度都有欄／列資料、數值確實改變且 verdict 符合公式。
這只證明「檢測標準接線與計算會對穩定輸入變化作出反應」，**不是正式 Mura 模擬，
不得拿來宣稱真實瑕疵檢出率或光學準確度**。

禁止：其他任何設定（IO／光源／儲存／一般顯示）不得觸發 Data 曲線 reload+重綁。
光源亮度只允許 arm S1 測試樣本，不得改寫檢測標準。

### S2 回顧強化（hd_EnableReviewEnhance）
```
T1: ui:設定[hd_EnableReviewEnhance]
單序號模式：RV loadGrab begin {當前grabId}
 → RV loadGrab curves=keep source=display {當前grabId}
 → RV pushFrames … chartView=keep
 → RV variantView keep beforeX=L~R beforeY=T~B afterX=L~R afterY=T~B maxDelta=D
 → RV loadGrab done
    判準：D ≤ 0.1 mm；禁止 RV prefit／RV curves／curves=load source=bin／RV mainRange。
    原圖 `/5` 與 HSM1 `/25` 像素密度不同時，必須保存並還原同一組物理 mm 視野，
    不得保存像素 zoom/pan，也不得重畫 Curve 或重新 fit。
時序模式：RV period load {當前時點} images=P/7 proc=True|False cfg=yes|no
 → RV pushFrames … chartView=keep → RV variantView keep … maxDelta=D
 → RV period curves=keep source=display
    ← 重載目前真正顯示的模式；不得一律假設單序號，也不得重畫 Curve
```

### S6 顯示裁切（cb_CropHead／cc_CropTail）
未 Grab：
```
T1: ui:設定[cb_CropHead|cc_CropTail]=N
T1: setting route {name} owner=LiveLayout effects=None
T1: displayCrop applied head=H tail=T mode=IC|WF content=WxH zoom=Z fit=True frames=dynamic
T1: displayCrop head=H tail=T scope=main+column-chart data=unchanged waterfallHistory=preserved
```
Grab 中：
```
T1: ui:設定[cb_CropHead|cc_CropTail]=N
T1: setting route {name} owner=LiveLayout effects=…
T1: capture layout pending grab=… setting={name} apply=display-now+stop-final
T1: displayCrop applied head=H tail=T mode=IC|WF content=WxH zoom=Z fit=True frames=dynamic
T1: displayCrop head=H tail=T scope=main+column-chart data=unchanged waterfallHistory=preserved
... 可繼續修改；每次 Crop intent 都立即更新顯示，但只記最後布局 ...
T1: StopGrab
T1: capture layout final grab=… ops=… start=… speed=N head=H tail=T path=…
T1: capture layout applied grab=… timing=stop ops=… start=… speed=N head=H tail=T render=already-applied source=unchanged
```
- Crop 是 **X／欄方向的純顯示布局**：先由完整合圖幾何算可視區，再用同一份可視布局驅動
  即時主畫面、瀑布主畫面、橘色相機框線及欄圖表；縮圖仍顯示各相機完整影像。
- `HorizontalDisplayCrop.Compute/Apply` 是裁切幾何唯一來源；fit 必須在裁切後寬度上計算，
  不得先 fit 完整寬度再隱藏頭尾。
- Live 使用目前 PropertyGrid；Review／Report 單序號使用該筆 CSV `#CFG` 的
  `TrimHead/TrimTail`，使歷史畫面與拍攝布局一致。
- **資料不變量**：Crop 不得進入 pipeline、相機 frame、Curve bin、圖片、`.acap` 或 CSV 資料列。
  Grab 中 Crop 立即更新主畫面、欄圖表與橘框，但拍攝布局只在 Stop 封存最後值；只有
  A輪速度等延後布局也曾改變時，Stop 才允許一次輕量 repaint。不得重讀檔、
  重跑演算法、重算 Curve 或清除既有瀑布歷史。
- 允許頭尾合計超過內容時保留最少一個顯示像素，不得產生零寬畫布或例外。

### S7 即時機台布局（OPS／Start）
```
T1: ui:設定[ab_OpsCam1..ah_OpsCam7|bb_StartCam1..bh_StartCam7]=N
T1: setting route {name} owner=LiveLayout effects=None
（Grab 中）capture layout pending grab=… setting={name} apply=display-now+stop-final
T1: WF layout remap storage=per-camera historyRows=R virtual=WxH slots=cam:srcW@x+w|… ms=N
Tn: WF layout presented storage=per-camera historyRows=R virtual=WxH latency=Nms
T1: displayLayout applied setting={name} refGrid=cam1 ops=… start=… speed=N head=H tail=T scope=main+column-chart source=unchanged
```
- CPU 合圖使用 Cam1 OPS 建立共同顯示格點；每台相機的顯示寬度為
  `來源寬度 × 該台 OPS / Cam1 OPS`。因此 Cam1 OPS 會改變整體格點，Cam2～7 OPS 也必須各自改變
  對應相機的物理寬度與重疊分界，禁止只讀 Cam1 或把七台 raw 像素寬直接當成相同實體寬度。
- Start 決定相機在共同格點的 X 起點。OPS／Start 在未 Grab 與 Grab 中都立即更新布局、橘框及
  欄圖表視野。瀑布歷史必須以 `layer × camera × chunk` 保存原始相機像素，LOD 顯示時才依當前
  OPS／Start／Crop 合成；因此 `historyRows>0` 的 remap 必須保留 write head 並重新定位既有歷史，
  不得只影響後續 band。`layout presented` 才代表新布局真正上畫；Stop 只把最後值封存進
  `#LAYOUT_FINAL`，不得再出現第二次布局跳動。
- OPS／Start 仍是純顯示／座標布局：不得縮放來源 frame、重跑 pipeline、重算 Curve bin 或改寫圖片。
  瀑布不得保存已套布局的整張合圖；每台相機只保存自己的原始像素。OPS／Start／Crop 改變時，
  只更新顯示取樣表並重建可見 LOD，既有歷史和後續 band 必須使用同一份最新布局。
  只有來源相機實際 pixel width 改變時，才允許清除無法再按原 stride 解讀的歷史並建立新世代。
- A輪速度只影響列方向物理座標，仍可維持 Grab 結束封存時套用，不屬於本條即時 X 布局。

### S4 監控強化（hc_EnableMuraEnhance）
```
T1: ui:設定[hc_EnableMuraEnhance]=True|False
T1: setting route hc_EnableMuraEnhance owner=Enhance effects=None
T1: live enhance enabled=True|False direction=raw|column|row cams=N scope=all-cameras waterfallHistory=preserved
T1: WF layer raw|column|row->raw|column|row writeRow=N history=preserved   ← 只在瀑布 view 已存在時
```
- 設定套到所有已配置相機；`AniloxCamera.OnMilFrameReady` 每幀重新讀
  `EnableImageProcessing`，因此 grab 中切換會影響後續全部幀，不是只處理按下當下的單張。
- GPU 每幀本來就同時計算欄／列強化；`OnWaterfallFrame` 把 raw／column／row 當成同一物理幀，
  `WaterfallView` 只做一次 tick 對齊，再把同一 band 寫入三個 lazy chunk layer。
- 瀑布切換強化或欄／列方向時 `waterfallHistory=preserved`：只准 `SetDisplayLayer` 換 LOD 讀取層；
  `_writeRow`、pending slots、tick 網格、zoom 與 pan 都必須不變，禁止呼叫 `Reset` 或重建 view。
- 三層固定上限：以 101171×30000 灰階計約 8.48 GiB；採 512-row lazy chunk，尚未寫到的區域不配置，
  view dispose 時全部釋放。這是使用者核准的記憶體換無縫切換政策，不得退回清空瀑布。
- `WaterfallView.Reset` 必須遞增 content generation；已被背景 writer 取走的舊 generation band，
  以及 Reset 前已開始計算的舊 LOD tile，在真正重新 Grab、改總高或佈局 Reset 後都不得安裝，
  避免清空後殘留一條舊來源影像。`ImageCanvas.ApplyLodTile` 是 LOD generation 的最後守門。
  新 Grab Reset 還必須以 `RefreshLod(clearCurrentTile:true)` 立即清除目前可見 tile；一般串流更新維持
  `clearCurrentTile:false`，保留上一 tile 等新內容完成，兩種語意不得混用。首個 `WF band first`
  的 generation 必須等於該輪 Reset，且 `startRow=0`。
- 新配置相機由 `AllocateCamerasAsync(enableEnhance)` 取得同一設定，不能另有預設值。

### S5 強化熱力圖（hda_EnhanceHeatmap）
```
T1: ui:設定[hda_EnhanceHeatmap]=Off|Cold|Warm|BlueYellowRed|Green
T1: setting route hda_EnhanceHeatmap owner=Enhance effects=None
T1: enhance heatmap mode=Off|Cold|Warm|BlueYellowRed|Green live=gray|cold|warm|blue-yellow-red|green review=gray|cold|warm|blue-yellow-red|green scope=main-only data=unchanged
```
- 熱力圖是 **8-bit 顯示調色盤**，固定以 0..255 映射；不得依每張圖的 min/max 自動拉伸，
  否則不同時間的顏色不能比較。四種模式都讓 0 保持純黑：Cold=黑→藍→青→白、
  Warm=黑→紅→黃→白、BlueYellowRed=黑→藍→黃→紅、Green=黑→綠→白。
- 只套用監控與回顧的主畫面強化圖；原圖、背景預覽與縮圖恒為灰階。滑鼠亮度仍回報原始 0..255 值。
- 切換只准重畫現有 bytes／瀑布 LOD tile；不得重讀圖片、重算 Curve、清空瀑布、改 fit／pan／zoom，
  也不得修改 raw/proc 圖檔、bin、CSV 或檢測結果。

### S3 上下方向（hee_VerticalDirection）
```
T1: ui:設定[hee_VerticalDirection]=TopToBottom|BottomToTop
T1: ApplyMainDisplayMode / 影像方向套用（依目前主畫面模式）
T1: LC row rowChart dir=… 或 LC row rowView dir=…     ← grab 停止後也要重套最後一組列圖表資料/視野
T1: RV row …（Review 有資料時）或 RV load/update row（依當前 Review 模式）
```
- 改上下方向不得新增 `AxisY.IsReversed`、`Reverse`、`total-` 等外層鏡像層；只准透過
  `RowCurveDisplayAdapter.ApplyDirection → RowCurveChartHelper.ZeroAtTop` 重畫既有資料/視野。
- 違規樣本：grab 停止後改 PropertyGrid 上下方向，主監控畫面跟著翻，但 `chartLiveRow`
  仍維持舊方向＝`RowCurveSyncCoordinator` 沒有對最後資料做方向刷新。

**自動校稿（`flow_checks/settings.py`）**：
- `S0.format`：所有 `ui:設定[]`／`set:[]` 必須同時帶屬性名與新值。
- `S0.route`：每個設定 intent 下一行必須有同名 `setting route`；並檢查
  `CapturePolicy` 只出現在允許的六個設定，防止無關設定重送相機參數。
- `S2.review-enhance`：切強化必須依當前回顧模式，以同一 grabId 完成
  `RV loadGrab begin → pushFrames(chartView=keep) → variantView keep(maxDelta≤0.1mm) → done`，
  或以同一時點完成 `RV period load`；尚無回顧資料則回 `NOT COVERED`。
- `S4.live-enhance`：有已配置相機時，切監控強化必須出現相同值且
  `scope=all-cameras waterfallHistory=preserved` 的狀態行；enabled=False 必須是 raw，
  enabled=True 必須是 column 或 row，證明全相機狀態一致且瀑布歷史未被清除。
- `S5.enhance-heatmap`：熱力圖 intent 後必須緊接 route 與 `scope=main-only data=unchanged`
  的狀態行，所選模式只能映射成同名 palette 或 gray。
- `S6.display-crop`：流程驗證模式下，每次去頭／去尾 intent 後必須緊接 route、
  `displayCrop applied` 與 `scope=main+column-chart data=unchanged waterfallHistory=preserved`，
  且 head/tail 必須立即反映新值。Grab 中另外要求
  `capture layout pending … apply=display-now+stop-final`；Stop 後 final/applied 必須等於最後值。
  若該輪只有 Crop 改變，必須是 `render=already-applied`，禁止停止時再跳一次。
  狀態行；Off 後 live/review 都必須回灰階，非灰階輸出必須與所選模式一致。固定相鄰行也保證
  切換中沒有插入圖片或 Curve 重載。
- `S7.live-layout`：流程驗證模式下，每次 OPS／Start intent 後必須有同名
  `displayLayout applied`，七台 OPS／Start 快照必須完整且改動槽位等於 PropertyGrid 新值；
  Grab 中若有 pending，必須是 `apply=display-now+stop-final`，不得延後到 Stop 才更新。瀑布模式
  另外要求 `WF layout remap storage=per-camera` 與對應 `layout presented`；已有歷史時
  `historyRows` 不得歸零，slot 必須位於 virtual width 內。
- `S3.direction`：方向變更後、下一次方向變更前，必須看見相同方向的
  `LC|RV row rowChart|rowView`，證明最後一組列資料/視野已重畫。

（其餘設定逐一補進：每補一個 UI 功能，順手寫它的 S 條目。）

## 報表 tab 契約（D 系列）

### D0 tab 切換（全域，不限報表）
```
T1: ui:tab → 監控|回顧|報表
（tab 切換本身不觸發顯示重建；切到回顧固定接 `RV tabVisible repaint view=True|False`，只補可見重繪；
  若 `_reviewDirty` 且有 Data pending selection，先接 R2 載入序列再重繪。
  開機 PrewarmAllTabs 的程式化 cycle 被 _suppressTabIntent 抑制不記——毫秒級三連發 tab 行
  ＝抑制失效（D 系列首輪誤報實例））
```

### D1 讀取資料（btnDataSelectFolder）
```
T1: ui:【讀取資料】鈕（Data）
T1: DT stats snapshot csv=N records=N grabs=N ms=N
    ← 報表 CSV 只掃一次，同時建立時間、序號、七台欄判定與整筆列判定索引
（會連動 Review → 接 R1 的 RV 序列）
（預設：單片=最新、序號範圍=最舊→最新）
```

**code-flow（一次解析，多個 view 共用）**
```
LoadDataFolder|SyncFromReviewFolderAsync@DataStatisticsPresenter.cs
 → LoadStatisticsSnapshot|Task.Run(LoadSnapshot)@InspectionStatisticsService.cs
    ├ 只讀 yyyyMMdd.csv（排除 _ticks.csv）
    └ 一次產生 AvailableTimes／GrabIdsDescending／DetailsByGrabId
 → PopulateAllGrabIdCombos|PopulateAllGrabIdCombosAsync／RefreshStats／YieldPeriodChartPresenter
    └ ComputeGroupedByMonthOfYear|DayOfMonth|HourOfDay（索引 overload，不再掃 CSV）
```
- **初始載入 SSoT**：同一資料夾＋同一門檻下，序號 List、色卡、年月日圖表必須共用同一份
  `InspectionStatisticsSnapshot`。初始【讀取資料】不得分別呼叫 `LoadAvailableTimes`、
  `LoadGrabIdInfosDescending`、`ScanCsvByDateRange`重複掃磁碟。
- **Curve／判定一致**：MeanC/MaxC/MeanR/MaxR bin 是 Curve 樣本 SSoT；CSV
  `MeanPeak/MaxPeak/MaxCMean/MeanRPeak/MaxRPeak`
  是 capture 當下從同份資料產生的可重建標量索引。改欄正規值／Mean／Max 門檻後，
  300ms debounce 只依新設定重判 snapshot／List／色卡／年月日圖表，不改寫歷史 CSV、
  不逐筆重讀 bin。bin 被外部修改時必須重建 CSV 索引；兩者分歧＝資料完整性 FAIL。

### D2 明細列表點選
```
T1: ui:【明細列表】→ {grabId}
T1: DT verdict click {grabId} cam=N mode=mean|max|both mean=M/T enabled=0|1 max=X/T enabled=0|1 result=pass|fail|unknown cause=none|mean|max|both list=pass|fail|unknown source=visible-curve-index|curve-index|missing
T1: DT verdict click done {grabId} cams=N
T1: DT curve display {grabId} mode=mean|max|both mean=M/T max=X/T scale=S points=N
    ← 每次點列逐相機列出實際峰值、當下門檻、啟用狀態、公式結果及 List 畫出的 O/X；
      `DATA/D1.verdict-click` 用同一公式重算，並要求 result=list、相機編號 1..N 完整。
      新資料必須是 `visible-curve-index`：先依 OPS／Start／重疊中線合成最終可見 Curve，
      再依合併器回傳的 owner 相機取峰值；`curve-index` 僅供舊版 Log 相容判讀。
      `DT curve display` 是 chart 真正畫出的縮點峰值；Mean/Max 縮點都必須保留桶內峰值，
      不得再次平均而藏掉會觸發 O/X 的窄尖峰。
T1: ui:【報表序號】→ {grabId}          ← 同步行（明細點選經 cbDataId commit，兩行成對＝正常）
T1: ui:【明細列表】同列再點 {grabId} → 回範圍模式
（範圍序號 cbDataIdStart/End 不因單片選取而變＝獨立。
  Review 同步＝標記 dirty **lazy**：切到回顧 tab 才接 R2 載入——明細點選後零 RV 行＝正確，
  D 系列首輪 log 實證；點選當下就出 RV 行反而是違規）
```

**code-flow（清單 View wiring 與報表 policy 分離）**
```
listViewGrabDetail.MouseUp
 → OnMouseUp@GrabDetailListBinder.cs（virtual row → grabId＋是否同列再點）
 → RowCommitted event
 → OnGrabDetailRowCommitted@DataStatisticsPresenter.cs
   ├ 同列再點＋SingleSheet → GrabDetailListBinder.ClearSelection → SetActiveStatGroupBox(range) → RefreshStats
   └ 一般點選 → CommitDataGrabIdFromDetailList@DataDateGrabIdNavigator.cs → D3 單片快路
```

### D3 報表序號 / 序號範圍
```
T1: ui:【報表序號】→ {grabId}          ← 單片切換（同 D2 的 cb 版）
（快速滾動合併）DT selected coalesced {grabId} skipped=N intervalMs=33
T1: DT curve load {grabId} captures=N source=shared storage=summary|bins|memory-summary|memory-bins configMs=N waitMs=N pathMs=N mergeMs=N summaryMs=N points=N drawMs=N totalMs=N
     ← 回顧／報表共用 `SingleGrabCurveDataLoader`；storage 表示持久匯總、原始 bin 或同來源的記憶體命中
T1: DT verdict {grabId} cam=N mode=mean|max|both mean=M/T enabled=0|1 max=X/T enabled=0|1 result=pass|fail cause=none|mean|max|both source=visible-merged-curve|merged-curve
     ← 報表欄 O/X 量實際上畫的最終可見合併曲線；多幀先合成每台相機 Curve，再與欄圖表共用
       `CurveOverviewMerger + MergeLayout(Midline)` 的 OPS／Start／重疊歸屬。每台相機只檢查 owner
       屬於自己的可見點，被相鄰相機遮掉的峰值不得造成 X。`merged-curve` 僅供舊版 Log 相容判讀。
       「欄曲線判定」同時控制顯示與判定：
       顯示平均只啟用 Mean、顯示最大只啟用 Max、顯示兩者則兩者任一超標即 X。
       Mean 只對平均閾值、Max 只對最大閾值，
       `cause` 直接標示觸發的閾值，不可因 Max 高於平均線就誤判為 Max 失敗。
     ← `DATA/D1.verdict` 逐行重算 result/cause；把 CurveMax 誤拿去比較平均閾值會直接 FAIL。
T1: DT verdict index apply=ok gen=G summaries=S bins=B missing=M/R cams=C verdicts=V ms=N
     ← 整份明細清單以同一套最終可見合併 Curve 重判。先讀 `_curve_summary`，舊資料缺 summary 時在背景
       批次掃一次 CSV 並合併原始 bin；`S+B+M=R`、`C=V`，完成後 List、卡片與期間統計一起刷新。
T1: DT row curve load {grabId} source=shared storage=summary|bins|memory-summary|memory-bins points=N pitch=Nmm
     ← 單片模式欄／列 Curve 同一刻套用同一份 load result，不得另掃一次路徑
（快速滾過的舊選取）DT curve stale-drop {grabId}
     ← 僅切模式／清快取等明確 invalidation 可使 running 結果失效；單純較新序號到達不得丟掉已開始的結果
（快速滾動合併）DT curve coalesced {grabId} skipped=N minCycleMs=33
     ← 前一筆已快速完成但仍在呈現週期內；中間 pending 被最新序號覆寫，N 為省略的未開始筆數
（舊資料缺 Row bin）DT row curve missing {grabId}，畫面清空
（切序號範圍／年/月/日）DT row curve clear mode=range
     ← 列沒有相機起始線，只對單序號反映；範圍模式不得保留上一筆列 Curve
T1: DT curve load policy latest-only shared-loader entries=512 maxMB=256 scale=merged-only minCycleMs=33
     ← Presenter 初始化一次；孤立選擇立即載入，連續快滾最多約 30 次/s 上畫；報表不再自行同步等待或相鄰預讀
Tn: DT curve summary {grabId} write=queued|ok|failed|dropped|evicted|skip-incomplete captures=N merged=N ms=N [reason=idle|pressure]
     ← merged=captures 才排入 bounded queue；一般於序號互動停止 750ms 後由單一背景 writer 寫入；
       pending 達 72 MB 壓力線時允許單線 pressure drain，防止持續滾動只排隊不落盤
     ← `check_all_flows.py` 的 `D3.summary-write` 將 failed/dropped/evicted 或格式錯誤直接判 FAIL；
       避免畫面因記憶體 cache 正常而掩蓋「匯總其實沒有落盤」
T1: DT selected {grabId} stats=cache|scan list=keep ms=N
     ← 單片快路：更新七台色卡＋列 O/X 色卡＋欄/列 Curve＋List 反白；`cache`＝從既有明細／全序號索引推導統計，`scan`＝索引找不到該序號時 fallback
T1: DT stats index rows=N ms=N
     ← 門檻設定變更使 snapshot 簽章失效時，重建一次全序號 Pass/Fail 索引；
       初始讀取已由 D1 snapshot 建立，同資料夾＋同閾值後續不得重掃 CSV
T1: ui:【序號範圍-起始|結束】變更       ← 手動拖範圍 → 期間高亮全滅（Custom）
T1: DT range policy listMs=33 curveMs=80 settleMs=150 curveMode=monotonic curveSamples=50 curveCacheEntries=2048 curveCacheMB=256
     ← DataRangePreviewCoordinator 初始化一次；目前上機驗綠的排程基準，改值必須同步本契約＋checker並重驗
T1: DT range list preview gen=G range={start}~{end} rows=N ms=N source=index
     ← 滾動期間 List＋7 台色卡預覽；獨立 33ms 節拍，只切記憶體完整索引，不讀磁碟
T1: DT range preview apply gen=G latest=L range={start}~{end} loadMs=N drawMs=N meanRows=N maxRows=M method=top-maxcmean|mixed|even coverage=S/R rankedCams=C/T index=H/B cache=H/M hmCoverage=K/R hmCurrent=V sampleLimit=50
     ← 單一 running 工作完成就上畫；若 `G<L` 表示滾動期間跳過中間選取，完成後下一輪直接接最新 generation
T1: DT range settle → refresh             ← 最後一次變更後 150ms；一串連續滾動只准一行
T1: DT list reload range={start}~{end} rows=N ms=N source=index
```
- **單片呈現節流**：ComboBox 選項逐格跟手；統計卡、List 高亮與 Curve 排程以 33ms 固定週期只套最新序號，
  中間序號記 `DT selected coalesced` 後省略，最後停住的序號不得省略。這避免每個滾輪事件都在 UI thread
  重繪七張統計卡與同步寫高密度 Trace。
- **List ownership**：明細 List 屬於範圍結果，不屬於單片序號；`ui:【報表序號】` 後只准 `list=keep`，
  不得出現 `DT list reload`／`GrabDetailListBinder.SetItems`／重設 `VirtualListSize`／欄寬。只有資料夾、範圍、期間、閾值改變才重算 List。
- **列判定索引**：明細第 8 判定欄固定為「列」；O/X 由同序號所有相機／capture 的
  `MeanRPeak/MaxRPeak` 一票否決。view-time 公式＝raw row peak × `HM_V_capture/HM_H_current`
  對當前列門檻。舊 CSV 沒列峰值只能顯示 `—`，不得當 Pass。篩選異常也必納入列 Fail。
- **List 捲動顯示**：資料已全在 `GrabDetailListBinder._visibleDetails`，VirtualMode 不需資料預載；ListView 啟用雙緩衝，選中列只在接近
  可視區上下邊界時以 margin 捲動，反白變更只重畫舊／新兩列，不得每格整窗 `Invalidate()`（跨視窗白閃的根因）。
  欄寬先依內容 fit；工作區縮放後若總欄寬小於可視寬度，剩餘寬度全補到序號欄，避免全寬模式留下白色空欄。
- **Virtual List 生命週期**：`SetItems` 必須先令 `VirtualListSize=0`，再替換私有資料快照並發布新列數；
  `Dispose` 必須先清空列數，最後才解除 `RetrieveVirtualItem`。每次 native request 都先得到完整 placeholder，禁止讓
  `e.Item=null` 逃出事件。`DT list virtual fallback ...` 是已恢復的結構異常，`DATA/D2.virtual-list` 仍判 FAIL；
  同一原因最多每 5 秒記一行，禁止例外／log 風暴。
- **跨 tab lazy**：報表序號只覆寫 pending selection 並標 `_reviewDirty`，不得逐格操作隱藏的 Review combo/date、
  `NavigateTo` 寫 session／重建日期清單，也不得當下載 Review 圖片；切到 Review tab 才一次套控制項並接 R2 完整載入。
- **跨 tab 曲線重用**：報表欄／列曲線完成後記 `DT curve share {grabId} target=Review`，保存同 root、同 grabId
  的原始 `SingleGrabCurveData`。切到回顧時，兩個回顧 chart 可各自套用一次記憶體資料，但圖片載入必記
  `RV loadGrab curves=reuse source=Data {grabId}` 並跳過四種 curve bin；不得反向通知隱藏報表再畫。
  無相符快照才准 `curves=load source=bin` fallback。`DATA/D3.review-reuse` 自動驗這條不變量。
- **時序索引只建一次**：`ImageRepository.LoadDirectory` 建立排序去重的 available-period index；報表／回顧每格同步
  只能對既有索引做查找，不得在 UI 執行緒重新解析全部影像檔名、`Distinct`、`OrderBy`。
- **單片 Curve latest-only**：快速滾動時 running request 可記 `DT curve stale-drop`，尚未開始的 request 只保留最新一筆；
  即使 `.mcsf` 快到兩次 UI intent 間已完成，也由 33ms 最短呈現週期保留 backpressure，並以
  `DT curve coalesced` 記錄省略筆數。這只省略中間上畫，不縮減任何一次實際上畫的 Curve 資料；
  已開始的 result 必須依序完成上畫，再接 pending latest，避免連續 intent 讓所有 running result 都 stale 而畫面停住。
  只有離開單片模式、清快取等明確 invalidation 才准 `DT curve stale-drop`；
  最後停住的 `ui:【報表序號】` 必有同 grabId 的 `DT curve load`，且列資料存在時必有 `DT row curve load`。
  欄／列必共用同一份 `SingleGrabCurveData`；快取保存 rescale 前欄 Mean/Max 與合併後列 Mean/Max，資料夾重載或 Presenter Dispose 必清空。
- **快 Curve／慢圖片可並行**：R2 圖片仍在 debounce 載入舊 grabId 時，後續序號的 `RV prefit/prefitPaint/prefitApply`
  可先推進 Curve；DVT 只把同 grabId 的證據歸入該次圖片 lifecycle，不得把不同 grabId 當格式錯誤。
- **單片 Curve 預排版**：報表選取序號後，以 `ReviewImageDataLoader.Prepare` 只讀該筆路徑、CFG 與 JPEG header
  （不解碼圖片），再經 `ReviewDisplayManager.TryComputeFitViewRange` 使用與回顧主畫面相同的合圖／fit 公式。
  `DT prefit {grabId} content=WxH viewX=L~R viewY=T~B source=main-geometry` 必須在同 grabId 的欄／列 Curve
  上畫前出現；報表不得讀取回顧頁上一筆 view，也不得維護第二套 fit 公式。切到回顧後，R2 的 `RV prefit`
  以同一公式先同步主畫面與欄／列圖表，完整回顧載入的 `ViewRangeMmChanged` 再覆核。
  `DT chartRange {grabId} chart=col|row axis=A~B/view=L~R` 是報表圖表 PostPaint 的實際狀態邊緣；
  同一選取在資料上畫後、且仍處於單序號模式時不得出現第二組 Axis／View。切入序號範圍或期間模式後，
  Curve 改顯示範圍統計，座標改變合法，DVT 必須停止追蹤前一筆單序號視野。不得以 `null` 視野覆寫成全幅。
- **單片 Curve cache 基準**：LRU 上限 `512 筆／256 MB`，納入列 Curve 後以目前 278 筆實測資料可整批容納，避免往返滾動時
  反覆淘汰／重載造成 Gen2 GC；30,000 筆時仍保持固定上限。view-time HM rescale 不得 clone 每台完整 raw Curve，
  只能在 `CurveOverviewMerger` 產生最多約 2,000 點的 merged result 後縮放。這些是可重驗調整的效能參數，非鐵則。
- `check_all_flows.py` 的 `DATA/D3.selected` 允許最多一筆真正缺 ID 的 `stats=scan` fallback；同 session 多筆
  `stats=scan` 代表全序號索引失效或未使用，直接 FAIL。
- **持久匯總不取代原始資料**：讀取順序＝記憶體 cache → `SingleGrabCurveSummaryStore` → 原始 MeanC/MaxC bins。
  `.mcsf` 是可重建 materialized view；格式版本、grabId、Earliest/Latest、cameraCount 任一不符或內容損壞時必退回 bins，
  完整重建後先顯示 Curve，再由 idle writer 以同目錄暫存檔原子替換；不得讓 UI 等待落盤。原始 bins 仍是 SSoT，
  匯總保存 rescale 前逐相機欄 Mean 平均／Max 最大，以及合併後列 Mean／Max；pending queue 上限 96 MB，
  72 MB 起由單一 writer pressure drain；writer 仍追不上時可 evict/drop，不可無界吃記憶體。列匯總是多幀串接，
  合法長度可超過單一 bin 的 200,000 點；匯總以 2,000,000 點／曲線及 96 MB／檔雙重守門。
  `merged != captures` 時只能回傳當下可讀結果並記 `write=skip-incomplete`，不得產生匯總；下次選取必重新嘗試原始 bins。
- **範圍 monotonic 跳讀 throttle**：起始／結束 combo 連續滾動時，33ms List timer 從完整明細索引切出當代範圍並更新
  List＋7 台色卡；獨立 80ms Curve timer 同時最多一個背景工作，固定使用完整 `50 Mean＋50 Max`。工作期間的新 intent
  不建立佇列、不取消 running 工作，只更新 latest generation；running 完成後照常上畫，再直接取最新範圍。因此快速滾動
  會看到 Curve 依完成速度跳著更新，中間序號自然略過，不會無界累積工作。停止後最終 Curve generation 必須追上 List。
  自動壓力情境以至少 100 筆 intent 為大量滾動門檻；必須觀察到至少兩次 Curve apply、至少一次 `G<L` 的中間跳讀，
  apply 數必須少於 intent 數，且最後一筆 Curve generation 追上最後一筆 List generation，否則 DVT FAIL。
  30,000 筆冷磁碟基準把背景計算與 UI 響應分開驗證：第一筆完整 `50 Mean＋50 Max` 冷讀可在 3000ms 內完成，
  後續快取已建立的 Curve apply 最慢 500ms；滾動證據窗內 UI stall 仍不得超過 1000ms。初始化資料夾、CSV 與明細索引
  必須在證據窗開始前完成，不得把載入資料的成本誤算成滾動卡頓，也不得因冷讀較慢而縮減 50/50 樣本。
  `DT range settle` 前允許 `DT range list preview`／`DT range preview apply`，但出現 `DT list reload` 或同步
  `DT curve candidates`＝逐格完整重算回歸。List 預覽只能使用已就緒且資料夾／閾值簽章相同的記憶體索引；
  索引未就緒時略過預覽，交由 settle 建立，禁止在 80ms 路徑掃 CSV。
- **節拍／樣本參數的效力**：`33/80/150ms` 與 `curveSamples=50` 是目前基準，不是不可修改的使用者鐵則。它們必須集中在
  `DataRangePreviewCoordinator` 的具名常數並由 `DT range policy` 量出實際執行值；任何調整都視為排程行為變更，
  同一批修改 DVT 與 checker 後重跑上機測試。不可散落 magic number，也不可只改文件或只改 code。
- **範圍曲線只有兩條**：每台相機各自選候選再合成全寬；`CurveMean`＝範圍 CSV 資料列均勻取樣最多 50 筆後
  對對應 `MeanC` bin 逐點平均；`CurveMax`＝依 `MaxCMean` 排序取前 50 筆，再對其 `MaxC` bin 做逐點最大值。
  候選必須保留 CSV `FileName` 並載入同一筆 bin，不得只選序號後誤讀該序號第一張。這個設計保留平坦趨勢，也不讓
  1/1000~1/10000 的凸波因均勻抽樣直接消失；畫面不得再增加操作員難以判讀的第三條曲線。
- **範圍正規值仍是 view-time 設定**：每日索引必須沿 `#CFG` 保存每筆 `HM_V_capture`；Mean 候選與
  Max 候選的原始 bin 都先乘 `HM_V_capture/HM_V_current` 再逐點彙總，`MaxCMean` 排名也套同一比例。
  因此改 PropertyGrid 欄正規值後只重畫目前範圍，不改寫 CSV/bin，也不得污染 immutable Curve cache。
  `hmCoverage=K/R` 是有拍攝 CFG 的資料列數；舊資料缺 CFG 時該筆比例退回 1 並明確留在 coverage 缺口，
  不得假造歷史設定。`hmCurrent` 必須等於本次上畫使用的目前欄正規值。
- **範圍 Curve bin 快取**：候選與統計規則完全不變；不同相機最多四路平行載入，已讀取的 immutable capture bin
  以完整路徑放入 LRU，固定上限 `2048 筆／256 MB`。`cache=H/M`＝本次命中／冷讀檔數；相同範圍或重疊範圍
  再次計算應逐步轉成 `H>0`。快取只保存原始 float 陣列並視為唯讀，原始 bin 仍是 SSoT，可隨時淘汰重建。
  這兩個容量值和四路平行度是可重驗的效能參數，不得用快取改變 50/50 候選集合或 Max 排名。
- `coverage=S/R` 是有 `MaxCMean` 的 CSV 資料列數／範圍資料列數；任一相機候選資料不完整時，該相機
  `CurveMax` 回退均勻取樣，避免拿新舊混合資料宣稱精確排名。
- **候選索引是衍生快取，不是資料真相**：`LoadRange` 以每日 CSV 完整路徑＋檔案長度＋最後修改時間驗證
  記憶體索引；簽章相同才可命中，CSV append/替換後必重建。原始 CSV 與 Curve bin 仍是 SSoT；索引只保存
  `grabId/camera/basePath/MaxCMean/HM_V_capture`，以 LRU 限制最多 25 萬筆／1024 日且可隨時丟棄。`index=H/B`＝本次命中／重建的日數；
  連續滾動在第一次暖索引後應以 `H>0,B=0` 為主。每日索引開始建立後允許完成並供下一代共用，
  不跟 range token 中途取消；token 仍在候選彙整／bin 載入／上畫前生效，故舊 generation 不得上畫。

**code-flow（單片快路 vs 範圍完整刷新）**
```
cbDataId.SelectedIndexChanged
 → OnSingleSheetComboChanged@DataDateGrabIdNavigator.cs
   → ScheduleSelectedGrabRefresh@DataStatisticsPresenter.cs
     └ 33ms Timer：合併中間選取，只將當下最新序號送入 RefreshSelectedGrab
   → RefreshSelectedGrab@DataStatisticsPresenter.cs
     ├ _currentDetails.FirstOrDefault（範圍內命中）／EnsureSingleGrabDetailIndex（範圍外只建一次全序號索引）
     │  └ 命中→BuildSingleGrabStats；索引仍無該 ID 才允許單 ID CSV scan fallback
     ├ InspectionStatsPresenter.Update／UpdateRowResult（7 台色卡＋camDataRow 列 O/X）
     ├ GrabDetailListBinder.Highlight（只移反白＋EnsureVisible＋RedrawItems）
     └ MuraProfileChartPresenter.Update（該 ID curve）
       ├ LatestGrabLoadCoordinator.Enqueue（running 依序完成上畫、pending 覆寫成最新；明確 invalidation 才 stale-drop）
       └ LoadSingleGrabCoreAsync → Task.Run
          ├ ReviewImageDataLoader.Prepare（只讀路徑／CFG／JPEG header，不解碼圖片）
          │  → ReviewFitViewRangeProvider → ReviewDisplayManager.TryComputeFitViewRange
          │    → ImageDisplayView.TryComputeMergeFitViewRange（報表／回顧主畫面 fit 公式唯一來源）
          │      → DT prefit（同 grabId，必在欄／列 Curve 上畫前）
          └ SingleGrabCurveDataLoader.Load（與回顧共用）
          ├ SingleGrabCurveCache.TryGet／GetOrLoadAsync（同 key in-flight 共用；rescale 前結果 LRU）
          ├ miss → SingleGrabCurveSummaryStore.TryLoad（命中＝一次 sequential read）
          ├ summary miss/stale/corrupt → InspectionImagePathRepository.LoadForGrabId
          │  ├ 各相機依序 CurveMergeHelper.MergeCurves（MeanC/MaxC）＋MergeRowCurves（MeanR/MaxR）
          │  └ MergeRowCurvesOverlap（所有相機同一列重疊）
          │     → CurveBinFile.Load（每個 bin bulk read；邊讀邊合併）→ SingleGrabCurveSummaryStore.QueueSave
          │       → idle 750ms／pending 72MB pressure → 單一 writer → TrySave（原子寫回）
          ├ cache raw Curve 唯讀直入 CurveOverviewMerger → 合併後小曲線套 Hessian ratio → UpdateOverviewChart
          │  （不得為 view-time rescale 複製每台完整 raw Curve；只縮放新建的 merged display result）
          │  └ ColumnCurveChartHelper.UpdateDataAndView（每兩個畫布像素一個顯示桶；Mean=桶平均、Max=桶最大，
          │       點數相同時原地更新既有 DataPoint；不得逐格 Clear＋DataBind 重建）
          └ row raw Curve clone＋HM_V_capture/HM_H_current rescale
             → RowCurveDisplayAdapter.UpdateData → RowCurveChartHelper（與監控／回顧列圖表同格式與方向規則）
     → DT selected … list=keep
     → GrabIdSelectedFromData → OnDataGrabIdSelected@AniloxRollForm.Data.cs
      └ 覆寫 pending selection＋_reviewDirty=true（不碰隱藏 Review 控制項）

tabMain.SelectedIndexChanged（進 Review 且有 pending）
 → cbReviewId＋DateTimeNavigator.SetPeriodToCombo＋UpdatePeriodNavigationState（只套最後一筆）
   → ImageRepository.GetAvailablePeriods（預建索引＋binary search）
 → DT review sync apply {grabId} → LoadGrabStitchedViewGuardRowRangeAsync（R2）
   ├ 已有 `DT curve share {grabId}`：套用報表的 `SingleGrabCurveData` 記憶體快照
   │  → RV loadGrab curves=reuse source=Data {grabId}
   │  → ReviewImageDataLoader.Load(includeCurves=false)（只載圖片；不得再讀 MeanC/MaxC/MeanR/MaxR）
   └ 無同 root／同 grabId 快照：RV loadGrab curves=load source=bin {grabId}（完整 fallback）

cbDataIdStart|End 手動變更
 → ScheduleRangeRefresh@DataStatisticsPresenter.cs
   → Start@DataRangePreviewCoordinator.cs（Timer／generation／cancellation 唯一 owner）
     ├ MuraProfileChartPresenter.ClearRow（列圖只屬單序號；範圍 mode 清空）
     │  └ LatestGrabLoadCoordinator.Invalidate（單片 running result 不得晚到覆蓋範圍 Curve）
     ├ generation++（running Curve 不取消；中間 intent 只保留 latest）
     ├ 33ms repeating throttle → ListTimer_Tick@DataRangePreviewCoordinator.cs
     │  └ ApplyRangeListPreview（只在完整 detail index 簽章有效時）
     │     └ 依 navigator 的範圍序號 SSoT 切片 → ApplyFailFilter／GrabDetailListBinder.SetItems
     │        ＋ ComputeStatsFromDetails／InspectionStatsPresenter.Update → DT range list preview
     ├ 80ms repeating throttle → CurveTimer_Tick@DataRangePreviewCoordinator.cs（同時最多一個 Curve 工作）
     │  └ UpdateRangePreviewAsync@MuraProfileChartPresenter.cs（sampleLimit=50；running 完成後上畫）
     │    → Task.Run → LoadRange@InspectionMuraProfileRepository.cs（可取消）
     │       ├ 每日 CSV 簽章相同 → DailyIndex 依 grabId 取候選資料＋拍攝 HM（不重掃 CSV）
     │       └ 簽章改變／首次使用 → 重建該日索引（LRU bounded 25 萬筆／1024 日）
     │    → 每筆候選以 HM_current/HM_capture 換算（含 MaxCMean 排名）→ 50 Mean／50 Max 彙總
     │    → 回 UI 執行緒；token 有效即 UpdateOverviewChart → DT range preview apply gen=G latest=L
     │       （G≤latest；若 G<latest，下一輪直接接最新；generation 不得倒退或重複）
     └ 重壓 150ms settle timer → DT range settle → RefreshStats(updateRangeCurve:false)（List／色卡最終對帳）
    ├ EnsureSingleGrabDetailIndex（資料夾／閾值未變時沿用完整 GrabDetail 衍生索引）
    ├ 依 navigator 的範圍序號 SSoT 切出 detail 子集 → ApplyFailFilter → GrabDetailListBinder.SetItems
    ├ ComputeStatsFromDetails（同一 detail 子集彙總 7 台色卡；不得再掃第二次 CSV）
    └ DT list reload …

期間變更／初次載入（非手動連續滾動）
 → RefreshStats(updateRangeCurve:true) → MuraProfileChartPresenter.Update
   → LoadRange@InspectionMuraProfileRepository.cs（同步精確結果；候選算法同上）
     ├ InspectionCsvReader.TryUpdateHmFromConfig＋TryParseRecord＋TryParseTimestamp＋TryExtractCameraId
     ├ Mean 候選＝EvenSample(rows,50) → HM view-time 換算 → 對應 MeanC 逐點平均
     └ Max 候選＝換算後 MaxCMean 排序前 50 → HM view-time 換算 → 對應 MaxC 逐點最大（缺分數→均勻 fallback）
```

### D4 年/月/日期間（lblChartNav 點選）
```
T1: ui:【期間-年|月|日】→ 範圍 {最舊}~{最新}   ← 取 cbDataYield 當前值設範圍 + 該期間綠高亮（互斥）
T1: ui:【期間-年|月|日】→ 取消綁定 保留範圍 {最舊}~{最新}
    ← 範圍模式再點同一 active 期間：轉 Custom、綠高亮熄滅；起訖序號不變且不重算
T1: ui:【期間-全局】→ 全範圍                    ← 點 groupBoxGrabIdRange
（active 期間改對應 cbDataYield → 範圍跟著更新；取消綁定或非 active 來源不觸發）
```

### D5 良率導航 / Y 軸暫時切換 / 篩選異常
```
T1: ui:【良率導航-年|月|日】→ {值}      ← 良率三圖跟著換週期
T1: ui:【良率圖-年|月|日】→ Y軸={Auto|Fixed} setting={Auto|Fixed} override={Auto|Fixed|off}
    ← 點圖表本體；暫時態不回寫 Chart.ScaleMode。有效模式＝該圖 override ?? setting；資料刷新與設定變更
      都從 YieldPeriodChartPresenter.ApplyScale 單點套用，禁止以 chart.Tag 另存狀態。
T1: ui:【篩選異常】→ 只顯示異常|顯示全部 dataOptions=N rangeOptions=N selected={序號|empty} range={最舊}~{最新}|empty
```
大量資料下，異常篩選重建 cbDataId／範圍起點／範圍終點時，必在各 ComboBox 批次填充之間
讓出 UI 訊息迴圈；30,000 筆 DVT 的回顧／報表互動期間 `UiStall` 仍不得超過 1000ms，
不得以放寬門檻掩蓋同步重填。

**code-flow（篩選異常）**
```
BtnShowFail.Click → BtnShowFail_Click@DataStatisticsPresenter.cs
 ├ EnsureSingleGrabDetailIndex（Pass/Fail 依目前欄／列門檻重算）
 ├ SelectFailRangeInfos（相機任一 Fail 或列 Fail）→ `_rangeGrabIdInfos`
 ├ await RefreshFilteredGrabIdCombosAsync@DataDateGrabIdNavigator.cs
 │  ├ 同一份 `_rangeGrabIdInfos` → cbDataId、cbDataIdStart、cbDataIdEnd
 │  ├ 三個 ComboBox 各自 `AddRange` 後讓出 UI 訊息迴圈 50ms
 │  ├ cbDataId 優先保留切換前序號；被篩掉時依全量清單位置選距離最近者
 │  └ cbReviewId 維持全量；跨頁同步依 GrabId 查找，不共用 filtered index
 ├ 單序號模式：cbDataId 當前 GrabId → 統計、欄／列 Curve
 └ 範圍模式：同一份 `_rangeGrabIdInfos` → 明細、統計、範圍 Curve
```
不變量：異常模式的 cbDataId 與起始／結束下拉只含異常序號，三者數量相同；單序號與範圍運算
不得以 filtered ComboBox index 直接索引全量 `_grabIdInfos`；切換篩選不得無條件跳到第一筆，且
listViewGrabDetail 高亮必須與 cbDataId 相同；再按一次「顯示全部」才恢復全量報表序號。

**code-flow（Y 軸暫時切換）**
```
Chart.MouseClick → PeriodChart_ToggleAutoScale@YieldPeriodChartPresenter.cs
 ├ GetEffectiveScaleMode＝該圖 override ?? Settings.Chart.ScaleMode
 ├ next＝Auto↔Fixed；next 等於 setting → override off，否則只改 Presenter 內該圖 override
 └ ApplyScale（唯一套用點）→ ApplyAutoScale｜ApplyFixedScale → SetChartYRange

FillPeriodChart（資料刷新）──────────────┐
ApplyChartScaleFromSettings（模式設定） ├→ GetEffectiveScaleMode → ApplyScale
ApplyChartScaleForChart（YMax 設定）────┘
```
不變量：`Chart.Tag` 不得保存 Y 軸模式；圖表點擊不得回寫 `Settings.Chart.ScaleMode`。
500ms 防連點以 `uint TickCount` elapsed 計算，必須可跨 24.9 天有號轉負與 49.7 天回繞；
不得用 `Environment.TickCount - 0 < 500`（2026-07-13 開機 26.74 天實測會永久吞點擊）。
`check_all_flows.py` 的 `DATA/D5.y-scale` 會檢查 log 格式與
`effective = override ?? setting` 關係；未操作圖表時為 `NOT COVERED`。

## 附錄：真 log 範例（導航用）

> 函式路徑已全數升級為各契約內的 code-flow（F1~F8），本附錄只留真機 log 範例；
> 範例取自真機（4 配置/2 在線），數值隨機台不同，**log 行才是判準**。

**F1 範例**：
```
10:35:07.029 T 1 AllocateCameras begin（expect 7 cams）
10:35:07.388 T 1 ApplyMainDisplayMode → ImageCanvas
10:35:07.436 T 1 EnsureImageDisplay create + subscribe 4 cams（merge=False）
10:35:07.438 T 1 SwitchMainDisplay cam=1 center=False mode=IC
10:35:07.439 T 1 AllocateCameras done（cams=4）
Tbg acquisition parameters ready cam1 cl=True lineRate=3001
Tbg acquisition standby start cam1
Tbg acquisition standby ready cam1 tick=T
Tbg acquisition parameters ready cam2 cl=True lineRate=3001
Tbg acquisition standby start cam2
Tbg acquisition standby ready cam2 tick=T
10:35:12.901 T 1 EnableGlobalMerge（slots=7）
```

**F2 範例**：
```
10:37:13.854 T 1 StartGrab（cams=4）
10:37:13.855 T 1 ApplyMainDisplayMode → ImageCanvas
T1 capture plan grab=… root=… imageDir=… csv=… archive=….acap assets=raw|proc_c|proc_r|hessian_c|hessian_r|mean_c|max_c|mean_r|max_r preview=1920x1080x3 scale=… hessianScale=25
T1 IO grab request stopCondition=IoSignal stopOnLow=True
T1 grab stop armed condition=IoSignal limit=Ns configured=10s grace=Gs source=io grab=…
T1 capture gate open cams=2 warm=True
10:37:15.170 T31 firstFrame cam1 16384x3000 → ImageDisplayView
10:37:15.207 T30 firstFrame cam2 16384x3000 → ImageDisplayView
```

**F3 範例**：
```
10:37:21.226 T 1 StopGrab
T1 capture gate closed standby=on
Tbg capture save drain done grab=… callbacks=0 pending=0
Tbg capture finalize grab=… archive=….acap atlas=3 atlasBytes=… remoteFiles=2
```

**F4 範例**：
```
10:13:40.107 T 1 ApplyMainDisplayMode → ImageCanvas
10:13:40.108 T 1 TeardownWaterfall（unsubscribe 4 cams）
10:13:40.124 T 1 EnsureImageDisplay create + subscribe 4 cams（merge=True）
```

**F6 範例**：
```
10:13:27.999 T 1 centerCam → cam3（WF）   ← 相鄰台階梯式、間隔不規則＝健康手拖
10:13:28.372 T 1 centerCam → cam2（WF）
```

**F6b 違規樣本**（2026-07-07 修復前，教學用）：
```
10:37:16.868 T 1 IC wheelZoom in → zoom=0.02（fit=0.01）
10:37:16.973 T 1 IC wheelZoom in → zoom=0.01   ← 滾放大卻回到 fit＝有人在重設（該次＝ClearFrame 空轉→lodRebind）
```

（F5/F7 範例待真機補。）

## 任意控制項 call chain 追蹤（F1~F8 以外的流程）

契約未涵蓋的控制項，用 `$verify-flows` 的通用追蹤法做迴歸驗證：

1. **查對照表**：repo 根 `AGENTS.md` §控制項速查 找程式碼 Name。
2. **定位 handler**：AniloxRollForm.* / presenter / coordinator 搜事件繫結。
3. **追呼叫鏈**：直接呼叫（含 async/await）→ 跨元件事件 → guard flag enter/exit → 更新的控制項（標準名稱）。
4. **輸出驗證結果**：
   ```
   [觸發] 讀取資料 (btnReviewSelectFolder)
     → [動作] ImageRepository.LoadDirectory
     → [輸出] 時段日期/時間 → 填入最早值 ✅
     → [同步] Data tab → 序號+時間+統計 ✅
   [結果] N/N 通過；斷裂處標 ❌ + 行號 + 修法
   ```

批次驗證（`--all`）的輸入清單：

| Tab | 需驗證的輸入 |
|-----|------------|
| Live | 開始抓取、監控強化、監控欄/列曲線圖點擊、監控縮圖點擊、取得背景、預覽背景 |
| Review | 讀取資料、時段導航、單片序號、回顧縮圖點擊、回顧強化、回顧欄/列曲線圖點擊 |
| Data | 讀取資料、序號範圍、序號選擇、年/月/日期間、良率圖導航、良率圖 Y 軸 Auto/Fixed、篩選異常 |
| 右側 | 檢測設定（Recipe/Algorithm/ChartScale）、相機參數滑桿 |
| 跨Tab | Review→Data 同步、Data→Review 同步 |

驗證中發現 skill 與 code 不一致 → 同步更新對應 skill（契約跟 code 對齊是本 skill 的存在意義）。
