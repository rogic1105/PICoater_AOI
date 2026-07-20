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

## 三種執行方式（同一份契約，三種檢查法——契約永遠只寫一份）

1. **function（模擬測試，不開 UI）**：順著 code 把每條 flow 追一遍，推導預期的 `[Flow]` 序列，對照契約。
   改壞接線（漏訂閱/漏 teardown/置中掛錯來源）在這一步就會現形。隨時可跑、免費，**預設第一道**。
2. **log-smoke（真機比對）**：使用者操作一輪 → 讀 `{AniloxRoot}\Logs\trace-*.log` 的 `[Flow]` 行對照契約。
   log 格式：`[Flow] HH:mm:ss.fff T{執行緒} 訊息`（唯一出口 `Services/FlowTrace.cs`）。
   只在動到「模擬蓋不住的三類」（硬體時序/native/視覺）才需要。
3. **log-nunit（headless 自動測試，🔲 未落地）**：NUnit 直接 new view/coordinator（不開視窗不接相機），
   程式扮相機 PushFrame + 扮使用者呼事件 → 收 `[Flow]` 行、契約規則寫成斷言。
   B1=sdk view 層（現有 code 即可測）；B2=coordinator 層（需開相機介面縫，與 Wave3 一起做）。

**自動化標記**：契約條目被 log-nunit 蓋到後，標題加 `[auto]`（一看便知哪些還要手測）。目前全部未自動化。

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
IC|WF stats paints=N/s paintMs=M statusEv=K/s ← canvas 每秒重繪組成（>5 次/秒才記；瀑布同儀器 WF 前綴）
```
`UiStallDetector` 在 Form ctor 建立，但必須等 `Shown` 的全 tab 預熱完成後才
`BeginInteractiveMeasurement`；建構期使用者尚不能互動，不得把 ctor／預熱時間算成第一筆 stall。
**判讀決策樹（2026-07-07 十輪教訓的結晶）**：
1. UiStall 有 GC 增量 → GC/LOH 問題；全零 → 往下。
2. UiStall 大 + UiPing 也大 → **阻塞型**（單件慢）→ 看 UiStack 點名。⚠ 按時間窗切開判讀
   （開機時段的大 ping 會污染整體結論）。
3. UiStall 大 + UiPing 靜默 → **飽和型**（件多不慢）→ 看 IC stats：paints > ~150/s＝paint 風暴回歸
   （限流後正常 ≤ ~130/s）。**飽和型用計數器抓、阻塞型用計時器抓——只裝計時器抓不到飽和**。
4. UiStack 點到的都是真 bug 但不一定是你要的 bug——修掉後重測，別急收工。
契約：拖曳中 `IC stats paints` 不得 >150/s（風暴回歸紅旗）；`[UiSlow] CamStatusTick/TelemetryTick`
出現＝MIL 查詢又回到 UI 執行緒（背景化被回退）。
- view 訂閱 `cam.OnDisplayFrame` 且 Enable* 冪等 → **相機批次換新（Allocate/Free）前後必有對稱 teardown**。

## 狀態快照儀器（方向/座標「機器可判」——2026-07-09 故障注入盲測 4 例定版）

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

**log-flow（執行期腳印＝判準）**
```
T1: acquisition sync begin reason=start attempt=A gate=closed cams=P
T1: acquisition sync paused reason=start attempt=A cams=P
Tbg: acquisition sync timing-reset reason=start lineRates=cam1=R,cam2=R,...
T1: acquisition sync resumed reason=start attempt=A cams=P
T1: acquisition sync ready reason=start attempt=A camN system=S tick=T freq=F       × P
T1: acquisition sync phase reason=start attempt=A system=S cams=... spreadTicks=D
    spreadMs=X limitMs=5.000 measurable=True aligned=True                          × 有在線相機的板
T1: acquisition sync complete reason=start attempts=A cams=P phase=True
T1: StartGrab（cams=M）
T1: ApplyMainDisplayMode → 同模式    ← 冪等：不得出現 create/teardown 行
T1: capture plan grab=… root=… imageDir=… csv=… files=… scale=…
T1: grab limit armed Ns grab=…       ← 正式監控 grab 成功後啟動 one-shot；背景採樣借用 grab 不武裝
T1: capture gate open cams=P warm=True   ← P=在線數；必晚於同步完成與 plan/limit，早於所有 firstFrame
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
 │   → await Task.Delay(LightWarmupMs)               ⚠ 序列埠寫入一律背景（UI 執行緒零 MIL/序列埠鐵則）
 ├（未抓取）ResetLiveChartsForDisplayTransition@AniloxRollForm.Live.cs ＋ _muraExceedLatch 歸零
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
 │      ├ IsLiveGrabbing = true
 │      ├ ApplyMainDisplayMode@LiveDisplayCoordinator.cs   ← 冪等（view 已存在早退）＝本 flow 不得出現 create/teardown 行
 │      ├ ResetWaterfallIfActive@LiveDisplayCoordinator.cs → Reset@WaterfallView.cs（清舊圖＋重置 tick 對齊，防新幀接舊網格錯位）
 │      └ per-cam SetUserGrabIntent(true)@AniloxCamera.cs
 │         └ SetUserGrabIntent@MilCamera.cs（同步 M_START 已完成；此處只開產品意圖）
 ├（啟動成功）NextGrabId@InspectionLogService.cs → _currentGrabId ＋ capture plan 行（C1）
 │  └ Arm(GrabLimitSeconds)@GrabDurationCoordinator.cs → grab limit armed 行
 │     ← 設定在本輪開始時拍快照；PropertyGrid 中途改值從下一輪生效
 ├ OpenCaptureGate@LiveCameraManager.cs
 │  └ _captureGateOpen=true                         ← 單一全域寫入點；資料 owner 已準備後，7 台 callback 才一起取得資格
 └ UpdateGrabButton@AniloxRollForm.Live.cs
（每幀幀流，MIL 回呼執行緒 Tn）
ProcessingFunction@MilCamera.cs（MdigProcess hook，static）
 └ FrameReady 事件 → OnMilFrameReady@AniloxCamera.cs
    ├ UserWantsGrab && CaptureGateOpen 才繼續；gate 關閉時不進 GPU／顯示／CSV／存檔
    ├ TryApplyPicoaterRidge@AniloxCamera.cs（GPU 檢測，一律跑）  ⚠ _picoaterLock＋尺寸守門（高度變更瞬間跳過幀防 AV）
    │  ├ ProcessImage@AoiService.cs（P/Invoke TanukiPipeline_Process；fused 存檔縮圖 wantResize＝grab-level 決策）
    │  ├ OnLiveCurveData 事件 → OnLiveCurveData@AniloxRollForm.Live.cs → CheckLiveMura("v")（M1）＋ _liveOverviewDirty=true
    │  └ OnLiveRowCurveData 事件 → OnLiveRowCurveData@AniloxRollForm.Live.cs → CheckLiveMura("h")
    │     ＋ pending latest-only cache（不得直接更新列 chart）
    ├ PutDisplayBytes@MilCamera.Display.cs（強化）｜CopyToDisplay@MilCamera.Display.cs（原圖）
    ├ OnDisplayFrame 事件（bytes）→ OnCameraDisplayFrame｜OnCameraWaterfallFrame@LiveDisplayCoordinator.cs
    │  ├ 模式錯掛自檢（⚠ 契約違規 行）＋ FlowFirstFrame（firstFrame 行，每台恰一）
    │  ├（即時）PushFrame@ImageDisplayView.cs（存快照＋餵 ThumbStrip＋_mainDirty）
    │  └（瀑布）PushFrame@WaterfallView.cs → PlaceFrame（tick 網格錨定）→ TryFlush → ComposeJob
    │      （佈局=MergeLayout.Compute 唯一來源）→ KickWriter → Task.Run WriteBand（背景 memcpy，不卡 UI）
    │      ＋ PushFrame@ThumbStrip.cs（縮圖一律即時，兩模式同源）
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

列曲線顯示不變量：GPU 完成只能更新 pending cache；必須等同一條主畫面 API 發出
`ContentPresented` 才可更新 chart。LOD 以 content generation 對應實際安裝的 tile，不能在
`RefreshLod` 僅提出請求時提前發布。每次發布留
`rowCurve present after=mainImage cams=N mode=IC|WF`；快速累積時只保留每台最新一筆，禁止
用固定延遲猜影像完成時間。

### F3 停止抓取

**log-flow（執行期腳印＝判準）**
```
T1: StopGrab
T1: capture gate closed standby=on
Tn: drop drainedFrame after StopGrab camN（可選；每台最多一行）
（之後不得再出現 firstFrame / 任何 [Flow] 顯示行，直到下一個動作；
  drop 行是 gate 關閉競態的觀測儀器，不是顯示更新）
```

**code-flow（靜態地圖＝責任鏈＋載重點；audit 時兩者都要對）**
```
手動：btnLiveGrab_Click@AniloxRollForm.Live.cs（wasGrabbing=true）→ ToggleLiveGrabAsync
                                                          intent 行 ui:【開始抓取】鈕
逾時：Elapsed@GrabDurationCoordinator.cs → SafeBeginInvoke → HandleGrabLimitElapsed@AniloxRollForm.Live.cs
                                                          intent 行 auto:抓取上限到時 limit=Ns grab=… → 停止
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
 ├ Disarm@GrabDurationCoordinator.cs                         ← 作廢 generation，舊 callback 不得停掉下一輪
 ├ Task.Run(LightTurnOff@AniloxRollForm.HardwareStatus.cs)   ⚠ [UiStack] 曾定罪停止時卡 SerialStream.Write → 一律背景
 ├ TriggerRetentionAndFlagAsync@AniloxRollForm.HardwareStatus.cs
 ├ UpdateMuraLed(false) ＋ ClearMura@IoGrabController.cs   ← MURA latch 清除時機＝檢測結束（M1；手動流程不經 FSM 必須自清 DO）
 ├ UpdateGrabButton@AniloxRollForm.Live.cs
 └（逾時來源）NotifyGrabStopped@IoGrabController.cs          ← 先把 DO_PC_INSPECT 拉低；FSM 在 DI START
                                                            維持 High 時刻意留 Running（沒有新上升緣，不會重啟），
                                                            等 START 真正下降後才走既有 falling-edge 收尾回 Idle
實體停止邊界只有兩種：
 - Start 同步／停止狀態的高度重配置：PauseAcquisition → M_STOP+M_WAIT＋M_GRAB_ABORT → 修改 → ResumeAcquisition
 - Release：先關 capture gate，再平行 PauseAcquisition，完成後才能釋放 merge／camera buffers
```

**StopGrab 校稿工具**
```
python tools/python/check_stopgrab_flow.py [trace.log]
```
- PASS 判準：每個 `StopGrab` 必接 `capture gate closed standby=on`；其後到下一個
  `ui:`/`StartGrab`/`AllocateCameras begin` 前只允許 `drop drainedFrame after StopGrab camN`，
  不得再出現 `firstFrame`、`LC row`、`capture csv`、IC/WF display 更新。

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

### F6b 滾輪縮放主畫面

**log-flow（執行期腳印＝判準）**
```
T1: IC|WF wheelZoom in|out → zoom=Z（fit=F）   ← 每手勢至少一行（100ms 節流）
T1: IC|WF|RV fit(double-click) / physical1x(triple-click)   ← 使用者 fit/1x 手勢（合法的視野重設主人）
（縮放/互動期間**不得出現 `autoFit(...)`/`lodRebind(...)` 行**——出現＝系統 fit 跟使用者縮放打架。
  zoom 突然回 fit 而無 fit(double-click) 行＝有東西在暗中重設（孤兒判讀）。
  `autoFit(firstFrame ...)` 只允許在 view 建立後首幀；`autoFit(sizeChanged@fitView ...)` 只允許在
  使用者「未動過視野」時的尺寸變更。centerCam 行在縮放中出現＝正常（中心相機隨視野變）。）
```

**code-flow（靜態地圖＝責任鏈＋載重點；audit 時兩者都要對）**
```
OnMouseWheel@ImageCanvas.cs   ← 滾輪一律 canvas 自理（app 無全域訊息濾鏡）
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
 ├（未配置）await EnsureAllocatedAndToggleGrabAsync@LiveCameraManager.cs（=F1＋F2 借道，不開影像處理）
 ├（未抓取）LightTurnOn@AniloxRollForm.HardwareStatus.cs → await Task.Delay(LightWarmupMs)
 │   → ToggleGrab@LiveCameraManager.cs ＋ UpdateGrabButton(true)   ← 借用現有 grab（啟停包夾）
 ├ 採集迴圈（await Task.Delay(100) × BackgroundSampleSeconds，UI 執行緒非阻塞、按鈕倒數）
 │   └ per-cam TryComputeColumnMean@AniloxCamera.cs → accum 累加
 ├ 產生 version=`yyyyMMdd-HHmmssfff`
 ├ per-cam 平均 → SaveBackgroundBin@AniloxRollForm.Background.cs
 │   → `bg_{width}_{cam}_{version}.bin`（MCBF v2；CreateNew＋WriteThrough＋Flush）
 ├ 全部在線相機成功 → ActivateBackgroundVersion 原子替換 `active-background.json`
 ├ LoadBackgroundBins@AniloxRollForm.Background.cs（manifest 指向的同一版 bin → TanukiCv_AllocPinned → cam.PrecomputedColMean）
 │   ← pinned 生命週期：舊 buffer 先 FreePinned 再換新（防漏）
 ├ 任一相機失敗 → 刪本次 version 檔、manifest 不動、上一組背景繼續生效
 │   → OutputHealth `BackgroundCaptureFailure` 深橘提示
 ├ finally：ToggleGrab 停止（=F3）＋ LightTurnOff ＋ UpdateStandardBgSubLockState@AniloxRollForm.Background.cs
 ├（_autoStartGrabAfterBg）await ReleaseAsync → btnLiveGrab_Click（IO 觸發自動回抓）→ return
 └ 尾端自動預覽：btnLiveViewBackground_Click（直呼）
時間設定不變量：`BackgroundSampleSeconds` 只管本段背景採樣；`GrabLimitSeconds` 只在 F2 正式監控啟動成功後
武裝，兩者不得互相中止。PropertyGrid 顯示於「時間設定」下：`背景採樣(sec)`、`抓取上限(sec)`。
預覽背景：
btnLiveViewBackground_Click@AniloxRollForm.Background.cs     intent 行 ui:【預覽背景】鈕
 ├（IsBgPreviewActive）ClearBackgroundPreview → return       ← 再按一次＝清除（toggle）
 ├ EnterBackgroundPreview@LiveDisplayCoordinator.cs（LCM forwarder 經過）
 │   └＝靜音鍵 _bgPreviewOverride=true → ApplyMainDisplayMode()   ⚠ 只改狀態→呼閘門，不自建/拆 view
 │       閘門 BgPreview 分支：DisableWaterfall＋EnsureImageDisplay＋ApplyBgPreviewLayout
 │                            （合圖未啟用→用設定 start/ops 餵佈局）
 ├ ReadActiveBackgroundVersion → per-cam Load@CurveBinFile.cs（同一 active version）
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
T1: bgPreview push camN WxH（view=True）× 有 bin 的台數
再按/開始抓取：ExitBackgroundPreview → ApplyMainDisplayMode → {Waterfall|ImageCanvas}（回設定模式）
```

## 硬體連線契約（H 系列）——邊緣觸發（同 MURA 模式：轉變才記，不洗版）

### H1 IO / 光源 / 儲存電腦 連線轉變
```
Tn: ⚠ IO 斷線 ／ IO 恢復連線            ← 光源/儲存分享 同格式
Tn: ⚠ IO 未連線（開機基線）             ← 首次觀測就不在線（拔線開機/初始化未完，恢復行會跟著出現）
Tn: 儲存程式 heartbeat 恢復 pid=N age=Ns
Tn: ⚠ 儲存程式 heartbeat 未回報 reason=…
```
- IO 重連倒數以 `IoGrabController.NextReconnectAtUtc` 為唯一來源，顯示到 `0s` 代表正在嘗試連線；
  不得在 `1s` 後退回沒有秒數的空白狀態。`ReconnectIntervalMs` 是 connect 嘗試起點間隔，
  TCP timeout 必須包含在週期內，不得 timeout 後再重複等待完整週期。
- **IO 恢復連線的定義＝TCP + 安全交握全過**：`ReconnectTick → ConnectAsync → TryAcceptConnectedModule`
  必須完成 `EnterIdle`（DO1=0 → DO2=0 → DO0=1，ALIVE 最後發布）及一次合法 `ReadDiStatuses`，之後才可
  `OnConnectionChanged(true)`；任一步失敗須 Dispose 並維持 Disconnected/CommLost，禁止假綠。
- **連線狀態 SSoT**：業務層一律讀 `IoGrabController.IsConnected`（accepted gate）；TCP 已接上但交握未過時，
  `NotifyGrabStarted/Stopped` 與 `NotifyMuraDetected/ClearMura` 必須靜默拒絕，不得用 `_plc.IsConnected` 繞過。
- **逾時收口**：`SendAndReceive` read/write 逾時先 `ObserveLateFault` 再關 transport；Connect timeout
  關 socket 後等待 SAEA completion 再釋放 args。全天 crash log 不得新增來源為 ConnectAsync/NetworkStream 的
  `UnobservedTaskException`。
  Connect 必須走 `SocketAsyncEventArgs`；debugger 不得再出現晚到 `TcpClient.EndConnect` 的
  `ObjectDisposedException/NullReferenceException` first-chance 例外。
- **冷開機恢復**：app 先開、IO 後上電不得要求重開 app；Bridge log 第 1 次及每 10 次失敗記
  `IO reconnect pending: attempt N, {TCP unavailable|handshake rejected}`，成功行必帶 attempt。
- **儲存電腦綠燈＝兩層都通**：第一層為 TCP 445 + `RemotePath` 建立/寫入/flush/刪除唯一探針檔；
  第二層為 `RemoteConfigPath\storage-app-heartbeat.json` 的 `LastSeenUtc` 不超過 15 秒。分享不可用顯示紅色；
  分享可寫但 heartbeat 缺少/過期顯示黃色；兩層都通才綠。斷線時每 2 秒重試，app 先開、儲存機後上電
  不得要求重開 app。
- **儲存程式自舉**：Release 根目錄 `setup.bat` 自動選擇完整安裝或更新；兩路都必須移除 payload 下載封鎖標記，並以
  `BUILTIN\Administrators` 群組主體安裝「任一使用者登入」＋「每分鐘保活」雙觸發排程；
  `RdpUser` 只供遠端登入，不得綁死排程。程式存活時 `MultipleInstances=IgnoreNew` 必須讓保活觸發靜默略過；
  正常關閉或異常退出後，最晚一分鐘內必須重新啟動。
  安裝當下必須由目前登入的系統管理員立即啟動，並在 10 秒內確認正確 EXE process 存活；
  `LastTaskResult=267011 (0x00041303)` 代表尚未執行，必須判定失敗而非歸因 DLL 封鎖。Storage role 每 5 秒原子發布
  heartbeat（PID、啟動/回報時間、磁碟與最後清理結果）。SMB 可寫只證明分享，不得當成程式存活。
- **儲存資料捷徑**：檢測電腦 `setup.bat` 必須冪等在 Public Desktop 建立
  `Anilox 儲存資料.lnk`，目標由 `\\VerifyPingTarget\StorageShareName` 計算，不得額外寫死 `192.168.10.20`。
- **程式桌面捷徑**：Storage 與 Inspection 的安裝及更新都必須冪等建立 Public Desktop
  `PICoater AOI.lnk`；TargetPath＝`AppDir\AniloxRoll.Monitor.exe`、WorkingDirectory＝`AppDir`、圖示＝同一 EXE，
  不得寫死磁碟位置。捷徑存在但目標過時必須覆寫修正。
- **遠端複製不丟資料**：`EnqueueFiles` 必須先在本機 `.remote-copy-pending` 持久化標記才進 worker；
  複製失敗保持 pending 並退避重試，程式重開從標記復原。禁止恢復「重試固定次數後丟棄」語意。
- **發布原子性**：遠端先寫同目錄 `.part-*`，確認來源前後長度穩定且遠端長度一致後，再原子
  move/replace 成正式檔名；正式發布且 pending 標記成功刪除後才算完成。
- **可變側車不漏尾筆**：`_ticks.csv` 追加期間若同一路徑已 pending/in-flight，必須標記 dirty；首輪發布後
  保留 durable marker 並再排一輪，直到遠端內容等於最新本機 snapshot 才可刪 marker。
- **Retention 以可持續寫入優先**：空間低於預留值時，檢測與儲存電腦都刪除「最舊的完整一天」；
  今天的資料不得刪。刪除集合＝日期資料夾內全部影像/bin/`_ticks.csv`/`_curve_summary`
  ＋月份層同日期 `yyyyMMdd.csv`；任一低空間清理成功後，非 active 的舊背景版本也列入候選。若該日仍有 pending 遠端檔案，必須先取消並移除
  對應 durable marker，再刪檔，禁止 worker 對已刪來源永久重試。刪到未送達資料須留下深橘狀態與 log。
- **光源釋放**：SerialPort ownership 必須先在 lock 內從 `_port` 移除，再對 detached instance 單次 Dispose；
  全天 crash log 不得新增 `SerialStream.Finalize → ObjectDisposedException（已關閉安全控制代碼）`。
- **光源重連防呆**：開機首次偵測 `AutoDetect` 全 port；離線後每 2 秒先 `TryConnect` 設定 COM，
  每 5 次失敗（約 10 秒）必須 `AutoDetect` 全 port 一次。找到不同 COM 要回寫 SSoT；禁止移除全掃描造成
  工廠現場無法自救，也禁止每輪全掃描造成 SerialPort handle churn。
- 光源停用（LightEnabled=false）/ 遠端路徑空 → 該項不觀測（靜默合法）。
- 開機常見「未連線（開機基線）→ 恢復連線」＝平行初始化的正常時序，非異常。

**儲存電腦 trace 判讀**：
```
[RemoteCopy] remote share unavailable: ...             ← TCP/路徑/寫入任一層未通
[RemoteCopy] remote share accepted (write verified)    ← 實際寫入交握通過
[RemoteCopy] retry pending attempt=N queue=N file=...  ← 保留待傳；第 1 次及每 10 次留痕
[RemoteCopy] restored pending queue count=N            ← 程式重開復原
[RemoteCopy] backlog drained: copied=N bytes=N          ← 斷線積壓清空
```
狀態轉換：`未排程 --持久標記成功--> 待傳 --複製失敗/重開--> 待傳
--長度驗證+原子發布+刪標記--> 完成`。任何失敗不得進完成態。

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
    files=*_raw.jpg|*_proc_c.jpg|*_proc_r.jpg|*_mean_c.bin|*_max_c.bin|*_mean_r.bin|*_max_r.bin
    scale={DefaultSaveResizeScale}
```
- `imageDir` 與 `csv` 必須由 `CaptureStoragePaths` 推導；檔名 suffix 必須由 `CaptureFileNaming` 推導。
- 曲線持久化新格式一律 C/R：`_mean_c/_max_c/_mean_r/_max_r.bin`；讀端依序 fallback 上一代
  `_mean_v/_max_v/_mean_h/_max_h.bin` 與最舊 `_mean/_max/_row_mean/_row_max.bin`，寫端不得再產生 V/H curve bin。
- 這行是每輪 grab 的「存放方式/位置」摘要；逐幀大小與資源量仍歸 `resource-monitor-*.csv`，不得用 `[Flow]` 洗版。

### C2 檢測 CSV 寫入（每個 grab 首筆 + CFG 變更）
```
Tn: capture csv open path=… cfg=yes|no              ← 新檔或換日首次開啟
Tn: capture csv cfg path=… HM=V/H ridge=N thrV=mean/max thrH=mean/max
Tn: capture csv firstRecord grab=… path=… file=… verdict=max0|1/mean0|1 peak=…/… rowPeak=…/… maxCMean=… thrV=…/…
```
- `firstRecord` 每個 grab 只出一行，用來確認檢測結果有落到哪一份 CSV；逐相機逐幀細節看 CSV 本體。
- `cfg` 行出現代表 `#CFG` 已寫入同一 CSV；`ridge` 是捕捉時的細線濾除值。
- `#CFG` 的機台佈局必須完整保存 `OPS + START(CamN_Pos) + CROP(TrimHead/TrimTail)`；
  檢測設定必須包含欄／列正規值、細線濾除與欄／列門檻。設定變更後，下一筆資料前必須出現新版 `#CFG`。
- `#CFG` 刻意與每日資料列同檔，不拆成平行設定檔：每筆資料以上方最近一行 `#CFG` 為設定版本，
  避免斷電或跨檔寫入失敗造成資料與設定失配。
- `verdict` 使用寫入 CSV 同一組 V 閾值，與 `AppendRecord@InspectionLogService.cs` 的 `MaxExceed/MeanExceed` 同源。
- CSV 資料列格式＝`Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs,MaxCMean,MeanRPeak,MaxRPeak`；
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
   → OnInspectionResult(camId,file,meanPeak,maxPeak,maxCMean,meanRPeak,maxRPeak)
     → LiveCameraManager forwarder → OnCameraInspectionResult@AniloxRollForm.Live.cs
       → AppendRecord@InspectionLogService.cs → CSV 第 10~12 欄 MaxCMean/MeanRPeak/MaxRPeak
```

### C3 遠端交付與循環儲存
```
CameraFrameSaver.SaveCapture
 → OnFilesSaved（本幀固定輸出 + 當日共享 `_ticks.csv`）
   → RemoteCopyService.EnqueueFiles
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
 → Storage role：TelemetryTimer → DriveInfo(`StorageMachineDataPath`) → `儲存電腦：剩餘/總容量`
 → Inspection role 本機：TelemetryTimer → DriveInfo(`CaptureRootPath`) → `檢測電腦：剩餘/總容量`
 → Inspection role 遠端：儲存 probe → heartbeat FreeBytes/TotalBytes → `儲存電腦：剩餘/總容量`
 → Inspection role 待傳：`RemoteCopyService.PendingBytes/QueueCount` → `待傳：N GB（M 檔）`
 → Inspection role 成功時間：`OnFilesSaved`／`RemoteCopyService.LastSuccessfulCopyUtc`
   → `最近存檔 HH:mm:ss`／`最近遠傳 HH:mm:ss`
 → heartbeat/磁碟不可讀 → 對應電腦顯示`無法讀取`
```
低磁碟整合測試使用隔離 volume/root，將門檻設為高於該測試磁碟目前可用空間、但低於磁碟總容量即可直接觸發，
不必真的填滿磁碟；禁止對正式 Captures 執行。

### C4 產出健康度與底部狀態列

`OutputHealthService` 是產出健康度唯一狀態機；writer、remote-copy、retention、設定與背景流程只回報事件，
不得各自在 UI 判斷顏色。`lblInfo` 使用 StatusStrip 預設背景，只顯示容量、待傳量與最近成功時間；每個未確認問題各自
對應一個 `ToolStripStatusLabel`，顯示 service 的完整 incident 清單，禁止只顯示最高嚴重度而藏掉其他問題。

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
- 背景版本只有在所有在線相機都完成 `CreateNew + WriteThrough + Flush` 後，才原子替換
  `active-background.json`。manifest 存在但無法解析時不得 fallback 到 legacy bin 混用；保留目前已載入背景，
  並回報 `BackgroundManifestInvalid`。
- 診斷檔只由 `LogFileCatalog` 納管：`trace`、`resource-monitor`、`dropdiag`、`phaselog`、
  `paramchange`、`ui-actions`、`io`、`crash`。`LogRetentionHours` 預設 168 小時，啟動後 5 秒及每小時清理；
  目前 process 建立的 log 與未知檔案不得刪。PropertyGrid 改保存時間後立即補跑一次清理。
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
- 現況（2026-07-20）：已掛 `GLOBAL`（任何 `契約違規` 行即 FAIL）＋`LIVE/F`＋`REVIEW/R`＋
  `DATA/D`＋`CAPTURE/C`＋`HARDWARE/H`＋`SETTINGS/S`＋`MURA/M`＋`PARAM/P`，registry 無待接 domain。
  `FULL` 只表示每個已登記 domain 都有 validator；單一 session 未操作到的 flow 仍必須回
  `NOT COVERED`，不可把「有檢查器」誤寫成「所有功能已實際測過」。
- domain 專用舊指令保留為薄 wrapper（例如 `check_review_flows.py`），規則實作只能存在
  `flow_checks/{domain}.py` 一份，避免 wrapper／總入口兩份判準分歧。

## 回顧 tab 契約（R 系列；儀器前綴 RV）

**回顧鏈自動校稿工具**（`flow_checks/review.py` 的相容入口；改回顧/跨 tab 同步後必跑）：
```
python tools/python/check_review_flows.py [trace.log]    # 預設抓最新 log；exit 0=全 PASS
```
判準：①R2 快路跟隨（最後選取的 grabId 必有成功 `RV curves`，全 drop=曲線沒跟上）②R2 token+begin/done 配對
③卡頓紅線（回顧互動期間 UiStall ≤1000ms）④讀取資料跳最新（第 2 次起不得停在舊序號）
⑤時段導航去重（同時點不得重複載入）⑥曲線 single-flight（兩個 paths 間必有 done/stale）
⑦方向對數（dataPhys↔dataChart 鏡射/直通，見§狀態快照儀器）⑧切入回顧必有
`RV tabVisible repaint view=True`，且其後必有 `RV visiblePaint ready=True lod=… size=WxH` 證明內容真正可畫；
只有 repaint intent 不算通過。
2026-07-10 基線：①③④⑤ 皆紅＝回顧戰役待修清單（兇手=每格序號同步觸發 Data 統計全重算於 UI 執行緒
〔SyncDataGrabIdFromReview→RefreshStats→掃目錄+CSV 全解析〕＋時段 date/time 串聯重複觸發）。

### R1 讀取資料（btnReviewSelectFolder）
```
T1: ui:【讀取資料】鈕（Review）
T1: RV folder selected root=…
T1: RV repo scan root=… files=N
T1: （首次）RV EnsureImageDisplay create（thumbs=7）
T1: RV loadGrab begin {grabId}（proc=…）
Tn: RV loadGrab paths {grabId} root=… images=N cams=P cfg=yes|no align=tick|filename
T1: RV pushFrames P/7（merge=True, feedScale=…）   ← P=該 grab 有影像的相機數；缺台=黑占位
T1: RV loadGrab done {grabId}（…ms）
（若由 Data 頁隱藏預載：切到回顧後延後一個 UI message，出現 `RV tabVisible repaint view=True` →
 `RV visiblePaint ready=True lod=… size=WxH`；以可見尺寸補 LOD tile + paint，不重讀檔、不重設視野）
（grab 中按：另會出現 DisableGlobalMerge 等監控行——歸本 intent 管，見孤兒判讀規則）
不變量：手按【讀取資料】＝刷新+跳最新（loadGrab 的 grabId=清單最新；2026-07-10 修「停在舊選取」）；
開機自動恢復上次位置不在此限。
載入 busy 視覺唯一 owner＝`BusyUiBinder`；`AniloxRollPresenter.BusyStateChanged` 與
`ReviewStitchCoordinator.LoadGrabStitchedViewAsync` 共用同一實例。圖片 loader 只有 latest token 可解除 busy，
stale loader 不得提早恢復游標或按鈕。
回顧 CFG 與螢幕校正 runtime state 唯一 owner＝`ReviewRuntimeState`；單片曲線快路、完整圖片載入與時段載入
都只更新/讀取此實例，不得在 Form、Presenter 或 Coordinator 另存第二份 CFG。
```

### R2 單片序號切換（cbReviewId）——分層載入（2026-07-07 定版）
```
T1: ui:【單片序號】→ {grabId}
Tn: RV curves paths {grabId} root=… images=N cams=P cfg=yes|no align=tick|filename
T1: RV curves {grabId}（…ms）          ← 快路：欄+列曲線+CFG 即時跟滾動（chart 先行，使用者掃異常）
（快速滾動：曲線 single-flight/latest-only，中間 intent 可無 paths；正在讀的舊筆完成後 stale-drop，再直接讀最新筆）
（影像 debounce 250ms：滾動中不發完整載入；停下才載「最後選取」；session 也只在 settle 落盤一次）
T1/Tn: RV loadGrab begin {grabId} → RV loadGrab paths … → RV lodRebind merge …（fit reset）→ RV pushFrames → RV loadGrab done
```
- **分層**：單步時曲線立即載；快速滾動時曲線最多「執行中 1 筆＋等待中最新 1 筆」，中間序號不讀檔；
  最後一個 intent 必有成功 `RV curves`。圖片只載 settle 後的最後一張；
  **Data tab 同步（統計/Mura 圖重算）也只在 settle 後做一次、排在影像之後**——唯一觸發點
  `SyncDataTabFromReviewSettled@AniloxRollForm.Data.cs`，不得回到逐格 inline
  （2026-07-10 定罪：逐格全量重算＝快撥 UiStall 5.7s＋曲線快路全餓死）。
- **日期/session 分層**：滾動中只走 `SetPeriodToCombo`（同日不重建 time items），不得走 `NavigateTo`
  的完整 Initialize/Save；`SaveCurrentSelection` 只在 250ms settle 執行一次。
- **換序號＝重設視野（fit）＝預期**（各 grab 高度不同 → lodRebind 合法出現）。
- **token 分治**：曲線與圖片各自最後贏，兩者不得共用 token（圖片開始載入不得讓同序號曲線 stale）；
  每個序號 intent 立即 invalidate 舊圖片，settle 回呼另以 selection token 守住 Data 同步。
- 最後一個非 stale 的 `curves`/`loadGrab done` 的 grabId 必須＝最後一個 intent 的 grabId——不符＝token 破了。
- begin 無對應 done/stale-drop＝載入中斷；pushFrames P 與 CSV 台數不符＝掉圖。

**code-flow（序號對應影像／CFG 查詢）**
```
LoadGrabStitchedViewGuardRowRangeAsync@ReviewStitchCoordinator.cs
 ├ LoadForGrabId@InspectionImagePathRepository.cs
 │  └ InspectionCsvReader.OpenShared＋TryParseRecord＋TryExtractCameraId（影像依 cam 分組）
 └ LoadForGrabId@InspectionConfigRepository.cs
    └ InspectionCsvReader.OpenShared＋TryParseRecord（取 grab 上方最近 #CFG）
```

### R3 時段導航（cbReviewDate/cbReviewTime 手動）
```
T1: ui:【時段導航】（cbReviewDate/Time）
T1: RV period begin {yyyy-MM-dd HH:mm:ss.fff}
T1: RV period load {yyyy-MM-dd HH:mm:ss.fff} images=P/7 proc=True|False cfg=yes|no
T1: RV pushFrames P/7（merge=True, feedScale=…）
T1: RV row … / RV state …（chart/狀態快照視資料而定）
T1: RV period done {yyyy-MM-dd HH:mm:ss.fff}
```
- 時段模式不進 `RV loadGrab begin/done`；它走 request 的 immutable period → `ApplyGlobalMergeForPeriod` → `StitchedImagesReady` → `ReviewDisplayManager.PushFrames`。
- **時序選擇 policy（使用者定版 2026-07-13）**：每個有效且不同的時點 intent 都刷新三項＝圖片＋欄曲線＋列曲線；
  資料量小，不套 R2 的 250ms debounce／latest-only。loader 由 `ReviewPeriodLoadCoordinator` 去重後 FIFO single-flight，
  各 request 持有 immutable period；不得並行後在 await 尾端重讀 ComboBox（會讓多筆都套用最新時點）。
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
      ├ 不同 period → FIFO single-flight（不得平行）
      └ LoadReviewPeriodRequestAsync@AniloxRollForm.Review.cs
         ├ RunWorkflowForPeriodAsync@AniloxRollPresenter.cs → GetImages(DateTime)@ImageRepository.cs
         ├ generation 失效 → stale-drop（不得 apply）
         └ ApplyPostLoadDisplay(period)
            ├ ApplyGlobalMergeForPeriod → PushFrames（圖片/LOD）
            ├ UpdateOverviewChartForPeriod（欄 curve）
            └ UpdateRowChartForPeriod（列 curve）
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
    ├ ApplySettingImpacts（只套 route 指定的跨功能影響）
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

### S1 檢測參數（dc_/dd_ 正規值、eb_ 檢出方向、ec_~ef_ 閾值）
```
T1: ui:設定[ec_ErrorValueMeanV]（例）
（反應：chart 閾值線/曲線坡度更新；Data 曲線重畫走 HandleDataStatsSettingsChanged）
禁止：**其他任何設定**（IO/光源/儲存/顯示…）不得觸發 Data 曲線 reload+重綁
      ——違規即「無關設定閃圖」（2026-07-03 修過的家族）。
```

### S2 回顧強化（hd_EnableReviewEnhance）
```
T1: ui:設定[hd_EnableReviewEnhance]
T1: RV loadGrab begin {當前grabId} → … → RV loadGrab done   ← 重載當前拼接視圖
```

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
- `S2.review-enhance`：已有回顧序號時，切強化必須以同一 grabId 完成
  `RV loadGrab begin → done`；尚無回顧資料則回 `NOT COVERED`。
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
LoadDataFolder|SyncFromReviewFolder@DataStatisticsPresenter.cs
 → LoadStatisticsSnapshot@DataStatisticsPresenter.cs
 → LoadSnapshot@InspectionStatisticsService.cs
    ├ 只讀 yyyyMMdd.csv（排除 _ticks.csv）
    └ 一次產生 AvailableTimes／GrabIdsDescending／DetailsByGrabId
 → PopulateAllGrabIdCombos／RefreshStats／YieldPeriodChartPresenter
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
T1: DT curve load {grabId} captures=N source=disk|prefetch|cache storage=summary|bins configMs=N waitMs=N pathMs=N mergeMs=N summaryMs=N points=N drawMs=N totalMs=N
     ← 每格都更新 Curve；storage=summary 讀持久匯總，storage=bins 代表匯總缺少／失效而由原始 bin 重建
T1: DT row curve load {grabId} source=disk|prefetch|cache points=N pitch=Nmm
     ← 單片模式列 Curve；與欄 Curve 共用同一個 profile/cache/prefetch，不得另掃一次路徑
（舊資料缺 Row bin）DT row curve missing {grabId}，畫面清空
（切序號範圍／年/月/日）DT row curve clear mode=range
     ← 列沒有相機起始線，只對單序號反映；範圍模式不得保留上一筆列 Curve
T1: DT curve cache policy entries=512 maxMB=256 prefetch=4 coldParallel=2 scale=merged-only
     ← Presenter 初始化一次；目前驗綠的 bounded LRU／配置策略，改值須同步契約＋checker 並重驗
Tn: DT curve prefetch {grabId} readyMs=N storage=summary|bins cacheEntries=N cacheMB=N
     ← 依滾動方向背景預讀下一個尚未快取的相鄰序號（前看最多 4 格，只允許一個預讀工作）
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
T1: DT range policy listMs=33 curveMs=80 settleMs=150 curveMode=monotonic
     ← Presenter 初始化一次；目前上機驗綠的排程基準，改值必須同步本契約＋checker 並重驗
T1: DT range list preview gen=G range={start}~{end} rows=N ms=N source=index
     ← 滾動期間 List＋7 台色卡預覽；獨立 33ms 節拍，只切記憶體完整索引，不讀磁碟
T1: DT range preview apply gen=G range={start}~{end} loadMs=N drawMs=N meanRows=N maxRows=M method=top-maxcmean|mixed|even coverage=S/R rankedCams=C/T index=H/B
     ← 滾動期間 Curve 背景取樣；與 List 共用 generation，已開始的樣本允許完成，但上畫 generation 只能向前
T1: DT range settle → refresh             ← 最後一次變更後 150ms；一串連續滾動只准一行
T1: DT list reload range={start}~{end} rows=N ms=N source=index
```
- **List ownership**：明細 List 屬於範圍結果，不屬於單片序號；`ui:【報表序號】` 後只准 `list=keep`，
  不得出現 `DT list reload`／`GrabDetailListBinder.SetItems`／重設 `VirtualListSize`／欄寬。只有資料夾、範圍、期間、閾值改變才重算 List。
- **列判定索引**：明細第 8 判定欄固定為「列」；O/X 由同序號所有相機／capture 的
  `MeanRPeak/MaxRPeak` 一票否決。view-time 公式＝raw row peak × `HM_V_capture/HM_H_current`
  對當前列門檻。舊 CSV 沒列峰值只能顯示 `—`，不得當 Pass。篩選異常也必納入列 Fail。
- **List 捲動顯示**：資料已全在 `GrabDetailListBinder._visibleDetails`，VirtualMode 不需資料預載；ListView 啟用雙緩衝，選中列只在接近
  可視區上下邊界時以 margin 捲動，反白變更只重畫舊／新兩列，不得每格整窗 `Invalidate()`（跨視窗白閃的根因）。
- **跨 tab lazy**：報表序號只覆寫 pending selection 並標 `_reviewDirty`，不得逐格操作隱藏的 Review combo/date、
  `NavigateTo` 寫 session／重建日期清單，也不得當下載 Review 圖片；切到 Review tab 才一次套控制項並接 R2 完整載入。
- **時序索引只建一次**：`ImageRepository.LoadDirectory` 建立排序去重的 available-period index；報表／回顧每格同步
  只能對既有索引做查找，不得在 UI 執行緒重新解析全部影像檔名、`Distinct`、`OrderBy`。
- **單片 Curve 不掠過**：每個 `ui:【報表序號】` 必有同 grabId 的 `DT curve load`，且列 bin 存在時
  必有同 grabId 的 `DT row curve load`。`source=disk`＝前景首次完整讀取；
  `source=prefetch`＝選取加入已在背景完整讀取的同一工作；`source=cache`＝使用之前完整合併完成的原始 Curve。
  快取保存 rescale 前欄 Mean/Max 與合併後列 Mean/Max；資料夾重載或 Presenter Dispose 必清空。
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
- **範圍雙速 monotonic throttle**：起始／結束 combo 連續滾動時，33ms List timer 從完整明細索引切出當代範圍並更新
  List＋7 台色卡；獨立 80ms Curve timer 最多啟動一個背景預覽。兩者共用 generation；新 intent 不取消已經開始的
  Curve 樣本，避免輸入間隔短於 12~21ms 計算時間時 Curve 永遠無法完成。樣本可落後當前選擇，但只能按 generation
  單調向前上畫；完成後下一輪取最新範圍，停止後最終 Curve generation 必須追上 List。150ms timer 只跑一次最終對帳。
  `DT range settle` 前允許 `DT range list preview`／`DT range preview apply`，但出現 `DT list reload` 或同步
  `DT curve candidates`＝逐格完整重算回歸。List 預覽只能使用已就緒且資料夾／閾值簽章相同的記憶體索引；
  索引未就緒時略過預覽，交由 settle 建立，禁止在 80ms 路徑掃 CSV。
- **節拍參數的效力**：`33/80/150ms` 是目前驗綠基準，不是不可修改的使用者鐵則。它們必須集中在
  `DataStatisticsPresenter` 的具名常數並由 `DT range policy` 量出實際執行值；任何調整都視為排程行為變更，
  同一批修改 DVT 與 checker 後重跑上機測試。不可散落 magic number，也不可只改文件或只改 code。
- **範圍曲線只有兩條**：每台相機各自選候選再合成全寬；`CurveMean`＝範圍 CSV 資料列均勻取樣最多 50 筆後
  對對應 `MeanC` bin 逐點平均；`CurveMax`＝依 `MaxCMean` 排序取前 50 筆，再對其 `MaxC` bin 做逐點最大值。
  候選必須保留 CSV `FileName` 並載入同一筆 bin，不得只選序號後誤讀該序號第一張。這個設計保留平坦趨勢，也不讓
  1/1000~1/10000 的凸波因均勻抽樣直接消失；畫面不得再增加操作員難以判讀的第三條曲線。
- `coverage=S/R` 是有 `MaxCMean` 的 CSV 資料列數／範圍資料列數；任一相機候選資料不完整時，該相機
  `CurveMax` 回退均勻取樣，避免拿新舊混合資料宣稱精確排名。
- **候選索引是衍生快取，不是資料真相**：`LoadRange` 以每日 CSV 完整路徑＋檔案長度＋最後修改時間驗證
  記憶體索引；簽章相同才可命中，CSV append/替換後必重建。原始 CSV 與 Curve bin 仍是 SSoT；索引只保存
  `grabId/camera/basePath/MaxCMean`，以 LRU 限制最多 25 萬筆／1024 日且可隨時丟棄。`index=H/B`＝本次命中／重建的日數；
  連續滾動在第一次暖索引後應以 `H>0,B=0` 為主。每日索引開始建立後允許完成並供下一代共用，
  不跟 range token 中途取消；token 仍在候選彙整／bin 載入／上畫前生效，故舊 generation 不得上畫。

**code-flow（單片快路 vs 範圍完整刷新）**
```
cbDataId.SelectedIndexChanged
 → OnSingleSheetComboChanged@DataDateGrabIdNavigator.cs
   → RefreshSelectedGrab@DataStatisticsPresenter.cs
     ├ _currentDetails.FirstOrDefault（範圍內命中）／EnsureSingleGrabDetailIndex（範圍外只建一次全序號索引）
     │  └ 命中→BuildSingleGrabStats；索引仍無該 ID 才允許單 ID CSV scan fallback
     ├ InspectionStatsPresenter.Update／UpdateRowResult（7 台色卡＋camDataRow 列 O/X）
     ├ GrabDetailListBinder.Highlight（只移反白＋EnsureVisible＋RedrawItems）
     └ MuraProfileChartPresenter.Update（該 ID curve）
       ├ SingleGrabCurveCache.TryGet／GetOrLoadAsync（同 key in-flight 共用；rescale 前結果 LRU）
       ├ miss → SingleGrabCurveSummaryStore.TryLoad（命中＝一次 sequential read）
       ├ summary miss/stale/corrupt → InspectionImagePathRepository.LoadForGrabId
       │        ├ Parallel.For（最多 2 台相機；降低冷讀等待且避免 7 路冷碟亂跳）
       │        │  ├ CurveMergeHelper.MergeCurves（MeanC/MaxC）
       │        │  └ MergeRowCurves（MeanR/MaxR；支援取消）
       │        └ MergeRowCurvesOverlap（所有相機同一列重疊）
       │        → CurveBinFile.Load（每個 bin bulk read；邊讀邊合併）→ SingleGrabCurveSummaryStore.QueueSave
       │          → idle 750ms／pending 72MB pressure → 單一 writer → TrySave（原子寫回）
       ├ cache raw Curve 唯讀直入 CurveOverviewMerger → 合併後小曲線套 Hessian ratio → UpdateOverviewChart
       │  （不得為 view-time rescale 複製每台完整 raw Curve；只縮放新建的 merged display result）
       │  └ ColumnCurveChartHelper.UpdateDataAndView（每兩個畫布像素一個顯示桶；Mean=桶平均、Max=桶最大，
       │       點數相同時原地更新既有 DataPoint；不得逐格 Clear＋DataBind 重建）
       ├ row raw Curve clone＋HM_V_capture/HM_H_current rescale
       │  → RowCurveDisplayAdapter.UpdateData → RowCurveChartHelper（與監控／回顧列圖表同格式與方向規則）
       └ ScheduleAdjacentPrefetch（依方向找下一個未命中項）→ DT curve prefetch
     → DT selected … list=keep
     → GrabIdSelectedFromData → OnDataGrabIdSelected@AniloxRollForm.Data.cs
      └ 覆寫 pending selection＋_reviewDirty=true（不碰隱藏 Review 控制項）

tabMain.SelectedIndexChanged（進 Review 且有 pending）
 → cbReviewId＋DateTimeNavigator.SetPeriodToCombo＋UpdatePeriodNavigationState（只套最後一筆）
   → ImageRepository.GetAvailablePeriods（預建索引＋binary search）
 → DT review sync apply {grabId} → LoadGrabStitchedViewGuardRowRangeAsync（R2）

cbDataIdStart|End 手動變更
 → ScheduleRangeRefresh@DataStatisticsPresenter.cs
   ├ MuraProfileChartPresenter.ClearRow（列圖只屬單序號；範圍 mode 清空）
   ├ generation++（不取消已開始的 Curve 樣本；資料夾／模式 teardown 才取消 token）
   ├ 33ms repeating throttle → RangeListPreviewTimer_Tick
   │  └ ApplyRangeListPreview（只在完整 detail index 簽章有效時）
   │     └ 依 `_grabIdInfos` 切片 → ApplyFailFilter／GrabDetailListBinder.SetItems
   │        ＋ ComputeStatsFromDetails／InspectionStatsPresenter.Update → DT range list preview
   ├ 80ms repeating throttle → RangePreviewTimer_Tick（同時最多一個 Curve 工作）
   │  └ UpdateRangePreviewAsync@MuraProfileChartPresenter.cs
   │    → Task.Run → LoadRange@InspectionMuraProfileRepository.cs（可取消）
   │       ├ 每日 CSV 簽章相同 → DailyIndex 依 grabId 取候選資料（不重掃 CSV）
   │       └ 簽章改變／首次使用 → 重建該日索引（LRU bounded 25 萬筆／1024 日）
   │    → 回 UI 執行緒；token 有效且 generation 大於前次上畫才 UpdateOverviewChart → DT range preview apply
   └ 重壓 150ms settle timer → DT range settle → RefreshStats(updateRangeCurve:false)（最終對帳）
    ├ EnsureSingleGrabDetailIndex（資料夾／閾值未變時沿用完整 GrabDetail 衍生索引）
    ├ 依 `_grabIdInfos` 選取範圍切出 detail 子集 → ApplyFailFilter → GrabDetailListBinder.SetItems
    ├ ComputeStatsFromDetails（同一 detail 子集彙總 7 台色卡；不得再掃第二次 CSV）
    └ DT list reload …

期間變更／初次載入（非手動連續滾動）
 → RefreshStats(updateRangeCurve:true) → MuraProfileChartPresenter.Update
   → LoadRange@InspectionMuraProfileRepository.cs（同步精確結果；候選算法同上）
     ├ InspectionCsvReader.TryParseRecord＋TryParseTimestamp＋TryExtractCameraId
     ├ Mean 候選＝EvenSample(rows,50) → 對應 MeanC 逐點平均
     └ Max 候選＝MaxCMean 排序前 50 → 對應 MaxC 逐點最大（缺分數→均勻 fallback）
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
T1: ui:【篩選異常】→ 只顯示異常|顯示全部
```

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
T1 capture plan grab=… root=… imageDir=… csv=… files=… scale=…
T1 grab limit armed 10s grab=…
T1 capture gate open cams=2 warm=True
10:37:15.170 T31 firstFrame cam1 16384x3000 → ImageDisplayView
10:37:15.207 T30 firstFrame cam2 16384x3000 → ImageDisplayView
```

**F3 範例**：
```
10:37:21.226 T 1 StopGrab
T1 capture gate closed standby=on
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
