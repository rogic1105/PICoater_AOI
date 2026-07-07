# verify-flows — UI 動作流程契約（EVT）與驗證

app ＝【監控/回顧/報表】三個 tab。改任何「使用者動作 → 接線 → 顯示/資料連動」流程後**必跑本 skill**：
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
函式流程屬於 CLAUDE.md/程式註解，不進契約。每條契約用三種型態組合：

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

## 偏序驗證規則（多執行緒鐵則）

- **同一執行緒（同 T）內驗全序**：順序必須完全符合契約。T1=UI 執行緒（按鈕/設定/接線），Tn=各相機 MIL 回呼。
- **跨執行緒只驗「因果 + 完整性」**：如 `StartGrab 必早於所有 firstFrame`、`每台在線相機首幀必出現`。
  **不驗**非決定性交錯（cam1/cam2 誰先本來就不定，驗了必誤報）。

## 不變量（任何 flow 都不得違反；見 app 巢狀 CLAUDE.md 顯示鐵則）

- app 內零 MIL 原生顯示視窗/滑鼠 hook（headless；MilCamera panelHandle=Zero）。
- 主畫面永遠合圖（即時=ImageDisplayView、瀑布=WaterfallView）；縮圖兩模式一律即時 ThumbStrip（橘框選中）。
- `SwitchMainDisplay` 的 `center=True` **只**允許出現在明確點縮圖/狀態字之後（拖曳/程式化路徑=False，否則回彈）。
- view 訂閱 `cam.OnDisplayFrame` 且 Enable* 冪等 → **相機批次換新（Allocate/Free）前後必有對稱 teardown**。

## Flow 契約

### F1 開機配置（AutoAllocateCameras）
```
T1: AllocateCameras begin（expect N）
T1: （前次 view 存在才有）TeardownImageDisplay / TeardownWaterfall
T1: ApplyMainDisplayMode → {ImageCanvas|Waterfall}
T1: {EnsureImageDisplay|EnableWaterfall} create + subscribe M cams
T1: SwitchMainDisplay cam=1 center=False
T1: AllocateCameras done（cams=M）
T1: （CLProtocol 就緒後）EnableGlobalMerge（slots=7）
```

### F2 開始抓取（btnLiveGrab，已配置）
```
T1: StartGrab（cams=M）
T1: ApplyMainDisplayMode → 同模式    ← 冪等：不得出現 create/teardown 行
Tn: firstFrame camX WxH → {ImageDisplayView|Waterfall}   ← 每台「在線」相機恰一行，順序不定
（首幀齊後進入穩態 → 適用「穩態靜默通則」：無互動下不得再有顯示狀態變更行）
```

### F3 停止抓取
```
T1: StopGrab
（之後不得再出現 firstFrame / 任何 [Flow] 顯示行，直到下一個動作）
```

### F4 切「主畫面顯示」設定（即時↔瀑布，即時生效）
```
T1: ApplyMainDisplayMode → 新模式
T1: Teardown{舊 view}（unsubscribe M）
T1: {新 view} create + subscribe M
（grab 中切換：接著每台在線相機 firstFrame → 新 view）
```

### F5 點縮圖/狀態字（縮圖→主畫面連動）
```
即時模式點縮圖：T1: SwitchMainDisplay cam=N center=False
  ← 置中由 ImageDisplayView「內部」完成（thumb click → CenterOnCamera），外部呼叫只同步選中 → False 是對的
瀑布模式點縮圖／任一模式點狀態字：T1: SwitchMainDisplay cam=N center=True
  ← 置中由 coordinator 做（WaterfallView/ImageDisplayView.CenterOnCamera）
（兩者皆：主畫面置中到 cam N、橘框=N）
```

### F6 拖動主畫面（主畫面→縮圖反向連動）
```
（不得出現任何 center=True 行——出現即回彈 bug）
T1: centerCam → camX（IC|WF）   ← 中心相機每跨一台一行（快拖連續數行=正常；跳號=補刷失效）
（即時=ImageDisplayView.UpdateReverseThumbSync、瀑布=WaterfallView.CenterCamChanged；
  兩者皆有 30/33ms timer 補刷，快拖不跳格）
```

### F6b 滾輪縮放主畫面
```
T1: IC|WF wheelZoom in|out → zoom=Z（fit=F）   ← 每手勢至少一行（100ms 節流）
T1: IC|WF|RV fit(double-click) / physical1x(triple-click)   ← 使用者 fit/1x 手勢（合法的視野重設主人）
（縮放/互動期間**不得出現 `autoFit(...)`/`lodRebind(...)` 行**——出現＝系統 fit 跟使用者縮放打架。
  zoom 突然回 fit 而無 fit(double-click) 行＝有東西在暗中重設（孤兒判讀）。
  `autoFit(firstFrame ...)` 只允許在 view 建立後首幀；`autoFit(sizeChanged@fitView ...)` 只允許在
  使用者「未動過視野」時的尺寸變更。centerCam 行在縮放中出現＝正常（中心相機隨視野變）。）
```

### F7 重配置（FreeCameras → 再配置）
```
T1: FreeCameras（cams=M）
T1: TeardownImageDisplay / TeardownWaterfall（有哪個拆哪個）
T1: （再配置時）F1 全序重跑——view 必須重建+重訂閱新相機批次
```

### F8 取得背景 / 預覽背景
現況：取得背景=借用現有 grab 採集（啟停包夾）、預覽=ImageCanvas overlay 蓋最上層（**不得動 MIL 顯示開關**）。
Wave3 改與 grab 共用顯示 API 後更新本節。

## 相機參數契約（P 系列）

### P1 滑桿/數字框調參（曝光/線掃/高度，放開才套用）
```
T1: ui:【相機參數】camN {param}={v}｜All {param}={v}    ← 帶參數名+值單行自足（Exp/LineRate/Height…）
（之後的 SwitchMainDisplay center=False（refresh）等程式化行歸此 intent 管；
  滑桿拖曳 vs 數字框輸入同一路徑，log 不區分）
（⚠ 判讀例外：開機後 ~1 秒內的「全部套用」×3（曝光/線掃/高度）＝初始值塞進 All 控制項觸發
  ValueChanged→debounce 套用的**副作用**（出處查證 2026-07-07：7a017a3/993d8cc 皆無「防跑掉」設計記錄；
  **有記錄的防跑掉機制**＝①AllocateCameras/Initialize 套 settings 參數 ②CLProtocol 就緒自動重套線掃）。
  非使用者動作；**行為保留勿抑制**——重複寫同值無害，且無法排除它在替①②兜底。
  口述歷史（2026-07-07 使用者）：很早期「開程式時曝光會亂飄」、修正無 commit 記錄可追——
  三連發可能正是實際擋住它的兜底，動它有復發風險。毫秒級連發＋緊跟開機序列＝辨識特徵。）
禁止：調參數不得出現任何 MIL 視窗——headless 鐵則：**每一個 MdispSelectWindow 呼叫點都必須帶
`_panelHandle != IntPtr.Zero` 守門**（MIL 對 Zero handle 會自開獨立浮動視窗；2026-07-07 實例：
改高度 realloc 路徑漏守門 → 4 台各跳一個視窗）。新增 MdispSelectWindow 呼叫點＝必帶守門。
```

## Mura 警告契約（M 系列）

### M1 曲線超過門檻（grab 中）
```
Tn: ⚠ MURA 超標（v|h）mean=…/max=…（thr …/…，IO已連線|未連線→僅畫面警告）   ← 邊緣觸發（進入超標一行）
Tn: MURA 恢復（v|h）                                                        ← 離開超標一行
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
- 違規樣本：chart 明顯超標卻無「MURA 超標」行＝判定鏈斷（2026-07-07 盲測抓到：舊版被
  IO 未連線 early-return 整段跳過＝操作員零警告）。

## 回顧 tab 契約（R 系列；儀器前綴 RV）

### R1 讀取資料（btnReviewSelectFolder）
```
T1: ui:【讀取資料】鈕（Review）
T1: （首次）RV EnsureImageDisplay create（thumbs=7）
T1: RV loadGrab begin {grabId}（proc=…）
T1: RV pushFrames P/7（merge=True, feedScale=…）   ← P=該 grab 有影像的相機數；缺台=黑占位
T1: RV loadGrab done {grabId}（…ms）
（grab 中按：另會出現 DisableGlobalMerge 等監控行——歸本 intent 管，見孤兒判讀規則）
```

### R2 單片序號切換（cbReviewId）——分層載入（2026-07-07 定版）
```
T1: ui:【單片序號】→ {grabId}
T1: RV curves {grabId}（…ms）          ← 快路：欄+列曲線+CFG 即時跟滾動（chart 先行，使用者掃異常）
（影像 debounce 250ms：滾動中不發完整載入；停下才載「最後選取」）
T1: RV loadGrab begin {grabId} → RV lodRebind merge …（fit reset）→ RV pushFrames → RV loadGrab done
```
- **分層**：曲線每個 intent 都跟（`RV curves`，舊的記 `RV curves stale-drop`）；影像只載 settle 後的最後一張。
- **換序號＝重設視野（fit）＝預期**（各 grab 高度不同 → lodRebind 合法出現）。
- **最後贏 token（快路+完整共用）**：最後一個非 stale 的 `curves`/`loadGrab done` 的 grabId
  必須＝最後一個 intent 的 grabId——不符＝token 破了。
- begin 無對應 done/stale-drop＝載入中斷；pushFrames P 與 CSV 台數不符＝掉圖。

### R3 時段導航（cbReviewDate/cbReviewTime 手動）
```
T1: ui:【時段導航】（cbReviewDate/Time）
T1: RV pushFrames …（時段模式載入；依實測補完整序列）
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
（之後的反應行歸此 intent 管）
```
**⚠ 非 PropertyGrid 的使用者入口（點 label/chart 走 Hub.Set）會被記成 set:（程式來源）**——
這類入口必須自帶 `ui:` intent 行（如【暫停Mura檢測】【IO暫停】），否則盲測會認錯兇手身份
（2026-07-07 盲測實例：點 lblIoDoMura 被誤判為程式動作、點 lblIoConn 完全無痕漏抓）。

### S 系列不變量：view 互斥
**任一時刻主畫面 view 唯一**：設定[主畫面顯示]=即時 期間，不得出現任何 WF 前綴行/EnableWaterfall；
=瀑布 期間反之（不得出現 IC 主畫面 view 建立行）。切換瞬間走 F4（teardown 舊→create 新）。
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

（其餘設定逐一補進：每補一個 UI 功能，順手寫它的 S 條目。）

## 報表 tab 契約（D 系列）

### D0 tab 切換（全域，不限報表）
```
T1: ui:tab → 監控|回顧|報表
（tab 切換本身不觸發顯示重建——例外：切到回顧且 _reviewDirty → 接 R2 載入序列。
  開機 PrewarmAllTabs 的程式化 cycle 被 _suppressTabIntent 抑制不記——毫秒級三連發 tab 行
  ＝抑制失效（D 系列首輪誤報實例））
```

### D1 讀取資料（btnDataSelectFolder）
```
T1: ui:【讀取資料】鈕（Data）
（Data 統計載入無顯示儀器＝靜默合法；會連動 Review → 接 R1 的 RV 序列）
（預設：單片=最新、序號範圍=最舊→最新）
```

### D2 明細列表點選
```
T1: ui:【明細列表】→ {grabId}
T1: ui:【報表序號】→ {grabId}          ← 同步行（明細點選經 cbDataId commit，兩行成對＝正常）
T1: ui:【明細列表】同列再點 {grabId} → 回範圍模式
（範圍序號 cbDataIdStart/End 不因單片選取而變＝獨立。
  Review 同步＝標記 dirty **lazy**：切到回顧 tab 才接 R2 載入——明細點選後零 RV 行＝正確，
  D 系列首輪 log 實證；點選當下就出 RV 行反而是違規）
```

### D3 報表序號 / 序號範圍
```
T1: ui:【報表序號】→ {grabId}          ← 單片切換（同 D2 的 cb 版）
T1: ui:【序號範圍-起始|結束】變更       ← 手動拖範圍 → 期間高亮全滅（Custom）
```

### D4 年/月/日期間（lblChartNav 點選）
```
T1: ui:【期間-年|月|日】→ 範圍 {最舊}~{最新}   ← 取 cbDataYield 當前值設範圍 + 該期間綠高亮（互斥）
T1: ui:【期間-全局】→ 全範圍                    ← 點 groupBoxGrabIdRange
（active 期間改對應 cbDataYield → 範圍跟著更新；非 active 來源不觸發）
```

### D5 良率導航 / 篩選異常
```
T1: ui:【良率導航-年|月|日】→ {值}      ← 良率三圖跟著換週期
T1: ui:【篩選異常】→ 只顯示異常|顯示全部
```

## 附錄：函式路徑（導航用）與真 log 範例

> 路徑僅供快速定位，**重構會變、log 行才是判準**；範例取自真機（4 配置/2 在線），數值隨機台不同。

**F1 路徑**：`AutoAllocateCameras(Form)` → `LiveCameraManager.AllocateCameras`（teardown 兩 view）→
`ApplyMainDisplayMode` → `EnsureImageDisplay|EnableWaterfallDisplay`（subscribe）→ `SwitchMainDisplay` →
（CLProtocol 就緒）`OnCamerasHwReady` → `EnableGlobalMerge`
```
10:35:07.029 T 1 AllocateCameras begin（expect 7 cams）
10:35:07.388 T 1 ApplyMainDisplayMode → ImageCanvas
10:35:07.436 T 1 EnsureImageDisplay create + subscribe 4 cams（merge=False）
10:35:07.438 T 1 SwitchMainDisplay cam=1 center=False mode=IC
10:35:07.439 T 1 AllocateCameras done（cams=4）
10:35:12.901 T 1 EnableGlobalMerge（slots=7）
```

**F2 路徑**：`btnLiveGrab_Click` → `ToggleGrab` → `StartGrab`（ResetFlowFirstFrame）→ `ApplyMainDisplayMode`
→ `cam.SetUserGrabIntent(true)` →（每幀，MIL 回呼執行緒）`MilCamera.FrameReady` → `AniloxCamera.OnMilFrameReady`
→ `OnDisplayFrame` → `OnCameraDisplayFrame|OnCameraWaterfallFrame` → `PushFrame`
```
10:37:13.854 T 1 StartGrab（cams=4）
10:37:13.855 T 1 ApplyMainDisplayMode → ImageCanvas
10:37:15.170 T31 firstFrame cam1 16384x3000 → ImageDisplayView
10:37:15.207 T30 firstFrame cam2 16384x3000 → ImageDisplayView
```

**F3 路徑**：`btnLiveGrab_Click` → `ToggleGrab` → `StopGrab` → 並行 `SetUserGrabIntent(false)`
```
10:37:21.226 T 1 StopGrab
```

**F4 路徑**：PropertyGrid → `SettingsHub.Set(he_MainDisplay)` → `OnSettingChanged` →
`HandleLiveLayoutSettingsChanged` → `ApplyMainDisplayMode` → Teardown(舊) + Enable(新)
```
10:13:40.107 T 1 ApplyMainDisplayMode → ImageCanvas
10:13:40.108 T 1 TeardownWaterfall（unsubscribe 4 cams）
10:13:40.124 T 1 EnsureImageDisplay create + subscribe 4 cams（merge=True）
```

**F5 路徑**：`ThumbStrip.SelectRequested` → 即時＝ImageDisplayView 內部 `CenterOnCamera` 再轉外部
`ImageSelectCamera→SwitchMainDisplay(center=False)`；瀑布＝coordinator `SwitchMainDisplay(center=True)`
→ `WaterfallView.CenterOnCamera`。（範例待真機補）

**F6 路徑**：ImageCanvas 拖曳 → `StatusChanged` → 即時＝`ImageDisplayView.UpdateReverseThumbSync`
→ `SelectedCamChanged`；瀑布＝`WaterfallView.UpdateCenterCam` → `CenterCamChanged` → `OnWaterfallCenterCam`。
兩者另有 30/33ms timer 補刷。
```
10:13:27.999 T 1 centerCam → cam3（WF）   ← 相鄰台階梯式、間隔不規則＝健康手拖
10:13:28.372 T 1 centerCam → cam2（WF）
```

**F6b 路徑**：`ImageCanvas.OnMouseWheel`（FlowLog 100ms 節流）。違規樣本（2026-07-07 修復前，教學用）：
```
10:37:16.868 T 1 IC wheelZoom in → zoom=0.02（fit=0.01）
10:37:16.973 T 1 IC wheelZoom in → zoom=0.01   ← 滾放大卻回到 fit＝有人在重設（該次＝ClearFrame 空轉→lodRebind）
```

**F7 路徑**：`FreeCameras` → `DisableGlobalMerge` → `TeardownImageDisplay`+`TeardownWaterfall` →
`cam.Free()`×N →（再配置＝F1 全序）。（範例待真機補）

## 任意控制項 call chain 追蹤（F1~F8 以外的流程）

契約未涵蓋的控制項（如 `/verify-flows 讀取資料`），用通用追蹤法做迴歸驗證：

1. **查對照表**：CLAUDE.md §控制項速查 找程式碼 Name。
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
| Data | 讀取資料、序號範圍、序號選擇、年/月/日期間、良率圖導航、篩選異常 |
| 右側 | 檢測設定（Recipe/Algorithm/ChartScale）、相機參數滑桿 |
| 跨Tab | Review→Data 同步、Data→Review 同步 |

驗證中發現 skill 與 code 不一致 → 同步更新對應 skill（契約跟 code 對齊是本 skill 的存在意義）。
