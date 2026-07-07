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
**intent 行清單**（使用者動作入口各記一行 `ui:...`）：【開始抓取】【取得背景】【預覽背景】
【讀取資料】(Review/Data)、設定[主畫面顯示]變更。新增會動到顯示的入口 → intent 行一併加。

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
（縮放/互動期間**不得出現 `autoFit(...)` 行**——出現＝系統 fit 跟使用者縮放打架（fit 打架回彈家族）。
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

## 模擬測試的極限（誠實邊界）

模擬蓋得住：步驟順序/訂閱生命週期/置中來源/事件接線。蓋不住：硬體時序（stall/掉幀）、native 行為、
視覺呈現（配色/佈局/重繪）。這三類改動仍需真機 spot check——但頻率遠低於逐項手測。

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
