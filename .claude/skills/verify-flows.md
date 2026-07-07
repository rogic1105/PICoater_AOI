# verify-flows — UI 動作流程契約（EVT）與驗證

改「顯示/接線」code（LiveCameraManager / LiveDisplayCoordinator / ImageDisplayView / WaterfallView /
AniloxRollForm.Live|Background）後**必跑本 skill**：先「模擬測試」再（必要時）真機比對 log。

## 兩種驗證方式

1. **模擬測試（不開 UI）**：順著 code 把下面每條 flow 追一遍，推導預期的 `[Flow]` 序列，對照契約。
   改壞接線（漏訂閱/漏 teardown/置中掛錯來源）在這一步就會現形。
2. **真機比對**：使用者操作一輪 → 讀 `{AniloxRoot}\Logs\trace-*.log` 的 `[Flow]` 行對照契約。
   log 格式：`[Flow] HH:mm:ss.fff T{執行緒} 訊息`（唯一出口 `Services/FlowTrace.cs`）。

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
