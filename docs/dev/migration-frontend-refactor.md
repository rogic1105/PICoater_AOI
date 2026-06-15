# 重構 Plan：前端 UI 落入架構（協調層整理 + sdk 化）

> 狀態：**規劃完成、待執行**。基準＝前端架構已定（`src/dotnet/AniloxRoll.Monitor/CLAUDE.md` 四層 + 協調層五角色）。
> 原則沿用前幾次大重構：**分波 + 每步 build+抽測 + 每波結束是可 merge 乾淨點 + 增量絕不 big-bang**。

## 0. 目標
前端三層已乾淨（①View 元件 / ③State Hub / ④Service）；**只有②協調層還亂**。把協調層整理成五角色
（Coordinator / ControlAdapter / Service / Helper / View，判準見 app CLAUDE.md），拆掉 god object，
把純機制搬進 sdk。

## 1. Misfit 盤點（沒落入架構的，按嚴重度）
| Misfit | 行數 | 問題 | 目標 | 波 |
|---|---|---|---|---|
| `AniloxRollForm`（+9 partials） | 999+ | View 自己做協調（45 處接線 + `OnSettingChanged` god-dispatcher） | View 瘦身、協調搬 coordinator | 3 |
| `FormInteractionHelper` | ~~269~~→ 縮水 | Wave2 已丟 canvas 代理+gallery；剩資料夾載入/忙碌鎖/設定套用/螢幕校正/ReviewConfig | 拆法重估（CanvasCoordinator 已不適用，canvas 沒了） | 3 |
| `LiveCameraManager` | 1327 | Service+Coordinator 混血 | 拆 Service + Coordinator | 3 |
| `DataStatisticsPresenter` | 1394 | Coordinator 但太肥（統計+chart+跨tab+mura分布） | 內部再拆 | 3 |
| `MultiClickDetector` | — | 唯一來源債：app 一份（`AniloxRollForm.cs` panelClicker，MIL 直繪雙擊）vs SmartCanvas 內建一份 | 抽 sdk + SmartCanvas 共用 or 留；**未解** | 3 |
| ~~`CanvasInteractionHelper`~~ | ~~389~~ | ✅ Wave2 已 whole-delete | — | ✅2 |
| ~~`ThumbnailGridPresenter`~~ | ~~133~~ | ✅ Wave2 已刪（LiveDisplayView ThumbStrip 接管） | — | ✅2 |
| `CurveMergeHelper`/`GrabImageStitcher` | 193/191 | 半搬：數學在 sdk、app 剩讀檔殼 | 留薄殼或榨乾 | 4 |

## 2. 解耦風險分級（誠實）
- 🟢 低（機械、build 立驗）：純機制搬 sdk
- 🟡 中：完成 review 解耦（已追 70%）
- 🔴 較高：拆 god object（大塊、載重、working code）→ 只能增量、每步 build+抽測，絕不 big-bang
  （同先前拆 MilCamera/AniloxRollForm 成 partial 的成功招式）

## 3. 四波（低→高風險，每波獨立可交付/可 merge）

### Wave 1 — 純機制 → sdk 🟢
把 View 層通用積木搬進 sdk UI 工具箱（建 `TanukiCv.Controls/{Input,Layout,Interaction,Imaging}`）：
- [ ] `TrackBarWheelInterceptor` → Input/
- [ ] `ComboBoxWheelReverser` → Input/
- [ ] `ProportionalScaler` → Layout/（已有 RoundedLabel）
- [ ] `EventGuard` → Interaction/
- [ ] `BitmapPool` → Imaging/
- [ ] `MultiClickDetector`：**評估**——app 獨立那份可能隨 camReviewMain 死（Wave 2）；若留，抽 sdk + SmartCanvas 改用它（收斂內建重複那份＝唯一來源）
- 每個：移檔+改 namespace+去產品語意+補 XML 註解+csproj 移轉+呼叫端改 using（見 sdk/CLAUDE.md 6 步 SOP）
- 風險：低。app 改 `using` 即可；build 立驗。

### Wave 2 — 完成 review 解耦 🟡
（接續 `refactor/review-cleanup-oldpath` 未完的耦合控制項簇）
- [ ] 把 chart 視野來源從舊 canvas 重導到新路徑快取（`TryComputeCurrentViewRange` 3 用點 → `_reviewViewLeftMm`/`SameSourceViewRange`；Form 的 `ViewRangeProvider` 已死可刪）
- [ ] 新路徑補 **fit-on-load**（取代 `camReviewMain.FitToScreen()` 死條件；解「先顯示再 fit 閃」）
- [ ] 刪死接線：`camReviewMain.StatusChanged/EdgeReached` → `UpdateCanvasInfo`/`NavigateCamera`、`SaveCanvasView`×4 no-op、`RefreshChartRange` 死門面
- [ ] 評估 review 畫布的 UiActionLogger（drag/fit）是否要接到 LiveDisplayView.Canvas（功能 parity）
- [ ] 刪 `CanvasInteractionHelper` 顯示部分 + `ThumbnailGridPresenter` + `camReviewMain`/`camReview1~7` 控制項 + Designer
- [ ] `ReviewDisplayManager` 改直接吃 Panel（Designer 放 Panel）
- 風險：中。每項 build+上機抽測（單片/時序/監控三條）。

### Wave 3 — 拆 god object（按判準）🔴 增量、最謹慎
**拆序（審查重排）：dispatcher 先 → 混血 → facade → 自包含最後。理由：先拆 SSoT 中樞，後面三個碰它時才有地方掛；最孤立的放最後降風險。**
- [ ] **①** `OnSettingChanged` dispatcher → **選項1（已定）：不拆掉 dispatcher 本身，拆它順便做的 8 種 feature 細節邏輯。**
  - **dispatcher 留著**＝它是協調層的合法核心職責：持 `SemaphoreSlim` 序列化（防連點 reload race）+ 跑有序共用前段（閾值/Live/統計）。**這把鎖必須單一入口持有，不可散到 N 個獨立 Hub 訂閱者**（散 = 共用鎖又回到單一點 / 各鎖各的 = race 回來）。`SemaphoreSlim` 一行不動。
  - **搬出去**＝各 feature 副作用 → 對應 coordinator（LightCoordinator / EnhanceCoordinator / ChartScaleCoordinator / LayoutMergeCoordinator…），dispatcher 改成「拿鎖 → 跑前段 → 依 setting 群組呼 coordinator → 放鎖」。
  - 反模式釐清：根 CLAUDE.md「OnSettingChanged 勿拆」指的是**結構/鎖別碎**，與本項（搬 feature 邏輯）不衝突。
  - 新建的 coordinator **當下正名**（不等 Wave 4）。
- [ ] **②** `LiveCameraManager`（1327）→ `LiveCameraService`（相機/MIL/grab/生命週期，不碰 WinForms）+ `LiveDisplayCoordinator`（Panel/LiveDisplayView/chart/view-range）。先 partial 物理拆、再抽類。**獨立子分支 + partial 階段就上機**——碰「先關 M_UPDATE 再 select」順序敏感邏輯，錯一步殘影/凍結。
- [ ] **③** `FormInteractionHelper`（**Wave2 後縮水**）→ 拆法重估：CanvasCoordinator **已不適用**（canvas 沒了）。剩 `ReviewFolderCoordinator` + `BusyUiBinder` + `ImageCacheService` + 設定套用殘留 Helper。`FormInteractionContext` 改回純 DTO（不准 service-locator）。
- [ ] **④** `DataStatisticsPresenter`（1394）→ 內部拆（統計計算 service / chart coordinator / 跨 tab 同步）。**放最後＝對其他三個依賴最少、跨 tab 邊界清楚、風險孤立。**
- [ ] **MultiClickDetector 唯一來源債**：app 一份（`AniloxRollForm.cs` panelClicker，MIL 直繪雙擊）vs SmartCanvas 內建一份。動到相關 code 時收斂（抽 sdk 讓 SmartCanvas 共用 or 評估留）。
- 風險：高。每個 god object 一條子分支，先 partial 再抽，每步 build + 完整上機回歸。**絕不一次拆完。**
- **每波結束 doc 同步**（新增步驟）：改完即更新根 `CLAUDE.md` 關鍵檔表 + 控制項速查 + 對應 skill（避免文件漂移誤導下次對話）。

### Wave 4 — 命名收斂 + namespace=資料夾 🟢
- [ ] 後綴按角色正名：`*Presenter`/`*Manager`/`*Navigator` → `*Coordinator`/`*ControlAdapter`/`*Service`（依實際職責）
- [ ] namespace = 資料夾 1:1（修 `Core.Data`/`Core.Services` 等錯位；刪空的 `Core/` 資料夾）
- [ ] `CurveMergeHelper`/`GrabImageStitcher` 榨乾（數學已在 sdk，留最薄讀檔殼）
- 風險：低（機械改名，build 立驗）。但**放最後**——名字對了之前的搬移才不會白做。

## 4. 分支策略
- 先把 `refactor/review-cleanup-oldpath`（review 清理 + RoundedLabel + 架構 docs，全驗過）**merge 回 main**＝乾淨基準。
- 每波一條分支（`refactor/fe-wave1-mechanism` …），波結束 build+測 OK → merge main。
- Wave 3 的每個 god object 再開子分支。

## 5. 提問順序（落實判準，避免邊做邊漂）
動每個類前問：「它是五角色的哪個？」→「它現在混了幾個角色？」→「先 partial 物理拆，再把散落的職責歸位。」
（= [[feedback_modularize_not_size]]：看模組化非大小；連續區塊物理拆只是第一步，必做散落歸位。）
