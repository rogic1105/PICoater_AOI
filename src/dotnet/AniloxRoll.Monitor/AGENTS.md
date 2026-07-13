# src/dotnet/AniloxRoll.Monitor — 前端（C# WinForms 應用）架構

> 此檔在編輯 `src/dotnet/AniloxRoll.Monitor/` 下任何檔時由 Codex 疊加載入，鏡像 `sdk/AGENTS.md`。
> 專案總規則見 repo 根 `AGENTS.md`；可重用 UI 元件（TanukiCv.Controls）規則見 `sdk/AGENTS.md`。
> 本檔放「這個 app 怎麼組裝 UI」＝ State 模型（SSoT）+ 四層分工 + 協調層角色。
>
> **分工提醒**：sdk/TanukiCv.Controls = 可重用 UI 樂高（mechanism，「有哪些積木」）；
> 本檔 = 這個產品怎麼拼積木（policy）。兩者都談 UI，但不同層次。

## 架構原則：SSoT 原子結構

所有「設定變更 → 副作用」流程必須遵守這個三層分工：

```
                   SettingsHub (state, SSoT)
                          │
                Changed event ↓
        ┌─────────────┬────┴────┬─────────────┐
        ↓             ↓         ↓             ↓
   PropertyGrid   image       chart 閾值    其他副作用
   (顯示)        (Mura on/off) (StripLines)  (save disk、reload、Live merge)
```

**規則：**

1. **state 集中在 `SettingsHub`** — 所有 setting 變更走 `Set` / `SetBatch` / `NotifyExternalChange`。沒有任何路徑直接 `_settings.X = ...`（bootstrap 階段例外，但加註解標記）。

2. **每個 UI 元件都是 view（搖桿）** — 按鈕、chart、滑桿、PropertyGrid 都是改 setting 的入口，不是邏輯擁有者。「點 chart 切 enhance」= 改 `EnableMuraEnhance`，副作用由 event 訂閱者跑，**不在 click handler 內 inline 跑副作用**。

3. **副作用是 view 對 event 的反應** — `FitToScreen`、`OnStitchModeChangedAsync`、`ApplyMuraEnhance`、save disk、PropertyGrid 同步顯示 — 全部訂閱 `Changed` event。view layer 自己決定怎麼更新，**不互相直接呼叫**。

4. **嚴格 transition 順序例外** — 多個 setting 同時變更且需要 atomic transition（如 chart click 同時改 StitchMode + EnableEnhance），用 `SetBatch`（save once、不 raise event），caller 自己 inline await transition 順序。這條 trade-off 要寫註解說明。

5. **變更來源要可區分** — `SettingChange.Source` 標示 `PropertyGrid`（UI 自己已 paint）vs `Programmatic`（程式碼路徑，view 要被動 refresh）。避免重複刷新造成閃爍。

**反模式：**

- `click handler` 內 inline 改多個 setting + 呼多個 apply（過去 chart click 邏輯）
- 跨層直接呼叫（如 chart click 直接 `await ApplyReviewEnhance(...)`，繞過 event）
- view 之間互相 invalidate（image view 知道 chart 存在）
- setting setter 寫 disk（save 屬 Hub 職責）

**討論 / 設計時的提問順序：**
「這是改哪個 setting？」 → 「副作用是什麼？」 → 「哪些 view 要更新？」— 而不是「按下按鈕跑哪些函式？」

## 顯示鐵則（2026-07-06 使用者定版）：顯示一律 CPU、MIL 只取像、主畫面永遠合圖

0. **主畫面（camLiveMain）永遠顯示「合圖」**——模式差別只在：即時合圖（ImageDisplayView）vs
   瀑布合圖（WaterfallView 捲動）。點縮圖＝選中/置中該相機。預覽背景的主畫面＝7 台背景合圖。
1. **監控縮圖（camLive1~7）在兩種主畫面模式下一律顯示即時相機影像**——同一來源
   （`AniloxCamera.OnDisplayFrame` bytes → CPU ThumbStrip），兩模式同源、行為一致（點選/高亮）。
2. **顯示一律 CPU 元件**：ImageDisplayView / ThumbStrip / WaterfallView / ImageCanvas。
   **MIL 只負責取像**（grab hook）；`_milDisplayBuffer` 保留（合圖 merge target 幀源）。
3. 滑鼠座標/點選/縮放/視野查詢一律走 ImageCanvas 事件（StatusChanged / Click / wheel）。
4. sample（sdk/MIL/samples）不在此限（原生直繪示範保留在範例）。

## 架構原則：前端 UI 分層（View / 協調 / State / Service）

UI 通用思想 =「**長什麼樣 / 做什麼 / 真相是什麼** 三件事分開」。本專案是 **MVP 變體**
（View + SettingsHub + Coordinator）—— 別硬套 MVVM/MVU（WinForms 資料綁定弱，硬套變樣板地獄）。

```
① View 呈現層      使用者碰的（Form 9 partials + 控制項）          搖桿/螢幕
        ↕
② 協調層（中間層）  翻譯：點擊→改 state、state 變→更新畫面 + 副作用    ← 唯一還在整理的層
        ↕
③ State 真相層     SettingsHub（SSoT，見上一節）                  帳本
        ↓
④ Service 業務層   檢測/統計/IO FSM/儲存（不碰 WinForms）          後勤
```
**鐵律（同後端依賴單向）**：View 不知業務（只喊「我被點了」）；Service 不知 View 長相（只回傳資料／事件）；中間靠協調層翻譯。各層仍在漸進收斂，**不得假設某層已經完全乾淨**；以本節 ownership map 與現行 code 交叉稽核。

### 協調層的五個角色（每個有判準，勿用一個名字吃掉全部）
| 角色 | 判準（一句話） | 例 |
|---|---|---|
| **Coordinator**（每 feature 一個） | 一個功能流程的指揮：訂閱 Hub、更新多 view、呼 service、可有副作用 | `ReviewStitchCoordinator` |
| **ControlAdapter / Binder**（每控制項群一個） | 只包「單一控制項群」的事件 / 格式 / guard | `DateTimeNavigator`（combo adapter）、忙碌鎖鈕、滾輪攔截 |
| **Service** | 業務 / 硬體能力，**不引用 System.Windows.Forms** | 統計 / IO FSM / 儲存 |
| **Helper** | `static`、純函式、無 Form/control/service 欄位 | 座標換算、計時 |
| **View** | 使用者點/拖/選，只把 intent 寫進 state 或呼 coordinator command | Form partials + 控制項 |

> ⚠ **別把 Presenter/Manager/Coordinator/Navigator 機械改名成一個** —— 要**按職責重分**。少了 ControlAdapter，Coordinator 會變新 god object（Codex 審查結論）。

### 分層依賴與責任邊界（監控／回顧／報表重構的統管）

本節是 app 分層與 ownership 的**唯一規範來源**。`$verify-flows` 的 DVT 契約管「行為有沒有變」，本節管
「行為應由哪一層擁有」；skill 只寫操作 SOP，不得另抄一份架構。契約是當下設計意圖而非真理：
要改邊界時先有意識更新本節與理由，再改 code；與 code 衝突時先考古，不直接信任任一邊。

```
View/Form ─→ ControlAdapter/Binder ─→ Feature Coordinator/Presenter
                                             ├─→ State Hub / feature state
                                             └─→ Application Service/Repository ─→ SDK/Bridge
結果／事件  ←───────────────────────────────────────────────────────────┘
```

依賴只准向右；結果以 return value／DTO／event 回來，**下層不得持有或呼叫上層物件**。

| 層 | 可以擁有 | 禁止擁有 |
|---|---|---|
| **View / Form partial** | 讀控制項、顯示結果、把 click/change 轉成 intent、單一控制項局部視覺更新 | 掃目錄、解析 CSV/bin、硬體通訊、跨頁流程狀態、重試／debounce policy |
| **ControlAdapter / Binder** | 一組控制項的事件接線、guard、格式轉換、enable/highlight/selection | 統計／檢測、檔案 IO、硬體 IO、跨 feature 協調 |
| **Coordinator / Presenter** | 一個 feature 的流程、非同步 token、latest-only/debounce、組合 state + service + view command | protocol 細節、直接解析檔案、承接不相關 feature、成為全 Form service-locator |
| **State** | `SettingsHub`、使用者 session、明確命名的 feature 暫時態；每個真相一份 | Form/control、檔案／網路／硬體 IO、畫面重繪 |
| **Application Service / Repository** | 產品業務規則、統計、檢測、CSV/bin 查詢、存檔、硬體 FSM；不碰 WinForms | 讀寫控制項、MessageBox、知道畫面布局、反向呼 coordinator |
| **SDK / Bridge** | 通用機制、演算法、硬體 protocol／transport；以介面與基本型別供 app 組合 | `InspectionSettings`、GrabId/Mura 等產品 policy、引用 `AniloxRoll.Monitor` |

`Helper` 不是第七層：只有 `static`、純函式、無 Form/control/service 欄位才可叫 Helper；否則按責任歸入
Adapter、Coordinator、Service 或 Repository。

### 三大功能 ownership map（2026-07-13 現況與目標）

| Feature | View intent / render | 協調 owner | Service / Repository owner | 重構邊界 |
|---|---|---|---|---|
| **監控** | `AniloxRollForm.Live.cs`、`Background.cs` | `LiveCameraManager`＝相機/grab 對外 facade；`LiveDisplayCoordinator`＝主畫面/縮圖/瀑布/背景預覽顯示狀態 | `AniloxCamera`、`CameraFrameSaver`、`InspectionEngine`、`InspectionLogService` | 顯示狀態不得回流 `LiveCameraManager`；先稽核 acquisition facade 剩餘責任，再決定是否續拆，禁止只因檔案大就拆 |
| **回顧** | `AniloxRollForm.Review.cs` | `ReviewStitchCoordinator`＝單片/時段載入生命週期與 latest-only；`DateTimeNavigator`＝日期時間控制項 adapter | `ImageRepository`、`FrameTickIndex`、曲線/影像載入服務 | `FormInteractionHelper` 不得再擴張；資料夾、busy UI、cache、設定套用各歸獨立 owner。`ReviewStitchCoordinator` 先盤責任再判是否拆 |
| **報表** | `AniloxRollForm.Data.cs` | `DataStatisticsPresenter`＝報表 feature 門面；`DataDateGrabIdNavigator`＝序號/期間 adapter；`YieldPeriodChartPresenter`、`MuraProfileChartPresenter`＝各自圖表 | `InspectionStatisticsService`（現為過渡 god：CSV parse/query、統計、curve bin、CFG） | 先拆 service 的資料責任；Presenter 只保留協調，明細虛擬 List 可獨立 Presenter。圖表暫時態留各圖 Presenter，不回寫 setting |
| **跨功能設定** | PropertyGrid / Form setting intent | 各 feature 的 setting handler/coordinator | `SettingsHub`、settings store | `OnSettingChanged` 只做路由過渡；副作用逐步下放 feature owner，不新增中央 case 堆積 |

### God object 判準與拆分驗收

- **行數不是判準**：同一完整生命週期即使大，也可能是單一責任；同時擁有 UI、IO、state、不同 feature 才是 god。
- 拆之前先列「現有責任｜caller｜state owner｜副作用｜DVT」；沒有黃金 log／測試的責任先補最小儀器。
- 每刀只搬一個責任，預設**行為不變**；build + 對應 `verify-flows`／`check_all_flows.py` + 上機煙測綠後立即 commit。
- 新 owner 必須能用一句話命名；需要拿整個 Form 或萬能 Context 才能工作＝邊界仍錯。
- 拆完的舊 facade 若只剩無價值 forwarder 就刪；若保留做穩定 API，文件要明說 facade，不得暗藏 state／policy。

### app-UI vs sdk-UI 邊界：機制 vs 政策
- **機制（mechanism）→ sdk `TanukiCv.Controls`**：通用能力，不知資料意義（zoom/pan 畫布、縮圖列、滾輪攔截、圓角 Label、DPI 縮放、bitmap 池）。**換任何影像產品都能用**。
- **政策（policy）→ 留 app**：出現 Anilox / InspectionSettings / GrabId / Mura / StitchMode / CSV / IO/Light → 留 app。
- **灰區**：sdk 給 mechanism、app 包 policy（`ReviewDisplayManager` 包 sdk `ImageDisplayView` + overlay + 橘框 + review 預設 = 正確示範）。
- sdk UI 工具箱按種類歸檔（**第二個軸，非架構層**）：`TanukiCv.Controls/{Display, Input, Layout, Interaction, Imaging}`。

### SSoT 嚴格度（別追求零 inline 副作用，WinForms 成本太高）
- 設定值變更 → **必須**走 `SettingsHub.Set/SetBatch/NotifyExternalChange`
- 跨 view 副作用 → Hub event / coordinator
- **單一控制項的視覺副作用**（button enabled、label text、combo guard）→ **inline 可以**，但**不准改業務狀態**
- 硬體 / IO / camera callback → 進 coordinator + SafeBeginInvoke，不散落改 UI
- bootstrap 例外 OK，但加註解

### 反模式
- ❌ **god object**：一個類同時抓 View、IO、state 與多個 feature（現存重點：`FormInteractionHelper`、`InspectionStatisticsService`、`OnSettingChanged` 路由堆積）→ 按 ownership 拆，不按行數拆
- ❌ **Context 變 service-locator**：`*Context` DTO 只能當 constructor 參數傳依賴，不能變「什麼都拿得到」的萬能袋
- ❌ 機制留在 app（純通用元件沒進 sdk）
- ❌ 協調層後綴動物園（Presenter/Manager/Coordinator/Navigator 無判準混用）

### 已知重構目標（按上述判準）
- `FormInteractionHelper` god-facade → `ReviewFolderCoordinator`（folder/session/repository+navigator）+
  `BusyUiBinder`（cursor/buttons）+ `ImageCacheService`（bitmap 生命週期）+
  `InspectionSettingsCoordinator`（pipeline/chart setting 副作用）；review config/calibration state 回各 review owner。
- `InspectionStatisticsService` 過渡 god → CSV parser/repository、統計 query、range curve repository、CFG repository；
  相容格式判讀留 repository，不散到 Presenter。
- `DataStatisticsPresenter` → 保留報表流程門面；明細 ListView virtualization/selection 抽獨立 Presenter，導航與圖表已有 owner 不再搬回。
- `ReviewStitchCoordinator` → 先做責任/caller 盤點；只有載入生命週期、幀對齊、顯示協調能形成獨立 owner 時才拆。
- `LiveCameraManager` 的 display 拆分已完成；目前作 acquisition facade。後續只依責任稽核結果續拆，顯示不得回流。
- `OnSettingChanged` 中央路由逐 feature 下放；不要求一次砍完，但新增 setting 優先由 feature owner 訂閱。
