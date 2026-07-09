# src/dotnet/AniloxRoll.Monitor — 前端（C# WinForms 應用）架構

> 此檔在編輯 `src/dotnet/AniloxRoll.Monitor/` 下任何檔時載入（巢狀 CLAUDE.md，鏡像 sdk/CLAUDE.md）。
> 專案總規則見 repo 根 `CLAUDE.md`；可重用 UI 元件（TanukiCv.Controls）規則見 `sdk/CLAUDE.md`。
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
**鐵律（同後端依賴單向）**：View 不知業務（只喊「我被點了」）；Service 不知 View 長相（只說「真相變了」）；中間靠協調層翻譯。**①View 元件 / ③State Hub / ④Service 已乾淨；亂的只有 ②協調層。**

### 協調層的五個角色（每個有判準，勿用一個名字吃掉全部）
| 角色 | 判準（一句話） | 例 |
|---|---|---|
| **Coordinator**（每 feature 一個） | 一個功能流程的指揮：訂閱 Hub、更新多 view、呼 service、可有副作用 | `ReviewStitchCoordinator` |
| **ControlAdapter / Binder**（每控制項群一個） | 只包「單一控制項群」的事件 / 格式 / guard | `DateTimeNavigator`（combo adapter）、忙碌鎖鈕、滾輪攔截 |
| **Service** | 業務 / 硬體能力，**不引用 System.Windows.Forms** | 統計 / IO FSM / 儲存 |
| **Helper** | `static`、純函式、無 Form/control/service 欄位 | 座標換算、計時 |
| **View** | 使用者點/拖/選，只把 intent 寫進 state 或呼 coordinator command | Form partials + 控制項 |

> ⚠ **別把 Presenter/Manager/Coordinator/Navigator 機械改名成一個** —— 要**按職責重分**。少了 ControlAdapter，Coordinator 會變新 god object（Codex 審查結論）。

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
- ❌ **god object**：一個類什麼都抓、什麼都做（現存：`FormInteractionHelper` god-facade、`OnSettingChanged` 唯一 dispatcher、`LiveCameraManager` = Service+Coordinator 混血）→ 按職責拆（同後端拆肥 module）
- ❌ **Context 變 service-locator**：`*Context` DTO 只能當 constructor 參數傳依賴，不能變「什麼都拿得到」的萬能袋
- ❌ 機制留在 app（純通用元件沒進 sdk）
- ❌ 協調層後綴動物園（Presenter/Manager/Coordinator/Navigator 無判準混用）

### 已知重構目標（按上述判準）
- `FormInteractionHelper` god-facade → 拆 `CanvasCoordinator` + `ReviewFolderCoordinator` + `BusyUiBinder` + `ImageCacheService` + `InspectionSettingsCoordinator` + 殘留 Helper
- `LiveCameraManager` → 拆 `LiveCameraService`（相機/MIL/grab）+ `LiveDisplayCoordinator`（Panel/display/chart）
- `OnSettingChanged` 唯一 dispatcher → 拆成多個 feature coordinator 各自訂 Hub
- 純機制搬 sdk：MultiClickDetector（收斂 ImageCanvas 內建重複那份）/ WheelInterceptor / ProportionalScaler / EventGuard / BitmapPool（RoundedLabel 已搬）
