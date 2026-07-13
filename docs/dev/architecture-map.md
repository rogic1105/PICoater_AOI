# PICoater AOI — Repo 架構地圖

> 高層「東西住哪 + 怎麼流」的鳥瞰圖。**穩定、不隨 UI 改動漂移**（取代已廢除的 ui-flow.html）。
> 細節（每個檔的職責、控制項名、參數預設）在 repo 根 `AGENTS.md` 的速查表 + `.agents/skills/`；
> 本檔只給「層次 / 依賴方向 / 資料流」三張地圖，看完知道去哪找。

---

## 1. 分層與依賴方向（單向 `src → sdk → third_party`，絕不反向）

```
┌─────────────────────────────────────────────────────────────┐
│ src/dotnet/AniloxRoll.Monitor   應用層（產品交付；只剩 UI）   │
│   View(Form 9 partials) ─ 協調層 ─ State(SettingsHub) ─ Service │
└───────────────┬─────────────────────────────────────────────┘
                │ 只能往下依賴
┌───────────────▼─────────────────────────────────────────────┐
│ sdk/  可獨立 split 的 library（無 GUI、無 exe）              │
│  ├─ TanukiCv/   影像 SDK（durable，跨產品共用）              │
│  │    native(CUDA 引擎 + pipeline) + dotnet(Core 純算 / Controls WinForms) │
│  ├─ MIL/        ★拋棄層：grabber 封裝（換硬體整區換）        │
│  └─ Bridges/    設備橋接（IO/Light/Storage，純函式庫 + 介面） │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│ third_party/ + 各 sdk 元件 vendor/   外部純 build-time lib    │
└─────────────────────────────────────────────────────────────┘

旁支：tools/（跨元件工具）、tests/（NUnit 三層）、benchmark（跟被測對象住）、deploy/（現場部署）
```

**鐵則**：`grep -r "using AniloxRoll" sdk/` 應為 0（sdk 不反向依賴 app）。
durable（演算法/合圖/像素↔mm）放 `TanukiCv/`；throwaway（MIL）放 `MIL/`，可重用 IP 不困在拋棄層。

---

## 2. 執行期資料流（取像 → 檢測 → 顯示/存檔）

```
  IO/光源（觸發）                          每台相機一條鏈
  IoGrabController(ET-7044 Modbus)   ┌──────────────────────────────────────┐
  LightController(RS-232) ──開燈──▶  │ MilCamera(sdk/MIL)  grab → FrameReady │
                                     │        ↓ 訂閱                          │
                                     │ AniloxCamera(app)  hook 內：           │
                                     │   ① GPU 檢測 pipeline                  │
                                     │   ② 顯示 bytes / 合圖貼圖              │
                                     │   ③ 存檔                               │
                                     └───┬───────────┬───────────┬───────────┘
                                         │①          │②          │③
                  ┌──────────────────────▼──┐   ┌────▼─────┐  ┌──▼──────────────┐
                  │ tanuki_pipeline_api.dll  │   │ 顯示層   │  │ CameraFrameSaver │
                  │ (P/Invoke→CUDA)          │   │ (見下圖) │  │  → 本地 + 遠端   │
                  │ find_stream_ridgeline    │   └──────────┘  │  RemoteCopyService│
                  │ = mura 脊線檢測          │                 │  → 儲存機(SMB)   │
                  │ → 欄/列 mean/max 曲線    │                 └──────────────────┘
                  └───────────┬──────────────┘
                              │ 曲線資料
                  ┌───────────▼──────────────────────────────┐
                  │ 欄曲線 ColumnCurveChartHelper (X/切向)     │
                  │ 列曲線 RowCurveChartHelper  (Y/法向)       │ ← 軸命名見 AGENTS.md「術語標準」
                  │  經 RowCurveDisplayAdapter（live/回顧共用）│
                  └───────────────────────────────────────────┘

設定真相：SettingsHub(SSoT) → InspectionSettings；UI 改值走 Hub event 扇出（見 src 巢狀 AGENTS.md）。
硬體無 encoder/外部觸發 → 相機靠 CLProtocol 套線掃才同頻（free-run 偶發不同步靠 tick 對齊補黑）。
```

---

## 3. 顯示 pipeline（多相機監控的「秀」三選一互斥）

```
LiveCameraManager(app)  ── 編排 + 生命週期，「秀」全委派 ──┐
  │  he_MainDisplay（主畫面顯示）三選一：                   │
  ├─ SmartCanvas/ImageCanvas ─▶ LiveDisplayCoordinator(app) │
  │                              └▶ ImageDisplayView(sdk)    │  主畫布 ImageCanvas
  │                                  + ThumbStrip 縮圖       │  + LOD + mm overlay
  ├─ MilDirect（MIL 直繪合圖） ─▶ GlobalMergeCoordinator(app)│  MIL 合圖 display
  │                              └▶ MultiCameraMerger(sdk/MIL)│  「拼」工頭
  └─ Waterfall（瀑布捲動） ─────▶ WaterfallView(sdk)         │  全幅合圖往下接

「拼」= MultiCameraMerger / MergeLayout（佈局唯一來源，sdk/TanukiCv.Core，純算術）
「秀」= 上面三條各自的 display 元件
合圖永遠 Global（hb_StitchMode 寫死，Vertical 已退場）；黑槽（沒影像相機）一致參與中線分界。
```

---

## 4. 找細節去哪

| 想找 | 看 |
|------|-----|
| 每個檔的職責 | `AGENTS.md` §關鍵檔案速查 |
| 控制項標準名↔程式名 | `AGENTS.md` §控制項速查 |
| PropertyGrid 參數/預設 | `AGENTS.md` §檢測參數速查 |
| 軸命名（欄/列 ↔ col/row） | `AGENTS.md` §術語標準 |
| sdk 元件地圖 / 分層鐵則 | `sdk/AGENTS.md` |
| app UI 四層分工 / 協調層角色 | `src/dotnet/AniloxRoll.Monitor/AGENTS.md` |
| 改某範圍的注意事項 | `.agents/skills/`（modify-ui / modify-acquisition / modify-pipeline …） |
| 演算法分層（kernel→pipeline） | `sdk/AGENTS.md` §演算法分層 |
| MIL 取像/stall/併發踩坑 | `sdk/MIL/docs/` + `docs/dev/MIL_API_Reference.md` |
```
