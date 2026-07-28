# PICoater AOI repository lookup reference

> Sole owner of file, API, setting, and control lookup facts. It does not define architecture rules,
> feature ownership, or refactor plans. Start with
> [`architecture-overview.md`](architecture-overview.md) for logical owners and use
> [`repository-layout.md`](repository-layout.md) for directory responsibilities.
> Produced-file locations and copy/retention policy are owned by
> [`output-storage-map.md`](output-storage-map.md).

Paths in the tables are repository-relative unless a shorter prefix is stated. Confirm every entry
against current code with `rg` before editing because lookup data can become stale.
## Native API

兩組 DLL，均宣告於 `src/dotnet/AniloxRoll.Monitor/Interop/NativeMethods.cs`：

| DLL | 函式 | 用途 |
|-----|------|------|
| `tanuki_pipeline_api.dll` | `TanukiPipeline_Create` / `Process` / `GetLastError` / `Destroy` / `ComputeColumnMean` | GPU 檢測 pipeline；output 可選擇回傳存檔縮圖 |
| `tanuki_cv_api.dll` | `TanukiCv_AllocPinned` / `TanukiCv_FreePinned` | CUDA pinned memory 管理 |
| `tanuki_cv_api.dll` | `TanukiCv_FastReadBMP` | 快速讀取 BMP（繞過 GDI+） |
| `tanuki_cv_api.dll` | `TanukiCv_Resize_GPU` | GPU 縮圖（**LOD 顯示用**；存檔縮圖已改走 pipeline fused 輸出，不再走這支） |

## 關鍵檔案速查

| 路徑 | 職責 |
|------|------|
| `Interop/NativeMethods.cs` | 唯一 P/Invoke 宣告點 |
| `ImageProcessing/NativeBufferPool.cs` | CUDA pinned buffer 管理 |
| `ImageProcessing/InspectionEngine.ImageProcessing.cs` | 縮圖/全解析度影像處理 |
| `ImageProcessing/InspectionEngineConfig.cs` | MaxWidth=16384, MaxHeight=10000, DefaultSaveResizeScale=5 |
| `ImageProcessing/BatchInspectionService.cs` | Parallel.For 批次縮圖 |
| `UI/Form/AniloxRollForm.cs` | App bootstrap、依賴組裝、Form 生命週期與 setting change 路由 |
| `UI/Form/AniloxRollForm.Live.cs` | Live 監控：grab 流程（`btnLiveGrab_Click`）、即時曲線（`OnLiveCurveData`/`OnLiveRowCurveData`）、Mura 判定（`CheckLiveMura`）、強化/合圖切換（`ApplyMuraEnhance`/`SwitchStitchModeWithEnhanceSequence`） |
| `UI/Form/AniloxRollForm.Review.cs` | 回顧：資料夾/時段載入（`btnReviewSelectFolder_Click`/period 導航/`ApplyReviewEnhance`/`LoadImagesWithReviewConfig`） |
| `UI/Form/AniloxRollForm.Background.cs` | 背景取得/載入/預覽 + 背景判斷（`IsBgBinReady`/`IsStandardBgSubEnabled`） |
| `UI/Form/AniloxRollForm.SettingsTabs.cs` | 右側設定面板 tab 建構（`SetupCameraTab`/`SetupSystemTab`/`Bind*Sync`）+ 相機參數硬體同步（`SyncCameraParamsFromHardware`） |
| `UI/Form/AniloxRollForm.HardwareStatus.cs` | IO、光源與儲存連線狀態的 UI 接線 |
| `UI/Form/AniloxRollForm.DirectionStitch.cs` | V/H 方向/ridge/合圖模式切換（`SwitchRidgeDirection`/`OnStitchModeChangedAsync`） |
| `UI/Form/AniloxRollForm.Data.cs` | 檢測數據 Tab（`SetupDataTab`/grabId 選擇） |
| `UI/Form/AniloxRollForm.Telemetry.cs` | Telemetry timer 與資源監控畫面更新 |
| `UI/Form/AniloxRollForm.Helpers.cs` | PG refresh（`RefreshGridItem`）/Review 座標（`ViewRangeProvider`）/通用（`FindCameraById`/`IsCanvasFitToScreen`） |
| `UI/Form/AniloxRollForm.Designer.cs` | Form 控制項佈局（VS Designer）。狀態列順序：相機→儲存→光源→IO連線→IO狀態→DIO |
| `Program.cs` | 程式進入點、全域例外處理與 trace listener 初始化 |
| `UI/Binders/BusyUiBinder.cs` | 回顧載入忙碌視覺唯一 owner：等待游標、命令按鈕鎖與 UI-thread marshal。Presenter workflow 與 stitched image loader 共用同一實例。 |
| `UI/Coordinators/ReviewFolderCoordinator.cs` | 回顧資料夾選擇、路徑修正、ImageRepository refresh、DateTimeNavigator 初始化。 |
| `UI/Coordinators/InspectionSettingsCoordinator.cs` | InspectionSettings 到 BatchInspectionService 的 pipeline 副作用唯一 owner。 |
| `UI/Coordinators/LatestGrabLoadCoordinator.cs` | 回顧／報表／預覽共用的單序號 latest-only／single-flight 排程與 stale token owner；不負責讀檔或畫畫面。 |
| `UI/Coordinators/ReviewPeriodLoadCoordinator.cs` | 回顧時段載入的 FIFO single-flight、重複 request 去重與 generation 失效 owner。 |
| `UI/Services/ImageCacheService.cs` | ProcessBatch 產出但不直接顯示的 Bitmap 生命週期唯一 owner；下一次 workflow 前統一 Dispose。 |
| `UI/Services/ReviewImageDataLoader.cs` | 回顧單片完整載入 service：查影像/CFG、幀對齊、每台拼接、欄列 curve 合併與灰階轉換；背景執行且不持有 WinForms 狀態。 |
| `UI/Services/ReviewPeriodDataLoader.cs` | 回顧時段投影 service：將某時點的每台影像轉灰階 frame、讀欄 curve、合併列 curve；不持有設定或 WinForms 狀態。 |
| `UI/Services/BitmapGrayConverter.cs` | 回顧 Bitmap 轉 8-bit 灰階 frame 的純轉換 helper。 |
| `UI/Services/SingleGrabCurveDataLoader.cs` | 回顧／報表共用的單序號 Curve application service：讀 CFG，依記憶體快取→`_curve_summary`→原始 bin 取得欄／列資料，且不持有 WinForms 狀態。 |
| `UI/State/ReviewRuntimeState.cs` | 回顧 CSV CFG 快照與螢幕 mm/px 的 runtime SSoT；Form 與 ReviewStitchCoordinator 共用。 |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/Interaction/EventGuard.cs` | 可重入事件 guard 與 scope |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/BaseCurveChartHelper.cs` | Mean/Max、閾值與 plot lifecycle 的共用曲線圖基底 |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/ColumnCurveChartHelper.cs` | 欄（X 軸）曲線圖 helper；供 Live、Review、Data 的欄圖共用 |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/RowCurveChartHelper.cs` | 列（Y 軸）曲線圖 helper；供 Live、Review 的列圖共用 |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/Input/TrackBarWheelInterceptor.cs` | TrackBar 滑鼠滾輪攔截器 |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/Input/ComboBoxWheelReverser.cs` | ComboBox 滑鼠滾輪方向轉換 |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/Interaction/MultiClickDetector.cs` | 雙擊與三擊手勢辨識 |
| `UI/Widgets/CurveMergeHelper.cs` | 曲線 bin 讀取與全覽曲線合併的 app adapter |
| `UI/Presenters/DataStatisticsPresenter.cs` | 報表統計、導航、明細清單與圖表的 feature presenter |
| `UI/Presenters/ReviewStitchCoordinator.cs` | 回顧單片／時段載入指揮：async token、busy lease、prepared layout、報表 Curve 共用與顯示事件發布；不持有圖表公式。 |
| `UI/Presenters/ReviewChartPresenter.cs` | 回顧欄／列圖表唯一套用 owner：正規值轉換、單片合圖、時序 Curve 與視野連動。 |
| `UI/State/ReviewDisplayContent.cs` | 當前回顧 Bitmap／欄列 Curve 的所有權與清理；不排程 IO、不操作控制項。 |
| `UI/Presenters/LiveTelemetryPresenter.cs` | 相機 telemetry 擷取與 UI snapshot 套用 |
| `Acquisition/AniloxCamera.cs` | 單台產品相機 composition：取像事件、檢測、顯示資料與存檔協調 |
| `sdk/MIL/MilGrabber.Core/MilCamera.cs` | 單台相機的 MIL 資源、取像、參數、CLProtocol 與 telemetry wrapper |
| `sdk/MIL/MilGrabber.Core/MultiCameraMerger.cs` | 多相機 MIL 合併 buffer、layout 與 merge target 管理 |
| `Acquisition/CameraFrameSaver.cs` | 擷取影像、曲線、資源紀錄與 frame tick sidecar 的背景持久化 |
| `Services/FlowTrace.cs` | 產品 `[Flow]` trace 的單一輸出介面 |
| `Services/FrameTickIndex.cs` | 跨相機時間槽對齊唯一決策點：硬體 tick 優先，任一 tick 缺失時整批 fallback 檔名，並回報實際模式。 |
| `UI/Widgets/WaterfallView.cs` | App 對瀑布顯示流程的相容入口 |
| `Services/CaptureArchiveStore.cs` | 每序號 `.acap` 容器、CRC、虛擬路徑、隨機記錄讀寫與舊資料轉換 owner。 |
| `Services/CapturePreviewAtlasCodec.cs` | `.acap` raw／欄／列 1080p 預覽合圖的生成、metadata 編解碼與相機切片 owner；完整 JPEG 仍是真實來源。 |
| `Services/CaptureFileNaming.cs` | 擷取影像、曲線與背景檔名規則及 legacy 解析 |
| `Services/CaptureStoragePaths.cs` | 每日 CSV 與日期影像目錄的路徑規則 |
| `Acquisition/CaptureTimestampCoordinator.cs` | 多相機存檔時間戳同步 |
| `UI/Managers/GlobalMergeCoordinator.cs` | MultiCameraMerger 的 app lifecycle coordinator |
| `UI/Managers/LiveDisplayCoordinator.cs` | 即時、瀑布、背景預覽、縮圖、選取與 LOD 的顯示 coordinator |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/WaterfallView.cs` | 可重用瀑布顯示、chunk storage 與 LOD control |
| `UI/Managers/LiveCameraManager.cs` | 多相機 acquisition facade、生命週期與參數套用協調 |
| `Settings/InspectionSettings.cs` | 根設定物件 |
| `Settings/Models/ChartSettings.cs` | 圖表 Y 軸範圍設定（ChartScaleMode + YMax）；StitchMode enum（Vertical / Global） |
| `Settings/Models/ImageViewSettings.cs` | 合圖方式設定（StitchMode） |
| `Settings/Models/MuraChartConfig.cs` | Mura 圖表閾值 PropertyGrid 展開代理 |
| `Settings/Models/CameraParamSettings.cs` | DCF 設定檔路徑 |
| `Settings/Models/LightSettings.cs` | 光源控制器設定（COM Port、Channel、Brightness） |
| `Settings/Models/Defaults/InspectionDefaults.cs` | Inspection 所有預設常數集中（CamOps/IoEnabled/AniloxRootPath/DcfPath/LightChannel…）— 預設值唯一來源 |
| `Settings/Models/Defaults/AcquisitionDefaults.cs` | Acquisition 預設值集中（GrabHeight/ExposureTimeUs/LineRateHz × 7 cam）；NewArray 工廠避免散落 |
| `Settings/Models/Defaults/AppModeDefaults.cs` | AppMode 預設值（Role/StorageMachineConfigFolder/StorageMachineDataPath） |
| `Settings/Models/Defaults/SystemDefaults.cs` | SystemSettings 預設值（7 cam 拓樸 NewCameraDevices；dcf 共用 InspectionDefaults.DcfPath） |
| `Settings/Utilities/DcfPathHelper.cs` | DcfPath 路徑解析：相對 `Config\Radient_Config.dcf` → 絕對 `BaseDir\Config\…`（進 MIL 前呼） |
| `Settings/Stores/SettingsStoreHelper.cs` | Settings Load/Save 共用 helper：JSON 檔案 I/O、regex 解析工具方法 |
| `Settings/Stores/AcquisitionSettingsStore.cs` | 讀寫 acquisition-settings.json |
| `UI/State/UserSessionState.cs` | UI session 持久化 → session-state.json |
| `ImageCatalog/ImageRepository.cs` | 掃描目錄建立索引 |
| `Services/AoiService.cs` | C# ↔ Native P/Invoke wrapper（ProcessImage + ComputeColumnMean） |
| `Services/InspectionLogService.cs` | 每日 CSV 寫入；GrabId = `yyMMdd-HHmmss` 時間戳格式 |
| `Services/InspectionCsvReader.cs` | CSV 資料列唯一 parser、檔名時間/相機解析、`FileShare.ReadWrite` 共用讀取 |
| `Services/InspectionConfigRepository.cs` | 依 grabId／日期／檔案／最新位置查詢 `#CFG` 快照 |
| `Services/InspectionImagePathRepository.cs` | 依 grabId 查詢、分組並排序實際存在的擷取影像路徑 |
| `Services/InspectionMuraProfileRepository.cs` | 讀取 MeanC/MaxC bin，執行均勻平均與 MaxCMean 候選排名聚合 |
| `Services/CurveBinFile.cs` | MCBF 曲線檔唯一讀取器；驗證 header 後 bulk read float payload |
| `Services/SingleGrabCurveSummaryStore.cs` | 報表單 grab 的版本化 `.mcsf` 匯總；驗證來源身分、一次順序讀取、原子寫回，原始 bin 仍是 SSoT |
| `UI/Services/SingleGrabCurveCache.cs` | 報表單序號 Curve 的 64 筆／64 MB LRU；共用同 key in-flight 載入並保存 rescale 前合併結果 |
| `Services/InspectionStatisticsService.cs` | Pass/Fail、序號明細、可用時段與期間統計（CSV 格式委派 `InspectionCsvReader`） |
| `Services/IoState.cs` | IoState enum（FSM 狀態）+ IoSnapshot struct（IO 快照） |
| `Services/IoGrabController.cs` | IO-Grab 連動：IoState FSM、IO 追蹤、Watchdog keepalive；支援 IModbusTcpClient 注入測試。`ReadWriteTimeoutMs`（可設）= 斷線偵測下限（拔線 OS 即報錯近 0ms；斷電靠逾時 ~500ms）；`NextReconnectAtUtc` 供 UI 重連倒數 |
| `Services/CsvConfigSnapshot.cs` | 不可變 `#CFG` 快照（OPS + START/CamPos + CROP/TrimHead/TrimTail、擷取參數、欄列正規值、RidgeSigma、門檻） |
| `Services/HessianRescaleHelper.cs` | View-time HM rescale 共用：Ratio / IsNoOp / RescaleInPlace1D\|2D / CloneAndRescale1D\|2D — 5 個公式單一來源 |
| `sdk/Bridges/StorageBridge/StorageBridge.Core/StorageRetentionService.cs` | 循環儲存：事件驅動（grab 結束/每 10 grab/watchdog），磁碟可用空間低於門檻時刪最舊日期資料夾影像，保留 CSV 與仍待遠端發布的日期資料夾 |
| `Services/CleanupFlagWatcher.cs` | Storage PC 專用：每 10 秒自主查空間 + 清理；同時輪詢 cleanup-request.flag（Inspection PC 寫入）立即觸發 |
| `Settings/Models/AppModeConfig.cs` | 機台角色設定：Role（Inspection/Storage）、StorageMachineConfigFolder、StorageMachineDataPath；Load/Save → Config\app-mode.json |
| `sdk/Bridges/StorageBridge/StorageBridge.Core/RemoteCopyService.cs` | 背景遠端複製：持久 pending 佇列、斷線退避重試、重開復原、`.part-*` 長度驗證後原子發布、分享路徑可寫探針 |
| `Services/LightController.cs` | LTS-3DPA24 光源控制器 RS-232 通訊：AutoDetect（先試設定 COM 再掃描）、嚴格 probe（PDF §4.1.4 表-4 驗證：8-byte、cmd/ch echo、XOR checksum）、TurnOn/Off/SetBrightness，跟隨 IO Grab 開關 |
| `UI/Widgets/GrabImageStitcher.cs` | 多張影像垂直拼接 + MergeHorizontal 全域合圖（佈局 xOffset + 重疊中點分界委派 sdk `TanukiCv.Controls.MergeLayout.Compute(Midline)` 單一來源，totalW 保留自家 ALL-slots 含空缺占位版）；LoadCameraImage（internal） |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/Layout/ProportionalScaler.cs` | WinForms 控制項樹的等比例 layout scaler |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/Layout/RoundedLabel.cs` | 可重用圓角狀態 Label |
| `sdk/TanukiCv/dotnet/TanukiCv.Controls/UI/ImageCanvas.cs` | 可重用影像畫布：zoom、pan、overlay、LOD 與多擊手勢 |

> 路徑前綴 `src/dotnet/AniloxRoll.Monitor/` 省略以節省空間。

### 測試專案

| 路徑 | 職責 |
|------|------|
| `sdk/Bridges/IoBridge/IoBridge.Core/IModbusTcpClient.cs` | Modbus TCP 介面（供 IoGrabController mock 注入） |
| `sdk/Bridges/IoBridge/IoBridge.Core/IoModuleFactory.cs` | 型號→client 單一決策點：`Create(model)`；新增型號加 case。`Modules/` 按廠商分實作 |
| `sdk/Bridges/IoBridge/IoBridge.Core/Modules/IcpDasModbusTcpClient.cs` | ICP DAS 標準 Modbus（ET 系列通用）；ET-7044 實作 |
| `tests/AniloxRoll.Monitor.Tests/` | NUnit 3.x + Moq 4.x 測試專案 |
| `CsvConfigSnapshotTests.cs` | #CFG round-trip、ContentKey |
| `AcquisitionSettingsTests.cs` | Validate fallback、JSON Save/Load |
| `InspectionLogServiceTests.cs` | CSV 寫入、#CFG 插入、Pass/Fail 判定 |
| `InspectionStatisticsServiceTests.cs` | 時間/序號統計、veto 邏輯、Period 分組 |
| `InspectionCsvReaderTests.cs` | 4/10 欄 CSV、CFG、檔名時間與相機解析 |
| `InspectionConfigRepositoryTests.cs` | 同檔最後 CFG、grabId 最近 CFG、跨日最新 CFG |
| `InspectionImagePathRepositoryTests.cs` | 影像格式 fallback、去重排序、日期提示縮限 |
| `InspectionMuraProfileRepositoryTests.cs` | 多序號 MeanC 平均與 MaxC 逐點最大聚合 |
| `SingleGrabCurveCacheTests.cs` | 同 key single-flight、LRU 淘汰、重載世代隔離與 raw Curve 複製隔離 |
| `SingleGrabCurveSummaryStoreTests.cs` | `.mcsf` round-trip、來源時間失效、損壞退回與原子覆寫 |
| `IoGrabControllerTests.cs` | FSM 狀態機：連線、邊緣偵測、故障恢復、CommLost |
| `StressTests.cs` | 長時間壓力：PLC 100 萬循環、CSV 50 萬筆、Settings 14.5 萬讀寫；STRESS_MINUTES 環境變數控制時長 |

---

## 檢測參數速查（PropertyGrid 屬性）

使用者在【檢測設定】看到的參數。溝通格式：「屬性名-值」（例如「欄正規值-0.2」「存檔-T」）。

**範圍**：此表只列「**what** — 參數是什麼 / 預設值 / 屬性名映射」。
**互動行為與 chart 聯動**＝以 code 為真相，配合對應 skill（`$modify-ui` 等）查閱；不在表格內重複描述。

### 0. 機台設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| 機台角色 | `AppRole` | Inspection | Inspection / Storage；變更後寫 app-mode.json，重開程式生效 |

### 1. 機台佈局

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| ── OPS (um) ── | （分隔列，唯讀） | — | — |
| Cam 1~7 | `ab_OpsCam1~ah_OpsCam7` | 24.4140625 | 各相機像素尺寸 |
| A輪速度 (m/min) | `ai_OpsSpeed` → `AniloxRollSpeedMPerMin` | 40.0 | Anilox 輪速 |
| ── Start (mm) ── | （分隔列，唯讀） | — | — |
| Cam 1~7 | `bb_StartCam1~bh_StartCam7` | 0/345/690/1035/1380/1725/2070 | 各相機起始位置 |
| ── Crop (mm) ── | （分隔列，唯讀） | — | — |
| 去頭 | `cb_CropHead` → `Crop.TrimHeadMm` | 0.0 | CAM1 左側裁切 |
| 去尾 | `cc_CropTail` → `Crop.TrimTailMm` | 0.0 | CAM7 右側裁切 |

### 2. 檢測設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| ── 演算法 ── | （分隔列，唯讀） | — | — |
| 去背演算法 | `db_Algorithm` → `Algorithm` | SingleFrameBgSub | SingleFrameBgSub / StandardBgSub |
| 背景採樣(秒) | `fb_BackgroundSampleSeconds` → `BackgroundSampleSeconds` | 3 | StandardBgSub 背景採樣時間 |
| ── 檢出標準 ── | （分隔列，唯讀） | — | — |
| 檢出方向 | `eb_RidgeDir` → `RidgeDir` | Both | 欄 / 列 / 全部 |
| 欄正規值 | `dc_HessianMaxFactorV` → `HessianMaxFactorV` | 0.3 | V Hessian 正規化係數（capture-time baked-in） |
| 列正規值 | `dd_HessianMaxFactorH` → `HessianMaxFactorH` | 0.3 | H Hessian 正規化係數（view-time only） |
| 細線濾除 | `de_RidgeSigma` → `RidgeSigma` | 9.0 | Ridge 前 Gaussian blur sigma；越大→濾掉越多細線/雜訊（較不敏感），越小→越敏感。走每幀 json 送 native；改設定下次 grab 生效。唯一預設 `InspectionEngineConfig.DefaultRidgeSigma` |
| 欄平均閾值 | `ec_ErrorValueMeanV` → `ErrorValueMeanV` | 0.2 | V chart Mean 閾值線 |
| 欄最大閾值 | `ed_ErrorValueMaxV` → `ErrorValueMaxV` | 0.6 | V chart Max 閾值線 |
| 列平均閾值 | `ee_ErrorValueMeanH` → `ErrorValueMeanH` | 0.2 | H chart Mean 閾值線 |
| 列最大閾值 | `ef_ErrorValueMaxH` → `ErrorValueMaxH` | 0.6 | H chart Max 閾值線 |
| ── 畫布設定 ── | （分隔列，唯讀） | — | — |
| 停止條件 | `fb_CaptureStopCondition` → `CaptureStopCondition` | IO | IO / 時間 / 高度；三者都由 IO High 啟動，選擇本輪由 Low、總時間或共同完成列數停止 |
| 總時間(秒) | `fc_GrabLimitSeconds` → `GrabLimitSeconds` | 10 | 時間模式的停止值；IO 模式仍作防止無限抓取的安全上限 |
| 總高度 | `hg_WaterfallTotalHeight` → `ImageView.WaterfallTotalHeight` | 30000 | 瀑布虛擬長圖總高度；高度模式同時以各在線相機共同完成列數達此值為停止條件 |

### 3. 圖表設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| ── 檢測報表 ── | （分隔列，唯讀） | — | — |
| y座標 | `gb_ChartScaleMode` → `ChartScaleMode` | Auto | Auto / Fixed |
| 月產量 | `gc_YearlyYMax` → `ChartDataYieldYearlyYMax` | 50000 | 良率年圖 Y 軸上限 |
| 日產量 | `gd_MonthlyYMax` → `ChartDataYieldMonthlyYMax` | 2000 | 良率月圖 Y 軸上限 |
| 時產量 | `ge_DailyYMax` → `ChartDataYieldDailyYMax` | 300 | 良率日圖 Y 軸上限 |
| ── 主畫面 ── | （分隔列，唯讀） | — | — |
| 監控強化 | `hc_EnableMuraEnhance` → `EnableMuraEnhance` | false | 即時影像強化 Mura |
| 回顧強化 | `hd_EnableReviewEnhance` → `EnableReviewEnhance` | false | 回顧影像強化 Mura |
| 強化熱力圖 | `hda_EnhanceHeatmap` → `EnhanceHeatmap` | Off | 關閉 / 冷色 / 暖色 / 藍黃紅；只影響主畫面強化顯示 |
| 主畫面顯示 | `he_MainDisplay` → `ImageView.MainDisplay` | Waterfall | ImageCanvas（即時合圖）/ Waterfall（瀑布合圖） |
| 上下方向 | `hee_VerticalDirection` → `ImageView.VerticalDirection` | BottomToTop | 由下而上 / 由上而下；監控與回顧共用 |
| 動態LOD | `hf_LiveLod` → `ImageView.LiveLod` | CPU | Off / GPU（TanukiCv GPU 縮）/ CPU（GrayResizeCpu 純 CPU 縮）。ImageCanvas 模式放大巨圖看細節用（顯示成本 ~180ms→~1ms），即時生效。預設 CPU＝無 GPU 機也能跑。`LiveCameraManager.SetLodMode` 套到 `ImageDisplayView.EnableLod`/`DisableLod` |
| 瀑布滿了 | `hh_WaterfallFullMode` → `ImageView.WaterfallFullMode` | Restart | 重來 / 循環 |

### 4. 儲存設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| ── 本機設定 ── | （分隔列，唯讀） | — | — |
| Anilox 根目錄 | `AniloxRootPath` | D:\Anilox | 資料根目錄；磁碟不存在時自動 fallback 到 C:\Anilox + MessageBox + 寫回 settings |
| 預留空間 (GB) | `LocalMinFreeGB` | 100 | 磁碟可用空間低於此值時刪除最舊完整一天的全部產出（含月份 CSV）；輸入超過磁碟容量時自動調整 |
| 存檔 | `EnableAutoCapture` | true | 取像時自動存檔 |
| 存原圖 | `SaveOriginalBmp` | false | 額外存原始 BMP |
| ── 遠端設定 ── | （分隔列，唯讀） | — | — |
| 遠端路徑 | `RemotePath` | \\192.168.10.20\Anilox\Captures | 遠端複製目標路徑（空=不複製）。單一 SMB share `Anilox` 子目錄 |
| 存檔目錄 | （computed）| `{AniloxRoot}\Captures` | 每序號 `.acap` + 統計 CSV；不顯示於 PropertyGrid |
| 存背景目錄 | （computed）| `{AniloxRoot}\Bg` | StandardBgSub 背景影像；不顯示 |
| Logs 目錄 | （computed）| `{AniloxRoot}\Logs` | Resource Log；不顯示 |
| Dcf 檔 | （跟 exe 走）| `{ExeDir}\Config\Radient_Config.dcf` | MIL DCF；repo 唯一來源 `sdk/MIL/Config/Radient_Config.dcf`，app/sample build 連結複製，PG 隱藏 |
| 遠端設定路徑 | `RemoteConfigPath` | \\192.168.10.20\Anilox\Config | [Browsable(false)] 開發者設定；cleanup-request.flag 寫入位置 |

### 5. LOG 設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| 記錄範圍 | `LogMode` | 日常運行 | 日常／流程驗證／完整診斷三級；用途直接顯示在下拉名稱中 |
| 保留時間 (小時) | `LogRetentionHours` | 168 | 只清理 Log catalog 內的診斷檔；目前 process 的 log 與未知檔案保留 |

### 6. 光源設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| COM Port | `LightComPort` | COM17 | RS-232 連接埠；啟動時先試此 port，失敗則自動掃描所有 port（找到後更新此欄位） |
| 通道 | `LightChannel` | 1 | 使用通道（單通道機型固定 1） |
| 亮度 | `LightBrightness` | 255 | 亮度（0~255） |
| 啟用光源 | `LightEnabled` | true | 啟用 LTS-3DPA24 光源控制器 |

### 7. IO 設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| IO IP | `IoIp` | 192.168.255.1 | ET-7044 IP |
| IO Port | `IoPort` | 502 | Modbus TCP port |
| 啟用 IO | `IoEnabled` | true | 啟用 IO Modbus TCP |
| 暫停檢出 | `MuraDetectPaused` | false | 是=暫停 Mura 檢出與 DO1；每次啟動恢復為否，不寫入 JSON |
| IO 型號（硬體資訊表） | `IoModel` | ET-7044 | PropertyGrid 隱藏；唯讀顯示於 `listViewHardware`，仍由 JSON 供 `IoModuleFactory` 選型 |

---

## 控制項速查（標準名稱 → 程式名稱）

每個控制項有一個**標準名稱**（中文），用於對話溝通。

### 監控（tabPageLiveView）

| 標準名稱 | Name | 類型 | 畫面文字 |
|---------|------|------|---------|
| 開始抓取 | `btnLiveGrab` | Button | 開始抓取 / 停止抓取 |
| 取得背景 | `btnLiveGetBackground` | Button | 取得背景 |
| 預覽背景 | `btnLiveViewBackground` | Button | 預覽背景 |
| 監控主畫面 | `camLiveMain` | Panel | — |
| 監控縮圖1~7 | `camLive1~7` | Panel | — |
| 監控欄曲線圖（全覽） | `chartLiveColumn` | Chart | — |
| 監控列曲線圖 | `chartLiveRow` | Chart | — |
| 暫停 Mura 檢測 | `lblIoDoMura`（點擊切換） | Label | DO1 MURA_DET / DO1 MURA ⏸（黃底=暫停中） |

### 回顧（tabPageReview）

| 標準名稱 | Name | 類型 | 畫面文字 |
|---------|------|------|---------|
| 讀取資料 | `btnReviewSelectFolder`（Review）/ `btnDataSelectFolder`（Data） | Button | 讀取資料 |
| 回顧縮圖1~7 | `camReview1~7` | Panel | — |
| 回顧主畫面 | `camReviewMain` | Panel | — |
| 回顧欄曲線圖（全覽） | `chartReviewColumn` | Chart | — |
| 回顧列曲線圖 | `chartReviewRow` | Chart | — |
| 時段群組 | `grpReviewTimePeriod` | GroupBox | 時序 |
| 時段日期（時序cb） | `cbReviewDate` | ComboBox | — |
| 時段時間（時序cb） | `cbReviewTime` | ComboBox | — |
| 單片群組 | `grpReviewGrabNav` | GroupBox | 單片 |
| 單片序號（序號cb） | `cbReviewId` | ComboBox | — |

### 報表（tabPageData）

| 標準名稱 | Name | 類型 | 畫面文字 |
|---------|------|------|---------|
| 篩選異常 | `btnDataShowFail` | Button | 篩選異常 / 顯示全部 |
| 良率卡片1~7 | `camData1~7` | Panel | — |
| Mura 空間分布圖 | `chartDataColumn` | Chart | — |
| 明細列表 | `listViewGrabDetail` | ListView | — |
| 序號範圍群組 | `groupBoxGrabIdRange` | GroupBox | 序號範圍 |
| 起始序號 | `cbDataIdStart` | ComboBox | — |
| 結束序號 | `cbDataIdEnd` | ComboBox | — |
| 序號選擇群組 | `grpDataSingleSheet` | GroupBox | 序號選擇 |
| 報表序號 | `cbDataId` | ComboBox | — |
| 良率年圖 | `chartDataYieldYearly` | Chart | — |
| 良率月圖 | `chartDataYieldMonthly` | Chart | — |
| 良率日圖 | `chartDataYieldDaily` | Chart | — |
| 年圖導航 | `cbDataYieldYear` | ComboBox | — |
| 月圖導航 | `cbDataYieldMonth` | ComboBox | — |
| 日圖導航 | `cbDataYieldDay` | ComboBox | — |

### 設定面板

| 標準名稱 | Name | 類型 | 畫面文字 |
|---------|------|------|---------|
| 檢測設定 | `propertyGridSettings` | PropertyGrid | — |
| 說明文字 | `helpRichText` | RichTextBox | — |
| 曝光滑桿1~7 | `trackBarExpCam1~7` + `numExpCam1~7` | TrackBar+NUD | — |
| 線掃滑桿1~7 | `trackBarLrCam1~7` + `numLrCam1~7` | TrackBar+NUD | — |
| 高度滑桿1~7 | `trackBarHtCam1~7` + `numHtCam1~7` | TrackBar+NUD | — |
| Telemetry 列表 | `listViewCameras` | ListView | — |
| 引擎常數列表 | `listViewEngine` | ListView | — |
| 圖表常數列表 | `listViewChartConst` | ListView | — |
| 硬體參數列表 | `listViewHardware` | ListView | — |
| 電腦容量狀態列 | `lblInfo` | Label | 檢測電腦：剩餘 N / Total GB｜儲存電腦：剩餘 N / Total GB（座標與亮度由 ImageCanvas 顯示在滑鼠旁） |
| 相機數狀態 | `lblCamCount` | Label | 相機: N/7 |
| IO 狀態 | `lblIoState` | RoundedLabel | -- → 待機/取像/停止/故障/斷線/關閉/未連線（FSM 運作狀態，非連線燈；已移入 `panelIo` IO 運作區當開頭，與 DIO 燈號同組，皆圓角晶片） |
| IO 連線狀態 | `lblIoConn` | Label | ● IO: -- |
| 光源連線狀態 | `lblLightConn` | Label | ● 光源: -- |
| 儲存電腦連線狀態 | `lblStorageConn` | Label | TCP 445 + 分享可寫 + 遠端 app heartbeat 新鮮才顯示已連線；分享通但 app 未回報顯示黃燈 |
| IO 燈號 | `lblIoDiAlive~lblIoDoPcBusy` | Label×5 | DI0~DO2 |

---

### 參考文件（僅供查閱，不自動載入）

| 文件 | 用途 |
|------|------|
| [`docs/user-manual/io_diagrams.html`](../../../../docs/user-manual/io_diagrams.html) | IO FSM 視覺化（ET-7044 ↔ 設備 Nakan）|
| [`docs/user-manual/storage-flow.html`](../../../../docs/user-manual/storage-flow.html) | 逐步退場中的 Storage 操作/部署視覺說明；工程契約以 verify-flows 與 output-storage-map 為準 |
| [`docs/user-manual/hardware-specs.html`](../../../../docs/user-manual/hardware-specs.html) | 7 相機 + Grabber + 光源 + PLC 硬體規格 |
| [`modify-acquisition/references/mil-api-reference.md`](../../modify-acquisition/references/mil-api-reference.md) | MIL .NET API 完整參考 |
| [`runtime-resources.md`](runtime-resources.md) | 現行資源儀器與量測方法 |
| [`add-test/references/stress-and-soak.md`](../../add-test/references/stress-and-soak.md) | 壓測、soak 與失效注入方法 |
| `.agents/skills/*` | 可重複工程流程、架構 reference 與 DVT 契約 |

### docs/ 目錄定位

```text
docs/
├── dev/            非 Markdown 的 vendor 對照表與互動 checklist
└── user-manual/    操作員 HTML（hardware / IO / storage）
```

Agent 需要的 Markdown 一律歸 owning skill；完成的 migration/review 紀錄不保留為規則。操作員或 vendor artifact 留 `docs/`。
