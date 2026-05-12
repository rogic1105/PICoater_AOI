# PICoater AOI — Claude Code Rules

## 專案結構

```
PICoater_AOI/
├── src_dotnet/AniloxRoll.Monitor/   ← C# WinForms 應用程式
├── src_dotnet/PlcBridge/            ← PLC Modbus TCP 通訊模組
│   ├── PlcBridge.Core/              ← 共用 Modbus TCP Client + Logger（IModbusTcpClient 介面）
│   ├── PlcBridge.ManualControl/     ← 手動 DI/DO 控制工具
│   └── PlcBridge.Automation/        ← FSM 狀態機自動控制工具
├── tests/dotnet_test/AniloxRoll.Monitor.Tests/ ← NUnit 單元 + 壓力測試
├── tests/python_test/               ← Python 測試/工具腳本
├── TestRunner/                      ← 測試啟動器（雙擊 TestRunner.bat）
├── deploy/                          ← 現場部署腳本（PowerShell + JSON 參數）
│   ├── storage-pc/                  ← 儲存機：固定 IP + SMB 共用 + 防火牆 + Guest 匿名（secedit）
│   └── inspection-pc/               ← 檢測機：單 NIC 雙 IP 別名 + Client 端匿名 Guest SMB
├── src_native/                      ← C++ pipeline 實作
└── sdk/AOI_SDK/                     ← 共用 SDK (core_cv_api / AOI.SDK)
```

## Native API

兩組 DLL，均宣告於 `src_dotnet/AniloxRoll.Monitor/Interop/NativeMethods.cs`：

| DLL | 函式 | 用途 |
|-----|------|------|
| `picoater_api.dll` | `PICoaterAPI_CreatePipeline` / `ProcessPipeline` / `DestroyPipeline` / `ComputeColumnMean` | GPU 檢測 pipeline |
| `core_cv_api.dll` | `CoreCV_AllocPinned` / `CoreCV_FreePinned` | CUDA pinned memory 管理 |
| `core_cv_api.dll` | `CoreCV_FastReadBMP` | 快速讀取 BMP（繞過 GDI+） |
| `core_cv_api.dll` | `CoreCV_Resize_GPU` | GPU 縮圖 |

## P/Invoke 架構規則

**所有 P/Invoke 宣告只能在 `AniloxRoll.Monitor/Interop/NativeMethods.cs`**，不得跨層使用 SDK 的 `AOI.SDK.Core.CoreCVWrapper`。

## 關鍵檔案速查

| 路徑 | 職責 |
|------|------|
| `Interop/NativeMethods.cs` | 唯一 P/Invoke 宣告點 |
| `ImageProcessing/NativeBufferPool.cs` | CUDA pinned buffer 管理 |
| `ImageProcessing/InspectionEngine.ImageProcessing.cs` | 縮圖/全解析度影像處理 |
| `ImageProcessing/InspectionEngineConfig.cs` | MaxWidth=16384, MaxHeight=10000, DefaultSaveResizeScale=5 |
| `ImageProcessing/BatchInspectionService.cs` | Parallel.For 批次縮圖 |
| `UI/Form/AniloxRollForm.cs` | Form 邏輯：事件、InitializeSystem；內含 helpers: `BindBidirectionalSync`、`SetChartYRange`、`FindCameraById`、`CheckLiveMura`（Live 即時閾值→DO_MURA_DETECTED） |
| `UI/Form/AniloxRollForm.Designer.cs` | Form 控制項佈局（VS Designer） |
| `UI/Widgets/FormInteractionHelper.cs` | UI 互動、gallery 選擇、計時；ReviewConfig 代理 |
| `UI/Widgets/CanvasInteractionHelper.cs` | Canvas zoom/pan 事件、mm 座標換算；ReviewConfig → GetEffectiveOps/Pos |
| `UI/Widgets/EventGuard.cs` | 可重入 bool 旗標（EventGuard + EventGuardScope），using 語法自動還原 |
| `UI/Widgets/BaseCurveChartHelper.cs` | 曲線圖抽象基底：共用欄位、Build 骨架、閾值線、PostPaint 事件 |
| `UI/Widgets/ColumnCurveChartHelper.cs` | 切向（X 軸）mura 曲線圖：繼承 BaseCurveChartHelper，含 zoom 同步 |
| `UI/Widgets/RowCurveChartHelper.cs` | 法向（Y 軸）mura 曲線圖：繼承 BaseCurveChartHelper，座標反轉、InnerPlot 補償 |
| `UI/Widgets/TrackBarWheelInterceptor.cs` | TrackBar 滑鼠滾輪攔截器（從 AniloxRollForm 提取） |
| `UI/Widgets/ComboBoxWheelReverser.cs` | ComboBox 滑鼠滾輪方向反轉（從 AniloxRollForm 提取） |
| `UI/Widgets/MultiClickDetector.cs` | 多擊偵測器：雙擊/三擊辨識（從 AniloxRollForm 提取） |
| `UI/Widgets/CurveMergeHelper.cs` | 全覽圖合併演算法 + .bin 曲線讀取（UpdateOverviewChart、MergeCurves、MergeRowCurves、GetCurveBasePath） |
| `UI/Presenters/DataStatisticsPresenter.cs` | Data tab 統計邏輯：統計計算、combo 串聯、Period Charts、Mura 空間分布圖（chartMuraProfile）、跨 Tab 同步事件 |
| `UI/Presenters/ReviewStitchCoordinator.cs` | Review tab 拼接管理：LoadGrabStitchedViewAsync、合圖、ClearStitchedMode、overview chart 聯動 |
| `UI/Presenters/LiveTelemetryPresenter.cs` | 16 欄即時 Telemetry |
| `Acquisition/AniloxCamera.cs` | 單台相機 MIL 資源封裝；Global merge 時每幀 MbufChild2d + MbufCopyClip（含 overlap 裁切）到合併 buffer；SetMergeTarget/ClearMergeTarget 封裝 merge 欄位 |
| `Acquisition/CameraFrameSaver.cs` | 存檔 I/O：SaveCapture（背景執行緒）、SaveJpegFromBytes、SaveCurveBinFromArray、Resource Log（CSV: CPU%/RAM/VRAM/GPU ms/Live/Review/StitchMode） |
| `Acquisition/CaptureTimestampCoordinator.cs` | 多相機存檔時間戳同步 |
| `UI/Managers/LiveCameraManager.cs` | 多台相機生命週期管理、連線數監控（OnCameraCountChanged）、即時全域合圖（EnableGlobalMerge / DisableGlobalMerge / RefreshGlobalMergeLayout）、合圖 zoom/pan（WheelZoomFilter）、合圖滑鼠座標（MergedMouseStatusHandler）、合圖 overview 聯動（TryGetMergedViewRange）、顯示同步 Timer（_mergedDisplayTimer） |
| `Settings/InspectionSettings.cs` | 根設定物件 |
| `Settings/Models/ChartSettings.cs` | 圖表 Y 軸範圍設定（ChartScaleMode + YMax）；StitchMode enum（Vertical / Global） |
| `Settings/Models/ImageViewSettings.cs` | 合圖方式設定（StitchMode） |
| `Settings/Models/MuraChartConfig.cs` | Mura 圖表閾值 PropertyGrid 展開代理 |
| `Settings/Models/CameraParamSettings.cs` | DCF 設定檔路徑 |
| `Settings/Models/LightSettings.cs` | 光源控制器設定（COM Port、Channel、Brightness） |
| `Settings/Stores/SettingsStoreHelper.cs` | Settings Load/Save 共用 helper：JSON 檔案 I/O、regex 解析工具方法 |
| `Settings/Stores/AcquisitionSettingsStore.cs` | 讀寫 acquisition-settings.json |
| `UI/State/UserSessionState.cs` | UI session 持久化 → session-state.json |
| `ImageCatalog/ImageRepository.cs` | 掃描目錄建立索引 |
| `Services/AoiService.cs` | C# ↔ Native P/Invoke wrapper（ProcessImage + ComputeColumnMean） |
| `Services/InspectionLogService.cs` | 每日 CSV 寫入；GrabId = `yyMMdd-HHmmss` 時間戳格式 |
| `Services/InspectionStatisticsService.cs` | CSV 統計服務；LoadConfigForDate（按日期載入 #CFG）；LoadAvgMuraProfile（抽樣 .bin 平均空間曲線） |
| `Services/IoState.cs` | IoState enum（FSM 狀態）+ IoSnapshot struct（IO 快照） |
| `Services/IoGrabController.cs` | IO-Grab 連動：IoState FSM、IO 追蹤、Watchdog keepalive；支援 IModbusTcpClient 注入測試 |
| `Services/CsvConfigSnapshot.cs` | 不可變設定快照（CamOps/CamPos/CamGrabHeight/CamExposureUs/CamLineRateHz/Hessian/ErrorValue/TrimHead/TrimTail） |
| `Services/StorageRetentionService.cs` | 循環儲存：事件驅動（grab 結束/每 10 grab/watchdog），磁碟可用空間低於門檻時刪最舊日期資料夾影像，保留 CSV |
| `Services/CleanupFlagWatcher.cs` | Storage PC 專用：每 10 秒自主查空間 + 清理；同時輪詢 cleanup-request.flag（Inspection PC 寫入）立即觸發 |
| `Settings/Models/AppModeConfig.cs` | 機台角色設定：Role（Inspection/Storage）、LocalConfigFolder、StorageFolderPath；Load/Save → Config\app-mode.json |
| `Services/RemoteCopyService.cs` | 背景遠端複製：ConcurrentQueue + 背景執行緒，File.Copy 含重試（3 次） |
| `Services/LightController.cs` | LTS-3DPA24 光源控制器 RS-232 通訊：AutoDetect（先試設定 COM 再掃描）、嚴格 probe（PDF §4.1.4 表-4 驗證：8-byte、cmd/ch echo、XOR checksum）、TurnOn/Off/SetBrightness，跟隨 IO Grab 開關 |
| `UI/Widgets/GrabImageStitcher.cs` | 多張影像垂直拼接 + MergeHorizontal 全域合圖；LoadCameraImage（internal） |
| `UI/Widgets/ProportionalScaler.cs` | Form 等比例縮放 |
| `sdk/AOI_SDK/src_dotnet/AOI.SDK/UI/SmartCanvas.cs` | PictureBox 子類：zoom/pan/edge/ClampPan；自訂白底黑邊十字游標 |

> 路徑前綴 `src_dotnet/AniloxRoll.Monitor/` 省略以節省空間。

### 測試專案

| 路徑 | 職責 |
|------|------|
| `src_dotnet/PlcBridge/PlcBridge.Core/IModbusTcpClient.cs` | Modbus TCP 介面（供 PlcGrabController mock 注入） |
| `tests/dotnet_test/AniloxRoll.Monitor.Tests/` | NUnit 3.x + Moq 4.x 測試專案 |
| `CsvConfigSnapshotTests.cs` | #CFG round-trip、ContentKey |
| `AcquisitionSettingsTests.cs` | Validate fallback、JSON Save/Load |
| `InspectionLogServiceTests.cs` | CSV 寫入、#CFG 插入、Pass/Fail 判定 |
| `InspectionStatisticsServiceTests.cs` | 時間/序號統計、veto 邏輯、Period 分組 |
| `IoGrabControllerTests.cs` | FSM 狀態機：連線、邊緣偵測、故障恢復、CommLost |
| `StressTests.cs` | 長時間壓力：PLC 100 萬循環、CSV 50 萬筆、Settings 14.5 萬讀寫；STRESS_MINUTES 環境變數控制時長 |

---

## Harness 架構

三方同步機制確保控制項命名在所有層級一致：

| 層 | 檔案 | 內容 |
|----|------|------|
| Form | `AniloxRollForm.Designer.cs` | `.Text` 畫面文字 |
| 速查表 | `CLAUDE.md` §控制項速查 | 標準名稱 + Name |
| 流程圖 | `docs/user-manual/ui-flow.html` | 【】包裹的控制項名稱 |

**規則：**
1. 改名時三層同時改，不可只改一層
2. 【】= 可操作控制項（使用者會點/選/拖的），被動顯示的不加
3. 同功能控制項共用標準名稱（如 Review/Data 的【讀取資料】）
4. Commit 前跑 `/update-docs` — 驗證速查表的 Name 全部存在於 Designer.cs

---

## 檢測參數速查（PropertyGrid 屬性）

使用者在【檢測設定】看到的參數。溝通格式：「屬性名-值」（例如「正規值-0.2」「存檔-T」）。

### 0. 機台設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| 機台角色 | `AppRole` | Inspection | Inspection / Storage；變更後寫 app-mode.json，重開程式生效 |

### 1. 機台佈局

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| ── OPS (um) ── | （分隔列，唯讀） | — | — |
| Cam 1~7 | `ab_OpsCam1~ah_OpsCam7` | 33.0 | 各相機像素尺寸 |
| A輪速度 (m/min) | `ai_OpsSpeed` → `AniloxRollSpeedMPerMin` | 40.0 | Anilox 輪速 |
| ── Start (mm) ── | （分隔列，唯讀） | — | — |
| Cam 1~7 | `bb_StartCam1~bh_StartCam7` | 0/400/800/1200/1600/2000/2400 | 各相機起始位置 |
| ── Crop (mm) ── | （分隔列，唯讀） | — | — |
| 去頭 | `cb_CropHead` → `Crop.TrimHeadMm` | 0.0 | CAM1 左側裁切 |
| 去尾 | `cc_CropTail` → `Crop.TrimTailMm` | 0.0 | CAM7 右側裁切 |

### 2. 檢測設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| ── 演算法 ── | （分隔列，唯讀） | — | — |
| 去背演算法 | `db_Algorithm` → `Algorithm` | SingleFrameBgSub | None / SingleFrameBgSub / StandardBgSub |
| 正規值 | `dc_HessianMaxFactor` → `HessianMaxFactor` | 0.3 | Hessian 正規化係數 |
| ── 檢出標準 ── | （分隔列，唯讀） | — | — |
| 檢出方向 | `eb_RidgeDir` → `RidgeDir` | Both | 垂直 / 水平 / 全部；決定哪個方向 Mura 超標才觸發 DO1 |
| 平均閾值 | `ec_ErrorValueMean` → `ErrorValueMean` | 0.2 | 曲線圖 Mean 閾值線 |
| 最大閾值 | `ed_ErrorValueMax` → `ErrorValueMax` | 0.4 | 曲線圖 Max 閾值線 |
| ── 背景校正 ── | （分隔列，唯讀） | — | — |
| 取時間 (sec) | `fb_BackgroundSampleSeconds` → `BackgroundSampleSeconds` | 3 | StandardBgSub 採集時間 |

### 3. 圖表設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| ── 檢測報表 ── | （分隔列，唯讀） | — | — |
| y座標 | `gb_ChartScaleMode` → `ChartScaleMode` | Auto | Auto / Fixed |
| 月產量 | `gc_YearlyYMax` → `ChartYearlyYMax` | 50000 | 良率年圖 Y 軸上限 |
| 日產量 | `gd_MonthlyYMax` → `ChartMonthlyYMax` | 2000 | 良率月圖 Y 軸上限 |
| 時產量 | `ge_DailyYMax` → `ChartDailyYMax` | 300 | 良率日圖 Y 軸上限 |
| ── 主畫面 ── | （分隔列，唯讀） | — | — |
| 合圖方式 | `hb_StitchMode` → `StitchMode` | Global | Vertical / Global |
| 監控強化 | `hc_EnableMuraEnhance` → `EnableMuraEnhance` | false | 即時影像強化 Mura |
| 回顧強化 | `hd_EnableReviewEnhance` → `EnableReviewEnhance` | false | 回顧影像強化 Mura |

### 4. 儲存設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| 存檔 | `EnableAutoCapture` | true | 取像時自動存檔 |
| 存原圖 | `SaveOriginalBmp` | false | 額外存原始 BMP |
| 存檔目錄 | `CaptureRootPath` | D:\AniloxCaptures | 存檔根目錄 |
| 存背景目錄 | （自動推算） | `CaptureRootPath\bg` | 只讀，不顯示於 PropertyGrid |
| 預留空間 (GB) | `LocalMinFreeGB` | 100 | 磁碟可用空間低於此值觸發循環儲存，刪最舊日期影像（CSV 保留） |
| 遠端路徑 | `RemotePath` | \\192.168.10.20\AniloxStorage | 遠端複製目標路徑（空=不複製） |
| 遠端設定路徑 | `RemoteConfigPath` | （由 RemotePath 自動推算） | [Browsable(false)] 開發者設定，改 JSON 生效；空值時自動從 RemotePath 的伺服器 IP 推算 |

### 5. 相機設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| 設定檔 | `DcfPath` | D:\AniloxCaptures\dcf\Radient_Config.dcf | MIL Digitizer DCF 檔案路徑 |

### 6. 光源設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| 啟用光源 | `LightEnabled` | true | 啟用 LTS-3DPA24 光源控制器 |
| COM Port | `LightComPort` | COM17 | RS-232 連接埠；啟動時先試此 port，失敗則自動掃描所有 port（找到後更新此欄位） |
| 通道 | `LightChannel` | 1 | 使用通道（單通道機型固定 1） |
| 亮度 | `LightBrightness` | 255 | 亮度（0~255） |
| 暖機延遲 (ms) | `LightWarmupMs` | 300 | 開燈後等待光源穩定的延遲；Grab 啟動前插入此延遲 |

### 7. IO 設定

| 顯示名稱 | 屬性 | 預設值 | 說明 |
|---------|------|--------|------|
| 啟用 IO | `PlcEnabled` | true | 啟用 PLC Modbus TCP |
| IO IP | `PlcIp` | 192.168.255.1 | ET-7044 IP |
| IO Port | `PlcPort` | 502 | Modbus TCP port |

---

## 控制項速查（標準名稱 → 程式名稱）

每個控制項有一個**標準名稱**（中文），用於對話溝通。流程圖（`docs/user-manual/ui-flow.html`）也使用這些名稱。

### 即時監控（tabPageLiveView）

| 標準名稱 | Name | 類型 | 畫面文字 |
|---------|------|------|---------|
| 開始抓取 | `btnCameraGrab` | Button | 開始抓取 / 停止抓取 |
| 取得背景 | `btnGetBackground` | Button | 取得背景 |
| 預覽背景 | `btnViewBackground` | Button | 預覽背景 |
| 監控主畫面 | `panelMainDisplay` | Panel | — |
| 監控縮圖1~7 | `panelLiveCam1~7` | Panel | — |
| 監控切向曲線圖 | `muraChartVerticalLive` | Chart | — |
| 監控法向曲線圖 | `muraChartHorizontalLive` | Chart | — |
| 監控全覽圖 | `chartLiveOverview` | Chart | — |
| 暫停 Mura 檢測 | `lblIoDoMura`（點擊切換） | Label | DO1 MURA_DET / DO1 MURA ⏸（黃底=暫停中） |

### 歷史查詢（tabPageReview）

| 標準名稱 | Name | 類型 | 畫面文字 |
|---------|------|------|---------|
| 讀取資料 | `btnSelectFolder`（Review）/ `btnSelectDataFolder`（Data） | Button | 讀取資料 |
| 回顧縮圖1~7 | `pbCam1~7` | PictureBox | — |
| 回顧主畫面 | `canvasMain` | SmartCanvas | — |
| 回顧切向曲線圖 | `chartMuraVertical` | Chart | — |
| 回顧法向曲線圖 | `chartMuraHorizontal` | Chart | — |
| 回顧全覽圖 | `chartOverview` | Chart | — |
| 時段群組 | `grpReviewTimePeriod` | GroupBox | 時序 |
| 時段日期（時序cb） | `cbDate` | ComboBox | — |
| 時段時間（時序cb） | `cbTime` | ComboBox | — |
| 上一時段（上下鍵） | `btnPeriodPrev` | Button | < |
| 下一時段（上下鍵） | `btnPeriodNext` | Button | > |
| 單片群組 | `grpReviewGrabNav` | GroupBox | 單片 |
| 單片序號（序號cb） | `cbReviewGrabId` | ComboBox | — |
| 上一序號（上下鍵） | `btnGrabIdPrev` | Button | < |
| 下一序號（上下鍵） | `btnGrabIdNext` | Button | > |

### 檢測報表（tabPageData）

| 標準名稱 | Name | 類型 | 畫面文字 |
|---------|------|------|---------|
| 篩選異常 | `btnShowFail` | Button | 篩選異常 / 顯示全部 |
| 良率卡片1~7 | `panelStatCam1~7` | Panel | — |
| Mura 空間分布圖 | `chartMuraProfile` | Chart | — |
| 明細列表 | `listViewGrabDetail` | ListView | — |
| 序號範圍群組 | `groupBoxGrabIdRange` | GroupBox | 序號範圍 |
| 起始序號 | `cbGrabIdStart` | ComboBox | — |
| 結束序號 | `cbGrabIdEnd` | ComboBox | — |
| 序號選擇群組 | `grpDataSingleSheet` | GroupBox | 序號選擇 |
| 報表序號 | `cbDataGrabId` | ComboBox | — |
| 報表上一序號 | `btnGrabIdDataPrev` | Button | < |
| 報表下一序號 | `btnGrabIdDataNext` | Button | > |
| 時序範圍群組 | `groupBoxTimeRange` | GroupBox | 時序範圍 |
| 起始日期/時間 | `cbStartDate/cbStartTime` | ComboBox | — |
| 結束日期/時間 | `cbEndDate/cbEndTime` | ComboBox | — |
| 良率年圖 | `chartYearly` | Chart | — |
| 良率月圖 | `chartMonthly` | Chart | — |
| 良率日圖 | `chartDaily` | Chart | — |
| 年圖導航 | `cbChartYear` | ComboBox | — |
| 月圖導航 | `cbChartMonth` | ComboBox | — |
| 日圖導航 | `cbChartDay` | ComboBox | — |

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
| 座標狀態列 | `lblPixelInfo` | Label | 位置:... |
| 相機數狀態 | `lblCamCount` | Label | 相機: N/7 |
| PLC 狀態 | `lblPlcState` | Label | ● 狀態: -- |
| IO 連線狀態 | `lblPlcConn` | Label | ● IO: -- |
| 光源連線狀態 | `lblLightConn` | Label | ● 光源: -- |
| 儲存電腦連線狀態 | `lblStorageConn` | Label | ● 儲存電腦: -- |
| IO 燈號 | `lblIoDiAlive~lblIoDoPcBusy` | Label×5 | DI0~DO2 |

---

## Skills 路由（取代 docs/dev）

開發知識已整合至 `.claude/skills/`，按修改範圍觸發對應 skill：

| 修改範圍 | Skill | 涵蓋內容 |
|---------|-------|---------|
| UI 控制項、事件、Chart、Canvas | `/modify-ui` | Guard flags、V/H 決策矩陣、StitchMode、Chart 對齊、跨倍率 View、ProportionalScaler |
| Data tab 統計、CSV、Period Charts | `/modify-data-stats` | 統計三模式、CSV 格式、Period Charts、跨 Tab 同步 |
| GPU pipeline、Buffer、存檔格式 | `/modify-pipeline` | CUDA pinned memory、V/H ridge、.bin 格式、ImageRepository、StandardBgSub |
| MIL 取像、相機、CLProtocol、PLC | `/modify-acquisition` | 初始化順序、CLProtocol 延遲啟動、資源釋放、SetGrabHeight、PLC FSM |
| C# / WinForms 通用開發 | `/csharp-patterns` | 命名規則、Settings 持久化、WinForms 陷阱、Designer 規則 |
| Native C API 新增/修改 | `/add-native-api` | P/Invoke 宣告、C++ 實作範本 |
| 效能瓶頸排查 | `/perf-diagnose` | Stopwatch 計時、IO/GPU/UI 分層診斷 |
| 追蹤 btn/cb/event I/O 流程 | `/review-flow <控制項>` | 完整 call chain 追蹤 + 文件比對 |
| Build 驗證 | `/build` | Release+Debug x64 完整 build |
| Commit 前文件更新 | `/update-docs` | 批次更新 CLAUDE.md + skills |
| 提交推送 | `/commit` | build + 文件 + conventional commit |
| 控制項別名記錄 | `/alias-log` | 對話中新稱呼 → 更新速查表 + 建議標準名稱 |
| 現場部署 / 網路 / SMB | `/deploy-network` | 雙網段架構、單 NIC 雙 IP、匿名 Guest SMB、編碼陷阱（bat ASCII / ps1 UTF-8 BOM / JSON UTF-8 讀法）、secedit SeDenyNetworkLogonRight |

### 參考文件（僅供查閱，不自動載入）

| 文件 | 用途 |
|------|------|
| [`docs/dev/MIL_API_Reference.md`](docs/dev/MIL_API_Reference.md) | MIL .NET API 完整參考（常數、方法、範例） |
| [`docs/dev/system-resources.md`](docs/dev/system-resources.md) | 系統資源用量（GPU/CPU/RAM 評估） |
| [`docs/user-manual/io_diagrams.html`](docs/user-manual/io_diagrams.html) | IO FSM 視覺化（ET-7044 ↔ 設備 Nakan，瀏覽器開啟） |

### docs/ 目錄定位

```
docs/
├── dev/            ← 開發者/部署參考（API、硬體規格評估）
└── user-manual/    ← 操作員說明書（UI 流程、IO 圖、硬體規格）
```

開發知識統一放 `.claude/skills/`，`docs/dev/` 放大型參考文件。

**撰寫/更新 docs/ 規則**：必須比對實際程式碼確認功能狀態，不可只參考 skills（skills 本身可能滯後）。

### Skills ↔ docs 同步規則

修改功能後，若同時影響開發注意事項和使用者操作流程，**兩邊都要更新**：

| 改動 | 更新 `.claude/skills/` | 更新 `docs/` |
|------|:---:|:---:|
| 新增/修改 UI 控制項行為 | 對應 modify-* skill 的注意事項 | user-manual 操作說明 |
| 新增/修改設定參數 | csharp-patterns（持久化）+ 對應 skill | user-manual 參數說明 |
| 修改 pipeline/存檔格式 | modify-pipeline | — |
| 修改 MIL/PLC 行為 | modify-acquisition | user-manual PLC 操作 |

---

## 實作指引

### 狀態機邏輯

實作 click counting、mode transition、status flag 等狀態機時：
1. 先列出完整的**狀態轉移表**（State + Event → Next State + Action）
2. 與使用者確認後再寫 code
3. 避免用 AND 條件做安全檢查（容易漏邊界情況），優先用比值/閾值比較

### Build 驗證

- 修改 `.cs`、`.csproj`、`.sln` 後**立即 build** 確認零錯誤
- 不得在 VS 的 reserved ImportGroup 放自訂 Import
- Build 命令（**必須帶 Platform=x64**，本專案依賴 AMD64 MIL SDK）：
  ```
  cat > /tmp/build.bat << 'EOFBAT'
  @echo off
  "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" %1 /p:Configuration=Release /p:Platform=x64 /v:minimal
  EOFBAT
  cmd //c "$(cygpath -w /tmp/build.bat)" "專案路徑"
  ```

---

## Git Workflow 規則

**未經使用者明確說「commit/push」，不得主動執行任何 git commit 或 git push。**

**每次 commit / push 前，必須先更新相關文件：**

1. `CLAUDE.md` — 更新關鍵檔案速查、Skills 路由
2. `.claude/skills/*.md` — 更新對應的 skill 注意事項（根據改動內容選擇）
3. `README.md` — 更新對外專案說明（若有重大功能變更）

確保文件反映最新的程式碼狀態，讓下次對話能快速上手。
