using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Management;
using System.Windows.Forms;
using StorageBridge.Core;
using LightBridge.Core;
using MilGrabber.Core;
using TanukiCv.Controls;
using TanukiCv.Utils;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Interop;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.State;
using AniloxRoll.Monitor.UI.Managers;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Presenters;
using AniloxRoll.Monitor.UI.Widgets;

namespace AniloxRoll.Monitor.Forms
{
    public partial class AniloxRollForm : Form
    {
        // 螢幕 mm/px 的 GetDeviceCaps P/Invoke 已收進 TanukiCv.Core.SystemInfo（唯一來源）

        // --- 核心服務 ---
        private readonly ImageRepository _imageRepository = new ImageRepository();
        private BatchInspectionService _inspectionService;

        // --- UI Helpers ---
        private DateTimeNavigator _dateTimeNavigator;
        private ReviewDisplayManager _reviewDisplayManager;   // 回顧同源顯示（sdk ImageDisplayView，絞殺榕收官）
        private double _reviewViewLeftMm = double.NaN, _reviewViewRightMm, _reviewViewTopMm, _reviewViewBotMm; // 新畫布視野快取（chart 原子更新用）
        private int _reviewSyncCount; private long _reviewSyncOvMax, _reviewSyncRowMax;   // [ReviewSync] 拖曳跟隨計時儀器
        private AniloxRollPresenter _presenter;
        private FormInteractionHelper _interactionHelper;
        private ColumnCurveChartHelper _reviewOverviewHelper;
        private ColumnCurveChartHelper _liveOverviewHelper;
        private RowCurveChartHelper _liveRowChartHelper;
        private RowCurveChartHelper _reviewRowChartHelper;
        private RowCurveDisplayAdapter _liveRowDisplay;
        private RowCurveDisplayAdapter _reviewRowDisplay;
        private RowCurveSyncCoordinator _liveRowSync;
        private RowCurveSyncCoordinator _reviewRowSync;
        private LiveCameraManager _liveCameraManager;
        // Global merge 用：快取各相機 row curve 資料，合併後更新圖表
        private readonly Dictionary<int, float[]> _liveRowMeanCache = new Dictionary<int, float[]>();
        private readonly Dictionary<int, float[]> _liveRowMaxCache  = new Dictionary<int, float[]>();
        private readonly Dictionary<int, float[]> _waterfallRowMeanPending = new Dictionary<int, float[]>();
        private readonly Dictionary<int, float[]> _waterfallRowMaxPending  = new Dictionary<int, float[]>();
        private float[] _waterfallRowMean;
        private float[] _waterfallRowMax;
        private int _waterfallRowWrite;
        private ProportionalScaler _scaler;

        // --- 相機參數控制項陣列（供 SyncFromCamera 存取）---
        private TrackBar[]      _expBars;
        private NumericUpDown[] _expNums;
        private TrackBar[]      _lrBars;
        private NumericUpDown[] _lrNums;
        private TrackBar[]      _htBars;
        private NumericUpDown[] _htNums;
        // --- CAM All 控制項 ---
        private TrackBar      _expAllBar;
        private NumericUpDown _expAllNum;
        private TrackBar      _lrAllBar;
        private NumericUpDown _lrAllNum;
        private TrackBar      _htAllBar;
        private NumericUpDown _htAllNum;

        // --- 拖曳偵測：拖曳中時抑制硬體寫入 ---
        private readonly HashSet<TrackBar> _dragging = new HashSet<TrackBar>();

        // --- TrackBar 滾輪攔截器（每格 = 1）---
        private readonly List<NativeWindow> _wheelInterceptors = new List<NativeWindow>();

        // --- Hardware → UI 同步：防止 SyncFromHardware 觸發 ValueChanged 再回寫硬體 ---
        private bool _syncingFromHw = false;

        // --- Telemetry ---
        private LiveTelemetryPresenter _telemetryPresenter;
        private System.Windows.Forms.Timer _telemetryTimer;
        private System.Windows.Forms.Timer _liveOverviewTimer;

        // --- Resource Monitor ---
        private ListViewItem _resMonRawSize, _resMonGpuTime, _resMonSaveSize;
        private ListViewItem _resMonDiskWrite, _resMonFrames, _resMonRamUsed, _resMonVramEst;
        private ListViewItem _storageDiskFreeRow, _storageLastCleanRow;

        // --- 檢測日誌 ---
        private InspectionLogService _inspectionLogService;
        private string _currentGrabId;

        // --- App Mode ---
        private AppModeConfig _appMode;
        private CleanupFlagWatcher _cleanupFlagWatcher;

        // --- 儲存管理 ---
        private StorageRetentionService _retentionService;
        private RemoteCopyService _remoteCopyService;
        private int _completedGrabCount;
        private DateTime _lastGrabEventTime;

        // --- IO 連動 ---
        private IoGrabController _ioGrabController;
        private LightController _lightController;

        // --- 統計 ---
        private DataStatisticsPresenter _dataStatsPresenter;

        // --- 資料緩存 ---
        private readonly List<Image> _thumbnailCache = new List<Image>();
        private InspectionSettings _settings;
        private AniloxRoll.Monitor.Settings.Services.SettingsHub _settingsHub;

        /// <summary>"v" = vertical ridge（預設），"h" = horizontal ridge。控制 Live 顯示方向。</summary>
        private string _liveDisplayDirection = "v";

        /// <summary>Data tab 變更 cbDataId 後 cbReviewId 已同步但 camReviewMain 尚未更新；
        /// 待使用者切到 Review tab 時才載圖。</summary>
        private bool _reviewDirty = false;

        /// <summary>
        /// H3 fix：PropertyValueChanged 觸發的 full CSV scan（RefreshStats + RefreshPeriodCharts）
        /// 用 debounce 合併多次連續變更（如 slider 拖拽）。300ms 內未再觸發才執行，避免 UI 凍結。
        /// </summary>
        private Timer _statsRefreshDebouncer;

        // --- 常數 ---
        private const int CameraCount = 7;
        private const int TickFreq = 1000;

        // --- IEC 60073 色碼 ---
        private static readonly Color IecGreen    = Color.FromArgb(56, 142, 60);
        private static readonly Color IecBlue     = Color.FromArgb(0, 122, 204);
        private static readonly Color IecYellow   = Color.FromArgb(249, 168, 37);
        private static readonly Color IecRed      = Color.FromArgb(198, 40, 40);
        private static readonly Color IecGray     = Color.FromArgb(117, 117, 117);
        private static readonly Color IecDarkGray = Color.FromArgb(60, 60, 60);

        // --- Live 全覽圖：每台相機最新曲線快取 ---
        private readonly float[][] _liveCurveMean = new float[CameraCount][];
        private readonly float[][] _liveCurveMax  = new float[CameraCount][];
        private volatile bool _liveOverviewDirty;
        private bool _isIoSuspended;

        // --- Review tab 拼接管理 ---
        private ReviewStitchCoordinator _stitchCoordinator;


        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);
            FlowTrace.Log("ui:關閉程式");   // session 收尾行——log 無此行而中斷＝異常終止（crash）的訊號
            // Closing 階段：只「停止」非 UI 執行緒活動（避免 Handle 銷毀後它們還在 BeginInvoke）。
            // Dispose 留到 FormClosed 統一處理，避免雙路徑釋放重疊。
            try { if (_liveCameraManager?.IsLiveGrabbing == true) _liveCameraManager.StopGrab(); } catch { }
            try { _telemetryTimer?.Stop(); } catch { }
            try { _liveOverviewTimer?.Stop(); } catch { }
            try { _statsRefreshDebouncer?.Stop(); _statsRefreshDebouncer?.Dispose(); _statsRefreshDebouncer = null; } catch { }  // H3 + round-2 H3 補 Dispose
            try { _reviewLoadDebounce?.Stop(); _reviewLoadDebounce?.Dispose(); _reviewLoadDebounce = null; } catch { }  // 回顧序號載入 debounce
            try { _cleanupFlagWatcher?.Dispose(); _cleanupFlagWatcher = null; } catch { }  // M3: 10 秒輪詢提前停
            try { _reviewDisplayManager?.Dispose(); _reviewDisplayManager = null; } catch { }  // #13 同源顯示（內含 33ms timer）
            try { System.Net.NetworkInformation.NetworkChange.NetworkAddressChanged -= OnNetworkAddressChanged; } catch { }
        }

        /// <summary>
        /// 背景執行緒（Modbus 輪詢、MIL mouse hook、retention service 等）回 UI 執行緒更新時的安全 marshal。
        /// 關閉時序：FormClosing/FormClosed 銷毀 Handle 後，背景 callback 可能仍在跑並呼叫此處 →
        /// 守 IsHandleCreated/IsDisposed/Disposing 早退，並 try/catch 吞掉競態窗口（guard 通過後 Handle 才銷毀）拋出的例外。
        /// 已在 UI 執行緒（!InvokeRequired）時直接執行 action。
        /// </summary>
        private void SafeBeginInvoke(Action action)
        {
            if (action == null) return;
            if (!InvokeRequired) { action(); return; }
            if (!IsHandleCreated || IsDisposed || Disposing) return;
            // ObjectDisposedException 繼承自 InvalidOperationException，單一 catch 即涵蓋兩者
            try { BeginInvoke(action); }
            catch (InvalidOperationException) { /* guard 通過後 Handle 已銷毀的競態窗口 */ }
        }

        public AniloxRollForm()
        {
            InitializeComponent();
            SetupMainTabRendering();   // 監控/回顧/報表 頁籤文字置中（預設渲染器留 icon 位偏左）
            try
            {
                using (var stream = System.Reflection.Assembly.GetExecutingAssembly()
                    .GetManifestResourceStream("AniloxRoll.Monitor.Resources.app.ico"))
                    if (stream != null) this.Icon = new System.Drawing.Icon(stream);
            }
            catch { }
            InitializeSystem();
            _scaler = new ProportionalScaler(this);
            _scaler.Initialize();
            Shown += (s, e) =>
            {
                // 開窗即最大化後，補縮一次「作用中 tab」（tabPageLiveView 等）的內容：
                // WinForms TabControl lazy-layout 讓作用中 tab 在 maximize 當下沒被 ScaleRecursive 套到
                // （原本要切到別 tab 再切回才放大）。
                _scaler?.RescaleActiveTabs();

                // 消除「第一次切 tab 整個版面一塊一塊放大冒出」的分塊：老架構是設計小尺寸 → 開窗放大到
                // 最大化，ProportionalScaler 在每個 tab「首次顯示」才逐控制項放大重排（= 分塊）。此時尺寸
                // 已最大化，PrewarmAllTabs 把每頁放大重排一次做完（過程 LockWindowUpdate 壓住整棵樹繪製、
                // 不可見）。cycle 會觸發 tabMain→Review 的 SelectedIndexChanged，故用 _reviewDirty 守衛。
                if (_scaler != null)
                {
                    bool savedReviewDirty = _reviewDirty;
                    _reviewDirty = false;
                    try { _scaler.PrewarmAllTabs(); }
                    finally { _reviewDirty = savedReviewDirty; }
                }
                _suppressTabIntent = false;   // 預熱完成 → 之後的 tab 切換才是使用者動作

                // PropertyGrid 字體維持 DPI 原生大小（使用者要求 1.0，不另外收小）
                AutoFitPropertyGridLabelColumn(propertyGridSettings);
                // 選取第一個 category 的第一個屬性，PropertyGrid 會自動捲動到頂
                // 層級：SelectedGridItem → parent(category) → parent.Parent(root) → [0](第一 category) → [0](第一屬性)
                try
                {
                    var root = propertyGridSettings.SelectedGridItem?.Parent?.Parent;
                    if (root?.GridItems?.Count > 0)
                    {
                        var firstCat = root.GridItems[0];
                        if (firstCat.GridItems.Count > 0)
                            propertyGridSettings.SelectedGridItem = firstCat.GridItems[0];
                    }
                }
                catch { }
            };
        }

        private Font _tabSelFont;     // 選中頁籤＝大字（快取，避免每次重繪 new Font）
        private Font _tabNormFont;    // 未選中＝小字
        private bool _suppressTabIntent = true;   // 開機預設抑制（PrewarmAllTabs 的程式化 cycle 不記 ui:）；Shown 預熱完解除

        /// <summary>tabMain（監控/回顧/報表）頁籤文字置中：預設渲染器保留 icon 位造成偏移 →
        /// OwnerDraw 固定尺寸 + 置中繪字；選中頁大字+白底、未選小字。</summary>
        private void SetupMainTabRendering()
        {
            _tabNormFont = tabMain.Font;
            _tabSelFont  = new Font(tabMain.Font.FontFamily, tabMain.Font.Size + 3f, FontStyle.Bold);
            tabMain.SizeMode = TabSizeMode.Fixed;
            tabMain.ItemSize = new Size(104, 34);
            tabMain.DrawMode = TabDrawMode.OwnerDrawFixed;
            tabMain.DrawItem += (s, e) =>
            {
                if (e.Index < 0 || e.Index >= tabMain.TabPages.Count) return;
                bool sel = e.Index == tabMain.SelectedIndex;
                using (var back = new SolidBrush(sel ? Color.White : SystemColors.Control))
                    e.Graphics.FillRectangle(back, e.Bounds);
                TextRenderer.DrawText(e.Graphics, tabMain.TabPages[e.Index].Text,
                    sel ? _tabSelFont : _tabNormFont, e.Bounds, Color.Black,
                    TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.NoPrefix);
            };
            // 切換時左右兩頁都要重繪字級 + tab 切換 intent（盲測輪5：5 次切 tab 全隱形＝盲區）
            // _suppressTabIntent：開機 PrewarmAllTabs 的程式化 cycle 不記 ui:（D 系列首輪抓到的誤報）
            tabMain.SelectedIndexChanged += (s, e) =>
            {
                tabMain.Invalidate();
                if (!_suppressTabIntent) FlowTrace.Log($"ui:tab → {tabMain.SelectedTab?.Text}");
            };
        }

        private void InitializeSystem()
        {
            _appMode = AppModeConfig.Load();
            if (_settings == null) _settings = ConfigManager.LoadInspectionSettings();
            // Bootstrap 階段例外：SettingsHub 在下行才建構，這條直接賦值是不可避免的（合理 SSoT 違反）。
            // AppRole 來自 app-mode.json（外部檔），啟動時同步進 InspectionSettings 記憶體；
            // 後續若使用者透過 PG 改 AppRole，會走 Hub 正常管線。
            _settings.AppRole = _appMode?.Role ?? MachineRole.Inspection;
            // L2 SettingsHub：所有 setting 變更走 Changed event，OnSettingChanged 接管 Apply* 副作用。
            _settingsHub = new AniloxRoll.Monitor.Settings.Services.SettingsHub(_settings);
            _settingsHub.Changed += OnSettingChanged;
            // FSM Action Logger（runtime flag，預設 Off 零 overhead）
            UiActionLogger.Init(_settings);
            UiActionLogger.Enabled = _settings.DebugUiActionLog;
            if (UiActionLogger.Enabled) _settingsHub.Changed += UiActionLogger.OnSettingChanged;
            EnsureAniloxFolderStructure();
            CameraFrameSaver.InitResourceLog(_settings?.Storage?.LogsPath);
            CameraFrameSaver.GetUiStateCallback = () =>
            {
                bool live = _liveCameraManager?.IsLiveGrabbing ?? false;
                bool review = _imageRepository?.FileCount > 0;
                string stitch = _settings?.StitchMode.ToString() ?? "?";
                return $"{(live ? "T" : "F")},{(review ? "T" : "F")},{stitch}";
            };
            InitServiceLayer();
            InitUiLayer();
            if (_appMode?.Role != MachineRole.Storage)
                InitCameraLayer();
            // Storage 模式：cleanup 統一在主 FormClosed handler 處理（H1 + B-M1 修正）
            InitializeRightPanelControls();
            SetupDataTab();
            ApplyStorageModeUi();

            // 硬體狀態列啟動先顯示「初始化中」（灰）；各硬體連線/偵測完成後由各自 Update*Label 接手。
            if (_appMode?.Role != MachineRole.Storage)
                ShowHardwareStatusInitializing();

            // DCF 缺失警語：UI 已建立，立即顯示
            if (_dcfMissing && _appMode?.Role != MachineRole.Storage)
                UpdateCamCountLabel(0, CameraCount);
        }

        /// <summary>確保 Anilox 資料根目錄與子目錄存在。
        /// AniloxRootPath 的磁碟不存在時，把磁碟換成 C:（如 D:\Anilox → C:\Anilox），
        /// MessageBox 告知 + 寫回 settings.json。建立 Captures/Logs/Bg/Dcf 子目錄。</summary>
        private void EnsureAniloxFolderStructure()
        {
            try
            {
                string aniloxRoot = _settings?.Storage?.AniloxRootPath;
                if (string.IsNullOrWhiteSpace(aniloxRoot)) return;

                // 檢查磁碟是否存在
                string drive = Path.GetPathRoot(aniloxRoot);
                if (!string.IsNullOrEmpty(drive) && !Directory.Exists(drive))
                {
                    // 換成 C 槽，保留後段路徑
                    string newRoot = "C:" + aniloxRoot.Substring(drive.Length - 1);
                    MessageBox.Show(
                        $"{drive} 不存在，Anilox 根目錄改用：{newRoot}\n請至檢測設定 → 儲存設定 修正路徑。",
                        "資料夾 fallback", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    _settings.Storage.AniloxRootPath = newRoot;
                    aniloxRoot = newRoot;
                    ConfigManager.SaveInspectionSettings(_settings);
                }

                // 建立所有子目錄
                Directory.CreateDirectory(aniloxRoot);
                Directory.CreateDirectory(_settings.Storage.CaptureRootPath);
                Directory.CreateDirectory(_settings.Storage.LogsPath);
                Directory.CreateDirectory(_settings.Storage.BackgroundPath);

                // 一次性遷移：舊版 D:\AniloxCaptures\bg → D:\Anilox\Bg
                // 只在新目錄為空時遷移，避免覆蓋使用者已建立的新內容
                string rootDrive = Path.GetPathRoot(aniloxRoot)?.TrimEnd('\\') ?? "D:";
                string legacyCaptures = Path.Combine(rootDrive + "\\", "AniloxCaptures");
                if (Directory.Exists(legacyCaptures))
                {
                    MigrateLegacySubdir(Path.Combine(legacyCaptures, "bg"),  _settings.Storage.BackgroundPath);
                }

                // 檢查 DCF 檔是否存在；不存在時設旗標，lblCamCount 之後會顯示警語
                string dcfPath = DcfPathHelper.Resolve(_settings?.CameraParam?.DcfPath);
                _dcfMissing = !string.IsNullOrWhiteSpace(dcfPath) && !File.Exists(dcfPath);
                if (_dcfMissing)
                    Trace.WriteLine($"[EnsureAniloxFolderStructure] DCF 缺失: {dcfPath}");
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"[EnsureAniloxFolderStructure] {ex}");
            }
        }

        /// <summary>把 legacy 目錄內所有檔案拷貝到 new 目錄（只在 new 為空時執行）。</summary>
        private static void MigrateLegacySubdir(string legacyDir, string newDir)
        {
            try
            {
                if (!Directory.Exists(legacyDir)) return;
                if (Directory.Exists(newDir) && Directory.GetFiles(newDir).Length > 0) return; // 新目錄非空，不覆蓋

                Directory.CreateDirectory(newDir);
                foreach (var src in Directory.GetFiles(legacyDir))
                {
                    string dst = Path.Combine(newDir, Path.GetFileName(src));
                    if (!File.Exists(dst))
                        File.Copy(src, dst);
                }
                Trace.WriteLine($"[Migrate] {legacyDir} → {newDir}");
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"[Migrate] {legacyDir} failed: {ex.Message}");
            }
        }

        /// <summary>純業務服務：不依賴任何 UI 控制項。</summary>
        private void InitServiceLayer()
        {
            try
            {
                _inspectionService = new BatchInspectionService();
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"[InitServiceLayer] GPU 初始化失敗（無獨顯？），BMP 處理功能不可用: {ex.Message}");
                _inspectionService = null;
            }
            _inspectionLogService = new InspectionLogService(
                () => _settings?.CaptureRootPath ?? string.Empty);

            // 循環儲存（事件驅動：grab 結束 / watchdog / 每 10 grab / 啟動時各觸發一次）
            _retentionService = new StorageRetentionService(
                getRootPath:     () => GetStorageRetentionRoot(),
                getMinFreeBytes: () => (long)(_settings?.LocalMinFreeGB ?? 100) * 1024L * 1024L * 1024L);

            if (_appMode?.Role == MachineRole.Storage)
            {
                // Storage 模式：輪詢旗標觸發清理
                _cleanupFlagWatcher = new CleanupFlagWatcher(
                    () => _appMode.StorageMachineConfigFolder,
                    _retentionService);
                _cleanupFlagWatcher.Start();
            }
            else
            {
                // Inspection 模式：遠端複製 + IO + 光源
                _remoteCopyService = new RemoteCopyService(
                    getRemotePath: () => _settings?.RemotePath ?? string.Empty,
                    getLocalRoot:  () => _settings?.CaptureRootPath ?? string.Empty);
                // 初始化順序對齊狀態列由左至右（相機→儲存→光源→IO）：光源先於 IO，
                // IO（快速 TCP）最後啟動，避免它最先亮綠讓人誤以為系統已就緒。
                InitLightController();
                InitIoController();
                // 本機網路介面變動（拔/插網路線）→ 立即重探儲存，不等探測週期（事件驅動、零輪詢成本）
                System.Net.NetworkInformation.NetworkChange.NetworkAddressChanged += OnNetworkAddressChanged;
            }

            // 啟動時執行一次清理（雙模式共用）
            Task.Run(() => _retentionService.RunCleanup());
        }


        /// <summary>DCF 檔不存在時設為 true，UpdateCamCountLabel 改顯示警語而非相機數量。</summary>
        private bool _dcfMissing = false;

        private void UpdateCamCountLabel(int connected, int expected)
        {
            if (_dcfMissing)
            {
                lblCamCount.Text = "⚠ DCF 缺失";
                lblCamCount.BackColor = IecRed;
                return;
            }
            // 相機已分配但 CLProtocol 尚未就緒（曝光/線掃未套）→ 顯示「初始化中」而非誤導的暫態連線數。
            if (_liveCameraManager != null && _liveCameraManager.IsAllocated && !_liveCameraManager.AreCamerasHwReady)
            {
                lblCamCount.Text = "相機: 初始化中…";
                lblCamCount.BackColor = IecGray;
                return;
            }
            lblCamCount.Text = $"相機: {connected}/{expected}";
            if (connected >= expected)
                lblCamCount.BackColor = IecGreen;   // 綠：全連
            else if (connected > 0)
                lblCamCount.BackColor = IecYellow;  // 黃：部分連線
            else
                lblCamCount.BackColor = IecRed;   // 紅：全斷
        }

        /// <summary>UI 層：Presenter、Helper、PropertyGrid、Canvas 事件。</summary>
        private void InitUiLayer()
        {
            _dateTimeNavigator = new DateTimeNavigator(
                _imageRepository, cbReviewDate, cbReviewTime);

            // 2b-ii-B：ThumbnailGridPresenter（舊回顧縮圖畫廊）已刪——縮圖顯示/選取全由 sdk ImageDisplayView
            //   的 ThumbStrip 承接（camReview1~7 現為 Panel 宿主，見 ReviewDisplayManager）。
            _presenter = new AniloxRollPresenter(
                _imageRepository, _inspectionService, _dateTimeNavigator);



            _reviewOverviewHelper = new ColumnCurveChartHelper(this.chartReviewColumn);
            _reviewOverviewHelper.SetThresholds(_settings.ErrorValueMeanV, _settings.ErrorValueMaxV);
            if (chartReviewColumn.ChartAreas.Count > 0)
                chartReviewColumn.ChartAreas[0].AxisX.ScaleView.Zoomable = false;

            _liveOverviewHelper = new ColumnCurveChartHelper(this.chartLiveColumn);
            _liveOverviewHelper.SetThresholds(_settings.ErrorValueMeanV, _settings.ErrorValueMaxV);
            if (chartLiveColumn.ChartAreas.Count > 0)
                chartLiveColumn.ChartAreas[0].AxisX.ScaleView.Zoomable = false;

            if (chartLiveRow == null)
                throw new InvalidOperationException("chartLiveRow is not initialized. Ensure InitializeComponent runs before UI layer initialization.");
            if (chartReviewRow == null)
                throw new InvalidOperationException("chartReviewRow is not initialized. Ensure InitializeComponent runs before UI layer initialization.");

            _liveRowChartHelper = new RowCurveChartHelper(this.chartLiveRow);
            _liveRowDisplay = new RowCurveDisplayAdapter(_liveRowChartHelper, GetVerticalDisplayDirection);
            _liveRowSync = new RowCurveSyncCoordinator(_liveRowDisplay);
            _liveRowDisplay.SetThresholds(_settings.ErrorValueMeanH, _settings.ErrorValueMaxH);

            _reviewRowChartHelper = new RowCurveChartHelper(this.chartReviewRow);
            _reviewRowDisplay = new RowCurveDisplayAdapter(_reviewRowChartHelper, GetVerticalDisplayDirection);
            _reviewRowSync = new RowCurveSyncCoordinator(_reviewRowDisplay);
            _reviewRowDisplay.SetThresholds(_settings.ErrorValueMeanH, _settings.ErrorValueMaxH);

            UpdateRowChartPitch();

            // Review tab 欄 chart 點選 —— 過渡語意（mode/強化切換 FSM 待 #13 接入後定案）：
            //   點全覽圖（接位後的 chartReviewColumn）＝切檢出方向 v；StitchMode/強化暫走 PropertyGrid。TODO-FSM
            chartReviewColumn.MouseClick += (s, e) =>
            {
                UiActionLogger.SetSource("chartReviewColumn.Click");
                SwitchRidgeDirection("v");
            };
            chartReviewRow.MouseClick += (s, e) =>
            {
                UiActionLogger.SetSource("chartReviewRow.Click");
                UiActionLogger.RecordViewOnly("chartReviewRow.Click");
                SwitchRidgeDirection("h");
            };

            // Live tab 欄 chart 點選 —— 過渡語意（mode/強化切換 FSM 待 #13 接入後定案）：
            //   點全覽圖（接位後的 chartLiveColumn）＝切檢出方向 v（與 Horizontal chart 對稱）；
            //   StitchMode / 強化切換暫時只走 PropertyGrid（SSoT 正路）。TODO-FSM
            chartLiveColumn.MouseClick += (s, e) =>
            {
                UiActionLogger.SetSource("chartLiveColumn.Click");
                SwitchLiveDisplayDirection("v");
            };
            chartLiveRow.MouseClick += (s, e) =>
            {
                UiActionLogger.SetSource("chartLiveRow.Click");
                UiActionLogger.RecordViewOnly("chartLiveRow.Click");
                SwitchLiveDisplayDirection("h");
            };

            // PropertyGrid：動態標題說明（點選 ─ X ─ 時，底部說明欄顯示當前參數值）
            TypeDescriptor.AddProvider(
                new InspectionSettingsDescriptionProvider(TypeDescriptor.GetProvider(_settings), _settings),
                _settings);

            // PropertyGrid：Categorized 排序（維持宣告順序），預設摺疊
            propertyGridSettings.SelectedObject = _settings;
            propertyGridSettings.ToolbarVisible = false;
            propertyGridSettings.PropertySort   = System.Windows.Forms.PropertySort.Categorized;
            propertyGridSettings.ExpandAllGridItems();
            // 展開第一層後，收合所有子項目（第二層以下）
            foreach (GridItem cat in propertyGridSettings.SelectedGridItem?.Parent?.GridItems
                     ?? (System.Collections.IEnumerable)Array.Empty<GridItem>())
            {
                foreach (GridItem prop in cat.GridItems)
                {
                    if (prop.GridItemType == GridItemType.Category || prop.Expandable)
                        prop.Expanded = false;
                }
            }
            propertyGridSettings.PropertyValueChanged -= _propertyGrid_PropertyValueChanged;
            propertyGridSettings.PropertyValueChanged += _propertyGrid_PropertyValueChanged;
            propertyGridSettings.SelectedGridItemChanged -= PropertyGridSettings_SelectedGridItemChanged;
            propertyGridSettings.SelectedGridItemChanged += PropertyGridSettings_SelectedGridItemChanged;
            AutoFitPropertyGridLabelColumn(propertyGridSettings);


            _interactionHelper = new FormInteractionHelper(new FormInteractionContext
            {
                Form             = this,
                ButtonsToLock    = new Button[] { btnReviewSelectFolder },
                ThumbnailCache   = _thumbnailCache,
                Presenter        = _presenter,
                InspectionService = _inspectionService,
                ImageRepository  = _imageRepository,
                TimeNavigator    = _dateTimeNavigator,
                Settings         = _settings,
                RowChartHelper = _reviewRowChartHelper,
            });
            _interactionHelper.ApplySettingsToService();

            _stitchCoordinator = new ReviewStitchCoordinator(new ReviewStitchContext
            {
                ChartReviewPatch             = chartReviewColumn,
                ChartReviewHorizontal       = chartReviewRow,
                InteractionHelper         = _interactionHelper,
                RowChartHelper            = _reviewRowChartHelper,
                RowChartDisplay           = _reviewRowDisplay,
                RowChartSync              = _reviewRowSync,
                OverviewHelper            = _reviewOverviewHelper,
                InspectionService         = _inspectionService,
                ImageRepository           = _imageRepository,
                DataStatsPresenter        = _dataStatsPresenter,
                Settings                  = _settings,
                DateTimeNavigator         = _dateTimeNavigator,
                CameraCount               = CameraCount,
            });

            // 絞殺榕收官（Wave2 2b-ii-B）：回顧主畫面/縮圖＝Designer 上的 Panel，直接交給 ImageDisplayView 落地生根。
            {
                _reviewDisplayManager = new ReviewDisplayManager(camReviewMain,
                    new System.Windows.Forms.Panel[] { camReview1, camReview2, camReview3, camReview4, camReview5, camReview6, camReview7 });
                // 選中相機 index 來源＝ImageDisplayView（取代舊 ThumbnailGridPresenter.SelectedIndex）
                _stitchCoordinator.SelectedCamIndexProvider = () => _reviewDisplayManager?.SelectedCamIndex ?? 0;
                _stitchCoordinator.StitchedImagesReady += (gray, ws, hs, ops, pos, isGlobal) =>
                    _reviewDisplayManager?.PushFrames(gray, ws, hs, ops, pos, isGlobal,
                        _interactionHelper?.ScreenMmPerPixel ?? 0,
                        AniloxRoll.Monitor.Core.Services.InspectionEngineConfig.DefaultSaveResizeScale,
                        _reviewRowDisplay?.RowPitchMm ?? 0,
                        ShouldFlipDisplayVertical());   // 灰階已在 RSC 解碼段轉好（零 race）；?.：關閉時序防 NRE
                // Stage2：新 canvas 視野 → 回顧曲線圖 zoom 連動（欄=全覽 X、列=Y；拖曳中即時）
                _reviewDisplayManager.ViewRangeMmChanged += (l, r, top, bot) =>
                {
                    _reviewViewLeftMm = l; _reviewViewRightMm = r; _reviewViewTopMm = top; _reviewViewBotMm = bot;
                    var swSync = System.Diagnostics.Stopwatch.StartNew();
                    _reviewOverviewHelper?.UpdateViewRange(l, r);
                    long ovMs = swSync.ElapsedMilliseconds;
                    _reviewRowSync?.SetViewRange(top, bot);
                    long rowMs = swSync.ElapsedMilliseconds - ovMs;
                    // [ReviewSync] 計時儀器：單次 >25ms 即時告警；每 120 次彙總（拖曳 ~4 秒）→ 看瓶頸在 overview/row/事件頻率
                    _reviewSyncCount++; _reviewSyncOvMax = Math.Max(_reviewSyncOvMax, ovMs); _reviewSyncRowMax = Math.Max(_reviewSyncRowMax, rowMs);
                    long gapMs = swSync.ElapsedMilliseconds; // handler 總耗時
                    if (gapMs > 25)
                        Trace.WriteLine($"[ReviewSync] SLOW ov={ovMs}ms row={rowMs}ms");
                    if (_reviewSyncCount >= 120)
                    {
                        Trace.WriteLine($"[ReviewSync] 120 events: ovMax={_reviewSyncOvMax}ms rowMax={_reviewSyncRowMax}ms");
                        _reviewSyncCount = 0; _reviewSyncOvMax = 0; _reviewSyncRowMax = 0;
                    }
                };
                // chart 重建（重載/強化切換）原子帶入當前視野 → 不先閃回預設（同 Live 解法）
                _stitchCoordinator.SameSourceViewRange = () =>
                    double.IsNaN(_reviewViewLeftMm) ? null
                    : new[] { _reviewViewLeftMm, _reviewViewRightMm, _reviewViewTopMm, _reviewViewBotMm };
                // 游標狀態 → 狀態列 lblPixelInfo（mm 換算同源在 ImageDisplayView，這裡只格式化＝app 政策）。
                // 取代舊 camReviewMain.StatusChanged→UpdateCanvasInfo（覆蓋後已死，#13 遷移時即斷）。
                _reviewDisplayManager.CursorStatusChanged += s =>
                {
                    if (lblPixelInfo == null) return;
                    lblPixelInfo.Text = CursorStatusTextFormatter.Format(s);
                };
                _reviewDisplayManager.SetFlipVertical(ShouldFlipDisplayVertical());
            }

            _stitchCoordinator.StitchedCurveUpdated += (mean, max, ops, pos, errMean, errMax) =>
                _dataStatsPresenter?.SyncMuraProfileFromReview(mean, max, ops, pos, errMean, errMax);

            _presenter.BusyStateChanged += _interactionHelper.SetUiLoadingState;
            _presenter.LogReported      += OnPresenterLogReported;
            // 4c：舊 gallery 選擇鏈已拆（PictureBox 被 sdk ThumbStrip 覆蓋＝點擊不可達；
            //     縮圖↔主畫面雙向連動由 ImageDisplayView 內建）。
            _dateTimeNavigator.PeriodSelectionChanged += _presenter.UpdatePeriodNavigationState;
            _dateTimeNavigator.PeriodSelectionChanged += () =>
            {
                var current = _dateTimeNavigator.GetCurrentPeriodOrDefault(DateTime.MinValue);
                if (current != DateTime.MinValue) _dataStatsPresenter.SyncGrabIdFromTime(current);
            };
            _dateTimeNavigator.PeriodSelectionChanged += OnPeriodComboChanged;
            _presenter.PeriodNavigationStateChanged   += (canLast, canNext) =>
            {
            };
            _presenter.UpdatePeriodNavigationState();

            // 絞殺榕全劇終（Wave2）：camReviewMain/camReview1~7 已是 Designer 上的 Panel，
            //   顯示/互動/手勢/座標 overlay/雙三擊/縮圖選取全由 sdk ImageDisplayView 內建承接
            //   （經 ReviewDisplayManager 落地生根）。舊 ImageCanvas/PictureBox/CanvasInteractionHelper/
            //   ThumbnailGridPresenter 顯示鏈已整棵砍除。

            UpdateLiveDirectionVisual();
            UpdateRidgeDirectionVisual(null); // dir=null：無強化橘框，底色依 StitchMode 上色
        }

        /// <summary>相機層：LiveCameraManager 與 FormClosed 清理。</summary>
        private void InitCameraLayer()
        {
            _liveCameraManager = new LiveCameraManager(
                this,
                new[] { camLive1, camLive2, camLive3,
                        camLive4, camLive5, camLive6, camLive7 },
                camLiveMain,
                pixelText => { if (lblPixelInfo != null) lblPixelInfo.Text = pixelText; }
            );
            _liveCameraManager.SetCaptureSettings(_settings);
            UpdateRowChartPitch();
            _liveCameraManager.OnFilesSaved = files => _remoteCopyService?.EnqueueFiles(files);
            _liveCameraManager.OnInspectionResult += OnCameraInspectionResult;
            btnLiveGetBackground.Click += btnLiveGetBackground_Click;
            btnLiveViewBackground.Click += btnLiveViewBackground_Click;
            UpdateStandardBgSubLockState();
            _liveCameraManager.OnLiveCurveData      += OnLiveCurveData;
            _liveCameraManager.OnLiveRowCurveData   += OnLiveRowCurveData;
            _liveCameraManager.OnLiveViewRange      += ApplyLiveViewRange; // 主畫面縮放/平移 → live 曲線圖 zoom 連動（bin↔主畫面對齊）
            _liveCameraManager.OnAfterVerticalZoom   = () =>
            {
                if (_settings?.StitchMode != StitchMode.Vertical) return;
                int camId = _liveCameraManager.SelectedMainCameraId;
                int idx   = camId - 1;
                if (idx < 0 || idx >= CameraCount) return;
                var mean = _liveCurveMean[idx];
                var max  = _liveCurveMax[idx];
                if (mean == null) return;
                OnLiveCurveData(camId, mean, max);
            };
            _liveCameraManager.OnCameraCountChanged += (connected, expected) =>
            {
                if (InvokeRequired) { if (!IsHandleCreated || IsDisposed || Disposing) return; SafeBeginInvoke(() => UpdateCamCountLabel(connected, expected)); return; }
                UpdateCamCountLabel(connected, expected);
            };
            // CLProtocol 就緒前「開始抓取」維持灰色 + 相機數顯示「初始化中」，避免 grab 期間才啟用
            // CLProtocol+重套線掃掉幀，也避免使用者在初始化中誤操作。
            btnLiveGrab.Enabled = false;
            btnLiveGrab.Text = "初始化中…";
            if (lblCamCount != null) { lblCamCount.Text = "相機: 初始化中…"; lblCamCount.BackColor = IecGray; }
            _liveCameraManager.OnHwReady += () =>
            {
                if (InvokeRequired) { if (!IsHandleCreated || IsDisposed || Disposing) return; SafeBeginInvoke(OnCamerasHwReady); return; }
                OnCamerasHwReady();
            };

            var panelClicker = new MultiClickDetector(
                300,
                new Size(SystemInformation.DoubleClickSize.Width, SystemInformation.DoubleClickSize.Width),
                MultiClickDistanceMode.Radius);
            camLiveMain.MouseDown += (s, e) =>
            {
                if (e.Button != MouseButtons.Left) return;
                int clicks = panelClicker.RegisterClick(e.Location);

                if (clicks == 2)
                {
                    if (_bgPreviewMainCanvas != null && _bgPreviewActive)
                        _bgPreviewMainCanvas.FitToScreen();
                    else if (_liveCameraManager.IsLiveGrabbing)
                        _liveCameraManager.ResetMainDisplayView();
                }
                else if (clicks >= 3)
                {
                    panelClicker.Consume();
                    if (_liveCameraManager.IsLiveGrabbing)
                        _liveCameraManager.SetPhysicalMagnification1x();
                    else if (_bgPreviewMainCanvas != null && _bgPreviewActive)
                        SetBgPreviewPhysicalMag1x();
                }
            };

            FormClosed += async (_, __) =>
            {
                // Closed 階段：統一 Dispose 路徑（Closing 已負責停止活動，這裡不重複 Stop）
                if (_ioGrabController != null)
                {
                    try { await _ioGrabController.StopAsync(); } catch { }
                    _ioGrabController.Dispose();
                    _ioGrabController = null;
                }
                FreePrecomputedColMeanBuffers();
                _liveCameraManager.FreeCameras();
                // 相機釋放後再 dispose CUDA pipeline（依賴關係安全；C2 修正）
                _inspectionService?.Dispose();
                _lightController?.Dispose();   _lightController = null;
                _retentionService?.Dispose();  _retentionService = null;
                _remoteCopyService?.Dispose(); _remoteCopyService = null;
            };

            // 程式啟動後自動分配相機（不 Grab），讓 lblCamCount 在按下【開始抓取】前就能顯示連線狀態。
            // 用 BeginInvoke 延後一個 UI 週期：讓最大化/layout/首幀 paint 先完成，再進 MIL 配置 + 啟動
            // CLProtocol，使視窗先變可互動（Codex 建議；不背景化以免動到 panel handle/timer/status UI）。
            Shown += (s, e) => BeginInvoke(new Action(AutoAllocateCameras));

            // commit 5b769f4 把 Live tab 加進 ProportionalScaler 後，camLiveMain 在 z-order 上層
            // 縮放時幾何疊到 chart 區、MIL window 吃掉 hit-test → chart click handler 完全不觸發。
            // 修法：Shown + Resize 後把 chart 提到 z-order 最上層，hit-test 順序就會正確。
            Shown    += (s, e) => BringLiveChartsToFront();
            Resize   += (s, e) => BringLiveChartsToFront();
        }

        private void BringLiveChartsToFront()
        {
            try
            {
                chartLiveColumn?.BringToFront();
                chartLiveRow?.BringToFront();
            }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[BringLiveChartsToFront] {ex.GetType().Name}: {ex.Message}"); }
        }

        /// <summary>
        /// 啟動時自動分配相機資源（不啟動 Grab）。
        /// 同時載入背景 .bin 與初始化 Global merge，使後續按【開始抓取】直接進入 ToggleGrab。
        /// </summary>
        private void AutoAllocateCameras()
        {
            if (_liveCameraManager == null || _liveCameraManager.IsAllocated) return;
            try
            {
                // 多相機相位量測 log（診斷）：設了路徑 → 每幀記 frame-start 硬體時戳 → Logs\phaselog-yyyyMMdd.csv。
                try
                {
                    string logsDir = _settings?.Storage?.LogsPath;
                    if (!string.IsNullOrEmpty(logsDir))
                    {
                        System.IO.Directory.CreateDirectory(logsDir);
                        MilGrabber.Core.MilCamera.PhaseLogPath =
                            System.IO.Path.Combine(logsDir, $"phaselog-{DateTime.Now:yyyyMMdd}.csv");
                        // 掉偵診斷 log（每 500ms 背景 Capture 記 frames/procMissed/grabMissed → 離線定位掉在哪層）
                        AniloxRoll.Monitor.UI.Presenters.LiveTelemetryPresenter.DropDiagLogPath =
                            System.IO.Path.Combine(logsDir, $"dropdiag-{DateTime.Now:yyyyMMdd_HHmmss}.csv");
                        // 參數變更 log（time,scope,cam,param,value → 對齊 _ticks.csv 掉偵時間，定位掉偵 vs 改參數）
                        ParamChangeLogPath =
                            System.IO.Path.Combine(logsDir, $"paramchange-{DateTime.Now:yyyyMMdd_HHmmss}.csv");
                    }
                }
                catch { }

                // max-buffer 模式已驗證不採用（2026-06-24）：grab 中拉高度真上限 ~12062（per-camera，板載 4 path 各自獨立），
                // 改走「高度一律 cap 到 MaxGrabHeightPx=12000」+ 安全的 buffer==source realloc。flag 維持預設 false。
                MilGrabber.Core.MilCamera.UseMaxHeightBuffers = false;

                _liveCameraManager.AllocateCameras(_settings.EnableMuraEnhance);
                LoadBackgroundBins();
                // 全域合圖（MIL 大 buffer alloc）延後到 CLProtocol 就緒後（OnCamerasHwReady）才建立：
                // 否則在 Shown 的 UI 執行緒上 alloc 會與剛啟動的背景 CLProtocol enable 搶 MIL 內部鎖，
                // UI 執行緒卡住直到 CLProtocol 釋放（~數秒）→ 視窗整段拖不動。
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"[AutoAllocateCameras] {ex.GetType().Name}: {ex.Message}");
            }
        }

        /// <summary>所有相機 CLProtocol 就緒（曝光/線掃已套）→ 更新相機數 + 解鎖「開始抓取」。</summary>
        private void OnCamerasHwReady()
        {
            if (IsDisposed || Disposing || _liveCameraManager == null) return;
            // CLProtocol 已就緒、背景不再佔 MIL 鎖 → 此時才建立全域合圖（從 AutoAllocateCameras 延後至此）。
            if (_settings.StitchMode == StitchMode.Global && !_liveCameraManager.IsGlobalMergeActive)
                _liveCameraManager.EnableGlobalMerge(
                    _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray());
            UpdateCamCountLabel(_liveCameraManager.ConnectedCameraCount, CameraCount);
            RefreshGrabButtonState();

            // CLProtocol 就緒後：把每張板的板載記憶體（總量/可用）標到 listViewHardware（grabber 記憶體大小）。
            // 記憶體是「每張板」共用（同板 channel 共池）→ 每個 unique System 顯示一列；同時 log 高度/線掃上限診斷。
            try
            {
                // 每張板（OwnerSystemKey）已配 buffer 台數（拔線不釋放→佔板載的台數），供顯示。
                var boardCamCount = new Dictionary<long, int>();
                foreach (var cam in _liveCameraManager.Cameras)
                {
                    if (cam == null || !cam.HasGrabBuffers) continue;
                    long key = cam.OwnerSystemKey;
                    boardCamCount[key] = (boardCamCount.TryGetValue(key, out var n) ? n : 0) + 1;
                }

                var seenSystems = new HashSet<long>();
                int boardIdx = 0;
                foreach (var cam in _liveCameraManager.Cameras)
                {
                    if (cam == null) continue;

                    // 每張板（System）只加一列板載記憶體（去重）：用量/總量 + 已配台數。存 item ref → telemetry timer 即時刷新（改參數後用量會變）。
                    if (listViewHardware != null && !IsDisposed && cam.HasGrabBuffers && seenSystems.Add(cam.OwnerSystemKey))
                    {
                        long key   = cam.OwnerSystemKey;
                        long total = cam.GetMemoryTotalMB();    // 板載總量（on-board，硬體固定）
                        long free  = cam.GetMemoryFreeMB();     // 即時可用
                        long used  = (total > 0 && free >= 0) ? total - free : -1;
                        int  nCam  = boardCamCount.TryGetValue(key, out var c) ? c : 1;
                        string val = (used >= 0) ? $"{used}/{total} MB" : $"{free} MB free";
                        var item = new ListViewItem(new[] { $"Grabber記憶體_板{boardIdx}（{nCam}台）", val });
                        listViewHardware.Items.Add(item);
                        _boardMemItems[key] = item;             // 存 ref 供 timer 更新
                        _boardMemInfo[key]  = (nCam, total);
                        boardIdx++;
                    }
                }

                // 診斷：log 各相機 GenICam Height feature 的合法範圍（Min/Max/Increment）。
                // 合法 grab 高度 = Min + k×Increment；判斷 9000 為何合法、8736 為何 stall（格點外）。
                foreach (var cam in _liveCameraManager.Cameras)
                    cam?.LogHeightFeatureInfo();
            }
            catch { }
        }

        /// <summary>btnLiveGrab 狀態唯一來源：依「相機就緒 / IO 連線 / 是否抓取中」決定顯示。
        /// 由 <see cref="OnCamerasHwReady"/> 與 <see cref="UpdateIoConnectionUi"/> 共同呼叫。
        /// 關鍵：相機就緒前一律「初始化中」，即使 IO 先連線也不顯示「IO 控制中」，
        /// 避免使用者誤以為系統已可操作（IO 是 TCP，通常比相機 CLProtocol 早就緒）。</summary>
        private void RefreshGrabButtonState()
        {
            if (IsDisposed || Disposing || btnLiveGrab == null) return;

            bool camReady      = _liveCameraManager?.AreCamerasHwReady ?? false;
            bool ioControlling = _ioGrabController != null && _ioGrabController.IsConnected && !_isIoSuspended;

            if (!camReady)
            {
                // 相機尚未就緒：一律「初始化中」（IO 即使已連線也不接管按鈕）
                btnLiveGrab.Enabled  = false;
                btnLiveGrab.Text     = "初始化中…";
                btnLiveGrab.BackColor = SystemColors.Control;
                btnLiveGrab.ForeColor = SystemColors.ControlText;
                return;
            }
            if (ioControlling)
            {
                btnLiveGrab.Enabled  = false;
                btnLiveGrab.Text     = "IO 控制中";
                btnLiveGrab.BackColor = IecBlue;
                btnLiveGrab.ForeColor = Color.White;
                return;
            }
            // 相機就緒、IO 未控制 → 可手動操作
            btnLiveGrab.Enabled  = true;
            btnLiveGrab.BackColor = SystemColors.Control;
            btnLiveGrab.ForeColor = SystemColors.ControlText;
            UpdateGrabButton(_liveCameraManager?.IsLiveGrabbing ?? false);
        }


        private void OnPresenterLogReported(string log)
        {
            Debug.WriteLine(log);
        }



        /// <summary>H3：debounce 統計重算 — 300ms 內合併多次 PropertyGrid 變更。</summary>
        private int _statsRefreshFailCount;
        private void ScheduleStatsRefresh()
        {
            if (_statsRefreshDebouncer == null)
            {
                _statsRefreshDebouncer = new Timer { Interval = 300 };
                _statsRefreshDebouncer.Tick += (_, __) =>
                {
                    _statsRefreshDebouncer.Stop();
                    try
                    {
                        _dataStatsPresenter?.RefreshStats();
                        _dataStatsPresenter?.RefreshPeriodCharts();
                        _statsRefreshFailCount = 0;
                    }
                    catch (Exception ex)
                    {
                        Trace.WriteLine($"[ScheduleStatsRefresh] {ex}");
                        // B-M2：連續失敗時通知 user（每 5 次彈一次，避免狂彈）
                        if (++_statsRefreshFailCount % 5 == 1)
                            MessageBox.Show($"統計刷新失敗（已失敗 {_statsRefreshFailCount} 次）：\n{ex.Message}",
                                "Data tab 統計", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }
                };
            }
            _statsRefreshDebouncer.Stop();
            _statsRefreshDebouncer.Start();
        }

        // Recipe property 名稱集（PropertyGrid wrapper aliases + InspectionRecipe.* 真名 + DisplayName）。
        // 注意：Q2 移除 RecipeChange reload 分支後目前**沒有 caller**，保留待「重算 .bin curve」實作後使用。
        private static readonly HashSet<string> RecipePropertyNames = new HashSet<string>(StringComparer.Ordinal)
        {
            // PropertyGrid wrapper aliases（實務上 PropertyGrid 改值送的就是這組）
            "dc_HessianMaxFactorV", "dd_HessianMaxFactorH",
            "db_Algorithm", "eb_RidgeDir",
            "ec_ErrorValueMeanV", "ed_ErrorValueMaxV",
            "ee_ErrorValueMeanH", "ef_ErrorValueMaxH",
            // InspectionRecipe.* 真名（程式碼直接改 Recipe 時用）
            nameof(InspectionRecipe.HessianMaxFactorV), "Hessian Max Factor V", "欄正規值",
            nameof(InspectionRecipe.HessianMaxFactorH), "Hessian Max Factor H", "列正規值",
            nameof(InspectionRecipe.ErrorValueMeanV),  "Error Value Mean V", "欄平均閾值",
            nameof(InspectionRecipe.ErrorValueMaxV),   "Error Value Max V",  "欄最大閾值",
            nameof(InspectionRecipe.ErrorValueMeanH),  "Error Value Mean H", "列平均閾值",
            nameof(InspectionRecipe.ErrorValueMaxH),   "Error Value Max H",  "列最大閾值",
            nameof(InspectionRecipe.Algorithm),        "去背演算法",
            nameof(InspectionRecipe.RidgeDir),         "Ridge 方向",
        };

        private bool _suppressGridSelChange;
        private void PropertyGridSettings_SelectedGridItemChanged(object sender, SelectedGridItemChangedEventArgs e)
        {
            if (_suppressGridSelChange) return;  // RefreshGridItem trick 暫時切 selection 不更新說明文字
            var item = e.NewSelection;
            helpRichText.Clear();
            if (item?.PropertyDescriptor == null) return;

            string title = string.IsNullOrEmpty(item.Label) ? item.PropertyDescriptor.DisplayName : item.Label;
            string desc = item.PropertyDescriptor.Description ?? string.Empty;

            var titleFont = new System.Drawing.Font(helpRichText.Font, System.Drawing.FontStyle.Bold);
            helpRichText.SelectionFont = titleFont;
            helpRichText.AppendText(title);
            helpRichText.SelectionFont = helpRichText.Font;
            if (!string.IsNullOrEmpty(desc))
            {
                helpRichText.AppendText(Environment.NewLine);
                helpRichText.AppendText(desc);
            }
        }

        // 序列化 OnSettingChanged：避免連點 chart click 觸發多個 reload 並行 race（Claude review A1）
        private readonly System.Threading.SemaphoreSlim _onSettingChangedSemaphore = new System.Threading.SemaphoreSlim(1, 1);

        /// <summary>
        /// L2 SettingsHub Changed event 的唯一訂閱者：所有 setting 變更的副作用都跑這個 switch。
        /// 來源不論：PropertyGrid（NotifyExternalChange）/ chart click（Set / SetBatch+inline）/ AutoDetect 回寫（Set）。
        /// 副作用順序：共用前段（chart 閾值、Live 設定、統計） → 個別 case dispatch（早退：AppRole）。
        /// SemaphoreSlim 序列化：連點時排隊處理，避免多個 reload 並行 race。
        /// </summary>
        private async void OnSettingChanged(AniloxRoll.Monitor.Settings.Services.SettingChange c)
        {
            await _onSettingChangedSemaphore.WaitAsync().ConfigureAwait(true);
            try
            {
                // 設定變更 intent（S0 通用）：使用者從 PropertyGrid 改＝ui: 前綴（孤兒判讀規則的主人）；
                // 程式化來源（自動掃描寫回等）＝set: 前綴（有主人但非使用者動作）。單一掛點蓋所有設定。
                // 帶新值（截 40 字防長值洗版）→ log 能還原「切到哪一檔」。
                string nv = c.NewValue?.ToString() ?? "null";
                if (nv.Length > 40) nv = nv.Substring(0, 40) + "…";
                FlowTrace.Log((c.Source == AniloxRoll.Monitor.Settings.Services.SettingSource.PropertyGrid
                    ? "ui:設定[" : "set:[") + c.Name + "]=" + nv);
                // ── 共用副作用（任何 setting 變更都跑） ────────────────────────
                // PropertyGrid 顯示同步：「程式碼路徑改值」時用精準 trick 重讀單 cell（不全 Refresh、不閃）。
                // PropertyGrid 自己改值已自我更新該 cell，不需要外部處理。
                if (c.Source == AniloxRoll.Monitor.Settings.Services.SettingSource.Programmatic)
                    RefreshGridItem(c.Name);
                _interactionHelper.HandleSettingsChanged();
                _liveCameraManager?.SetCaptureSettings(_settings);
                _reviewOverviewHelper?.SetThresholds(_settings.ErrorValueMeanV, _settings.ErrorValueMaxV);
                _liveOverviewHelper?.SetThresholds(_settings.ErrorValueMeanV, _settings.ErrorValueMaxV);
                _liveRowDisplay?.SetThresholds(_settings.ErrorValueMeanH, _settings.ErrorValueMaxH);
                _reviewRowDisplay?.SetThresholds(_settings.ErrorValueMeanH, _settings.ErrorValueMaxH);
                UpdateRowChartPitch();
                if (_stitchCoordinator?.IsStitchMode == true)
                {
                    _stitchCoordinator.UpdateStitchedOverviewChart();
                    _stitchCoordinator.RefreshCurrentCameraChartsForSettingsChange();
                }
                if (_liveCameraManager?.IsLiveGrabbing == true)
                    _inspectionLogService?.ForceWriteConfig(CsvConfigSnapshot.FromSettings(_settings));

                // ── 各 feature 副作用 dispatch（Wave3 選項1：邏輯搬各 feature partial；
                //    dispatcher 只「持鎖 + 跑共用前段 + 依序 fan-out」，不擁有 feature 細節）──────────
                if (HandleAppRoleSettingsChanged(c.Name)) return;  // 早退：機台角色（寫 app-mode.json）
                HandleLiveLayoutSettingsChanged(c.Name);           // 動態 LOD + OPS/Start 合圖佈局（Live.cs）
                HandleChartScaleSettingsChanged(c.Name);           // 檢測報表 Y 軸（Data.cs）
                HandleDataStatsSettingsChanged(c.Name);            // Data 曲線/統計重畫（僅檢測參數變更才跑，避免無關設定閃圖；Data.cs）
                HandleLightSettingsChanged(c.Name);                // 光源（HardwareStatus.cs）
                HandleIoSettingsChanged(c.Name);                   // IO IP/Port/型號/啟用 → 重啟 controller 立即生效（HardwareStatus.cs）
                await HandleEnhanceSettingsChanged(c.Name);        // 監控/回顧強化（Live.cs）
                HandleMuraPauseSettingsChanged(c.Name);            // IO 檢測暫停 LED（HardwareStatus.cs）
                HandleAlgorithmSettingsChanged(c.Name);            // 去背演算法（Background.cs）

                // 註：Recipe 變更（正規值/閾值/Ridge 方向）影響 PASS/FAIL + 閾值線 + 曲線坡度，
                //     已由共用前段（SetThresholds + UpdateStitchedOverviewChart）處理；不影響影像 bytes，無需 reload。
            }
            catch (Exception ex) { Trace.WriteLine($"[OnSettingChanged {c?.Name}] {ex}"); }
            finally { _onSettingChangedSemaphore.Release(); }
        }

        /// <summary>機台角色變更 → 寫 app-mode.json + 提示重開。回 true=已處理（dispatcher 早退、跳過其餘 feature）。</summary>
        private bool HandleAppRoleSettingsChanged(string name)
        {
            if (name != nameof(InspectionSettings.AppRole)) return false;
            if (_appMode == null) _appMode = new AppModeConfig();
            _appMode.Role = _settings.AppRole;
            _appMode.Save();
            MessageBox.Show("機台角色已儲存，重新開啟程式後生效。",
                "機台設定", MessageBoxButtons.OK, MessageBoxIcon.Information);
            return true;
        }

        // OPS/Start setting 名稱清單（用來判斷是不是「機台佈局」群組的 setting）
        private static readonly HashSet<string> OpsStartSettingNames = new HashSet<string>(StringComparer.Ordinal)
        {
            "ab_OpsCam1", "ac_OpsCam2", "ad_OpsCam3", "ae_OpsCam4", "af_OpsCam5", "ag_OpsCam6", "ah_OpsCam7",
            "bb_StartCam1", "bc_StartCam2", "bd_StartCam3", "be_StartCam4", "bf_StartCam5", "bg_StartCam6", "bh_StartCam7"
        };

        /// <summary>
        /// PropertyGrid 改值：setter 已寫 memory，這裡只負責把它導入 SettingsHub 走統一管線。
        /// </summary>
        private void _propertyGrid_PropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            try
            {
                string name = e?.ChangedItem?.PropertyDescriptor?.Name ?? string.Empty;
                object newVal = e?.ChangedItem?.Value;
                _settingsHub.NotifyExternalChange(name, e?.OldValue, newVal);
            }
            catch (Exception ex) { Trace.WriteLine($"[PropertyValueChanged] {ex}"); }
        }




    }
}
