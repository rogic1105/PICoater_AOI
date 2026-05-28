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
        [DllImport("gdi32.dll")] private static extern int GetDeviceCaps(IntPtr hdc, int index);
        [DllImport("user32.dll")] private static extern IntPtr GetDC(IntPtr hwnd);
        [DllImport("user32.dll")] private static extern int ReleaseDC(IntPtr hwnd, IntPtr hdc);

        // --- 核心服務 ---
        private readonly ImageRepository _imageRepository = new ImageRepository();
        private BatchInspectionService _inspectionService;

        // --- UI Helpers ---
        private DateTimeNavigator _dateTimeNavigator;
        private ThumbnailGridPresenter _galleryManager;
        private AniloxRollPresenter _presenter;
        private FormInteractionHelper _interactionHelper;
        private ColumnCurveChartHelper _reviewColumnChartHelper;
        private ColumnCurveChartHelper _liveColumnChartHelper;
        private ColumnCurveChartHelper _reviewOverviewHelper;
        private ColumnCurveChartHelper _liveOverviewHelper;
        private RowCurveChartHelper _liveRowChartHelper;
        private RowCurveChartHelper _reviewRowChartHelper;
        private LiveCameraManager _liveCameraManager;
        // Global merge 用：快取各相機 row curve 資料，合併後更新圖表
        private readonly Dictionary<int, float[]> _liveRowMeanCache = new Dictionary<int, float[]>();
        private readonly Dictionary<int, float[]> _liveRowMaxCache  = new Dictionary<int, float[]>();
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
        private bool IsStandardBgSubEnabled =>
            _settings?.Recipe?.Algorithm == BackgroundAlgorithm.StandardBgSub;

        private bool IsLightReadyForBg =>
            !(_settings?.LightEnabled == true) || (_lightController != null && _lightController.IsConnected);

        private bool _autoStartGrabAfterBg;

        private bool IsBgBinReady()
        {
            if (!IsStandardBgSubEnabled) return true;
            string bgDir = _settings.Storage.BackgroundPath;
            if (_liveCameraManager?.IsAllocated == true)
            {
                foreach (var cam in _liveCameraManager.Cameras)
                {
                    if (!cam.IsConnected) continue;
                    if (cam.FrameWidth <= 0) continue;
                    string binPath = Path.Combine(bgDir, CaptureFileNaming.BgBin(cam.FrameWidth, cam.CameraId));
                    if (!File.Exists(binPath)) return false;
                }
                return true;
            }
            return Directory.Exists(bgDir) && Directory.GetFiles(bgDir, CaptureFileNaming.BgGlob).Length > 0;
        }

        /// <summary>"v" = vertical ridge（預設），"h" = horizontal ridge。控制 Live 顯示方向。</summary>
        private string _liveDisplayDirection = "v";

        /// <summary>Data tab 變更 cbDataGrabId 後 cbReviewGrabId 已同步但 canvasMain 尚未更新；
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
        private bool _isMuraDetectPaused;
        private bool _isIoSuspended;

        // --- Review tab 拼接管理 ---
        private ReviewStitchCoordinator _stitchCoordinator;
        private PictureBox[] _cameraPanels;


        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);
            // Closing 階段：只「停止」非 UI 執行緒活動（避免 Handle 銷毀後它們還在 BeginInvoke）。
            // Dispose 留到 FormClosed 統一處理，避免雙路徑釋放重疊。
            try { if (_liveCameraManager?.IsLiveGrabbing == true) _liveCameraManager.StopGrab(); } catch { }
            try { _telemetryTimer?.Stop(); } catch { }
            try { _liveOverviewTimer?.Stop(); } catch { }
            try { _statsRefreshDebouncer?.Stop(); _statsRefreshDebouncer?.Dispose(); _statsRefreshDebouncer = null; } catch { }  // H3 + round-2 H3 補 Dispose
            try { _cleanupFlagWatcher?.Dispose(); _cleanupFlagWatcher = null; } catch { }  // M3: 10 秒輪詢提前停
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
            // 啟動 banner log — 用來驗證 user 跑的是不是新 build
            try
            {
                System.IO.File.AppendAllText(@"D:\Anilox\stitch-debug.log",
                    $"========== AniloxRoll.Monitor started at {DateTime.Now:yyyy-MM-dd HH:mm:ss.fff} ==========" + Environment.NewLine);
            }
            catch { }
            // 全域 mouse-down 攔截：記錄每次左鍵按下命中的控制項，用來診斷 Live chart click 失蹤。
            try { Application.AddMessageFilter(new GlobalMouseLogger()); } catch { }
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
                Directory.CreateDirectory(_settings.Storage.DcfDirPath);

                // 一次性遷移：舊版 D:\AniloxCaptures\{dcf,bg} → D:\Anilox\{Dcf,Bg}
                // 只在新目錄為空時遷移，避免覆蓋使用者已建立的新內容
                string rootDrive = Path.GetPathRoot(aniloxRoot)?.TrimEnd('\\') ?? "D:";
                string legacyCaptures = Path.Combine(rootDrive + "\\", "AniloxCaptures");
                if (Directory.Exists(legacyCaptures))
                {
                    MigrateLegacySubdir(Path.Combine(legacyCaptures, "dcf"), _settings.Storage.DcfDirPath);
                    MigrateLegacySubdir(Path.Combine(legacyCaptures, "bg"),  _settings.Storage.BackgroundPath);
                }

                // 檢查 DCF 檔是否存在；不存在時設旗標，lblCamCount 之後會顯示警語
                string dcfPath = _settings?.CameraParam?.DcfPath;
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
                    () => _appMode.LocalConfigFolder,
                    _retentionService);
                _cleanupFlagWatcher.Start();
            }
            else
            {
                // Inspection 模式：遠端複製 + IO + 光源
                _remoteCopyService = new RemoteCopyService(
                    getRemotePath: () => _settings?.RemotePath ?? string.Empty,
                    getLocalRoot:  () => _settings?.CaptureRootPath ?? string.Empty);
                InitIoController();
                InitLightController();
            }

            // 啟動時執行一次清理（雙模式共用）
            Task.Run(() => _retentionService.RunCleanup());
        }

        /// <summary>初始化 IO 連動：自動偵測連線，連上後以 DI START 控制 Grab。</summary>
        private void InitIoController()
        {
            if (!_settings.IoEnabled) return;

            _ioGrabController = new IoGrabController(_settings.IoModel);

            // 背景 Modbus 輪詢執行緒回 UI 更新；關閉時 Handle 已銷毀 → SafeBeginInvoke 守 guard 防 InvalidOperationException
            _ioGrabController.OnStartRequested += () => SafeBeginInvoke(IoStartGrab);

            _ioGrabController.OnStopRequested += () => SafeBeginInvoke(IoStopGrab);

            _ioGrabController.OnStateChanged += state => SafeBeginInvoke(() => UpdateIoStateLabel(state));

            _ioGrabController.OnConnectionChanged += connected => SafeBeginInvoke(() => UpdateIoConnectionUi(connected));

            _ioGrabController.OnIoUpdated += snapshot => SafeBeginInvoke(() => UpdateIoLeds(snapshot));

            // 背景嘗試連線（不阻塞 Form 顯示）
            _ = _ioGrabController.StartAsync(_settings.IoIp, _settings.IoPort);
        }

        private void InitLightController()
        {
            if (!_settings.LightEnabled) return;
            _lightController = new LightController();

            // 先試檢測設定的 COM，失敗則掃描所有 port
            string found = _lightController.AutoDetect(_settings.LightComPort, _settings.LightChannel);
            if (found == null)
            {
                System.Diagnostics.Trace.WriteLine("[Light] 光源控制器: NA（設定 " + _settings.LightComPort + " + 全 port 掃描均無回應）");
                _lightController.Dispose();
                _lightController = null;
                return;
            }

            // 掃描找到但非原設定 → 更新記錄的 COM（下次啟動直接命中）。
            // 用 SetBatch（save only no event）避免遞迴：Hub.Set 會 raise event → HandleLightSettingsChanged 重 Init → 又呼此 AutoDetect。
            if (!string.Equals(found, _settings.LightComPort, StringComparison.OrdinalIgnoreCase))
            {
                _settingsHub.SetBatch(s => s.LightComPort = found);
                RefreshGridItem(nameof(InspectionSettings.LightComPort));
            }
        }

        private void IoStartGrab()
        {
            if (_isIoSuspended) return;
            if (_liveCameraManager == null || _liveCameraManager.IsLiveGrabbing) return;
            if (IsStandardBgSubEnabled && !IsBgBinReady())
            {
                System.Diagnostics.Trace.TraceWarning("[IoStartGrab] StandardBgSub 無背景 bin，自動取得背景後接續 grab");
                _autoStartGrabAfterBg = true;
                btnGetBackground_Click(null, null);
                return;
            }
            btnCameraGrab_Click(null, null);
            _ = _ioGrabController?.NotifyGrabStarted();
        }

        private void IoStopGrab()
        {
            if (_isIoSuspended) return;
            if (_liveCameraManager == null || !_liveCameraManager.IsLiveGrabbing) return;
            btnCameraGrab_Click(null, null);
            _ = _ioGrabController?.NotifyGrabStopped();
        }

        private void LightTurnOn()
        {
            if (_lightController == null || !_lightController.IsConnected) return;
            _lightController.TurnOn(_settings.LightChannel, _settings.LightBrightness);
        }

        private void LightTurnOff()
        {
            if (_lightController == null || !_lightController.IsConnected) return;
            _lightController.TurnOff(_settings.LightChannel);
        }

        /// <summary>
        /// 光源 PropertyGrid 變更 → 立即生效：
        /// - LightEnabled false→true：啟動偵測；true→false：關閉連線
        /// - COM Port / 通道變更：重新偵測
        /// - 亮度變更：立即套用到硬體（若正在點燈，連同 TurnOn 更新輸出）
        /// </summary>
        private void HandleLightSettingsChanged(string changedPropertyName)
        {
            switch (changedPropertyName)
            {
                case nameof(InspectionSettings.LightEnabled):
                    if (_settings.LightEnabled)
                    {
                        if (_lightController == null) InitLightController();
                    }
                    else
                    {
                        _lightController?.Dispose();
                        _lightController = null;
                    }
                    break;

                case nameof(InspectionSettings.LightComPort):
                case nameof(InspectionSettings.LightChannel):
                    if (_settings.LightEnabled)
                    {
                        _lightController?.Dispose();
                        _lightController = null;
                        InitLightController();
                    }
                    break;

                case nameof(InspectionSettings.LightBrightness):
                    if (_lightController != null && _lightController.IsConnected)
                        _lightController.SetBrightness(_settings.LightChannel, _settings.LightBrightness);
                    UpdateLightConnLabel();
                    break;
            }
        }

        private void UpdateIoStateLabel(IoState state)
        {
            if (_isIoSuspended) return;
            string text;
            Color bgColor;
            switch (state)
            {
                case IoState.Idle:      text = "Idle 待機"; bgColor = IecGreen;  break;
                case IoState.Running:   text = "取像中";   bgColor = IecBlue;   break;
                case IoState.Stopping:  text = "停止中";   bgColor = IecYellow; break;
                case IoState.Faulted:   text = "設備離線"; bgColor = IecRed;    break;
                case IoState.CommLost:  text = "通訊中斷"; bgColor = IecRed;    break;
                case IoState.Closed:    text = "已關閉";   bgColor = IecGray;   break;
                default:                text = "未連線";   bgColor = IecGray;   break;  // Disconnected
            }
            lblIoState.Text = $"〔{text}〕";
            lblIoState.BackColor = bgColor;
        }

        private void UpdateIoConnectionUi(bool connected)
        {
            if (_isIoSuspended) return;
            if (connected)
            {
                lblIoConn.Text = "● IO 已連線";
                lblIoConn.BackColor = IecGreen;
                btnCameraGrab.Enabled = false;
                btnCameraGrab.Text = "IO 控制中";
                btnCameraGrab.BackColor = IecBlue;
                btnCameraGrab.ForeColor = Color.White;
            }
            else
            {
                lblIoConn.Text = "● IO 離線";
                lblIoConn.BackColor = IecGray;
                btnCameraGrab.Enabled = true;
                UpdateGrabButton(_liveCameraManager?.IsLiveGrabbing ?? false);
                btnCameraGrab.BackColor = SystemColors.Control;
                btnCameraGrab.ForeColor = SystemColors.ControlText;
            }
        }

        private void UpdateLightConnLabel()
        {
            if (_settings == null || !_settings.LightEnabled)
            {
                lblLightConn.Text = "● 光源 停用";
                lblLightConn.BackColor = IecGray;
                return;
            }
            if (_lightController != null && _lightController.IsConnected)
            {
                lblLightConn.Text = $"● 光源 已連線 ({_settings.LightBrightness})";
                lblLightConn.BackColor = IecGreen;
            }
            else
            {
                lblLightConn.Text = "● 光源 離線";
                lblLightConn.BackColor = IecGray;
            }

            UpdateStandardBgSubLockState();
        }

        private int _storageProbeTickCounter;
        private volatile bool _storageProbeInFlight;
        private int _lightProbeTickCounter;
        private volatile bool _lightProbeInFlight;

        private void UpdateStorageConnLabel(bool? connected)
        {
            string path = _settings?.RemotePath ?? string.Empty;
            if (string.IsNullOrWhiteSpace(path))
            {
                lblStorageConn.Text = "● 儲存電腦 停用";
                lblStorageConn.BackColor = IecGray;
                return;
            }
            if (connected == true)
            {
                lblStorageConn.Text = "● 儲存電腦 已連線";
                lblStorageConn.BackColor = IecGreen;
            }
            else if (connected == false)
            {
                lblStorageConn.Text = "● 儲存電腦 離線";
                lblStorageConn.BackColor = IecRed;
            }
            // connected == null：保留上次結果（probe 還沒回來）
        }

        /// <summary>
        /// 由 TelemetryTimer_Tick 每 500ms 呼叫。光源每 5 秒背景 probe 一次（SerialPort.IsOpen 偵測不到拔線，
        /// 必須實際送命令驗證）；儲存機每 5 秒背景 probe 一次（UNC Directory.Exists 可能阻塞，不可在 UI thread）。
        /// </summary>
        private void UpdateConnectionStatusLabels()
        {
            if (_appMode?.Role == MachineRole.Storage)
            {
                if (_storageDiskFreeRow != null)
                {
                    try
                    {
                        string root = GetStorageRetentionRoot();
                        if (!string.IsNullOrWhiteSpace(root))
                        {
                            var di = new System.IO.DriveInfo(
                                System.IO.Path.GetPathRoot(System.IO.Path.GetFullPath(root)));
                            double freeGb  = di.AvailableFreeSpace / (1024.0 * 1024 * 1024);
                            double totalGb = di.TotalSize           / (1024.0 * 1024 * 1024);
                            _storageDiskFreeRow.SubItems[1].Text = $"{freeGb:F1} / {totalGb:F1} GB";
                        }
                    }
                    catch { }
                }
                return;
            }

            // Grab watchdog：取像中超過 30 秒沒有 result callback → 觸發循環儲存
            if (_liveCameraManager?.IsLiveGrabbing == true &&
                _lastGrabEventTime != DateTime.MinValue &&
                (DateTime.UtcNow - _lastGrabEventTime).TotalSeconds > 30)
            {
                _lastGrabEventTime = DateTime.UtcNow;
                Task.Run(() => _retentionService?.RunCleanup());
            }

            // 光源：先同步更新（用 IsConnected 快取結果），再 2 秒背景實測 / 重連
            // （Probe 用 TryEnter，與取像時 SendCommand 不會競爭，可放心高頻）
            UpdateLightConnLabel();
            if (++_lightProbeTickCounter >= 4)
            {
                _lightProbeTickCounter = 0;
                if (_settings != null && _settings.LightEnabled && !_lightProbeInFlight)
                {
                    _lightProbeInFlight = true;
                    int channel = _settings.LightChannel;
                    string preferredPort = _settings.LightComPort;
                    var lc = _lightController;
                    System.Threading.Tasks.Task.Run(() =>
                    {
                        try
                        {
                            if (lc != null && lc.IsConnected)
                            {
                                // 已連線 → 實測（拔線會被 Probe 偵測，內部關 port）
                                lc.Probe(channel);
                            }
                            else
                            {
                                // 未連線 → 嘗試重連（背景 AutoDetect，成功才接管欄位）
                                var fresh = new LightController();
                                string found = fresh.AutoDetect(preferredPort, channel);
                                if (found != null && !IsDisposed && !Disposing)
                                {
                                    try
                                    {
                                        BeginInvoke(new Action(() =>
                                        {
                                            if (_settings != null && _settings.LightEnabled)
                                            {
                                                _lightController?.Dispose();
                                                _lightController = fresh;
                                                if (!string.Equals(found, _settings.LightComPort, StringComparison.OrdinalIgnoreCase))
                                                {
                                                    _settingsHub.SetBatch(s => s.LightComPort = found);
                                                    RefreshGridItem(nameof(InspectionSettings.LightComPort));
                                                }
                                            }
                                            else
                                            {
                                                fresh.Dispose();
                                            }
                                        }));
                                    }
                                    catch (InvalidOperationException) { fresh.Dispose(); }
                                }
                                else
                                {
                                    fresh.Dispose();
                                }
                            }
                        }
                        catch { /* Probe/AutoDetect 內已處理例外，這裡保險 */ }
                        finally { _lightProbeInFlight = false; }

                        if (IsDisposed || Disposing) return;
                        try { BeginInvoke(new Action(UpdateLightConnLabel)); }
                        catch (InvalidOperationException) { }
                    });
                }
            }

            // 儲存機：每 5 秒背景 probe UNC 路徑
            if (++_storageProbeTickCounter < 10) return;
            _storageProbeTickCounter = 0;

            string path = _settings?.RemotePath ?? string.Empty;
            if (string.IsNullOrWhiteSpace(path))
            {
                UpdateStorageConnLabel(null);
                return;
            }
            if (_storageProbeInFlight) return;
            _storageProbeInFlight = true;

            System.Threading.Tasks.Task.Run(() =>
            {
                bool ok;
                try { ok = System.IO.Directory.Exists(path); }
                catch { ok = false; }
                finally { _storageProbeInFlight = false; }

                if (IsDisposed || Disposing) return;
                try { BeginInvoke(new Action<bool?>(UpdateStorageConnLabel), (bool?)ok); }
                catch (InvalidOperationException) { }
            });
        }

        private void UpdateIoLeds(IoSnapshot io)
        {
            if (_isIoSuspended) return;
            SetIoLed(lblIoDiAlive,   io.DiNakanAlive);
            SetIoLed(lblIoDiStart,   io.DiInspectStart);
            SetIoLed(lblIoDoPcAlive, io.DoPcAlive);
            UpdateMuraLed(io.DoMuraDetected);
            SetIoLed(lblIoDoPcBusy,  io.DoPcInspect);
        }

        private static void SetIoLed(Label lbl, bool on)
        {
            string[] parts = lbl.Text.Split(new[] { "\r\n" }, StringSplitOptions.None);
            string id   = parts[0].TrimStart('◎', '×', ' ');
            string name = parts.Length > 1 ? parts[1] : "";
            lbl.Text = (on ? "◎ " : "× ") + id + "\r\n" + name;
            lbl.BackColor = on ? IecGreen : IecDarkGray;
        }

        private void UpdateMuraLed(bool doMuraOn)
        {
            if (_isMuraDetectPaused)
            {
                lblIoDoMura.BackColor = IecYellow;
                lblIoDoMura.ForeColor = Color.Black;
                lblIoDoMura.Text = "⏸ DO1\r\nMURA_DET";
            }
            else
            {
                lblIoDoMura.BackColor = doMuraOn ? IecGreen : IecDarkGray;
                lblIoDoMura.ForeColor = Color.White;
                lblIoDoMura.Text = (doMuraOn ? "◎ " : "× ") + "DO1\r\nMURA_DET";
            }
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
                _imageRepository, cbDate, cbTime);

            _cameraPanels = new PictureBox[] {
                pbCam1, pbCam2, pbCam3, pbCam4, pbCam5, pbCam6, pbCam7
            };
            _galleryManager = new ThumbnailGridPresenter();
            _galleryManager.Initialize(_cameraPanels);

            _presenter = new AniloxRollPresenter(
                _imageRepository, _inspectionService, _dateTimeNavigator, _galleryManager);

            _reviewColumnChartHelper = new ColumnCurveChartHelper(this.chartMuraVertical);
            _reviewColumnChartHelper.SetOps(_settings.Cam1_Ops);
            _reviewColumnChartHelper.SetThresholds(_settings.ErrorValueMeanV, _settings.ErrorValueMaxV);

            _liveColumnChartHelper = new ColumnCurveChartHelper(this.muraChartVerticalLive);
            _liveColumnChartHelper.SetOps(_settings.Cam1_Ops);
            _liveColumnChartHelper.SetThresholds(_settings.ErrorValueMeanV, _settings.ErrorValueMaxV);

            _reviewOverviewHelper = new ColumnCurveChartHelper(this.chartOverview);
            _reviewOverviewHelper.SetThresholds(_settings.ErrorValueMeanV, _settings.ErrorValueMaxV);
            if (chartOverview.ChartAreas.Count > 0)
                chartOverview.ChartAreas[0].AxisX.ScaleView.Zoomable = false;

            _liveOverviewHelper = new ColumnCurveChartHelper(this.chartLiveOverview);
            _liveOverviewHelper.SetThresholds(_settings.ErrorValueMeanV, _settings.ErrorValueMaxV);
            if (chartLiveOverview.ChartAreas.Count > 0)
                chartLiveOverview.ChartAreas[0].AxisX.ScaleView.Zoomable = false;

            _liveRowChartHelper = new RowCurveChartHelper(this.muraChartHorizontalLive);
            _liveRowChartHelper.SetThresholds(_settings.ErrorValueMeanH, _settings.ErrorValueMaxH);

            _reviewRowChartHelper = new RowCurveChartHelper(this.chartMuraHorizontal);
            _reviewRowChartHelper.SetThresholds(_settings.ErrorValueMeanH, _settings.ErrorValueMaxH);

            UpdateRowChartPitch();

            // Review tab chart 點選（統一語意：點 chart 等於設定「目標 StitchMode + 是否強化 + 方向」）：
            //   chartMuraVertical：
            //     同 mode (Vertical) → SwitchRidgeDirection("v") 切換強化方向
            //     不同 mode (Global) → 切回 Vertical（強化中也適用，順便關 enhance）
            //   chartOverview：對稱（同/不同 mode 行為對調）
            //   chartMuraHorizontal：永遠 toggle ridge dir = "h"
            chartMuraVertical.MouseClick += (s, e) =>
            {
                UiActionLogger.SetSource("chartMuraVertical.Click");
                LogClick("chartMuraVertical.MouseClick", e);
                if (_settings?.StitchMode == StitchMode.Vertical) SwitchRidgeDirection("v");
                else if (_settings?.StitchMode == StitchMode.Global) _ = SwitchReviewStitchModeAndDisableEnhance(StitchMode.Vertical);
            };
            chartOverview.MouseClick += (s, e) =>
            {
                UiActionLogger.SetSource("chartOverview.Click");
                LogClick("chartOverview.MouseClick", e);
                if (_settings?.StitchMode == StitchMode.Global) SwitchRidgeDirection("v");
                else if (_settings?.StitchMode == StitchMode.Vertical) _ = SwitchReviewStitchModeAndDisableEnhance(StitchMode.Global);
            };
            chartMuraHorizontal.MouseClick += (s, e) =>
            {
                UiActionLogger.SetSource("chartMuraHorizontal.Click");
                UiActionLogger.RecordViewOnly("chartMuraHorizontal.Click");
                SwitchRidgeDirection("h");
            };

            // Live tab chart 點選（同 Review tab 語意，只是底層 apply 函式不同）：
            //   muraChartVerticalLive：
            //     同 mode (Vertical) → SwitchLiveDisplayDirection("v")
            //     不同 mode (Global) → SwitchStitchModeWithEnhanceSequence(Vertical)（內含關 enhance）
            //   chartLiveOverview：對稱
            muraChartVerticalLive.MouseClick += (s, e) =>
            {
                UiActionLogger.SetSource("muraChartVerticalLive.Click");
                LogClick("muraChartVerticalLive.MouseClick", e);
                if (_settings?.StitchMode == StitchMode.Vertical) SwitchLiveDisplayDirection("v");
                else if (_settings?.StitchMode == StitchMode.Global) _ = SwitchStitchModeWithEnhanceSequence(StitchMode.Vertical);
            };
            chartLiveOverview.MouseClick += (s, e) =>
            {
                UiActionLogger.SetSource("chartLiveOverview.Click");
                LogClick("chartLiveOverview.MouseClick", e);
                if (_settings?.StitchMode == StitchMode.Global) SwitchLiveDisplayDirection("v");
                else if (_settings?.StitchMode == StitchMode.Vertical) _ = SwitchStitchModeWithEnhanceSequence(StitchMode.Global);
            };
            muraChartHorizontalLive.MouseClick += (s, e) =>
            {
                UiActionLogger.SetSource("muraChartHorizontalLive.Click");
                UiActionLogger.RecordViewOnly("muraChartHorizontalLive.Click");
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
                Canvas           = canvasMain,
                ButtonsToLock    = new Button[] { btnSelectFolder },
                ThumbnailCache   = _thumbnailCache,
                Presenter        = _presenter,
                InspectionService = _inspectionService,
                ImageRepository  = _imageRepository,
                TimeNavigator    = _dateTimeNavigator,
                GalleryManager   = _galleryManager,
                ColumnChartHelper  = _reviewColumnChartHelper,
                Settings         = _settings,
                StatusLabel      = lblPixelInfo,
                CameraPanels     = _cameraPanels,
                RowChartHelper = _reviewRowChartHelper,
            });
            _interactionHelper.ApplySettingsToService();

            _stitchCoordinator = new ReviewStitchCoordinator(new ReviewStitchContext
            {
                Canvas                    = canvasMain,
                ChartOverview             = chartOverview,
                ChartMuraVertical         = chartMuraVertical,
                ChartMuraHorizontal       = chartMuraHorizontal,
                InteractionHelper         = _interactionHelper,
                ColumnChartHelper         = _reviewColumnChartHelper,
                RowChartHelper            = _reviewRowChartHelper,
                OverviewHelper            = _reviewOverviewHelper,
                GalleryManager            = _galleryManager,
                InspectionService         = _inspectionService,
                ImageRepository           = _imageRepository,
                DataStatsPresenter        = _dataStatsPresenter,
                Settings                  = _settings,
                DateTimeNavigator         = _dateTimeNavigator,
                CameraCount               = CameraCount,
            });

            _stitchCoordinator.StitchedCurveUpdated += (mean, max, ops, pos, errMean, errMax) =>
                _dataStatsPresenter?.SyncMuraProfileFromReview(mean, max, ops, pos, errMean, errMax);

            _presenter.BusyStateChanged += _interactionHelper.SetUiLoadingState;
            _presenter.LogReported      += OnPresenterLogReported;
            _galleryManager.SelectionChanged += idx =>
            {
                if (_stitchCoordinator.IsGlobalMerged || _stitchCoordinator.IsPeriodMerged)
                {
                    PanCanvasToReviewCameraCenter(idx);
                    return;
                }
                if (_stitchCoordinator.IsStitchMode)
                    _stitchCoordinator.ShowStitchedCameraInCanvas(idx);
                else
                    _interactionHelper.OnGallerySelectionChanged(idx);
            };

            _dateTimeNavigator.PeriodSelectionChanged += _presenter.UpdatePeriodNavigationState;
            _dateTimeNavigator.PeriodSelectionChanged += () =>
            {
                var current = _dateTimeNavigator.GetCurrentPeriodOrDefault(DateTime.MinValue);
                if (current != DateTime.MinValue) _dataStatsPresenter.SyncGrabIdFromTime(current);
            };
            _dateTimeNavigator.PeriodSelectionChanged += OnPeriodComboChanged;
            _presenter.PeriodNavigationStateChanged   += (canLast, canNext) =>
            {
                btnPeriodPrev.Enabled = canLast;
                btnPeriodNext.Enabled = canNext;
            };
            _presenter.UpdatePeriodNavigationState();

            canvasMain.StatusChanged += _interactionHelper.UpdateCanvasInfo;
            canvasMain.StatusChanged += UpdateSelectedReviewCamFromViewCenter;
            canvasMain.EdgeReached   += _interactionHelper.NavigateCamera;
            var canvasClicker = new MultiClickDetector();
            canvasMain.MouseDown += (s, e) =>
            {
                if (e.Button != MouseButtons.Left) return;
                int clicks = canvasClicker.RegisterClick(e.Location);
                if (clicks == 1) UiActionLogger.RecordViewOnly("canvasMain.Drag");
                if (clicks == 2)
                {
                    UiActionLogger.RecordViewOnly("canvasMain.DoubleClick");
                    if (canvasMain.Image != null && !IsCanvasFitToScreen())
                    {
                        canvasMain.FitToScreen();
                        canvasClicker.Consume();   // 歸零，防止下一下誤觸三擊
                    }
                }
                else if (clicks >= 3)
                {
                    canvasClicker.Consume();
                    _interactionHelper?.SetCanvasPhysicalMag1x(e.Location);
                }
            };

            UpdateLiveDirectionVisual();
            UpdateRidgeDirectionVisual(null); // dir=null：無強化橘框，底色依 StitchMode 上色
        }

        /// <summary>相機層：LiveCameraManager 與 FormClosed 清理。</summary>
        private void InitCameraLayer()
        {
            _liveCameraManager = new LiveCameraManager(
                this,
                new[] { panelLiveCam1, panelLiveCam2, panelLiveCam3,
                        panelLiveCam4, panelLiveCam5, panelLiveCam6, panelLiveCam7 },
                panelMainDisplay,
                pixelText => { if (lblPixelInfo != null) lblPixelInfo.Text = pixelText; }
            );
            _liveCameraManager.SetCaptureSettings(_settings);
            _liveCameraManager.OnFilesSaved = files => _remoteCopyService?.EnqueueFiles(files);
            _liveCameraManager.OnInspectionResult += OnCameraInspectionResult;
            btnGetBackground.Click += btnGetBackground_Click;
            btnViewBackground.Click += btnViewBackground_Click;
            UpdateStandardBgSubLockState();
            _liveCameraManager.OnLiveCurveData      += OnLiveCurveData;
            _liveCameraManager.OnLiveRowCurveData   += OnLiveRowCurveData;
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
                if (InvokeRequired) { if (!IsHandleCreated || IsDisposed || Disposing) return; BeginInvoke(new Action<int, int>(UpdateCamCountLabel), connected, expected); return; }
                UpdateCamCountLabel(connected, expected);
            };

            var panelClicker = new MultiClickDetector();
            panelMainDisplay.MouseDown += (s, e) =>
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

            // 程式啟動後自動分配相機（不 Grab），讓 lblCamCount 在按下【開始抓取】前就能顯示連線狀態
            Shown += (s, e) => AutoAllocateCameras();

            // commit 5b769f4 把 Live tab 加進 ProportionalScaler 後，panelMainDisplay 在 z-order 上層
            // 縮放時幾何疊到 chart 區、MIL window 吃掉 hit-test → chart click handler 完全不觸發。
            // 修法：Shown + Resize 後把 chart 提到 z-order 最上層，hit-test 順序就會正確。
            Shown    += (s, e) => BringLiveChartsToFront();
            Resize   += (s, e) => BringLiveChartsToFront();
        }

        private void BringLiveChartsToFront()
        {
            try
            {
                chartLiveOverview?.BringToFront();
                muraChartVerticalLive?.BringToFront();
                muraChartHorizontalLive?.BringToFront();
                LogClick("BringLiveChartsToFront() called");
            }
            catch (Exception ex) { LogClick("BringLiveChartsToFront throw: " + ex.Message); }
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
                _liveCameraManager.AllocateCameras(_settings.EnableMuraEnhance);
                LoadBackgroundBins();
                if (_settings.StitchMode == StitchMode.Global)
                    _liveCameraManager.EnableGlobalMerge(
                        _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray());
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"[AutoAllocateCameras] {ex.GetType().Name}: {ex.Message}");
            }
        }


        private void OnPresenterLogReported(string log)
        {
            Debug.WriteLine(log);
        }

        // ==========================================
        // --- 相機按鈕事件 ---
        // ==========================================

        private async void btnCameraGrab_Click(object sender, EventArgs e)
        {
            // 背景預覽中按 Grab → 先清除預覽並 Free，讓 MIL 能重新初始化
            if (_bgPreviewActive)
            {
                ClearBackgroundPreview(restoreMilDisplay: true);
                _liveCameraManager.FreeCameras();
                _telemetryPresenter?.ResetAll();
            }

            bool wasGrabbing = _liveCameraManager.IsLiveGrabbing;

            // 啟動路徑：先亮燈 → 等光源穩定 → 再開始 grab
            if (!wasGrabbing)
            {
                LightTurnOn();
                int warmup = _settings?.LightWarmupMs ?? 0;
                if (warmup > 0) await Task.Delay(warmup);
            }

            if (!_liveCameraManager.IsAllocated)
            {
                try
                {
                    _liveCameraManager.EnsureAllocatedAndToggleGrab(_settings.EnableMuraEnhance);
                    LoadBackgroundBins();

                    // 初次分配即為 Global 模式 → 立即啟用即時合圖
                    if (_settings.StitchMode == StitchMode.Global)
                        _liveCameraManager.EnableGlobalMerge(
                            _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray());
                }
                catch (Exception ex)
                {
                    LightTurnOff();
                    MessageBox.Show($"相機配置失敗: {ex.Message}", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
            }
            else
            {
                _liveCameraManager.ToggleGrab();
            }

            // 剛從「未抓取」→「抓取中」：分配新的抓圖編號（燈已在上方開啟）
            if (!wasGrabbing && _liveCameraManager.IsLiveGrabbing)
            {
                _currentGrabId = _inspectionLogService.NextGrabId();
            }

            // 剛從「抓取中」→「停止」：關燈 + 觸發循環儲存 + 通知儲存機清理
            if (wasGrabbing && !_liveCameraManager.IsLiveGrabbing)
            {
                LightTurnOff();
                TriggerRetentionAndFlagAsync();
            }

            UpdateGrabButton(_liveCameraManager.IsLiveGrabbing);
        }

        /// <summary>
        /// 相機存檔後回呼（MIL 執行緒，非 UI 執行緒）。
        /// EnableAutoCapture=true 且抓取中時才會觸發。
        /// </summary>
        private void OnCameraInspectionResult(int camId, string fileNameNoExt, float meanPeak, float maxPeak)
        {
            if (string.IsNullOrEmpty(_currentGrabId)) return;
            int idx = camId - 1;
            if (_inspectionLogService != null)
            {
                // OnCameraInspectionResult 的 meanPeak/maxPeak 為 V 方向（pipeline 主處理方向），用 V 閾值記錄
                _inspectionLogService.AppendRecord(
                    _currentGrabId,
                    fileNameNoExt,
                    meanPeak,
                    maxPeak,
                    _settings.ErrorValueMeanV,
                    _settings.ErrorValueMaxV,
                    idx >= 0 && idx < _settings.Acquisition.CameraGrabHeight.Length
                        ? _settings.Acquisition.CameraGrabHeight[idx] : 0,
                    idx >= 0 && idx < _settings.Acquisition.CameraLineRateHz.Length
                        ? _settings.Acquisition.CameraLineRateHz[idx] : 0,
                    idx >= 0 && idx < _settings.Acquisition.CameraExposureTimeUs.Length
                        ? _settings.Acquisition.CameraExposureTimeUs[idx] : 0,
                    CsvConfigSnapshot.FromSettings(_settings));

                // CSV 寫完後排入遠端複製佇列（CSV 在 month 目錄，不在 OnFilesSaved 的 day 目錄）
                string csvPath = _inspectionLogService.LastCsvPath;
                if (!string.IsNullOrEmpty(csvPath))
                    _remoteCopyService?.EnqueueFile(csvPath);
            }

            // IO MURA 信號：任一相機超過閾值即通知
            if (_ioGrabController?.IsConnected == true)
            {
                // meanPeak/maxPeak 為 V 方向，按 V 閾值判定
                bool isMura = meanPeak > _settings.ErrorValueMeanV || maxPeak > _settings.ErrorValueMaxV;
                if (isMura) _ = _ioGrabController.NotifyMuraDetected();
            }

            // 抓圖計數器 + watchdog 時間戳（Inspection 模式）
            if (_appMode?.Role != MachineRole.Storage)
            {
                _lastGrabEventTime = DateTime.UtcNow;
                int count = System.Threading.Interlocked.Increment(ref _completedGrabCount);
                if (count % 10 == 0)
                    TriggerRetentionAndFlagAsync();
            }
        }

        private void TriggerRetentionAndFlagAsync()
        {
            Task.Run(() => _retentionService?.RunCleanup());
            WriteFlagToRemoteAsync();
        }

        private void WriteFlagToRemoteAsync()
        {
            // JSON 有設定就用，否則從 RemotePath 推算（同 IP，固定 AniloxConfig share）
            string configPath = _settings?.RemoteConfigPath ?? string.Empty;
            if (string.IsNullOrWhiteSpace(configPath))
                configPath = DeriveFlagSharePath(_settings?.RemotePath);
            if (string.IsNullOrWhiteSpace(configPath)) return;

            Task.Run(() =>
            {
                try
                {
                    string flagPath = Path.Combine(configPath, "cleanup-request.flag");
                    File.WriteAllText(flagPath, DateTime.UtcNow.ToString("O"),
                        System.Text.Encoding.UTF8);
                }
                catch (Exception ex)
                {
                    Trace.TraceWarning($"[RetentionFlag] 寫旗標失敗: {ex.Message}");
                }
            });
        }

        private void ApplyStorageModeUi()
        {
            if (_appMode?.Role != MachineRole.Storage) return;

            tabMain.TabPages.Remove(tabPageLiveView);
            tabControlRight.TabPages.Remove(tabPageCamera);

            // PropertyGrid：隱藏 IO / 相機 / 光源三個大類
            TypeDescriptor.AddProvider(
                new StorageModeSettingsFilter(TypeDescriptor.GetProvider(_settings)), _settings);
            propertyGridSettings.Refresh();

            lblCamCount.Visible      = false;
            lblStorageConn.Visible   = false;

            lblIoState.Visible    = false;
            lblIoConn.Visible     = false;
            lblLightConn.Visible   = false;
            lblIoDiAlive.Visible   = false;
            lblIoDiStart.Visible   = false;
            lblIoDoPcAlive.Visible = false;
            lblIoDoMura.Visible    = false;
            lblIoDoPcBusy.Visible  = false;
        }

        // \\server\share → \\server\AniloxConfig（cleanup-request.flag 目標）
        private static string DeriveFlagSharePath(string remotePath)
        {
            if (string.IsNullOrWhiteSpace(remotePath)) return "";
            var parts = remotePath.TrimStart('\\').Split('\\');
            return parts.Length < 1 || string.IsNullOrEmpty(parts[0])
                ? "" : $@"\\{parts[0]}\AniloxConfig";
        }

        private string GetStorageRetentionRoot()
        {
            if (_appMode?.Role == MachineRole.Storage &&
                !string.IsNullOrWhiteSpace(_appMode.StorageFolderPath))
                return _appMode.StorageFolderPath;
            return _settings?.CaptureRootPath ?? string.Empty;
        }

        /// <summary>
        /// Live 曲線閾值判斷（callback 執行緒呼叫）。
        /// direction: "v"=垂直, "h"=水平；依 CheckLiveMura 設定的「檢測方向」決定是否觸發 DO1。
        /// 陣列為 0-255，閾值為 0-1，取陣列 max 後除以 255 比較。
        /// </summary>
        private void CheckLiveMura(float[] meanArr, float[] maxArr, string direction)
        {
            if (_isMuraDetectPaused) return;
            if (_ioGrabController?.IsConnected != true) return;
            if (_settings == null) return;
            if (!_liveCameraManager.IsLiveGrabbing) return;

            var ridgeDir = _settings.RidgeDir;
            if (direction == "v" && ridgeDir == RidgeDirection.Horizontal) return;
            if (direction == "h" && ridgeDir == RidgeDirection.Vertical)   return;

            float meanPeak = 0f, maxPeak = 0f;
            if (meanArr != null) { for (int i = 0; i < meanArr.Length; i++) if (meanArr[i] > meanPeak) meanPeak = meanArr[i]; }
            if (maxArr  != null) { for (int i = 0; i < maxArr.Length;  i++) if (maxArr[i]  > maxPeak)  maxPeak  = maxArr[i];  }
            meanPeak /= 255f;
            maxPeak  /= 255f;

            // 依 direction 用對應方向閾值
            float thMean = direction == "h" ? _settings.ErrorValueMeanH : _settings.ErrorValueMeanV;
            float thMax  = direction == "h" ? _settings.ErrorValueMaxH  : _settings.ErrorValueMaxV;

            if (meanPeak > thMean || maxPeak > thMax)
            {
                // fire-and-forget; 寫入失敗不應影響取像流程
                _ = _ioGrabController.NotifyMuraDetected().ContinueWith(
                    t => { /* swallow — PollTick 會偵測真正的 CommLost */ },
                    TaskContinuationOptions.OnlyOnFaulted);
            }
        }

        private void OnLiveCurveData(int camId, float[] meanArr, float[] maxArr)
        {
            // 快取每台相機最新曲線（callback 執行緒，只是 ref 賦值）
            int cameraIndex = camId - 1;
            if (cameraIndex >= 0 && cameraIndex < CameraCount)
            {
                _liveCurveMean[cameraIndex] = meanArr;
                _liveCurveMax[cameraIndex]  = maxArr;
                // M8: memory barrier 確保 UI thread 透過 volatile _liveOverviewDirty 讀到 dirty=true 時，
                // array reference 寫入已完成（避免讀到舊指標）
                System.Threading.Interlocked.MemoryBarrier();
                _liveOverviewDirty = true;
            }

            // Live Mura 判斷（callback 執行緒，所有相機都檢查）
            CheckLiveMura(meanArr, maxArr, "v");

            // Global 模式不更新 Live mura 垂直圖（單台資料無意義）
            if (_settings.StitchMode == StitchMode.Global) return;

            // 只有選中相機才 marshal 到 UI 執行緒更新 muraChartLive
            if (camId != _liveCameraManager.SelectedMainCameraId) return;

            if (InvokeRequired)
            {
                if (!IsHandleCreated || IsDisposed || Disposing) return;
                BeginInvoke(new Action<int, float[], float[]>(OnLiveCurveData), camId, meanArr, maxArr);
                return;
            }

            if (_liveColumnChartHelper == null || _settings == null) return;

            double[] opsUmArr       = _settings.GetCameraOpsUmArray();
            double[] startPositions = _settings.GetCameraStartPositionMmArray();

            double opsUm = (cameraIndex >= 0 && cameraIndex < opsUmArr.Length)
                ? opsUmArr[cameraIndex] : _settings.Cam1_Ops;
            double opsInMm  = opsUm / 1000.0;
            double startPos = (cameraIndex >= 0 && cameraIndex < startPositions.Length)
                ? startPositions[cameraIndex] : 0;

            _liveColumnChartHelper.SetOps(opsUm);

            // 查詢 MIL 副顯示器的實際 zoom/pan（隨使用者滾輪操作即時變化）
            // panOffsetX = 面板左邊緣對應的 buffer pixel X
            // rightPixel = panOffsetX + panelWidth / zoomX
            double viewLeftMm = double.NaN, viewRightMm = double.NaN;

            var liveCam = FindCameraById(camId);

            if (liveCam != null && opsInMm > 0 &&
                liveCam.TryGetSecondaryDisplayGeometry(
                    out double milZoomX, out double milZoomY, out double milPanX, out double milPanY))
            {
                double panelW = panelMainDisplay.Width;
                double leftPixel  = milPanX;
                double rightPixel = milPanX + panelW / milZoomX;
                viewLeftMm  = startPos + leftPixel  * opsInMm;
                viewRightMm = startPos + rightPixel * opsInMm;
            }

            _liveColumnChartHelper.UpdateDataAndView(meanArr, maxArr,
                startPos, viewLeftMm, viewRightMm);
        }

        private void OnLiveRowCurveData(int camId, float[] meanArr, float[] maxArr)
        {
            // Live Mura 判斷（水平方向）
            CheckLiveMura(meanArr, maxArr, "h");

            if (InvokeRequired)
            {
                if (!IsHandleCreated || IsDisposed || Disposing) return;
                BeginInvoke(new Action<int, float[], float[]>(OnLiveRowCurveData), camId, meanArr, maxArr);
                return;
            }

            if (_liveRowChartHelper == null) return;

            bool isGlobal = _liveCameraManager?.IsGlobalMergeActive == true;

            if (isGlobal)
            {
                // 全域模式：快取每台相機資料，合併後更新（mean 取 mean, max 取 max）
                _liveRowMeanCache[camId] = meanArr;
                _liveRowMaxCache[camId]  = maxArr;
                MergeAndUpdateLiveRowChart();

                // 同步 Y 軸視野：查詢 _mergedDisplay 的 zoom/pan
                double rowPitch = _liveRowChartHelper.RowPitchMm;
                if (rowPitch > 0 && _liveCameraManager.TryGetMergedViewRangeY(
                    out double topPixel, out double botPixel))
                {
                    _liveRowChartHelper.UpdateViewRange(topPixel * rowPitch, botPixel * rowPitch);
                }
            }
            else
            {
                // 垂直模式：只顯示選中相機
                if (camId != _liveCameraManager.SelectedMainCameraId) return;
                _liveRowChartHelper.UpdateData(meanArr, maxArr);

                // 同步 Y 軸視野：查詢 MIL 副顯示器 zoom/pan
                var liveCam = FindCameraById(camId);
                double rowPitch = _liveRowChartHelper.RowPitchMm;
                if (liveCam != null && rowPitch > 0 &&
                    liveCam.TryGetSecondaryDisplayGeometry(
                        out double milZoomX, out double milZoomY, out double milPanX, out double milPanY))
                {
                    double panelH  = panelMainDisplay.Height;
                    double topPixel = milPanY;
                    double botPixel = milPanY + panelH / milZoomY;
                    _liveRowChartHelper.UpdateViewRange(topPixel * rowPitch, botPixel * rowPitch);
                }
            }
        }

        /// <summary>合併所有快取的 row curve 資料：mean 取平均、max 取最大值。</summary>
        private void MergeAndUpdateLiveRowChart()
        {
            if (_liveRowMeanCache.Count == 0) return;

            // 取最短長度對齊
            int minLen = int.MaxValue;
            foreach (var arr in _liveRowMeanCache.Values)
                if (arr.Length < minLen) minLen = arr.Length;
            if (minLen <= 0 || minLen == int.MaxValue) return;

            float[] mergedMean = new float[minLen];
            float[] mergedMax  = new float[minLen];

            int camCount = _liveRowMeanCache.Count;
            foreach (var arr in _liveRowMeanCache.Values)
                for (int i = 0; i < minLen; i++)
                    mergedMean[i] += arr[i];
            for (int i = 0; i < minLen; i++)
                mergedMean[i] /= camCount;

            foreach (var arr in _liveRowMaxCache.Values)
                for (int i = 0; i < minLen; i++)
                    if (arr[i] > mergedMax[i]) mergedMax[i] = arr[i];

            _liveRowChartHelper.UpdateData(mergedMean, mergedMax);
        }

        /// <summary>用 A輪速度 和選中相機的取樣頻率（Line Rate）更新法向圖表座標。</summary>
        private void UpdateRowChartPitch()
        {
            if (_settings == null) return;
            double lineRateHz = _settings.Acquisition.CameraLineRateHz[0]; // CAM1 master
            _liveRowChartHelper?.SetRowPitchFromSpeed(
                _settings.AniloxRollSpeedMPerMin, lineRateHz);
            _reviewRowChartHelper?.SetRowPitchFromSpeed(
                _settings.AniloxRollSpeedMPerMin, lineRateHz);
        }


        private Panel[] GetLivePanels() => new[]
        {
            panelLiveCam1, panelLiveCam2, panelLiveCam3,
            panelLiveCam4, panelLiveCam5, panelLiveCam6, panelLiveCam7
        };

        /// <summary>
        /// 將 float[] column mean 擴展為 width×height 的 8bpp 灰階 Bitmap。
        /// 每列（row）相同：pixel[x] = clamp(colMean[x], 0, 255)。
        /// </summary>
        private static Bitmap ExpandColMeanToBitmap(float[] colMean, int width, int height)
        {
            byte[] row = new byte[width];
            for (int x = 0; x < width; x++)
            {
                float v = colMean[x];
                row[x] = v <= 0 ? (byte)0 : v >= 255 ? (byte)255 : (byte)(v + 0.5f);
            }

            byte[] pixels = new byte[width * height];
            for (int y = 0; y < height; y++)
                Buffer.BlockCopy(row, 0, pixels, y * width, width);

            return ImageUtils.Create8bppBitmap(pixels, width, height);
        }

        private void UpdateGrabButton(bool isGrabbing)
        {
            btnCameraGrab.Text = isGrabbing ? "停止抓取" : "開始抓取";
            // 抓取中：凍結取得背景/預覽背景；停止後解鎖
            btnGetBackground.Enabled = !isGrabbing;
            btnViewBackground.Enabled = !isGrabbing;
            if (!isGrabbing)
            {
                UpdateStandardBgSubLockState(); // 停止後依 bin 狀態重新檢查
            }
        }

        private void ApplyMuraEnhance(bool enabled)
        {
            _liveCameraManager?.SetImageProcessingEnabled(enabled);
            UpdateLiveDirectionVisual();
        }

        /// <summary>
        /// 安全序列化：Live chart 點選切 StitchMode 時，若同時要關掉強化，
        /// 必須先把 callback thread 的 chart 更新訂閱斷開（C），避免轉場期間 callback
        /// BeginInvoke 到 chart handle 不穩定的視窗。
        /// 並把 UpdateLiveDirectionVisual 延後到 OnStitchModeChangedAsync 之後一次性執行（D），
        /// 減少 Border 屬性變更引起的 paint storm。
        ///
        /// **DEBUG**：每步驟 Trace.WriteLine + 寫 D:\Anilox\stitch-debug.log；
        /// 任何 exception 抓到後彈 MessageBox 顯示完整 stack trace，並寫 log 檔。
        /// </summary>
        private static void LogClick(string msg, MouseEventArgs e = null)
        {
            string suffix = e != null ? $" (Button={e.Button} Loc={e.X},{e.Y})" : "";
            string line = $"[{DateTime.Now:HH:mm:ss.fff}] [Click] {msg}{suffix}";
            try { System.IO.File.AppendAllText(@"D:\Anilox\stitch-debug.log", line + Environment.NewLine); } catch { }
        }

        /// <summary>
        /// 全域 IMessageFilter：log 每次 WM_LBUTTONDOWN 命中的控制項 + 螢幕座標。
        /// 用來診斷 Live chart click 為什麼沒觸發 MouseClick（panel/MIL native window 截獲？
        /// chart 內部吞掉？bounds 重疊？）。試一次就能看出 click 去了哪裡。
        /// </summary>
        private sealed class GlobalMouseLogger : IMessageFilter
        {
            private const int WM_LBUTTONDOWN = 0x0201;
            public bool PreFilterMessage(ref Message m)
            {
                if (m.Msg == WM_LBUTTONDOWN)
                {
                    try
                    {
                        var c = Control.FromHandle(m.HWnd);
                        var pt = Cursor.Position;
                        string name = c?.Name ?? "(null)";
                        string type = c?.GetType().Name ?? "(no-type)";
                        string line = $"[{DateTime.Now:HH:mm:ss.fff}] [MsgFilter] WM_LBUTTONDOWN hwnd=0x{m.HWnd.ToInt64():X} ctl={name}({type}) screen=({pt.X},{pt.Y})";
                        System.IO.File.AppendAllText(@"D:\Anilox\stitch-debug.log", line + Environment.NewLine);
                    }
                    catch { }
                }
                return false; // 不攔截，繼續傳遞
            }
        }

        /// <summary>
        /// Live chart 點選切 StitchMode 時，若同時要關掉強化，必須先把 callback thread 的 chart 更新訂閱斷開，
        /// 避免轉場期間 callback BeginInvoke 到 chart handle 不穩定的視窗。
        /// L2：setting 變更走 Hub.SetBatch 統一 save；副作用 transition 仍 inline await（避免 event race）。
        /// </summary>
        private async Task SwitchStitchModeWithEnhanceSequence(StitchMode newMode)
        {
            if (_settings == null) return;
            bool wasEnhanced = _settings.EnableMuraEnhance;
            try
            {
                _liveCameraManager.OnLiveCurveData    -= OnLiveCurveData;
                _liveCameraManager.OnLiveRowCurveData -= OnLiveRowCurveData;

                _settingsHub.SetBatch(s =>
                {
                    if (wasEnhanced) s.EnableMuraEnhance = false;
                    s.hb_StitchMode = newMode;
                });
                if (wasEnhanced) _liveCameraManager?.SetImageProcessingEnabled(false);
                if (wasEnhanced) RefreshGridItem(nameof(InspectionSettings.hc_EnableMuraEnhance));
                RefreshGridItem(nameof(InspectionSettings.hb_StitchMode));
                await OnStitchModeChangedAsync();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"切換 StitchMode 異常:\n{ex}", "StitchMode", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                _liveCameraManager.OnLiveCurveData    += OnLiveCurveData;
                _liveCameraManager.OnLiveRowCurveData += OnLiveRowCurveData;
                try { UpdateLiveDirectionVisual(); } catch (Exception ex) { Trace.WriteLine(ex); }
            }
        }

        private void lblIoDoMura_Click(object sender, EventArgs e)
        {
            _isMuraDetectPaused = !_isMuraDetectPaused;
            UpdateMuraLed(false);
        }

        private void lblIoConn_Click(object sender, EventArgs e)
        {
            if (_ioGrabController == null) return;
            _isIoSuspended = !_isIoSuspended;
            if (_isIoSuspended)
            {
                lblIoConn.BackColor = IecYellow;
                lblIoConn.ForeColor = Color.Black;
                lblIoConn.Text = "● IO 暫停 ⏸";
                btnCameraGrab.Enabled = true;
                UpdateGrabButton(_liveCameraManager?.IsLiveGrabbing ?? false);
                btnCameraGrab.BackColor = SystemColors.Control;
                btnCameraGrab.ForeColor = SystemColors.ControlText;
                // 暫停 = 等同 IO 離線：重置狀態燈和所有 IO 燈號
                lblIoState.Text = "〔已關閉〕";
                lblIoState.BackColor = IecGray;
                SetIoLed(lblIoDiAlive,   false);
                SetIoLed(lblIoDiStart,   false);
                SetIoLed(lblIoDoPcAlive, false);
                SetIoLed(lblIoDoPcBusy,  false);
                UpdateMuraLed(false);
            }
            else
            {
                UpdateIoConnectionUi(_ioGrabController.IsConnected);
            }
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
            nameof(InspectionRecipe.HessianMaxFactorV), "Hessian Max Factor V", "垂直正規值",
            nameof(InspectionRecipe.HessianMaxFactorH), "Hessian Max Factor H", "水平正規值",
            nameof(InspectionRecipe.ErrorValueMeanV),  "Error Value Mean V", "垂直平均閾值",
            nameof(InspectionRecipe.ErrorValueMaxV),   "Error Value Max V",  "垂直最大閾值",
            nameof(InspectionRecipe.ErrorValueMeanH),  "Error Value Mean H", "水平平均閾值",
            nameof(InspectionRecipe.ErrorValueMaxH),   "Error Value Max H",  "水平最大閾值",
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
                // ── 共用副作用（任何 setting 變更都跑） ────────────────────────
                // PropertyGrid 顯示同步：「程式碼路徑改值」時用精準 trick 重讀單 cell（不全 Refresh、不閃）。
                // PropertyGrid 自己改值已自我更新該 cell，不需要外部處理。
                if (c.Source == AniloxRoll.Monitor.Settings.Services.SettingSource.Programmatic)
                    RefreshGridItem(c.Name);
                _interactionHelper.HandleSettingsChanged();
                _liveCameraManager?.SetCaptureSettings(_settings);
                _reviewColumnChartHelper?.SetThresholds(_settings.ErrorValueMeanV, _settings.ErrorValueMaxV);
                _liveColumnChartHelper?.SetOps(_settings.Cam1_Ops);
                _liveColumnChartHelper?.SetThresholds(_settings.ErrorValueMeanV, _settings.ErrorValueMaxV);
                _reviewOverviewHelper?.SetThresholds(_settings.ErrorValueMeanV, _settings.ErrorValueMaxV);
                _liveOverviewHelper?.SetThresholds(_settings.ErrorValueMeanV, _settings.ErrorValueMaxV);
                _liveRowChartHelper?.SetThresholds(_settings.ErrorValueMeanH, _settings.ErrorValueMaxH);
                _reviewRowChartHelper?.SetThresholds(_settings.ErrorValueMeanH, _settings.ErrorValueMaxH);
                UpdateRowChartPitch();
                _dataStatsPresenter?.RefreshMuraProfileForSettingsChange();
                if (_stitchCoordinator?.IsStitchMode == true)
                {
                    _stitchCoordinator.UpdateStitchedOverviewChart();
                    _stitchCoordinator.RefreshCurrentCameraChartsForSettingsChange();
                }
                ScheduleStatsRefresh();
                if (_liveCameraManager?.IsLiveGrabbing == true)
                    _inspectionLogService?.ForceWriteConfig(CsvConfigSnapshot.FromSettings(_settings));

                // ── 機台角色：寫 app-mode.json（早退） ────────────────────────
                if (c.Name == nameof(InspectionSettings.AppRole))
                {
                    if (_appMode == null) _appMode = new AppModeConfig();
                    _appMode.Role = _settings.AppRole;
                    _appMode.Save();
                    MessageBox.Show("機台角色已儲存，重新開啟程式後生效。",
                        "機台設定", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return;
                }

                // ── StitchMode 變更 ────────────────────────────────────────────
                if (c.Name == nameof(InspectionSettings.hb_StitchMode))
                    await OnStitchModeChangedAsync();

                // ── OPS/Start 變更 → Live 全域合圖佈局即時更新 ────────────────
                if (OpsStartSettingNames.Contains(c.Name))
                {
                    if (_liveCameraManager?.IsGlobalMergeActive == true)
                        _liveCameraManager.RefreshGlobalMergeLayout(
                            _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray());
                }

                // ── 檢測報表設定 ──────────────────────────────────────────────
                if (c.Name == nameof(InspectionSettings.gb_ChartScaleMode))
                    _dataStatsPresenter.ApplyChartScaleFromSettings();
                else if (c.Name == nameof(InspectionSettings.gc_YearlyYMax))
                    _dataStatsPresenter.ApplyFixedScaleForChart("Yearly", _settings.Chart.YearlyYMax);
                else if (c.Name == nameof(InspectionSettings.gd_MonthlyYMax))
                    _dataStatsPresenter.ApplyFixedScaleForChart("Monthly", _settings.Chart.MonthlyYMax);
                else if (c.Name == nameof(InspectionSettings.ge_DailyYMax))
                    _dataStatsPresenter.ApplyFixedScaleForChart("Daily", _settings.Chart.DailyYMax);

                // ── 光源設定 ──────────────────────────────────────────────────
                HandleLightSettingsChanged(c.Name);

                // ── 強化 setting ──────────────────────────────────────────────
                if (c.Name == nameof(InspectionSettings.hc_EnableMuraEnhance))
                    ApplyMuraEnhance(_settings.EnableMuraEnhance);
                if (c.Name == nameof(InspectionSettings.hd_EnableReviewEnhance))
                    await ApplyReviewEnhance(_settings.EnableReviewEnhance);

                // ── Algorithm 變更 ────────────────────────────────────────────
                if (c.Name == "db_Algorithm" || c.Name == nameof(InspectionRecipe.Algorithm) || c.Name == "去背演算法")
                {
                    if (_liveCameraManager.IsAllocated) LoadBackgroundBins();
                    UpdateStandardBgSubLockState();
                }

                // ── Recipe 變更（正規值 / 閾值 / 演算法 / Ridge 方向） ─────────
                // 影響：PASS/FAIL 判定 + 閾值線 + 曲線坡度（共用前段的 SetThresholds + UpdateStitchedOverviewChart 已處理）
                // 不影響：影像 bytes（無需 reload 主畫面）。
                // TODO：未來實作「重算 .bin curve」時在此分支補上。原本 pre-existing 的 reload 行為移除。
            }
            catch (Exception ex) { Trace.WriteLine($"[OnSettingChanged {c?.Name}] {ex}"); }
            finally { _onSettingChangedSemaphore.Release(); }
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

        private async void btnSelectFolder_Click(object sender, EventArgs e)
        {
            try
            {
                _interactionHelper.SelectAndLoadFolder();
                _presenter.UpdatePeriodNavigationState();
                await ResetAndLoadReviewAfterFolderChanged(dataPresenterAlreadySynced: false);
            }
            catch (Exception ex) { Trace.WriteLine($"[btnSelectFolder_Click] {ex}"); }
        }

        /// <summary>
        /// 載入 Anilox 資料夾後共用的 Review 重置 + 主畫面載入：
        /// state reset（合圖方式=全域、回顧強化=否）、Live merge sync + chart clear、
        /// DataPresenter 同步、Review 主畫面載入。
        /// btnSelectFolder（Review tab）跟 OnDataFolderSelected（Data tab 觸發）共用。
        /// </summary>
        private async Task ResetAndLoadReviewAfterFolderChanged(bool dataPresenterAlreadySynced)
        {
            _stitchCoordinator.LastReviewProcessedMode = false;
            _settingsHub.SetBatch(s =>
            {
                s.EnableReviewEnhance = false;
                s.hb_StitchMode       = StitchMode.Global;
            });
            RefreshGridItem(nameof(InspectionSettings.hd_EnableReviewEnhance));
            RefreshGridItem(nameof(InspectionSettings.hb_StitchMode));

            // Live tab 副作用（SetBatch 沒 raise event，手動同步）
            if (_settings.StitchMode == StitchMode.Global && _liveCameraManager?.IsAllocated == true)
                _liveCameraManager.EnableGlobalMerge(
                    _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray());
            else
                _liveCameraManager?.DisableGlobalMerge();
            if (_settings.StitchMode == StitchMode.Global)
            {
                chartMuraVertical.Series["Mean"].Points.Clear();
                chartMuraVertical.Series["Max"].Points.Clear();
                muraChartVerticalLive.Series["Mean"].Points.Clear();
                muraChartVerticalLive.Series["Max"].Points.Clear();
            }
            UpdateLiveDirectionVisual();

            // Data tab 已 LoadDataFolder 時跳過 SyncFromReviewFolder（避免 duplicate load）。
            // SyncGrabIdFromTime 兩條路徑都需要（保持 DataPresenter 內部 _grabIdInfos 對齊 navigator 當前 period）。
            if (_imageRepository.FileCount > 0)
            {
                if (!dataPresenterAlreadySynced)
                {
                    var reviewPath = UserSessionState.LastDataPath;
                    if (!string.IsNullOrWhiteSpace(reviewPath))
                        _dataStatsPresenter.SyncFromReviewFolder(reviewPath);
                }
                var current = _dateTimeNavigator.GetCurrentPeriodOrDefault(DateTime.MinValue);
                if (current != DateTime.MinValue)
                    _dataStatsPresenter.SyncGrabIdFromTime(current);
            }

            _stitchCoordinator.ClearStitchedMode();
            _dataStatsPresenter.SetReviewGroupBoxes(true);
            _dataStatsPresenter.SelectLatestInSingleSheetMode();

            // 預設 grpReviewGrabNav（單片序號模式）→ 直接 LoadGrabStitchedViewAsync
            int reviewIdx = cbReviewGrabId.SelectedIndex;
            if (reviewIdx >= 0 && reviewIdx < _dataStatsPresenter.GrabIdInfos.Count)
            {
                var info = _dataStatsPresenter.GrabIdInfos[reviewIdx];
                await _stitchCoordinator.LoadGrabStitchedViewAsync(info.GrabId, info.Earliest, info.Latest);
                if (canvasMain.Image != null) canvasMain.FitToScreen();
                _reviewDirty = false;
            }
            else
            {
                // 資料夾無序號 → period 模式 fallback
                await _presenter.LoadImagesWithPeriodLockAsync(false, LoadImagesWithReviewConfig);
                ApplyPostLoadDisplay();
            }
        }

        private async Task ApplyReviewEnhance(bool enableProcess)
        {
            try
            {
            UpdateRidgeDirectionVisual(enableProcess ? _stitchCoordinator.ActiveRidgeDirection : null);
            if (_stitchCoordinator.IsStitchMode)
            {
                await ReloadCurrentStitchedView(enableProcess);
                return;
            }
            _stitchCoordinator.LastReviewProcessedMode = enableProcess;
            _stitchCoordinator.ClearStitchedMode();
            await _presenter.LoadImagesWithPeriodLockAsync(enableProcess, _interactionHelper.LoadImages);
            ApplyPostLoadDisplay();
            }
            catch (Exception ex) { Trace.WriteLine($"[ApplyReviewEnhance] {ex}"); }
        }

        private async Task ReloadCurrentStitchedView(bool enableProcess)
        {
            int idx = cbReviewGrabId.SelectedIndex;
            if (idx < 0 || idx >= _dataStatsPresenter.GrabIdInfos.Count) return;
            _interactionHelper.SaveCanvasView();
            var info = _dataStatsPresenter.GrabIdInfos[idx];
            await _stitchCoordinator.LoadGrabStitchedViewAsync(info.GrabId, info.Earliest, info.Latest, enableProcess);
        }

        /// <summary>
        /// 從當前 Period 日期的 CSV 載入 #CFG，更新 ReviewConfig。
        /// 應在每次 Period 切換或資料夾載入後呼叫。
        /// </summary>
        private void RefreshReviewConfigForCurrentPeriod()
        {
            string rootPath = UserSessionState.LastDataPath;
            if (string.IsNullOrWhiteSpace(rootPath)) { _interactionHelper.ReviewConfig = null; return; }

            var periodDate = _dateTimeNavigator.GetCurrentPeriodOrDefault(DateTime.MinValue);
            if (periodDate == DateTime.MinValue) { _interactionHelper.ReviewConfig = null; return; }

            var cfg = InspectionStatisticsService.LoadConfigForDate(rootPath, periodDate);
            _interactionHelper.ReviewConfig = cfg;
        }

        /// <summary>
        /// 包裝 LoadImages：先刷新 ReviewConfig（navigator 已指向新日期），再載入影像。
        /// 確保 OnGallerySelectionChanged 觸發時 ReviewConfig 已是正確的 CFG。
        /// </summary>
        private async Task LoadImagesWithReviewConfig(bool enableProcess)
        {
            RefreshReviewConfigForCurrentPeriod();
            await _interactionHelper.LoadImages(enableProcess);
        }

        /// <summary>
        /// 載入影像後，根據 StitchMode 決定顯示方式：
        /// Vertical → 觸發 gallery 選取顯示單台影像；Global → 合圖顯示。
        /// 最後更新全覽圖。
        /// </summary>
        private void ApplyPostLoadDisplay()
        {
            if (_settings.StitchMode == StitchMode.Global)
                _stitchCoordinator.ApplyGlobalMergeIfNeeded();
            else
                _interactionHelper.RefreshCurrentCanvasResult();
            _stitchCoordinator.UpdateOverviewChartFromRepository();
        }

        private async void btnPeriodPrev_Click(object sender, EventArgs e)
        {
            try
            {
                bool wasStitch = _stitchCoordinator.IsStitchMode;
                _interactionHelper.SaveCanvasView();
                _stitchCoordinator.ClearStitchedMode();
                await _presenter.MovePeriodAsync(-1, _stitchCoordinator.LastReviewProcessedMode, LoadImagesWithReviewConfig);
                ApplyPostLoadDisplay();
                if (wasStitch && canvasMain.Image != null) canvasMain.FitToScreen();
            }
            catch (Exception ex) { Trace.WriteLine($"[btnPeriodPrev] {ex}"); }
        }

        private async void btnPeriodNext_Click(object sender, EventArgs e)
        {
            try
            {
                bool wasStitch = _stitchCoordinator.IsStitchMode;
                _interactionHelper.SaveCanvasView();
                _stitchCoordinator.ClearStitchedMode();
                await _presenter.MovePeriodAsync(+1, _stitchCoordinator.LastReviewProcessedMode, LoadImagesWithReviewConfig);
                ApplyPostLoadDisplay();
                if (wasStitch && canvasMain.Image != null) canvasMain.FitToScreen();
            }
            catch (Exception ex) { Trace.WriteLine($"[btnPeriodNext] {ex}"); }
        }

        /// <summary>cbDate/cbTime 手動滾動時載入對應圖片（同 btnPeriodPrev/Next）。
        /// _dataStatsPresenter.GrabIdNavGuard 時跳過（由 OnReviewGrabIdChanged 等程式碼觸發的 NavigateToDateTime）。</summary>
        private async void OnPeriodComboChanged()
        {
            if (_dataStatsPresenter.GrabIdNavGuard.IsSet) return;
            if (_imageRepository.FileCount == 0) return;
            try
            {
            bool wasStitch = _stitchCoordinator.IsStitchMode;
            _interactionHelper.SaveCanvasView();
            _stitchCoordinator.ClearStitchedMode();
            _dataStatsPresenter.SetReviewGroupBoxes(false);
            await _presenter.LoadImagesWithPeriodLockAsync(_stitchCoordinator.LastReviewProcessedMode, LoadImagesWithReviewConfig);
            ApplyPostLoadDisplay();
            if (wasStitch && canvasMain.Image != null) canvasMain.FitToScreen();
            }
            catch (Exception ex) { Trace.WriteLine($"[OnPeriodComboChanged] {ex}"); }
        }



    }
}
