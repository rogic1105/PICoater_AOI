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
using AOI.SDK.UI;
using AOI.SDK.Utils;
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
        private IoGrabController _plcGrabController;
        private LightController _lightController;

        // --- 統計 ---
        private DataStatisticsPresenter _dataStatsPresenter;

        // --- 資料緩存 ---
        private readonly List<Image> _thumbnailCache = new List<Image>();
        private InspectionSettings _settings;
        private bool IsStandardBgSubEnabled =>
            _settings?.Recipe?.Algorithm == BackgroundAlgorithm.StandardBgSub;

        /// <summary>"v" = vertical ridge（預設），"h" = horizontal ridge。控制 Live 顯示方向。</summary>
        private string _liveDisplayDirection = "v";

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


        public AniloxRollForm()
        {
            InitializeComponent();
            InitializeSystem();
            _scaler = new ProportionalScaler(this);
            _scaler.Initialize();
            Shown += (s, e) => AutoFitPropertyGridLabelColumn(propertyGridSettings);
        }

        private void InitializeSystem()
        {
            _appMode = AppModeConfig.Load();
            if (_settings == null) _settings = ConfigManager.LoadInspectionSettings();
            _settings.AppRole = _appMode?.Role ?? MachineRole.Inspection;
            CameraFrameSaver.InitResourceLog(_settings?.CaptureRootPath);
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
            else
                RegisterStorageModeCleanup();
            InitializeRightPanelControls();
            SetupDataTab();
            ApplyStorageModeUi();
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
                // Inspection 模式：遠端複製 + PLC + 光源
                _remoteCopyService = new RemoteCopyService(
                    getRemotePath: () => _settings?.RemotePath ?? string.Empty,
                    getLocalRoot:  () => _settings?.CaptureRootPath ?? string.Empty);
                InitPlcController();
                InitLightController();
            }

            // 啟動時執行一次清理（雙模式共用）
            Task.Run(() => _retentionService.RunCleanup());
        }

        /// <summary>初始化 IO 連動：自動偵測連線，連上後以 DI START 控制 Grab。</summary>
        private void InitPlcController()
        {
            if (!_settings.PlcEnabled) return;

            _plcGrabController = new IoGrabController();

            _plcGrabController.OnStartRequested += () =>
            {
                if (InvokeRequired) { BeginInvoke(new Action(PlcStartGrab)); return; }
                PlcStartGrab();
            };

            _plcGrabController.OnStopRequested += () =>
            {
                if (InvokeRequired) { BeginInvoke(new Action(PlcStopGrab)); return; }
                PlcStopGrab();
            };

            _plcGrabController.OnStateChanged += state =>
            {
                if (InvokeRequired) { BeginInvoke(new Action<IoState>(UpdatePlcStateLabel), state); return; }
                UpdatePlcStateLabel(state);
            };

            _plcGrabController.OnConnectionChanged += connected =>
            {
                if (InvokeRequired) { BeginInvoke(new Action<bool>(UpdatePlcConnectionUi), connected); return; }
                UpdatePlcConnectionUi(connected);
            };

            _plcGrabController.OnIoUpdated += snapshot =>
            {
                if (InvokeRequired) { BeginInvoke(new Action<IoSnapshot>(UpdatePlcIoLeds), snapshot); return; }
                UpdatePlcIoLeds(snapshot);
            };

            // 背景嘗試連線（不阻塞 Form 顯示）
            _ = _plcGrabController.StartAsync(_settings.PlcIp, _settings.PlcPort);
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

            // 掃描找到但非原設定 → 更新記錄的 COM（下次啟動直接命中）
            if (!string.Equals(found, _settings.LightComPort, StringComparison.OrdinalIgnoreCase))
            {
                _settings.LightComPort = found;
            }
        }

        private void PlcStartGrab()
        {
            if (_isIoSuspended) return;
            if (_liveCameraManager == null || _liveCameraManager.IsLiveGrabbing) return;
            btnCameraGrab_Click(null, null);
            _ = _plcGrabController?.NotifyGrabStarted();
        }

        private void PlcStopGrab()
        {
            if (_isIoSuspended) return;
            if (_liveCameraManager == null || !_liveCameraManager.IsLiveGrabbing) return;
            btnCameraGrab_Click(null, null);
            _ = _plcGrabController?.NotifyGrabStopped();
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

        private void UpdatePlcStateLabel(IoState state)
        {
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
            lblPlcState.Text = $"● {text}";
            lblPlcState.BackColor = bgColor;
        }

        private void UpdatePlcConnectionUi(bool connected)
        {
            if (_isIoSuspended) return;
            if (connected)
            {
                lblPlcConn.Text = "● IO 已連線";
                lblPlcConn.BackColor = IecGreen;
                btnCameraGrab.Enabled = false;
                btnCameraGrab.Text = "IO 控制中";
                btnCameraGrab.BackColor = IecBlue;
                btnCameraGrab.ForeColor = Color.White;
            }
            else
            {
                lblPlcConn.Text = "● IO 離線";
                lblPlcConn.BackColor = IecGray;
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
                                                    _settings.LightComPort = found;
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

        private void UpdatePlcIoLeds(IoSnapshot io)
        {
            SetIoLed(lblIoDiAlive,   io.DiNakanAlive);
            SetIoLed(lblIoDiStart,   io.DiInspectStart);
            SetIoLed(lblIoDoPcAlive, io.DoPcAlive);
            UpdateMuraLed(io.DoMuraDetected);
            SetIoLed(lblIoDoPcBusy,  io.DoPcInspect);
        }

        private static void SetIoLed(Label lbl, bool on)
        {
            lbl.BackColor = on ? IecGreen : IecDarkGray;
        }

        private void UpdateMuraLed(bool doMuraOn)
        {
            if (_isMuraDetectPaused)
            {
                lblIoDoMura.BackColor = IecYellow;
                lblIoDoMura.ForeColor = Color.Black;
                lblIoDoMura.Text = "DO1\r\nMURA ⏸";
            }
            else
            {
                lblIoDoMura.BackColor = doMuraOn ? IecGreen : IecDarkGray;
                lblIoDoMura.ForeColor = Color.White;
                lblIoDoMura.Text = "DO1\r\nMURA_DET";
            }
        }

        private void UpdateCamCountLabel(int connected, int expected)
        {
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
            _reviewColumnChartHelper.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);

            _liveColumnChartHelper = new ColumnCurveChartHelper(this.muraChartVerticalLive);
            _liveColumnChartHelper.SetOps(_settings.Cam1_Ops);
            _liveColumnChartHelper.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);

            _reviewOverviewHelper = new ColumnCurveChartHelper(this.chartOverview);
            _reviewOverviewHelper.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            if (chartOverview.ChartAreas.Count > 0)
                chartOverview.ChartAreas[0].AxisX.ScaleView.Zoomable = false;

            _liveOverviewHelper = new ColumnCurveChartHelper(this.chartLiveOverview);
            _liveOverviewHelper.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            if (chartLiveOverview.ChartAreas.Count > 0)
                chartLiveOverview.ChartAreas[0].AxisX.ScaleView.Zoomable = false;

            _liveRowChartHelper = new RowCurveChartHelper(this.muraChartHorizontalLive);
            _liveRowChartHelper.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);

            _reviewRowChartHelper = new RowCurveChartHelper(this.chartMuraHorizontal);
            _reviewRowChartHelper.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);

            UpdateRowChartPitch();

            // Review tab chart 點選切換 V/H 處理圖方向
            chartMuraVertical.MouseClick += (s, e) => SwitchRidgeDirection("v");
            chartMuraHorizontal.MouseClick += (s, e) => SwitchRidgeDirection("h");

            // Live tab chart 點選切換 V/H 處理圖方向
            muraChartVerticalLive.MouseClick += (s, e) => SwitchLiveDisplayDirection("v");
            muraChartHorizontalLive.MouseClick += (s, e) => SwitchLiveDisplayDirection("h");

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

            _presenter.BusyStateChanged += _interactionHelper.SetUiLoadingState;
            _presenter.LogReported      += OnPresenterLogReported;
            _galleryManager.SelectionChanged += idx =>
            {
                if (_stitchCoordinator.IsGlobalMerged || _stitchCoordinator.IsPeriodMerged)
                    return; // 全域模式：canvasMain 顯示合併圖，gallery 點選不切換
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
            canvasMain.EdgeReached   += _interactionHelper.NavigateCamera;
            var canvasClicker = new MultiClickDetector();
            canvasMain.MouseDown += (s, e) =>
            {
                if (e.Button != MouseButtons.Left) return;
                int clicks = canvasClicker.RegisterClick(e.Location);
                if (clicks == 2)
                {
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
            _liveCameraManager.OnCameraCountChanged += (connected, expected) =>
            {
                if (InvokeRequired) { BeginInvoke(new Action<int, int>(UpdateCamCountLabel), connected, expected); return; }
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
                _telemetryTimer?.Stop();
                _liveOverviewTimer?.Stop();
                if (_plcGrabController != null)
                {
                    await _plcGrabController.StopAsync();
                    _plcGrabController.Dispose();
                }
                FreePrecomputedColMeanBuffers();
                _liveCameraManager.FreeCameras();
                _lightController?.Dispose();
                _retentionService?.Dispose();
                _remoteCopyService?.Dispose();
                _cleanupFlagWatcher?.Dispose();
            };

            // 程式啟動後自動分配相機（不 Grab），讓 lblCamCount 在按下【開始抓取】前就能顯示連線狀態
            Shown += (s, e) => AutoAllocateCameras();
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

        private void btnCameraGrab_Click(object sender, EventArgs e)
        {
            // 背景預覽中按 Grab → 先清除預覽並 Free，讓 MIL 能重新初始化
            if (_bgPreviewActive)
            {
                ClearBackgroundPreview(restoreMilDisplay: true);
                _liveCameraManager.FreeCameras();
                _telemetryPresenter?.ResetAll();
            }

            bool wasGrabbing = _liveCameraManager.IsLiveGrabbing;

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
                    MessageBox.Show($"相機配置失敗: {ex.Message}", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
            }
            else
            {
                _liveCameraManager.ToggleGrab();
            }

            // 剛從「未抓取」→「抓取中」：開燈 + 分配新的抓圖編號
            if (!wasGrabbing && _liveCameraManager.IsLiveGrabbing)
            {
                LightTurnOn();
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
                _inspectionLogService.AppendRecord(
                    _currentGrabId,
                    fileNameNoExt,
                    meanPeak,
                    maxPeak,
                    _settings.ErrorValueMean,
                    _settings.ErrorValueMax,
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
            if (_plcGrabController?.IsConnected == true)
            {
                bool isMura = meanPeak > _settings.ErrorValueMean || maxPeak > _settings.ErrorValueMax;
                if (isMura) _ = _plcGrabController.NotifyMuraDetected();
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
            string configPath = _settings?.RemoteConfigPath ?? string.Empty;
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

            lblPlcState.Visible    = false;
            lblPlcConn.Visible     = false;
            lblLightConn.Visible   = false;
            lblIoDiAlive.Visible   = false;
            lblIoDiStart.Visible   = false;
            lblIoDoPcAlive.Visible = false;
            lblIoDoMura.Visible    = false;
            lblIoDoPcBusy.Visible  = false;
        }

        private string GetStorageRetentionRoot()
        {
            if (_appMode?.Role == MachineRole.Storage &&
                !string.IsNullOrWhiteSpace(_appMode.StorageFolderPath))
                return _appMode.StorageFolderPath;
            return _settings?.CaptureRootPath ?? string.Empty;
        }

        private void RegisterStorageModeCleanup()
        {
            FormClosed += (_, __) =>
            {
                _telemetryTimer?.Stop();
                _retentionService?.Dispose();
                _cleanupFlagWatcher?.Dispose();
            };
        }

        /// <summary>
        /// Live 曲線閾值判斷（callback 執行緒呼叫，V/H 共用）。
        /// 陣列為 0-255，閾值為 0-1，取陣列 max 後除以 255 比較。
        /// 任一方向超標即觸發 DO_MURA_DETECTED H，維持到 grab 結束。
        /// </summary>
        private void CheckLiveMura(float[] meanArr, float[] maxArr)
        {
            if (_isMuraDetectPaused) return;
            if (_plcGrabController?.IsConnected != true) return;
            if (_settings == null) return;
            if (!_liveCameraManager.IsLiveGrabbing) return;

            float meanPeak = 0f, maxPeak = 0f;
            if (meanArr != null) { for (int i = 0; i < meanArr.Length; i++) if (meanArr[i] > meanPeak) meanPeak = meanArr[i]; }
            if (maxArr  != null) { for (int i = 0; i < maxArr.Length;  i++) if (maxArr[i]  > maxPeak)  maxPeak  = maxArr[i];  }
            meanPeak /= 255f;
            maxPeak  /= 255f;

            if (meanPeak > _settings.ErrorValueMean || maxPeak > _settings.ErrorValueMax)
            {
                // fire-and-forget; 寫入失敗不應影響取像流程
                _ = _plcGrabController.NotifyMuraDetected().ContinueWith(
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
                _liveOverviewDirty = true;
            }

            // Live Mura 判斷（callback 執行緒，所有相機都檢查）
            CheckLiveMura(meanArr, maxArr);

            // Global 模式不更新 Live mura 垂直圖（單台資料無意義）
            if (_settings.StitchMode == StitchMode.Global) return;

            // 只有選中相機才 marshal 到 UI 執行緒更新 muraChartLive
            if (camId != _liveCameraManager.SelectedMainCameraId) return;

            if (InvokeRequired)
            {
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
            CheckLiveMura(meanArr, maxArr);

            if (InvokeRequired)
            {
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

        /// <summary>
        /// 取得背景：啟動 grab → 採集 N 秒 → 多幀平均 column mean → 存 MCBF bin。
        /// </summary>
        private async void btnGetBackground_Click(object sender, EventArgs e)
        {
            if (!IsStandardBgSubEnabled)
            {
                MessageBox.Show("請先將去背演算法切換為「標準去背」。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            // 先清除舊的背景預覽（釋放 overlay + 恢復 MIL display）
            if (_bgPreviewActive) ClearBackgroundPreview();

            // 確保相機已 allocate
            if (!_liveCameraManager.IsAllocated)
            {
                try
                {
                    _liveCameraManager.EnsureAllocatedAndToggleGrab(false); // 不需影像處理
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"相機配置失敗: {ex.Message}", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
            }

            // 確保 grab 中
            if (!_liveCameraManager.IsLiveGrabbing)
            {
                _liveCameraManager.ToggleGrab();
                UpdateGrabButton(true);
            }

            btnGetBackground.Enabled = false;
            btnCameraGrab.Enabled = false;

            int sampleSeconds = Math.Max(1, _settings.Recipe.BackgroundSampleSeconds);
            string bgDir = _settings.Storage.BackgroundPath;

            try
            {
                if (!Directory.Exists(bgDir))
                    Directory.CreateDirectory(bgDir);

                var cameras = _liveCameraManager.Cameras;
                int camCount = cameras.Count;

                double[][] accum = new double[camCount][];
                int[] frameCount = new int[camCount];

                // 採集 sampleSeconds 秒，按鈕顯示倒數
                var sw = Stopwatch.StartNew();
                int lastShown = -1;
                while (sw.Elapsed.TotalSeconds < sampleSeconds)
                {
                    int remaining = sampleSeconds - (int)sw.Elapsed.TotalSeconds;
                    if (remaining != lastShown)
                    {
                        lastShown = remaining;
                        btnGetBackground.Text = $"採集中 {remaining}s";
                    }

                    await Task.Delay(100);

                    for (int i = 0; i < camCount; i++)
                    {
                        var cam = cameras[i];
                        if (!cam.IsConnected || cam.FrameWidth <= 0) continue;

                        if (accum[i] == null)
                            accum[i] = new double[cam.FrameWidth];

                        float[] colMean = new float[cam.FrameWidth];
                        if (cam.TryComputeColumnMean(colMean))
                        {
                            for (int c = 0; c < cam.FrameWidth; c++)
                                accum[i][c] += colMean[c];
                            frameCount[i]++;
                        }
                    }
                }

                // 平均並存檔
                for (int i = 0; i < camCount; i++)
                {
                    if (frameCount[i] == 0 || accum[i] == null) continue;

                    var cam = cameras[i];
                    float[] avgColMean = new float[cam.FrameWidth];
                    double invN = 1.0 / frameCount[i];
                    for (int c = 0; c < cam.FrameWidth; c++)
                        avgColMean[c] = (float)(accum[i][c] * invN);

                    string binPath = Path.Combine(bgDir, $"bg_{cam.FrameWidth}_{cam.CameraId}.bin");
                    SaveBackgroundBin(avgColMean, binPath);
                }

                // 載入到各相機
                LoadBackgroundBins();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"背景採集失敗: {ex.Message}", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                btnGetBackground.Text = "取得背景";
                btnGetBackground.Enabled = true;

                // 採集完成後一律停止 grab
                if (_liveCameraManager.IsLiveGrabbing)
                {
                    _liveCameraManager.ToggleGrab();
                    UpdateGrabButton(false);
                }

                UpdateStandardBgSubLockState();
            }

            // 採集完成後直接預覽（先清除舊預覽，確保每次都重新開啟）
            if (_bgPreviewActive) ClearBackgroundPreview();
            btnViewBackground_Click(btnViewBackground, EventArgs.Empty);
        }

        /// <summary>MCBF 格式存 background column mean。</summary>
        private static void SaveBackgroundBin(float[] data, string path)
        {
            using (var bw = new BinaryWriter(File.Open(path, FileMode.Create, FileAccess.Write)))
            {
                bw.Write(new byte[] { (byte)'M', (byte)'C', (byte)'B', (byte)'F' });
                bw.Write(1);                    // version
                bw.Write(1.0f);                 // scale_factor (1 = 全解析度)
                bw.Write(data.Length);           // array_length
                foreach (float v in data) bw.Write(v);
            }
        }

        /// <summary>
        /// 從 BackgroundPath 載入各相機的 bg bin → pinned buffer → 設定到 AniloxCamera.PrecomputedColMean。
        /// </summary>
        private void LoadBackgroundBins()
        {
            if (!IsStandardBgSubEnabled)
            {
                // 非 StandardBgSub 模式：清除所有預算背景
                foreach (var cam in _liveCameraManager.Cameras)
                    cam.PrecomputedColMean = IntPtr.Zero;
                return;
            }

            string bgDir = _settings.Storage.BackgroundPath;
            if (!Directory.Exists(bgDir)) return;

            foreach (var cam in _liveCameraManager.Cameras)
            {
                if (cam.FrameWidth <= 0) continue;

                string binPath = Path.Combine(bgDir, $"bg_{cam.FrameWidth}_{cam.CameraId}.bin");
                float[] colMean = InspectionEngine.LoadCurveBin(binPath);
                if (colMean != null && colMean.Length == cam.FrameWidth)
                {
                    // 分配 pinned memory 並複製
                    IntPtr pinned = NativeMethods.CoreCV_AllocPinned((ulong)(cam.FrameWidth * sizeof(float)));
                    if (pinned != IntPtr.Zero)
                    {
                        Marshal.Copy(colMean, 0, pinned, colMean.Length);

                        // 釋放舊的（如果有）
                        if (cam.PrecomputedColMean != IntPtr.Zero)
                            NativeMethods.CoreCV_FreePinned(cam.PrecomputedColMean);

                        cam.PrecomputedColMean = pinned;
                    }
                }
            }
        }

        /// <summary>釋放所有相機的 PrecomputedColMean pinned buffer。</summary>
        private void FreePrecomputedColMeanBuffers()
        {
            if (_liveCameraManager == null) return;
            foreach (var cam in _liveCameraManager.Cameras)
            {
                if (cam.PrecomputedColMean != IntPtr.Zero)
                {
                    NativeMethods.CoreCV_FreePinned(cam.PrecomputedColMean);
                    cam.PrecomputedColMean = IntPtr.Zero;
                }
            }
        }

        /// <summary>
        /// StandardBgSub 時檢查是否有 bin → 控制按鈕鎖定狀態。
        /// </summary>
        private void UpdateStandardBgSubLockState()
        {
            // IO 已連線且未暫停：btnCameraGrab 由 IO 連線邏輯控制，不覆寫
            if (_plcGrabController?.IsConnected == true && !_isIoSuspended) return;

            if (!IsStandardBgSubEnabled)
            {
                // 非 StandardBgSub：正常解鎖
                btnCameraGrab.Enabled = true;
                btnGetBackground.Enabled = true;
                return;
            }

            // 檢查已連線的相機是否都有對應的 bg bin
            string bgDir = _settings.Storage.BackgroundPath;
            bool allConnectedHaveBin = true;
            if (_liveCameraManager?.IsAllocated == true)
            {
                foreach (var cam in _liveCameraManager.Cameras)
                {
                    if (!cam.IsConnected) continue; // 斷線相機不考慮
                    if (cam.FrameWidth <= 0) continue;
                    string binPath = Path.Combine(bgDir, $"bg_{cam.FrameWidth}_{cam.CameraId}.bin");
                    if (!File.Exists(binPath)) { allConnectedHaveBin = false; break; }
                }
            }
            else
            {
                // 尚未 allocate：檢查目錄下是否有任何 bin
                allConnectedHaveBin = Directory.Exists(bgDir) && Directory.GetFiles(bgDir, "bg_*.bin").Length > 0;
            }

            btnGetBackground.Enabled = true;
            btnCameraGrab.Enabled = allConnectedHaveBin;
        }

        // --- 背景預覽狀態 ---
        private Bitmap[] _bgPreviewBitmaps;
        private bool _bgPreviewActive;
        private SmartCanvas[] _bgPreviewBoxes;      // panelLiveCam 上的 overlay（SmartCanvas with ClampPan）
        private SmartCanvas _bgPreviewMainCanvas;  // panelMainDisplay 上的 SmartCanvas（支援縮放/拖曳）
        private int _bgPreviewSelectedCamIndex;    // 目前預覽中的相機 index (0-based)

        /// <summary>
        /// 預覽背景：讀取各相機的 bg bin → 擴展為 width × grabHeight 灰階影像。
        /// 用 PictureBox 疊在 panelLiveCam 上方，SmartCanvas 疊在 panelMainDisplay 上方（支援縮放拖曳）。
        /// 點選 panelLiveCam 可切換 panelMainDisplay。再按一次清除預覽。
        /// </summary>
        private void btnViewBackground_Click(object sender, EventArgs e)
        {
            // 先清除舊預覽（釋放 overlay + 恢復 MIL display），再重新載入
            if (_bgPreviewActive)
                ClearBackgroundPreview();

            string bgDir = _settings.Storage.BackgroundPath;
            if (!Directory.Exists(bgDir))
            {
                MessageBox.Show("背景目錄不存在。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            // 先卸載 MIL primary + secondary display，避免 native window 殘影
            if (_liveCameraManager.IsAllocated)
            {
                foreach (var cam in _liveCameraManager.Cameras)
                {
                    cam.SetPrimaryDisplayVisible(false);
                    cam.SetSecondaryDisplay(IntPtr.Zero);
                }
            }

            // 清除殘留的最後一幀（MIL native window detach 後面板不會自動重繪）
            panelMainDisplay.Invalidate();
            panelMainDisplay.Update();

            Panel[] livePanels = GetLivePanels();
            foreach (var p in livePanels) { p.Invalidate(); p.Update(); }
            int[] grabHeights = _settings.Acquisition.CameraGrabHeight;
            _bgPreviewBitmaps = new Bitmap[livePanels.Length];
            _bgPreviewBoxes = new SmartCanvas[livePanels.Length];
            int firstValid = -1;

            for (int i = 0; i < livePanels.Length; i++)
            {
                int camId = i + 1;
                string[] matches = Directory.GetFiles(bgDir, $"bg_*_{camId}.bin");
                if (matches.Length == 0) continue;

                float[] colMean = InspectionEngine.LoadCurveBin(matches[0]);
                if (colMean == null || colMean.Length == 0) continue;

                int height = (i < grabHeights.Length && grabHeights[i] > 0) ? grabHeights[i] : 3000;
                Bitmap bmp = ExpandColMeanToBitmap(colMean, colMean.Length, height);
                _bgPreviewBitmaps[i] = bmp;

                // SmartCanvas 疊在 panel 最上層（ClampPan 模式，同 grab 的 MIL 顯示行為）
                var sc = new SmartCanvas
                {
                    Dock = DockStyle.Fill,
                    ClampPan = true,
                    Tag = i,
                    BackColor = Color.Black
                };
                sc.Image = bmp;
                livePanels[i].Controls.Add(sc);
                sc.BringToFront();
                sc.FitToScreen();
                sc.Click += BgPreviewPanel_Click;
                _bgPreviewBoxes[i] = sc;

                if (firstValid < 0) firstValid = i;
            }

            if (firstValid >= 0)
            {
                // SmartCanvas 覆蓋 panelMainDisplay：支援滑鼠滾輪縮放 + 左鍵拖曳
                _bgPreviewMainCanvas = new SmartCanvas { Dock = DockStyle.Fill, ClampPan = true };
                _bgPreviewMainCanvas.Image = _bgPreviewBitmaps[firstValid];
                _bgPreviewMainCanvas.StatusChanged += BgPreviewCanvas_StatusChanged;
                panelMainDisplay.Controls.Add(_bgPreviewMainCanvas);
                _bgPreviewMainCanvas.BringToFront();
                _bgPreviewMainCanvas.FitToScreen();
                _bgPreviewActive = true;
                _bgPreviewSelectedCamIndex = firstValid;
            }
            else
            {
                _bgPreviewBitmaps = null;
                _bgPreviewBoxes = null;
                MessageBox.Show("未找到背景 bin 檔。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }

        /// <summary>點選 panelLiveCam 上的 PictureBox → 切換 panelMainDisplay 顯示該相機背景。</summary>
        private void BgPreviewPanel_Click(object sender, EventArgs e)
        {
            if (!_bgPreviewActive || _bgPreviewBitmaps == null || _bgPreviewMainCanvas == null) return;

            var sc = sender as SmartCanvas;
            if (sc?.Tag is int idx && idx >= 0 && idx < _bgPreviewBitmaps.Length && _bgPreviewBitmaps[idx] != null)
            {
                _bgPreviewMainCanvas.Image = _bgPreviewBitmaps[idx];
                _bgPreviewMainCanvas.FitToScreen();
                _bgPreviewSelectedCamIndex = idx;
            }
        }

        /// <summary>SmartCanvas 滑鼠移動時更新 lblPixelInfo（與 canvasMain 同格式：mm 座標 + 範圍 + 倍率）。</summary>
        private void BgPreviewCanvas_StatusChanged(CanvasInfo info)
        {
            if (lblPixelInfo == null) return;

            int camIdx = _bgPreviewSelectedCamIndex;
            int camId = camIdx + 1;
            string text;
            if (info.ImageX < 0 || info.ImageY < 0 ||
                _bgPreviewMainCanvas?.Image == null ||
                info.ImageX >= _bgPreviewMainCanvas.Image.Width ||
                info.ImageY >= _bgPreviewMainCanvas.Image.Height)
            {
                text = $"背景預覽 [CAM {camId}] | 游標超出影像範圍";
            }
            else if (_settings == null)
            {
                int gray = info.PixelColor.R;
                text = $"背景預覽 [CAM {camId}] | 座標: ({info.ImageX}, {info.ImageY}) | 亮度: {gray} | 縮放: {info.Zoom:F2}x";
            }
            else
            {
                double[] opsUmArr  = _settings.GetCameraOpsUmArray();
                double[] startMmArr = _settings.GetCameraStartPositionMmArray();
                if (camIdx < 0 || camIdx >= opsUmArr.Length)
                {
                    int gray = info.PixelColor.R;
                    text = $"背景預覽 [CAM {camId}] | 座標: ({info.ImageX}, {info.ImageY}) | 亮度: {gray} | 縮放: {info.Zoom:F2}x";
                }
                else
                {
                    double opsInMm    = opsUmArr[camIdx] / 1000.0;
                    double startPosMm = startMmArr[camIdx];

                    // 背景圖為全解析度，scaleFactor = 1
                    double physicalX = startPosMm + info.ImageX * opsInMm;
                    double rowPitchMm = _interactionHelper?.RowPitchMm ?? 0;
                    double physicalY = info.ImageY * rowPitchMm;

                    // 視野範圍
                    double viewLeftMm = 0, viewRightMm = 0;
                    double viewTopMm = 0, viewBotMm = 0;
                    if (info.Zoom > 0)
                    {
                        double pixelLeft  = (0                            - info.PanOffset.X) / info.Zoom;
                        double pixelRight = (_bgPreviewMainCanvas.Width   - info.PanOffset.X) / info.Zoom;
                        viewLeftMm  = startPosMm + pixelLeft  * opsInMm;
                        viewRightMm = startPosMm + pixelRight * opsInMm;

                        if (rowPitchMm > 0)
                        {
                            double pixelTop = (0                            - info.PanOffset.Y) / info.Zoom;
                            double pixelBot = (_bgPreviewMainCanvas.Height  - info.PanOffset.Y) / info.Zoom;
                            viewTopMm = pixelTop * rowPitchMm;
                            viewBotMm = pixelBot * rowPitchMm;
                        }
                    }

                    // 實體倍率
                    double screenMmPerPx = _interactionHelper?.ScreenMmPerPixel ?? 0;
                    string magStr = "-";
                    if (info.Zoom > 0 && screenMmPerPx > 0 && opsInMm > 0)
                    {
                        double physicalMag = (info.Zoom * screenMmPerPx) / opsInMm;
                        magStr = $"{physicalMag:F2}x";
                    }

                    text = $"背景預覽 [CAM {camId}] | " +
                           $"位置:({physicalX:F2}, {physicalY:F2}) mm | " +
                           $"X範圍:{viewLeftMm:F1}~{viewRightMm:F1} mm | " +
                           $"Y範圍:{viewTopMm:F1}~{viewBotMm:F1} mm | " +
                           $"座標: ({info.ImageX}, {info.ImageY}) | " +
                           $"亮度: {info.PixelColor.R} | " +
                           $"實體倍率:{magStr}";
                }
            }

            if (InvokeRequired)
                BeginInvoke(new Action(() => lblPixelInfo.Text = text));
            else
                lblPixelInfo.Text = text;
        }

        /// <summary>背景預覽模式：將 panelMainDisplay 上的 SmartCanvas 設為實體倍率 1x（畫面中心不動）。</summary>
        private void SetBgPreviewPhysicalMag1x()
        {
            if (_bgPreviewMainCanvas?.Image == null || _settings == null) return;

            int camIdx = _bgPreviewSelectedCamIndex;
            double[] opsUmArr = _settings.GetCameraOpsUmArray();
            if (camIdx < 0 || camIdx >= opsUmArr.Length) return;

            double opsInMm = opsUmArr[camIdx] / 1000.0;
            double screenMmPerPx = _interactionHelper?.ScreenMmPerPixel ?? 0;
            if (opsInMm <= 0 || screenMmPerPx <= 0) return;

            // scaleFactor=1 for background preview
            float zoom1x = (float)(opsInMm / screenMmPerPx);

            // keep center of current view stable
            float oldZoom = _bgPreviewMainCanvas.Zoom;
            PointF oldPan = _bgPreviewMainCanvas.PanOffset;
            float cx = _bgPreviewMainCanvas.Width / 2f;
            float cy = _bgPreviewMainCanvas.Height / 2f;
            float imgCx = (cx - oldPan.X) / oldZoom;
            float imgCy = (cy - oldPan.Y) / oldZoom;
            float newPanX = cx - imgCx * zoom1x;
            float newPanY = cy - imgCy * zoom1x;

            _bgPreviewMainCanvas.SetView(zoom1x, new PointF(newPanX, newPanY));
        }

        /// <summary>
        /// 清除所有面板的背景預覽。
        /// restoreMilDisplay=true 時恢復 MIL display（用於 btnCameraGrab 等需要回到即時畫面的場景）。
        /// 預設 false，避免在即將重新進入預覽時產生殘影。
        /// </summary>
        private void ClearBackgroundPreview(bool restoreMilDisplay = false)
        {
            // 移除 panelLiveCam 上的 overlay SmartCanvas
            Panel[] livePanels = GetLivePanels();
            if (_bgPreviewBoxes != null)
            {
                for (int i = 0; i < _bgPreviewBoxes.Length; i++)
                {
                    var sc = _bgPreviewBoxes[i];
                    if (sc == null) continue;
                    sc.Click -= BgPreviewPanel_Click;
                    sc.Image = null;
                    livePanels[i].Controls.Remove(sc);
                    sc.Dispose();
                }
                _bgPreviewBoxes = null;
            }

            // 移除 panelMainDisplay 上的 SmartCanvas
            if (_bgPreviewMainCanvas != null)
            {
                _bgPreviewMainCanvas.StatusChanged -= BgPreviewCanvas_StatusChanged;
                _bgPreviewMainCanvas.Image = null;
                panelMainDisplay.Controls.Remove(_bgPreviewMainCanvas);
                _bgPreviewMainCanvas.Dispose();
                _bgPreviewMainCanvas = null;
            }

            // Dispose bitmaps
            if (_bgPreviewBitmaps != null)
            {
                foreach (var bmp in _bgPreviewBitmaps)
                    bmp?.Dispose();
                _bgPreviewBitmaps = null;
            }

            _bgPreviewActive = false;

            if (restoreMilDisplay && _liveCameraManager?.IsAllocated == true)
            {
                // 恢復 primary display（panelLiveCam）+ secondary display（panelMainDisplay）
                foreach (var cam in _liveCameraManager.Cameras)
                    cam.SetPrimaryDisplayVisible(true);
                _liveCameraManager.RefreshMainDisplay();
            }
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
            UpdateLiveDirectionVisual(enabled ? _liveDisplayDirection : null);
        }

        private void lblIoDoMura_Click(object sender, EventArgs e)
        {
            _isMuraDetectPaused = !_isMuraDetectPaused;
            UpdateMuraLed(false);
        }

        private void lblPlcConn_Click(object sender, EventArgs e)
        {
            if (_plcGrabController == null) return;
            _isIoSuspended = !_isIoSuspended;
            if (_isIoSuspended)
            {
                lblPlcConn.BackColor = IecYellow;
                lblPlcConn.ForeColor = Color.Black;
                lblPlcConn.Text = "● IO 暫停 ⏸";
                btnCameraGrab.Enabled = true;
                UpdateGrabButton(_liveCameraManager?.IsLiveGrabbing ?? false);
                btnCameraGrab.BackColor = SystemColors.Control;
                btnCameraGrab.ForeColor = SystemColors.ControlText;
            }
            else
            {
                UpdatePlcConnectionUi(_plcGrabController.IsConnected);
            }
        }

        // PropertyGrid 回傳的 ChangedItem.PropertyDescriptor.Name 可能是 MemberName 或 DisplayName 其中之一，
        // 因此兩種形式都放入集合，避免版本差異導致漏判。
        private static readonly HashSet<string> RecipePropertyNames = new HashSet<string>(StringComparer.Ordinal)
        {
            nameof(InspectionRecipe.HessianMaxFactor), "Hessian Max Factor", "正規值",
            nameof(InspectionRecipe.ErrorValueMean),   "Error Value Mean",   "平均閾值",
            nameof(InspectionRecipe.ErrorValueMax),    "Error Value Max",    "最大閾值",
            nameof(InspectionRecipe.Algorithm),        "去背演算法",
            nameof(InspectionRecipe.RidgeDir),         "Ridge 方向",
        };

        private async void _propertyGrid_PropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            try
            {
            _interactionHelper.HandleSettingsChanged();
            _liveCameraManager?.SetCaptureSettings(_settings);
            _reviewColumnChartHelper?.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            _liveColumnChartHelper?.SetOps(_settings.Cam1_Ops);
            _liveColumnChartHelper?.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            _liveOverviewHelper?.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            _liveRowChartHelper?.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            _reviewRowChartHelper?.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            UpdateRowChartPitch();

            // 抓圖進行中設定變更 → 立刻在 CSV 插入 #CFG
            if (_liveCameraManager?.IsLiveGrabbing == true)
                _inspectionLogService?.ForceWriteConfig(CsvConfigSnapshot.FromSettings(_settings));

            string changedPropertyName = e?.ChangedItem?.PropertyDescriptor?.Name ?? string.Empty;

            // 機台角色變更 → 寫 app-mode.json，不影響 inspection-settings.json
            if (changedPropertyName == nameof(InspectionSettings.AppRole))
            {
                if (_appMode == null) _appMode = new AppModeConfig();
                _appMode.Role = _settings.AppRole;
                _appMode.Save();
                MessageBox.Show("機台角色已儲存，重新開啟程式後生效。",
                    "機台設定", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            // StitchMode 變更 → 清除/恢復 mura 圖 + 重新載入回顧主畫面 + Live 合圖切換
            if (changedPropertyName == nameof(InspectionSettings.StitchMode))
            {
                // Live tab：即時全域合圖
                if (_settings.StitchMode == StitchMode.Global && _liveCameraManager?.IsAllocated == true)
                    _liveCameraManager.EnableGlobalMerge(
                        _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray());
                else
                {
                    _liveCameraManager?.DisableGlobalMerge();
                    _liveRowMeanCache.Clear();
                    _liveRowMaxCache.Clear();
                }

                if (_settings.StitchMode == StitchMode.Global)
                {
                    chartMuraVertical.Series["Mean"].Points.Clear();
                    chartMuraVertical.Series["Max"].Points.Clear();
                    muraChartVerticalLive.Series["Mean"].Points.Clear();
                    muraChartVerticalLive.Series["Max"].Points.Clear();
                }

                // 根據當前選中的回顧縮圖重新載入回顧主畫面
                if (_stitchCoordinator.IsStitchMode)
                {
                    int idx = _galleryManager?.SelectedIndex ?? 0;
                    if (_settings.StitchMode == StitchMode.Global)
                        _stitchCoordinator.ApplyGlobalMergeIfNeeded();
                    else
                        _stitchCoordinator.ShowStitchedCameraInCanvas(idx);
                }
                else if (_imageRepository.FileCount > 0)
                {
                    _stitchCoordinator.ClearStitchedMode();
                    await _presenter.LoadImagesWithPeriodLockAsync(
                        _stitchCoordinator.LastReviewProcessedMode, _interactionHelper.LoadImages);
                    ApplyPostLoadDisplay();
                }
            }

            // OPS/Start 變更 → 即時更新全域合圖佈局（下一幀生效）
            string parentLabel = e?.ChangedItem?.Parent?.Label ?? string.Empty;
            if (parentLabel == "OPS (um)" || parentLabel == "Start (mm)")
            {
                if (_liveCameraManager?.IsGlobalMergeActive == true)
                    _liveCameraManager.RefreshGlobalMergeLayout(
                        _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray());
            }

            // 檢測報表設定變更 → 立刻套用
            if (changedPropertyName == nameof(InspectionSettings.ChartScaleMode))
            {
                _dataStatsPresenter.ApplyChartScaleFromSettings();
            }
            else if (changedPropertyName == nameof(InspectionSettings.ChartYearlyYMax))
                _dataStatsPresenter.ApplyFixedScaleForChart("Yearly", _settings.Chart.YearlyYMax);
            else if (changedPropertyName == nameof(InspectionSettings.ChartMonthlyYMax))
                _dataStatsPresenter.ApplyFixedScaleForChart("Monthly", _settings.Chart.MonthlyYMax);
            else if (changedPropertyName == nameof(InspectionSettings.ChartDailyYMax))
                _dataStatsPresenter.ApplyFixedScaleForChart("Daily", _settings.Chart.DailyYMax);

            // 光源設定即時生效（啟用切換、COM/通道重新偵測、亮度立即套用）
            HandleLightSettingsChanged(changedPropertyName);

            if (changedPropertyName == nameof(InspectionSettings.EnableMuraEnhance))
                ApplyMuraEnhance(_settings.EnableMuraEnhance);

            if (changedPropertyName == nameof(InspectionSettings.EnableReviewEnhance))
                _ = ApplyReviewEnhance(_settings.EnableReviewEnhance);

            // 演算法切換 → 更新 UI 鎖定 + 載入/清除背景 bin
            if (changedPropertyName == nameof(InspectionRecipe.Algorithm) ||
                changedPropertyName == "去背演算法")
            {
                if (_liveCameraManager.IsAllocated) LoadBackgroundBins();
                UpdateStandardBgSubLockState();
            }

            bool isRecipeChange = RecipePropertyNames.Contains(changedPropertyName);

            // 有影像且為配方參數變更 → 重載（始終用 processed 模式，因為配方只影響演算法輸出）
            if (isRecipeChange && _imageRepository.FileCount > 0)
            {
                _stitchCoordinator.LastReviewProcessedMode = true;
                _settings.EnableReviewEnhance = true;
                propertyGridSettings.Refresh();
                _stitchCoordinator.ClearStitchedMode();
                await _presenter.LoadImagesWithPeriodLockAsync(true, _interactionHelper.LoadImages);
                ApplyPostLoadDisplay();
            }
            }
            catch (Exception ex) { Trace.WriteLine($"[PropertyValueChanged] {ex}"); }
        }

        private async void btnSelectFolder_Click(object sender, EventArgs e)
        {
            try
            {
            _interactionHelper.SelectAndLoadFolder();
            _presenter.UpdatePeriodNavigationState();
            _stitchCoordinator.LastReviewProcessedMode = false;
            _settings.EnableReviewEnhance = false;
            propertyGridSettings.Refresh();

            // 同步載入序號清單並填充所有序號 ComboBox（Review + Data）
            if (_imageRepository.FileCount > 0)
            {
                var reviewPath = UserSessionState.LastDataPath;
                if (!string.IsNullOrWhiteSpace(reviewPath))
                    _dataStatsPresenter.SyncFromReviewFolder(reviewPath);

                // Initialize 期間 _updating=true 不觸發 PeriodSelectionChanged，手動同步
                var current = _dateTimeNavigator.GetCurrentPeriodOrDefault(DateTime.MinValue);
                if (current != DateTime.MinValue)
                    _dataStatsPresenter.SyncGrabIdFromTime(current);
            }

            _stitchCoordinator.ClearStitchedMode();
            _dataStatsPresenter.SetReviewGroupBoxes(false);
            await _presenter.LoadImagesWithPeriodLockAsync(false, LoadImagesWithReviewConfig);
            ApplyPostLoadDisplay();
            }
            catch (Exception ex) { Trace.WriteLine($"[btnSelectFolder_Click] {ex}"); }
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
        { try { _interactionHelper.SaveCanvasView(); _stitchCoordinator.ClearStitchedMode(); await _presenter.MovePeriodAsync(-1, _stitchCoordinator.LastReviewProcessedMode, LoadImagesWithReviewConfig); ApplyPostLoadDisplay(); } catch (Exception ex) { Trace.WriteLine($"[btnPeriodPrev] {ex}"); } }

        private async void btnPeriodNext_Click(object sender, EventArgs e)
        { try { _interactionHelper.SaveCanvasView(); _stitchCoordinator.ClearStitchedMode(); await _presenter.MovePeriodAsync(+1, _stitchCoordinator.LastReviewProcessedMode, LoadImagesWithReviewConfig); ApplyPostLoadDisplay(); } catch (Exception ex) { Trace.WriteLine($"[btnPeriodNext] {ex}"); } }

        /// <summary>cbDate/cbTime 手動滾動時載入對應圖片（同 btnPeriodPrev/Next）。
        /// _dataStatsPresenter.GrabIdNavGuard 時跳過（由 OnReviewGrabIdChanged 等程式碼觸發的 NavigateToDateTime）。</summary>
        private async void OnPeriodComboChanged()
        {
            if (_dataStatsPresenter.GrabIdNavGuard.IsSet) return;
            if (_imageRepository.FileCount == 0) return;
            try
            {
            _interactionHelper.SaveCanvasView();
            _stitchCoordinator.ClearStitchedMode();
            _dataStatsPresenter.SetReviewGroupBoxes(false);
            await _presenter.LoadImagesWithPeriodLockAsync(_stitchCoordinator.LastReviewProcessedMode, LoadImagesWithReviewConfig);
            ApplyPostLoadDisplay();
            }
            catch (Exception ex) { Trace.WriteLine($"[OnPeriodComboChanged] {ex}"); }
        }
        // ==========================================
        // --- 右側面板：初始化 ---
        // ==========================================

        private void InitializeRightPanelControls()
        {
            SetupCameraTab();
            SetupSystemTab();
        }

        /// <summary>
        /// 用 reflection 調整 PropertyGrid 標籤欄寬度至最長屬性名稱剛好容納。
        /// PropertyGrid 無公開 API 可設欄寬，透過內部 gridView.MoveSplitterTo() 實現。
        /// 注意：MoveSplitterTo 的參數是整個標籤欄寬（含左側 indent 區域），
        /// 因此需在純文字寬度之外加上 indent（約 16px）＋ 右側留白。
        /// </summary>
        private static void AutoFitPropertyGridLabelColumn(System.Windows.Forms.PropertyGrid grid)
        {
            try
            {
                var gridViewField = grid.GetType().GetField("gridView",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var gridView = gridViewField?.GetValue(grid);
                if (gridView == null) return;

                // 以所有可見屬性的 DisplayName 量測最大文字寬度
                int maxTextWidth = 0;
                foreach (System.ComponentModel.PropertyDescriptor pd in
                    System.ComponentModel.TypeDescriptor.GetProperties(grid.SelectedObject))
                {
                    if (!pd.IsBrowsable) continue;
                    string label = pd.DisplayName ?? pd.Name;
                    int w = System.Windows.Forms.TextRenderer.MeasureText(
                        label, grid.Font).Width;
                    if (w > maxTextWidth) maxTextWidth = w;
                }

                // indent：PropertyGrid 標籤欄左側的展開箭頭區域固定約 16px
                // rightMargin：文字右側留白，避免緊貼分隔線
                const int indent      = 16;
                const int rightMargin = 8;
                var moveSplitter = gridView.GetType().GetMethod("MoveSplitterTo",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                moveSplitter?.Invoke(gridView, new object[] { indent + maxTextWidth + rightMargin });
            }
            catch { /* reflection 失敗時保留預設欄寬，不影響功能 */ }
        }

        private void SetupCameraTab()
        {
            const int ExpMin    =     1;   // μs
            const int ExpMaxCap = 10000;   // μs 硬上限
            const int LrMin     =   100;   // Hz
            const int LrMax     = 10000;   // Hz
            const int HtMin     =   100;   // px
            const int HtMax     = 10000;   // px

            // ── 7 台相機控制項陣列（存為 Form 欄位，供 SyncFromCamera 存取）────
            var acq = _settings.Acquisition;
            _expBars = new[] { trackBarExpCam1, trackBarExpCam2, trackBarExpCam3, trackBarExpCam4, trackBarExpCam5, trackBarExpCam6, trackBarExpCam7 };
            _expNums = new[] { numExpCam1,      numExpCam2,      numExpCam3,      numExpCam4,      numExpCam5,      numExpCam6,      numExpCam7      };
            _lrBars  = new[] { trackBarLrCam1,  trackBarLrCam2,  trackBarLrCam3,  trackBarLrCam4,  trackBarLrCam5,  trackBarLrCam6,  trackBarLrCam7  };
            _lrNums  = new[] { numLrCam1,       numLrCam2,       numLrCam3,       numLrCam4,       numLrCam5,       numLrCam6,       numLrCam7       };
            _htBars  = new[] { trackBarHtCam1,  trackBarHtCam2,  trackBarHtCam3,  trackBarHtCam4,  trackBarHtCam5,  trackBarHtCam6,  trackBarHtCam7  };
            _htNums  = new[] { numHtCam1,       numHtCam2,       numHtCam3,       numHtCam4,       numHtCam5,       numHtCam6,       numHtCam7       };

            // ── CAM All 控制項事件綁定（控制項已在 Designer.cs 定義）──────────
            _expAllBar = trackBarExpAll; _expAllNum = numExpAll;
            _lrAllBar  = trackBarLrAll;  _lrAllNum  = numLrAll;
            _htAllBar  = trackBarHtAll;  _htAllNum  = numHtAll;

            BindAllSync(_expAllBar, _expAllNum, _expBars, _expNums,
                (j, v) => _liveCameraManager?.SetExposureForCamera(j + 1, v),
                (j, v) => { acq.CameraExposureTimeUs[j] = v; ConfigManager.SaveAcquisitionSettings(acq); });

            BindAllSync(_lrAllBar, _lrAllNum, _lrBars, _lrNums,
                (j, v) => _liveCameraManager?.SetLineRateForCamera(j + 1, v),
                (j, v) => { acq.CameraLineRateHz[j] = v; ConfigManager.SaveAcquisitionSettings(acq); },
                () => {
                    // 同步所有 cam 的 exp max（每台 LR 都同值，算一次套到所有 cam）
                    int newMax = (int)acq.CameraLineRateHz[0];
                    int expMax = newMax <= 0 ? ExpMaxCap : Math.Max(ExpMin, Math.Min(ExpMaxCap, (int)(900000.0 / newMax)));
                    for (int i = 0; i < CameraCount; i++) UpdateExpMaxAndClampColor(i, expMax);
                    UpdateRowChartPitch();
                });

            BindAllSync(_htAllBar, _htAllNum, _htBars, _htNums,
                (j, v) => _liveCameraManager?.SetGrabHeightForCamera(j + 1, v),
                (j, v) => { acq.CameraGrabHeight[j] = v; ConfigManager.SaveAcquisitionSettings(acq); },
                () => {
                    _liveCameraManager?.RefreshMainDisplay();
                    if (_settings.StitchMode == StitchMode.Global && _liveCameraManager?.IsGlobalMergeActive == true)
                        _liveCameraManager.RefreshGlobalMergeLayout(_settings.Ops.ToArray(), _settings.StartPosition.ToArray());
                });

            for (int i = 0; i < CameraCount; i++)
            {
                int idx   = i;
                int camId = i + 1;

                // 動態曝光上限（依各台自己的 LR）
                int CalcExpMax()
                {
                    int lrHz = (int)acq.CameraLineRateHz[idx];
                    return lrHz <= 0 ? ExpMaxCap : Math.Max(ExpMin, Math.Min(ExpMaxCap, (int)(900000.0 / lrHz)));
                }

                // ── 曝光時間 ────────────────────────────────────────────
                int expMax = CalcExpMax();
                BindBidirectionalSync(_expBars[idx], _expNums[idx], camId,
                    ExpMin, expMax, (int)acq.CameraExposureTimeUs[idx],
                    v => { acq.CameraExposureTimeUs[idx] = v; ConfigManager.SaveAcquisitionSettings(acq); },
                    v => _liveCameraManager?.SetExposureForCamera(camId, v));

                // ── 線掃速率 ────────────────────────────────────────────
                BindBidirectionalSync(_lrBars[idx], _lrNums[idx], camId,
                    LrMin, LrMax, (int)acq.CameraLineRateHz[idx],
                    v => { acq.CameraLineRateHz[idx] = v; ConfigManager.SaveAcquisitionSettings(acq); },
                    v => _liveCameraManager?.SetLineRateForCamera(camId, v),
                    () => { UpdateExpMaxAndClampColor(idx, CalcExpMax()); if (idx == 0) UpdateRowChartPitch(); });

                // ── 擷取高度 ────────────────────────────────────────────
                BindBidirectionalSync(_htBars[idx], _htNums[idx], camId,
                    HtMin, HtMax, acq.CameraGrabHeight[idx],
                    v => { acq.CameraGrabHeight[idx] = v; ConfigManager.SaveAcquisitionSettings(acq); },
                    v => _liveCameraManager?.SetGrabHeightForCamera(camId, v),
                    () => {
                        _liveCameraManager?.RefreshMainDisplay();
                        if (_settings.StitchMode == StitchMode.Global && _liveCameraManager?.IsGlobalMergeActive == true)
                            _liveCameraManager.RefreshGlobalMergeLayout(_settings.Ops.ToArray(), _settings.StartPosition.ToArray());
                    });
                _htBars[idx].SmallChange = 64; _htBars[idx].LargeChange = 512;
            }

            // ── CAM All 範圍設定 ──────────────────────────────────────────
            int expAllMax = ExpMaxCap;
            for (int i = 0; i < CameraCount; i++)
            {
                int lrHz = (int)acq.CameraLineRateHz[i];
                int m = lrHz <= 0 ? ExpMaxCap : Math.Max(ExpMin, Math.Min(ExpMaxCap, (int)(900000.0 / lrHz)));
                if (m < expAllMax) expAllMax = m;
            }
            _expAllBar.Minimum = ExpMin; _expAllBar.Maximum = expAllMax;
            _expAllNum.Minimum = ExpMin; _expAllNum.Maximum = expAllMax;
            _expAllBar.Value = Math.Max(ExpMin, Math.Min(expAllMax, (int)acq.CameraExposureTimeUs[0]));
            _expAllNum.Value = _expAllBar.Value;

            _lrAllBar.Minimum = LrMin; _lrAllBar.Maximum = LrMax;
            _lrAllNum.Minimum = LrMin; _lrAllNum.Maximum = LrMax;
            _lrAllBar.Value = Math.Max(LrMin, Math.Min(LrMax, (int)acq.CameraLineRateHz[0]));
            _lrAllNum.Value = _lrAllBar.Value;

            _htAllBar.Minimum = HtMin; _htAllBar.Maximum = HtMax;
            _htAllNum.Minimum = HtMin; _htAllNum.Maximum = HtMax;
            _htAllBar.Value = Math.Max(HtMin, Math.Min(HtMax, acq.CameraGrabHeight[0]));
            _htAllNum.Value = _htAllBar.Value;
            _htAllBar.SmallChange = 64; _htAllBar.LargeChange = 512;

            // 滾輪每格移動 1（攔截原生 3 格行為）
            RegisterWheelInterceptors(_expBars);
            RegisterWheelInterceptors(_lrBars);
            RegisterWheelInterceptors(_htBars);
            RegisterWheelInterceptors(new[] { _expAllBar, _lrAllBar, _htAllBar });
        }

        /// <summary>
        /// TrackBar ↔ NumericUpDown 雙向同步綁定：
        /// - 拖曳中：抑制硬體寫入，MouseUp 立即寫入。
        /// - 滾輪 / 鍵盤箭頭 / NUD 輸入：1 秒 debounce 才寫硬體（避免高頻 MIL 寫入造成卡頓）。
        /// </summary>
        private void BindBidirectionalSync(
            TrackBar bar, NumericUpDown num, int camId,
            int min, int max, int initialValue,
            Action<int> saveSetting, Action<int> writeHardware,
            Action postAction = null)
        {
            int clamped = Math.Max(min, Math.Min(max, initialValue));
            bar.Minimum = min; bar.Maximum = max; bar.TickFrequency = TickFreq;
            num.Minimum = min; num.Maximum = max;
            bar.Value = clamped; num.Value = clamped;

            bool syncing = false;

            // 滾輪/鍵盤/NUD 等非拖曳輸入 → 1s debounce 才寫硬體
            var debounce = new System.Windows.Forms.Timer { Interval = 1000 };
            int pendingValue = clamped;
            bool hasPending = false;
            debounce.Tick += (s, e) =>
            {
                debounce.Stop();
                if (!hasPending) return;
                hasPending = false;
                writeHardware(pendingValue);
                postAction?.Invoke();   // 硬體寫完後再 refresh（例：HT 改變後重新載入主畫面 buffer）
            };
            void ScheduleWrite(int v)
            {
                pendingValue = v;
                hasPending = true;
                debounce.Stop();
                debounce.Start();
            }

            bar.MouseDown += (s, e) => _dragging.Add(bar);
            bar.MouseUp += (s, e) =>
            {
                _dragging.Remove(bar);
                // 拖曳結束 → 立即寫入並取消 debounce
                debounce.Stop();
                hasPending = false;
                writeHardware(bar.Value);
                postAction?.Invoke();   // 硬體寫完後再 refresh（例：HT 改變後重新載入主畫面 buffer）
            };
            bar.ValueChanged += (s, e) =>
            {
                if (syncing || _syncingFromHw) return; syncing = true;
                num.Value = bar.Value;
                saveSetting(bar.Value);
                if (!_dragging.Contains(bar)) ScheduleWrite(bar.Value);
                postAction?.Invoke();
                syncing = false;
            };
            num.ValueChanged += (s, e) =>
            {
                if (syncing || _syncingFromHw) return; syncing = true;
                int v = (int)num.Value;
                bar.Value = Math.Max(min, Math.Min(max, v));
                saveSetting(v);
                ScheduleWrite(v);
                postAction?.Invoke();
                syncing = false;
            };
        }

        /// <summary>
        /// CAM All → CAM1~7 同步：
        /// - 拖曳 All：MouseUp 才寫硬體；滾輪/鍵盤/NUD：1s debounce 才寫硬體。
        /// - 寫硬體完成後才同步 CAM1~7 的 bar/num 顯示（避免 UI 比硬體快）。
        /// </summary>
        private void BindAllSync(TrackBar barAll, NumericUpDown numAll,
            TrackBar[] bars, NumericUpDown[] nums,
            Action<int, int> writeHardwareForCam,    // (camIdx0based, value)
            Action<int, int> saveSettingForCam,      // (camIdx0based, value)
            Action postWriteAll = null)
        {
            bool allSyncing = false;
            var debounce = new System.Windows.Forms.Timer { Interval = 1000 };
            int pendingValue = barAll.Value;
            bool hasPending = false;

            void Apply(int v)
            {
                // 1. 寫硬體（所有 7 台）
                for (int j = 0; j < bars.Length; j++)
                    writeHardwareForCam(j, v);
                // 2. 寫 settings
                for (int j = 0; j < bars.Length; j++)
                    saveSettingForCam(j, v);
                // 3. 同步 cam 的 bar/num 顯示（用 _syncingFromHw 跳過 BindBidirectionalSync 的 ScheduleWrite/saveSetting）
                _syncingFromHw = true;
                try
                {
                    for (int j = 0; j < bars.Length; j++)
                    {
                        int clamped = Math.Max(bars[j].Minimum, Math.Min(bars[j].Maximum, v));
                        bars[j].Value = clamped;
                        nums[j].Value = clamped;
                    }
                }
                finally { _syncingFromHw = false; }
                postWriteAll?.Invoke();
            }

            debounce.Tick += (s, e) =>
            {
                debounce.Stop();
                if (!hasPending) return;
                hasPending = false;
                Apply(pendingValue);
            };
            void Schedule(int v)
            {
                pendingValue = v;
                hasPending = true;
                debounce.Stop();
                debounce.Start();
            }

            barAll.MouseDown += (s, e) => _dragging.Add(barAll);
            barAll.MouseUp += (s, e) =>
            {
                _dragging.Remove(barAll);
                debounce.Stop();
                hasPending = false;
                Apply(barAll.Value);
            };
            barAll.ValueChanged += (s, e) => {
                if (allSyncing || _syncingFromHw) return; allSyncing = true;
                numAll.Value = barAll.Value;
                if (!_dragging.Contains(barAll)) Schedule(barAll.Value);
                allSyncing = false;
            };
            numAll.ValueChanged += (s, e) => {
                if (allSyncing || _syncingFromHw) return; allSyncing = true;
                int v = (int)numAll.Value;
                barAll.Value = Math.Max(barAll.Minimum, Math.Min(barAll.Maximum, v));
                Schedule(v);
                allSyncing = false;
            };
        }

        /// <summary>
        /// 更新曝光 TrackBar/NUD 的 Maximum；若現有值被夾緊則將 NUD 背景色改為 OrangeRed，
        /// 否則恢復預設白色。由 LR ValueChanged 呼叫。
        /// </summary>
        private void UpdateExpMaxAndClampColor(int idx, int newMax)
        {
            _expBars[idx].Maximum = newMax;
            _expNums[idx].Maximum = newMax;
            if (_expBars[idx].Value > newMax)
            {
                _expBars[idx].Value = newMax;
                _expNums[idx].Value = newMax;
                _expNums[idx].BackColor = Color.OrangeRed;   // 夾緊警告
            }
            else
            {
                _expNums[idx].BackColor = SystemColors.Window;
            }
            UpdateExpAllMax();
        }

        private void UpdateExpAllMax()
        {
            if (_expAllBar == null || _expBars == null) return;
            int minMax = _expBars[0].Maximum;
            for (int i = 1; i < _expBars.Length; i++)
                if (_expBars[i].Maximum < minMax) minMax = _expBars[i].Maximum;
            _expAllBar.Maximum = minMax;
            _expAllNum.Maximum = minMax;
            if (_expAllBar.Value > minMax)
            {
                _expAllBar.Value = minMax;
                _expAllNum.Value = minMax;
            }
        }

        private void SetupSystemTab()
        {
            // ── 即時 Telemetry ListView（取代靜態 5 欄設定表）─────────────
            _telemetryPresenter = new LiveTelemetryPresenter(listViewCameras);
            _telemetryPresenter.Initialize(SystemSettings.CreateDefault().CameraDevices);

            // ── 影像引擎常數 ──────────────────────────────────────────────
            listViewEngine.Columns.Add("參數", 160);
            listViewEngine.Columns.Add("值",    90);

            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxWidth",            InspectionEngineConfig.MaxWidth.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxHeight",           InspectionEngineConfig.MaxHeight.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxThumbnailSide",    InspectionEngineConfig.MaxThumbnailSide.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultBgSigma",      InspectionEngineConfig.DefaultBgSigma.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultRidgeSigma",   InspectionEngineConfig.DefaultRidgeSigma.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultHessianMax",   InspectionEngineConfig.DefaultHessianMaxFactor.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultRidgeMode",    InspectionEngineConfig.DefaultRidgeMode }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "SaveResizeScale",     InspectionEngineConfig.DefaultSaveResizeScale.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "SaveJpgQuality",      InspectionEngineConfig.DefaultSaveJpgQuality.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "───", "── 圖表引擎 ──" }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxOverviewPoints", "2000" }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "TelemetryInterval", "500 ms" }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "OverviewRefresh",   "FPS-sync" }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DownsampleMode",    "Max-Window" }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "OverlapMean",       "Average" }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "OverlapMax",        "Maximum" }));
            AutoFitListViewColumns(listViewEngine);

            // ── 硬體參數 ─────────────────────────────────────────────────
            listViewHardware.Columns.Add("參數", 120);
            listViewHardware.Columns.Add("值",   120);

            // ── CPU / RAM ──
            try
            {
                using (var cpuSearcher = new ManagementObjectSearcher("SELECT Name, NumberOfCores, NumberOfLogicalProcessors FROM Win32_Processor"))
                foreach (var obj in cpuSearcher.Get())
                {
                    listViewHardware.Items.Add(new ListViewItem(new[] { "CPU",       obj["Name"]?.ToString().Trim() ?? "N/A" }));
                    listViewHardware.Items.Add(new ListViewItem(new[] { "CPU_Cores",  $"{obj["NumberOfCores"]}C / {obj["NumberOfLogicalProcessors"]}T" }));
                    break; // 只取第一顆
                }

                using (var memSearcher = new ManagementObjectSearcher("SELECT Capacity, Speed, SMBIOSMemoryType FROM Win32_PhysicalMemory"))
                {
                    var sticks = memSearcher.Get().Cast<ManagementObject>().ToArray();
                    int count = sticks.Length;
                    ulong totalBytes = 0;
                    int speed = 0;
                    int memType = 0;
                    foreach (var stick in sticks)
                    {
                        totalBytes += (ulong)stick["Capacity"];
                        if (speed == 0 && stick["Speed"] != null)
                            speed = Convert.ToInt32(stick["Speed"]);
                        if (memType == 0 && stick["SMBIOSMemoryType"] != null)
                            memType = Convert.ToInt32(stick["SMBIOSMemoryType"]);
                    }
                    double perStickGb = count > 0 ? (totalBytes / (double)count) / (1024.0 * 1024 * 1024) : 0;
                    string ddrGen = memType == 34 ? "DDR5" : memType == 26 ? "DDR4" : memType == 24 ? "DDR3" : "DDR";
                    string speedStr = speed > 0 ? $"-{speed}" : "";
                    listViewHardware.Items.Add(new ListViewItem(new[] { "RAM",
                        $"{totalBytes / (1024.0 * 1024 * 1024):F0} GB ({count}×{perStickGb:F0}GB {ddrGen}{speedStr})" }));
                }
            }
            catch { /* WMI 非關鍵，忽略 */ }

            // ── GPU ──
            try
            {
                // Registry 查 64-bit VRAM（qwMemorySize），避免 WMI uint32 溢位
                var regVram = new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase);
                try
                {
                    using (var videoKey = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(@"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"))
                    if (videoKey != null)
                    {
                        foreach (string sub in videoKey.GetSubKeyNames())
                        {
                            if (!int.TryParse(sub, out _)) continue;
                            using (var sk = videoKey.OpenSubKey(sub))
                            {
                                if (sk == null) continue;
                                string desc = sk.GetValue("DriverDesc") as string;
                                if (string.IsNullOrEmpty(desc)) continue;
                                object qw = sk.GetValue("HardwareInformation.qwMemorySize");
                                if (qw is long qwVal && qwVal > 0)
                                    regVram[desc] = qwVal;
                                else if (qw is byte[] qwBytes && qwBytes.Length >= 8)
                                    regVram[desc] = BitConverter.ToInt64(qwBytes, 0);
                            }
                        }
                    }
                }
                catch { /* registry 非關鍵 */ }

                using (var gpuSearcher = new ManagementObjectSearcher("SELECT Name, AdapterRAM FROM Win32_VideoController"))
                foreach (ManagementObject obj in gpuSearcher.Get())
                {
                    string gpuName = obj["Name"]?.ToString() ?? "N/A";
                    long vramBytes;
                    if (regVram.TryGetValue(gpuName, out long regBytes) && regBytes > 0)
                        vramBytes = regBytes;
                    else
                        vramBytes = Convert.ToUInt32(obj["AdapterRAM"]);

                    double vramGb = vramBytes / (1024.0 * 1024 * 1024);
                    string vramStr = vramGb >= 1.0 ? $"{vramGb:F1} GB" : $"{vramBytes / (1024.0 * 1024):F0} MB";
                    listViewHardware.Items.Add(new ListViewItem(new[] { "GPU",      gpuName }));
                    listViewHardware.Items.Add(new ListViewItem(new[] { "GPU_VRAM", vramStr }));
                }
            }
            catch { /* WMI 非關鍵，忽略 */ }

            // ── Grabber（PCIe frame grabber）──
            try
            {
                using (var grabSearcher = new ManagementObjectSearcher(
                    "SELECT Name, DeviceID FROM Win32_PnPEntity WHERE Name LIKE '%frame grabber%' OR Name LIKE '%Frame Grabber%'"))
                foreach (ManagementObject obj in grabSearcher.Get())
                {
                    string grabName = obj["Name"]?.ToString() ?? "N/A";
                    string devId = obj["DeviceID"]?.ToString() ?? "";
                    listViewHardware.Items.Add(new ListViewItem(new[] { "Grabber", grabName }));

                    if (!devId.StartsWith("PCI\\", StringComparison.OrdinalIgnoreCase)) continue;

                    // PCIe link speed/width via PowerShell Get-PnpDeviceProperty
                    try
                    {
                        var psi = new System.Diagnostics.ProcessStartInfo
                        {
                            FileName = "powershell.exe",
                            Arguments = $"-NoProfile -Command \"Get-PnpDeviceProperty -InstanceId '{devId}' | " +
                                "Where-Object { $_.KeyName -match 'CurrentLinkSpeed|CurrentLinkWidth' } | " +
                                "ForEach-Object { $_.KeyName + '=' + $_.Data }\"",
                            RedirectStandardOutput = true,
                            UseShellExecute = false,
                            CreateNoWindow = true
                        };
                        using (var proc = System.Diagnostics.Process.Start(psi))
                        {
                            string output = proc.StandardOutput.ReadToEnd();
                            proc.WaitForExit(5000);

                            int linkSpeed = 0, linkWidth = 0;
                            foreach (string line in output.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
                            {
                                var parts = line.Split('=');
                                if (parts.Length != 2) continue;
                                if (parts[0].Contains("CurrentLinkSpeed")) int.TryParse(parts[1].Trim(), out linkSpeed);
                                if (parts[0].Contains("CurrentLinkWidth")) int.TryParse(parts[1].Trim(), out linkWidth);
                            }

                            if (linkSpeed > 0 && linkWidth > 0)
                            {
                                string[] genNames = { "?", "Gen1", "Gen2", "Gen3", "Gen4", "Gen5" };
                                double[] genGTs = { 0, 2.5, 5, 8, 16, 32 };
                                string gen = linkSpeed < genNames.Length ? genNames[linkSpeed] : $"Gen{linkSpeed}";
                                double bwGBs = linkSpeed < genGTs.Length
                                    ? genGTs[linkSpeed] * linkWidth * 0.8 / 8.0   // 8b/10b for Gen1-2, 128b/130b for Gen3+
                                    : 0;
                                if (linkSpeed >= 3 && linkSpeed < genGTs.Length)
                                    bwGBs = genGTs[linkSpeed] * linkWidth * (128.0 / 130.0) / 8.0;

                                listViewHardware.Items.Add(new ListViewItem(new[] {
                                    "Grabber_PCIe", $"{gen} x{linkWidth} ({bwGBs:F1} GB/s)" }));
                            }
                        }
                    }
                    catch { /* PowerShell 非關鍵 */ }
                }
            }
            catch { }

            // ── 磁碟（所有固定碟） ──
            try
            {
                string capRoot = _settings?.CaptureRootPath ?? @"D:\AniloxCaptures";
                string capDrive = Path.GetPathRoot(capRoot)?.TrimEnd('\\') ?? "";
                foreach (var di in DriveInfo.GetDrives())
                {
                    if (di.DriveType != DriveType.Fixed || !di.IsReady) continue;
                    double totalGb = di.TotalSize / (1024.0 * 1024 * 1024);
                    double freeGb  = di.AvailableFreeSpace / (1024.0 * 1024 * 1024);
                    string label   = di.Name.TrimEnd('\\');
                    string suffix  = label.Equals(capDrive, StringComparison.OrdinalIgnoreCase) ? " [存圖]" : "";
                    listViewHardware.Items.Add(new ListViewItem(new[] {
                        $"Disk_{label}", $"{di.DriveFormat}  {freeGb:F1} / {totalGb:F1} GB free{suffix}" }));
                }
            }
            catch { /* 非關鍵，忽略 */ }

            // ── 螢幕 ──
            try
            {
                IntPtr hdc = GetDC(IntPtr.Zero);
                int horzMm   = GetDeviceCaps(hdc, 4);   // HORZSIZE (mm)
                int vertMm   = GetDeviceCaps(hdc, 6);   // VERTSIZE (mm)
                int horzPx   = GetDeviceCaps(hdc, 8);   // HORZRES (px, 含 DPI 縮放)
                int vertPx   = GetDeviceCaps(hdc, 10);  // VERTRES (px, 含 DPI 縮放)
                int logDpiX  = GetDeviceCaps(hdc, 88);  // LOGPIXELSX
                int logDpiY  = GetDeviceCaps(hdc, 90);  // LOGPIXELSY
                ReleaseDC(IntPtr.Zero, hdc);

                int nativeW  = (int)Math.Round(horzPx * logDpiX / 96.0);
                int nativeH  = (int)Math.Round(vertPx * logDpiY / 96.0);
                int scalePct = (int)Math.Round(logDpiX / 96.0 * 100);

                double screenMmPerPx = (double)horzMm / horzPx;
                listViewHardware.Items.Add(new ListViewItem(new[] { "ScreenSize",   $"{horzMm / 10.0:F1} × {vertMm / 10.0:F1} cm" }));
                listViewHardware.Items.Add(new ListViewItem(new[] { "NativeRes",    $"{nativeW} × {nativeH}" }));
                listViewHardware.Items.Add(new ListViewItem(new[] { "EffectiveRes", $"{horzPx} × {vertPx}" }));
                listViewHardware.Items.Add(new ListViewItem(new[] { "DpiScale",     $"{scalePct}%" }));
                listViewHardware.Items.Add(new ListViewItem(new[] { "mm/px",        $"{screenMmPerPx:F4}" }));

                _interactionHelper?.SetScreenMmPerPixel(screenMmPerPx);
                _liveCameraManager?.SetScreenMmPerPixel(screenMmPerPx);
            }
            catch { /* 非關鍵資訊，忽略 */ }

            // ── Storage 模式：磁碟 + 清理狀態（即時，Timer 更新）──
            if (_appMode?.Role == MachineRole.Storage)
            {
                listViewHardware.Items.Add(new ListViewItem(new[] { "───", "── Storage 狀態 ──" }));
                _storageDiskFreeRow  = AddResMonItem("Disk_Free",    "—");
                _storageLastCleanRow = AddResMonItem("Last_Cleanup", "—");
                _retentionService.OnCleanupCompleted += r =>
                {
                    if (_storageLastCleanRow == null) return;
                    string text = r.FreedBytes > 0
                        ? $"{r.DeletedDayFolders} folders, {r.FreedBytes / (1024.0 * 1024):F1} MB  ({DateTime.Now:HH:mm:ss})"
                        : $"OK  ({DateTime.Now:HH:mm:ss})";
                    BeginInvoke((Action)(() => _storageLastCleanRow.SubItems[1].Text = text));
                };
            }
            else
            {
                // ── Resource Monitor（即時資源用量，Timer 更新）──
                listViewHardware.Items.Add(new ListViewItem(new[] { "───", "── Resource Monitor ──" }));
                _resMonRawSize     = AddResMonItem("RawSize",     "—");
                _resMonGpuTime     = AddResMonItem("GPU_Time",    "—");
                _resMonSaveSize    = AddResMonItem("Save/Frame",  "—");
                _resMonDiskWrite   = AddResMonItem("DiskWrite",   "—");
                _resMonFrames      = AddResMonItem("Frames",      "—");
                _resMonRamUsed     = AddResMonItem("RAM_Used",    "—");
                _resMonVramEst     = AddResMonItem("VRAM_Est",    "—");
            }
            AutoFitListViewColumns(listViewHardware);

            // ── Telemetry Timer（每 500ms 更新 ListView + SyncFromHardware）─
            _telemetryTimer = new System.Windows.Forms.Timer { Interval = 500 };
            _telemetryTimer.Tick += TelemetryTimer_Tick;
            _telemetryTimer.Start();

            // ── Live Overview Timer（chartLiveOverview 全覽圖，動態跟隨最大 FPS）──
            _liveOverviewTimer = new System.Windows.Forms.Timer { Interval = 100 };
            _liveOverviewTimer.Tick += LiveOverviewTimer_Tick;
            _liveOverviewTimer.Start();
        }

        // ==========================================
        // --- Telemetry Timer ---
        // ==========================================

        private bool _telemetryFitDone;

        private void TelemetryTimer_Tick(object sender, EventArgs e)
        {
            // 連線狀態不受相機釋放影響，先於 gate 更新
            UpdateConnectionStatusLabels();

            if (_liveCameraManager == null || _liveCameraManager.IsReleasing) return;

            _telemetryPresenter?.Update(_liveCameraManager.Cameras);

            if (!_telemetryFitDone)
            {
                AutoFitListViewColumns(listViewCameras);
                _telemetryFitDone = true;
            }

            if (_liveCameraManager.IsAllocated)
            {
                SyncCameraParamsFromHardware();

                // 動態調整 Live Overview Timer：跟隨最大 FPS，下限 50ms（20Hz），上限 500ms（2Hz）
                double maxFps = 0;
                foreach (var cam in _liveCameraManager.Cameras)
                {
                    double fps = cam.CurrentFps;
                    if (fps > maxFps) maxFps = fps;
                }
                if (maxFps > 0.1 && _liveOverviewTimer != null)
                {
                    int interval = Math.Max(50, Math.Min(500, (int)(1000.0 / maxFps)));
                    if (_liveOverviewTimer.Interval != interval)
                        _liveOverviewTimer.Interval = interval;
                }
            }

            // ── Resource Monitor 更新 ──
            UpdateResourceMonitor();
        }

        private ListViewItem AddResMonItem(string key, string value)
        {
            var item = new ListViewItem(new[] { key, value });
            listViewHardware.Items.Add(item);
            return item;
        }

        private void UpdateResourceMonitor()
        {
            try
            {
                var cameras = _liveCameraManager?.Cameras;
                if (cameras == null || cameras.Count == 0) return;

                // 取第一台有效相機的 frame size
                int w = 0, h = 0;
                long maxGpuMs = 0;
                long totalSaveBytes = 0;
                long totalFrames = 0;
                long lastSaveBytes = 0;

                foreach (var cam in cameras)
                {
                    if (cam == null) continue;
                    if (cam.FrameWidth > 0 && w == 0) { w = cam.FrameWidth; h = cam.FrameHeight; }
                    if (cam.LastGpuTimeMs > maxGpuMs) maxGpuMs = cam.LastGpuTimeMs;
                    if (cam.LastSaveBytesTotal > lastSaveBytes) lastSaveBytes = cam.LastSaveBytesTotal;
                    totalSaveBytes += cam.SessionSaveBytes;
                    totalFrames += cam.SessionFrameCount;
                }

                long rawBytes = (long)w * h;
                double rawMB = rawBytes / (1024.0 * 1024);

                if (_resMonRawSize == null) return;
                _resMonRawSize.SubItems[1].Text = w > 0 ? $"{w}×{h} = {rawMB:F1} MB" : "—";
                _resMonGpuTime.SubItems[1].Text = maxGpuMs > 0 ? $"{maxGpuMs} ms" : "—";
                _resMonSaveSize.SubItems[1].Text = lastSaveBytes > 0 ? $"{lastSaveBytes / 1024.0:F0} KB" : "—";
                _resMonDiskWrite.SubItems[1].Text = totalSaveBytes > 0
                    ? $"{totalSaveBytes / (1024.0 * 1024 * 1024):F2} GB ({totalFrames} frames)"
                    : "—";
                _resMonFrames.SubItems[1].Text = totalFrames > 0 ? $"{totalFrames}" : "—";

                // RAM: process working set
                long ramBytes = System.Diagnostics.Process.GetCurrentProcess().WorkingSet64;
                _resMonRamUsed.SubItems[1].Text = $"{ramBytes / (1024.0 * 1024):F0} MB";

                // VRAM: 根據演算法計算（6×W×H + Gaussian workspace 3×W×H×4）
                if (w > 0)
                {
                    long pixels = (long)w * h;
                    long fixedBuf = pixels * 6;                      // 6 個 uint8 buffer
                    long workspace = pixels * 4 * 3;                 // Gaussian: 3 個 float buffer
                    long vramTotal = fixedBuf + workspace + 200L * 1024 * 1024; // + CUDA runtime ~200MB
                    _resMonVramEst.SubItems[1].Text = $"~{vramTotal / (1024.0 * 1024):F0} MB (est.)";
                }
            }
            catch { /* 非關鍵，忽略 */ }
        }

        private void LiveOverviewTimer_Tick(object sender, EventArgs e)
        {
            if (_liveCameraManager == null || _liveCameraManager.IsReleasing) return;
            if (!_liveOverviewDirty || _liveOverviewHelper == null || _settings == null) return;
            _liveOverviewDirty = false;
            CurveMergeHelper.UpdateOverviewChart(_liveCurveMean, _liveCurveMax,
                _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray(),
                _settings.ErrorValueMean, _settings.ErrorValueMax,
                _liveOverviewHelper, CameraCount, _settings.StitchMode, LiveViewRangeProvider);
        }

        // ==========================================
        // --- Hardware → UI 反向同步 ---
        // ==========================================

        /// <summary>
        /// 每次 Telemetry Timer Tick 呼叫。從相機硬體讀回曝光與線掃速率，
        /// 若差異超過 5% 且使用者未拖曳，則更新 TrackBar/NUD（帶 hysteresis 防抖）。
        /// </summary>
        // ==========================================
        // --- 檢測數據 Tab ---
        // ==========================================

        private void SetupDataTab()
        {
            _dataStatsPresenter = new DataStatisticsPresenter(new DataStatisticsContext
            {
                CbStartDate = cbStartDate, CbStartTime = cbStartTime,
                CbEndDate = cbEndDate, CbEndTime = cbEndTime,
                CbGrabIdStart = cbGrabIdStart, CbGrabIdEnd = cbGrabIdEnd,
                CbDataGrabId = cbDataGrabId, CbReviewGrabId = cbReviewGrabId,
                BtnGrabIdPrev = btnGrabIdPrev, BtnGrabIdNext = btnGrabIdNext,
                BtnGrabIdDataPrev = btnGrabIdDataPrev, BtnGrabIdDataNext = btnGrabIdDataNext,
                BtnSelectDataFolder = btnSelectDataFolder, BtnShowFail = btnShowFail,
                GroupBoxGrabIdRange = groupBoxGrabIdRange, GrpDataSingleSheet = grpDataSingleSheet,
                GroupBoxTimeRange = groupBoxTimeRange,
                GrpReviewGrabNav = grpReviewGrabNav, GrpReviewTimePeriod = grpReviewTimePeriod,
                ListViewStats = listViewStats, ListViewGrabDetail = listViewGrabDetail,
                PanelStatCams = new[] { panelStatCam1, panelStatCam2, panelStatCam3,
                                        panelStatCam4, panelStatCam5, panelStatCam6, panelStatCam7 },
                ChartYearly = chartYearly, ChartMonthly = chartMonthly, ChartDaily = chartDaily,
                CbChartYear = cbChartYear, CbChartMonth = cbChartMonth, CbChartDay = cbChartDay,
                Settings = _settings, CameraCount = CameraCount,
            });
            _dataStatsPresenter.Initialize();

            // 延遲注入：_stitchCoordinator 在 InitUiLayer 初始化時 _dataStatsPresenter 尚未建立
            _stitchCoordinator.SetDataStatsPresenter(_dataStatsPresenter);

            // 滾輪上滾 = 數值增加（反轉 ComboBox 預設行為）——僅用於升序排列的 ComboBox
            foreach (var cb in new[] { cbChartYear, cbChartMonth, cbChartDay })
                _wheelInterceptors.Add(new ComboBoxWheelReverser(cb));

            // 跨 Tab 事件
            _dataStatsPresenter.GrabIdSelectedFromData += OnDataGrabIdSelected;
            _dataStatsPresenter.GrabIdSelectedFromReview += OnReviewGrabIdSelected;
            _dataStatsPresenter.PeriodComboManualChanged += OnPeriodComboChanged;
            _dataStatsPresenter.DataFolderSelected += OnDataFolderSelected;
        }

        private async void OnDataGrabIdSelected(string grabId, DateTime earliest, DateTime latest, int idx)
        {
            try
            {
                using (_dataStatsPresenter.GrabIdCrossGuard.Enter())
                {
                    using (_dataStatsPresenter.GrabIdNavGuard.Enter())
                    {
                        cbReviewGrabId.SelectedIndex = idx;
                        _interactionHelper.NavigateToDateTime(earliest);
                    }
                    _presenter.UpdatePeriodNavigationState();
                    _dataStatsPresenter.UpdateGrabIdNavState();
                    _dataStatsPresenter.SetReviewGroupBoxes(true);
                    await _stitchCoordinator.LoadGrabStitchedViewAsync(grabId, earliest, latest);
                }
            }
            catch (Exception ex) { Trace.WriteLine($"[OnDataGrabIdSelected] {ex}"); }
        }

        private async void OnReviewGrabIdSelected(string grabId, DateTime earliest, DateTime latest, int idx)
        {
            try
            {
                _interactionHelper.SaveCanvasView();
                using (_dataStatsPresenter.GrabIdNavGuard.Enter())
                    _interactionHelper.NavigateToDateTime(earliest);
                _presenter.UpdatePeriodNavigationState();

                await _stitchCoordinator.LoadGrabStitchedViewAsync(grabId, earliest, latest);

                // 同步 Data tab
                if (!_dataStatsPresenter.GrabIdCrossGuard.IsSet
                    && cbDataGrabId.Items.Count > 0 && idx < cbDataGrabId.Items.Count)
                {
                    var info = _dataStatsPresenter.GrabIdInfos[idx];
                    _dataStatsPresenter.SyncDataGrabIdFromReview(idx, info);
                }
            }
            catch (Exception ex) { Trace.WriteLine($"[OnReviewGrabIdSelected] {ex}"); }
        }

        private void OnDataFolderSelected(string path)
        {
            // 同步 Review tab
            UserSessionState.SetLastDataPath(path);
            UserSessionState.Save();
            _interactionHelper.LoadDirectoryAndInitNavigator(path);
            _presenter.UpdatePeriodNavigationState();
        }


        /// <summary>
        /// 切換 Live 顯示的 V/H 處理圖方向，點選 muraChartVerticalLive/HorizontalLive 觸發。
        /// 三態邏輯同 Review tab 的 SwitchRidgeDirection：
        /// 未勾選 → 自動勾選 + 設方向；同方向 → 取消勾選；不同方向 → 切換。
        /// </summary>
        private void SwitchLiveDisplayDirection(string dir)
        {
            if (!_settings.EnableMuraEnhance)
            {
                _liveDisplayDirection = dir;
                _settings.EnableMuraEnhance = true;
                ApplyMuraEnhance(true);
                propertyGridSettings.Refresh();
                return;
            }

            if (dir == _liveDisplayDirection)
            {
                _settings.EnableMuraEnhance = false;
                ApplyMuraEnhance(false);
                propertyGridSettings.Refresh();
                return;
            }

            _liveDisplayDirection = dir;
            UpdateLiveDirectionVisual(dir);
        }

        private void UpdateLiveDirectionVisual(string dir)
        {
            muraChartVerticalLive.BackColor = (dir == "v")
                ? System.Drawing.Color.FromArgb(230, 240, 255) : System.Drawing.SystemColors.Control;
            muraChartHorizontalLive.BackColor = (dir == "h")
                ? System.Drawing.Color.FromArgb(230, 240, 255) : System.Drawing.SystemColors.Control;
        }

        /// <summary>
        /// 切換 canvasMain 的 V/H 處理圖方向，點選 chartMuraVertical/Horizontal 觸發。
        /// 未勾選強化圖時：自動勾選 + 設方向。
        /// 已勾選強化圖且點同方向：取消勾選（回原圖）。
        /// 已勾選強化圖且點不同方向：切換方向。
        /// </summary>
        private async void SwitchRidgeDirection(string dir)
        {
            try
            {
            if (!_stitchCoordinator.LastReviewProcessedMode)
            {
                _stitchCoordinator.ActiveRidgeDirection = dir;
                _interactionHelper.SetRidgeDirection(dir);
                UpdateRidgeDirectionVisual(dir);
                _settings.EnableReviewEnhance = true;
                propertyGridSettings.Refresh();
                _ = ApplyReviewEnhance(true);
                return;
            }

            if (dir == _stitchCoordinator.ActiveRidgeDirection)
            {
                UpdateRidgeDirectionVisual(null);
                _settings.EnableReviewEnhance = false;
                propertyGridSettings.Refresh();
                _ = ApplyReviewEnhance(false);
                return;
            }

            // 不同方向 → 切換（重新載入處理圖）
            _stitchCoordinator.ActiveRidgeDirection = dir;
            _interactionHelper.SetRidgeDirection(dir);
            UpdateRidgeDirectionVisual(dir);
            _interactionHelper.SaveCanvasView();

            if (_stitchCoordinator.IsStitchMode)
            {
                int idx = cbReviewGrabId.SelectedIndex;
                if (idx >= 0 && idx < _dataStatsPresenter.GrabIdInfos.Count)
                {
                    var info = _dataStatsPresenter.GrabIdInfos[idx];
                    await _stitchCoordinator.LoadGrabStitchedViewAsync(info.GrabId, info.Earliest, info.Latest, true);
                }
            }
            else
            {
                _stitchCoordinator.ClearStitchedMode();
                await _presenter.LoadImagesWithPeriodLockAsync(true, _interactionHelper.LoadImages);
                ApplyPostLoadDisplay();
            }
            }
            catch (Exception ex) { Trace.WriteLine($"[SwitchRidgeDirection] {ex}"); }
        }

        private void UpdateRidgeDirectionVisual(string dir)
        {
            chartMuraVertical.BackColor = (dir == "v")
                ? System.Drawing.Color.FromArgb(230, 240, 255) : System.Drawing.SystemColors.Control;
            if (chartMuraHorizontal != null)
                chartMuraHorizontal.BackColor = (dir == "h")
                    ? System.Drawing.Color.FromArgb(230, 240, 255) : System.Drawing.SystemColors.Control;
        }

        // ── TrackBar 滾輪：每格僅移動 1 ──────────────────────────────────
        private void RegisterWheelInterceptors(TrackBar[] bars)
        {
            foreach (var bar in bars)
                _wheelInterceptors.Add(new TrackBarWheelInterceptor(bar));
        }


        private void SyncCameraParamsFromHardware()
        {
            if (_expBars == null || _lrBars == null) return;

            var cameras = _liveCameraManager.Cameras;
            var acq     = _settings?.Acquisition;
            if (acq == null) return;

            for (int idx = 0; idx < CameraCount; idx++)
            {
                try
                {
                    var cam = FindCameraById(idx + 1);
                    if (cam == null) continue;
                    if (!cam.IsHwParamsStable) continue;

                    SyncHardwareParam(_expBars[idx], _expNums[idx],
                        cam.GetMeasuredExposureUs(), v => acq.CameraExposureTimeUs[idx] = v);

                    SyncHardwareParam(_lrBars[idx], _lrNums[idx],
                        cam.GetLineRateHz(), v => acq.CameraLineRateHz[idx] = v);
                }
                catch (Exception ex) { Debug.WriteLine($"[SyncHw] CAM{idx + 1}: {ex.Message}"); }
            }
        }

        // ── Helper Methods ──────────────────────────────────────────

        private void SyncHardwareParam(TrackBar bar, NumericUpDown nud, double hwValue, Action<int> saveSetting)
        {
            if (_dragging.Contains(bar) || hwValue <= 0) return;
            int clamped = Math.Max(bar.Minimum, Math.Min(bar.Maximum, (int)hwValue));
            double diff = Math.Abs(clamped - bar.Value) / (double)Math.Max(1, bar.Value);
            if (diff <= 0.05) return;
            _syncingFromHw = true;
            bar.Value = clamped;
            nud.Value = clamped;
            saveSetting(clamped);
            _syncingFromHw = false;
        }

        /// <summary>
        /// Live overview 用：Global 模式從合併 display 取視野，否則返回 NaN（Vertical 模式 X 軸固定）。
        /// </summary>
        private double LiveViewRangeProvider(int cameraIndex, bool isLeft, double defaultValue)
        {
            if (_liveCameraManager?.IsGlobalMergeActive == true &&
                _liveCameraManager.TryGetMergedViewRange(out double left, out double right))
                return isLeft ? left : right;
            return defaultValue;
        }

        /// <summary>
        /// CurveMergeHelper 用的 viewRange 代理：將 TryComputeCurrentViewRange 包裝為 Func。
        /// </summary>
        private double ViewRangeProvider(int cameraIndex, bool isLeft, double defaultValue)
        {
            if (_interactionHelper == null) return defaultValue;
            if (!_interactionHelper.TryComputeCurrentViewRange(cameraIndex, out double left, out double right))
                return defaultValue;
            return isLeft ? left : right;
        }

        private AniloxCamera FindCameraById(int camId)
        {
            if (_liveCameraManager?.Cameras == null) return null;
            foreach (var c in _liveCameraManager.Cameras)
                if (c.CameraId == camId) return c;
            return null;
        }

        // ── Helper Methods ──────────────────────────────────────────

        private static void AutoFitListViewColumns(ListView lv)
        {
            for (int i = 0; i < lv.Columns.Count; i++)
            {
                lv.AutoResizeColumn(i, ColumnHeaderAutoResizeStyle.ColumnContent);
                int contentWidth = lv.Columns[i].Width;
                lv.AutoResizeColumn(i, ColumnHeaderAutoResizeStyle.HeaderSize);
                if (contentWidth > lv.Columns[i].Width)
                    lv.Columns[i].Width = contentWidth;
            }
        }

        // ── Inner Classes ───────────────────────────────────────────

        /// <summary>
        /// Storage 模式 PropertyGrid 過濾器：隱藏 IO / 相機 / 光源三個大類。
        /// 使用 TypeDescriptor instance-level provider，不影響 Inspection 模式。
        /// </summary>
        private sealed class StorageModeSettingsFilter : TypeDescriptionProvider
        {
            private static readonly HashSet<string> Hidden = new HashSet<string>
            {
                "5. IO 模組設定",
                "6. 相機參數設定",
                "7. 光源設定"
            };

            public StorageModeSettingsFilter(TypeDescriptionProvider parent) : base(parent) { }

            public override ICustomTypeDescriptor GetTypeDescriptor(Type objectType, object instance)
                => new FilteredDescriptor(base.GetTypeDescriptor(objectType, instance));

            private sealed class FilteredDescriptor : CustomTypeDescriptor
            {
                public FilteredDescriptor(ICustomTypeDescriptor parent) : base(parent) { }

                public override PropertyDescriptorCollection GetProperties()
                    => Filter(base.GetProperties());
                public override PropertyDescriptorCollection GetProperties(Attribute[] attributes)
                    => Filter(base.GetProperties(attributes));

                private static PropertyDescriptorCollection Filter(PropertyDescriptorCollection all)
                {
                    var visible = all.Cast<PropertyDescriptor>()
                        .Where(p => !Hidden.Contains(p.Category))
                        .ToArray();
                    return new PropertyDescriptorCollection(visible);
                }
            }
        }

        private bool IsCanvasFitToScreen()
        {
            if (canvasMain.Image == null) return false;
            float ratioW = (float)canvasMain.Width / canvasMain.Image.Width;
            float ratioH = (float)canvasMain.Height / canvasMain.Image.Height;
            float fitZoom = Math.Min(ratioW, ratioH) * 0.95f;
            if (Math.Abs(canvasMain.Zoom - fitZoom) > 0.001f) return false;

            float drawW = canvasMain.Image.Width * fitZoom;
            float drawH = canvasMain.Image.Height * fitZoom;
            float fitPanX = (canvasMain.Width - drawW) / 2f;
            float fitPanY = (canvasMain.Height - drawH) / 2f;
            var pan = canvasMain.PanOffset;
            return Math.Abs(pan.X - fitPanX) < 1f && Math.Abs(pan.Y - fitPanY) < 1f;
        }

    }
}
