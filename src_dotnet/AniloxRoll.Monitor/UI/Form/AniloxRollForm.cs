using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
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
        private MuraChartHelper _muraChartHelper;
        private MuraChartHelper _muraChartLiveHelper;
        private MuraChartHelper _stitchedOverviewHelper;
        private MuraChartHelper _liveOverviewHelper;
        private RowMuraChartHelper _rowChartLiveHelper;
        private RowMuraChartHelper _muraChartHorizontalHelper;
        private LiveCameraManager _liveCameraManager;
        private ProportionalScaler _scaler;

        // --- 相機參數控制項陣列（供 SyncFromCamera 存取）---
        private TrackBar[]      _expBars;
        private NumericUpDown[] _expNums;
        private TrackBar[]      _lrBars;
        private NumericUpDown[] _lrNums;
        private TrackBar[]      _htBars;
        private NumericUpDown[] _htNums;

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

        // --- 檢測日誌 ---
        private InspectionLogService _inspectionLogService;
        private string _currentGrabId;

        // --- 統計 ---
        private InspectionStatsPresenter    _statsPresenter;
        private string                      _statsDataRootPath   = string.Empty;
        private SortedSet<DateTime>         _statAvailableTimes  = new SortedSet<DateTime>();
        private List<GrabIdInfo>            _grabIdInfos         = new List<GrabIdInfo>();
        private bool                        _statComboUpdating;
        private GroupBox                    _activeStatMode;
        private bool                        _syncingGrabIdNav;
        private bool                        _syncingGrabIdCross;
        private List<GrabDetail>            _currentDetails      = new List<GrabDetail>();
        private bool                        _showFailOnly        = false;
        // --- 圖表導航狀態 ---
        private List<int> _chartYears  = new List<int>();
        private List<int> _chartMonths = new List<int>();
        private List<int> _chartDays   = new List<int>();
        private bool      _chartNavUpdating = false;

        // --- 資料緩存 ---
        private readonly List<Image> _thumbnailCache = new List<Image>();
        private InspectionSettings _settings;
        private bool _lastReviewProcessedMode = false;
        private bool _syncingProcessedCheckbox = false;
        /// <summary>"v" = vertical ridge（預設），"h" = horizontal ridge。控制 canvasMain 處理圖方向。</summary>
        private string _activeRidgeDirection = "v";
        /// <summary>"v" = vertical ridge（預設），"h" = horizontal ridge。控制 Live 顯示方向。</summary>
        private string _liveDisplayDirection = "v";

        // --- Live 全覽圖：每台相機最新曲線快取 ---
        private readonly float[][] _liveCurveMean = new float[7][];
        private readonly float[][] _liveCurveMax  = new float[7][];
        private volatile bool _liveOverviewDirty;
        private const int MaxOverviewPoints = 2000;

        // --- Grab ID 拼接模式（null = 一般模式）---
        private Bitmap[] _stitchedImages;
        private float[][] _stitchedCurveMean;
        private float[][] _stitchedCurveMax;
        private float[][] _stitchedRowCurveMean;
        private float[][] _stitchedRowCurveMax;
        private CsvConfigSnapshot _currentGrabConfig;


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
            if (_settings == null) _settings = ConfigManager.LoadInspectionSettings();
            InitServiceLayer();
            InitUiLayer();
            InitCameraLayer();
            InitializeRightPanelControls();
            SetupDataTab();
        }

        /// <summary>純業務服務：不依賴任何 UI 控制項。</summary>
        private void InitServiceLayer()
        {
            _inspectionService    = new BatchInspectionService();
            _inspectionLogService = new InspectionLogService(
                () => _settings?.CaptureRootPath ?? string.Empty,
                UserSessionState.LastGrabIdNum);
        }

        /// <summary>UI 層：Presenter、Helper、PropertyGrid、Canvas 事件。</summary>
        private void InitUiLayer()
        {
            _dateTimeNavigator = new DateTimeNavigator(
                _imageRepository, cbDate, cbTime);

            _galleryManager = new ThumbnailGridPresenter();
            _galleryManager.Initialize(new PictureBox[] {
                pbCam1, pbCam2, pbCam3, pbCam4, pbCam5, pbCam6, pbCam7
            });

            _presenter = new AniloxRollPresenter(
                _imageRepository, _inspectionService, _dateTimeNavigator, _galleryManager);

            _muraChartHelper = new MuraChartHelper(this.chartMuraVertical);
            _muraChartHelper.SetOps(_settings.Cam1_Ops);
            _muraChartHelper.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);

            _muraChartLiveHelper = new MuraChartHelper(this.muraChartVerticalLive);
            _muraChartLiveHelper.SetOps(_settings.Cam1_Ops);
            _muraChartLiveHelper.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);

            _stitchedOverviewHelper = new MuraChartHelper(this.chartOverview);
            _stitchedOverviewHelper.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            if (chartOverview.ChartAreas.Count > 0)
                chartOverview.ChartAreas[0].AxisX.ScaleView.Zoomable = false;

            _liveOverviewHelper = new MuraChartHelper(this.chartLiveOverview);
            _liveOverviewHelper.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            if (chartLiveOverview.ChartAreas.Count > 0)
                chartLiveOverview.ChartAreas[0].AxisX.ScaleView.Zoomable = false;

            _rowChartLiveHelper = new RowMuraChartHelper(this.muraChartHorizontalLive);
            _rowChartLiveHelper.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);

            _muraChartHorizontalHelper = new RowMuraChartHelper(this.chartMuraHorizontal);
            _muraChartHorizontalHelper.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);

            UpdateRowChartPitch();

            // Review tab chart 點選切換 V/H 處理圖方向
            chartMuraVertical.MouseClick += (s, e) => SwitchRidgeDirection("v");
            chartMuraHorizontal.MouseClick += (s, e) => SwitchRidgeDirection("h");

            // Live tab chart 點選切換 V/H 處理圖方向
            muraChartVerticalLive.MouseClick += (s, e) => SwitchLiveDisplayDirection("v");
            muraChartHorizontalLive.MouseClick += (s, e) => SwitchLiveDisplayDirection("h");

            checkBoxEnableImageProcessing.Checked =
                UserSessionState.GetLastEnableImageProcessing(checkBoxEnableImageProcessing.Checked);

            // PropertyGrid：Categorized 排序（維持宣告順序），預設摺疊
            propertyGridSettings.SelectedObject = _settings;
            propertyGridSettings.ToolbarVisible = false;
            propertyGridSettings.PropertySort   = System.Windows.Forms.PropertySort.Categorized;
            propertyGridSettings.CollapseAllGridItems();
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
                MuraChartHelper  = _muraChartHelper,
                Settings         = _settings,
                StatusLabel      = lblPixelInfo,
                CameraPanels     = new[] { pbCam1, pbCam2, pbCam3, pbCam4, pbCam5, pbCam6, pbCam7 },
                MuraChartHorizontalHelper = _muraChartHorizontalHelper,
            });
            _interactionHelper.ApplySettingsToService();

            _presenter.BusyStateChanged += _interactionHelper.SetUiLoadingState;
            _presenter.LogReported      += OnPresenterLogReported;
            _galleryManager.SelectionChanged += idx =>
            {
                if (_stitchedImages != null)
                    ShowStitchedCameraInCanvas(idx);
                else
                    _interactionHelper.OnGallerySelectionChanged(idx);
            };

            _dateTimeNavigator.PeriodSelectionChanged += _presenter.UpdatePeriodNavigationState;
            _dateTimeNavigator.PeriodSelectionChanged += SyncGrabIdFromTimeCombos;
            _dateTimeNavigator.PeriodSelectionChanged += OnPeriodComboChanged;
            _presenter.PeriodNavigationStateChanged   += (canLast, canNext) =>
            {
                btnPeriodPrev.Enabled = canLast;
                btnPeriodNext.Enabled = canNext;
            };
            _presenter.UpdatePeriodNavigationState();

            canvasMain.StatusChanged += _interactionHelper.UpdateCanvasInfo;
            canvasMain.EdgeReached   += _interactionHelper.NavigateCamera;
            canvasMain.DoubleClick   += (s, e) => { if (canvasMain.Image != null) canvasMain.FitToScreen(); };
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
            _liveCameraManager.OnInspectionResult += OnCameraInspectionResult;
            btnGetBackground.Click += btnGetBackground_Click;
            btnViewBackground.Click += btnViewBackground_Click;
            UpdateStandardBgSubLockState();
            _liveCameraManager.OnLiveCurveData      += OnLiveCurveData;
            _liveCameraManager.OnLiveRowCurveData   += OnLiveRowCurveData;

            FormClosed += (_, __) =>
            {
                _telemetryTimer?.Stop();
                _liveOverviewTimer?.Stop();
                FreePrecomputedColMeanBuffers();
                _liveCameraManager.FreeCameras();
            };
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
                    _liveCameraManager.EnsureAllocatedAndToggleGrab(checkBoxEnableImageProcessing.Checked);
                    LoadBackgroundBins();
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

            // 剛從「未抓取」→「抓取中」：分配新的抓圖編號
            if (!wasGrabbing && _liveCameraManager.IsLiveGrabbing)
                _currentGrabId = _inspectionLogService.NextGrabId();

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
            _inspectionLogService?.AppendRecord(
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
        }

        private void OnLiveCurveData(int camId, float[] meanArr, float[] maxArr)
        {
            // 快取每台相機最新曲線（callback 執行緒，只是 ref 賦值）
            int cameraIndex = camId - 1;
            if (cameraIndex >= 0 && cameraIndex < 7)
            {
                _liveCurveMean[cameraIndex] = meanArr;
                _liveCurveMax[cameraIndex]  = maxArr;
                _liveOverviewDirty = true;
            }

            // 只有選中相機才 marshal 到 UI 執行緒更新 muraChartLive
            if (camId != _liveCameraManager.SelectedMainCameraId) return;

            if (InvokeRequired)
            {
                BeginInvoke(new Action<int, float[], float[]>(OnLiveCurveData), camId, meanArr, maxArr);
                return;
            }

            if (_muraChartLiveHelper == null || _settings == null) return;

            double[] opsUmArr       = _settings.GetCameraOpsUmArray();
            double[] startPositions = _settings.GetCameraStartPositionMmArray();

            double opsUm = (cameraIndex >= 0 && cameraIndex < opsUmArr.Length)
                ? opsUmArr[cameraIndex] : _settings.Cam1_Ops;
            double opsInMm  = opsUm / 1000.0;
            double startPos = (cameraIndex >= 0 && cameraIndex < startPositions.Length)
                ? startPositions[cameraIndex] : 0;

            _muraChartLiveHelper.SetOps(opsUm);

            // 查詢 MIL 副顯示器的實際 zoom/pan（隨使用者滾輪操作即時變化）
            // panOffsetX = 面板左邊緣對應的 buffer pixel X
            // rightPixel = panOffsetX + panelWidth / zoomX
            double viewLeftMm = double.NaN, viewRightMm = double.NaN;

            AniloxCamera liveCam = null;
            foreach (var c in _liveCameraManager.Cameras)
                if (c.CameraId == camId) { liveCam = c; break; }

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

            _muraChartLiveHelper.UpdateDataAndView(meanArr, maxArr,
                startPos, viewLeftMm, viewRightMm);
        }

        private void OnLiveRowCurveData(int camId, float[] meanArr, float[] maxArr)
        {
            if (camId != _liveCameraManager.SelectedMainCameraId) return;

            if (InvokeRequired)
            {
                BeginInvoke(new Action<int, float[], float[]>(OnLiveRowCurveData), camId, meanArr, maxArr);
                return;
            }

            if (_rowChartLiveHelper == null) return;
            _rowChartLiveHelper.UpdateData(meanArr, maxArr);

            // 同步 Y 軸視野：查詢 MIL 副顯示器 zoom/pan，以 panel 上下邊緣的 mm 值對齊
            AniloxCamera liveCam = null;
            foreach (var c in _liveCameraManager.Cameras)
                if (c.CameraId == camId) { liveCam = c; break; }

            double rowPitch = _rowChartLiveHelper.RowPitchMm;
            if (liveCam != null && rowPitch > 0 &&
                liveCam.TryGetSecondaryDisplayGeometry(
                    out double milZoomX, out double milZoomY, out double milPanX, out double milPanY))
            {
                double panelH  = panelMainDisplay.Height;
                double topPixel = milPanY;
                double botPixel = milPanY + panelH / milZoomY;
                _rowChartLiveHelper.UpdateViewRange(topPixel * rowPitch, botPixel * rowPitch);
            }
        }

        /// <summary>用 A輪速度 和選中相機的取樣頻率（Line Rate）更新法向圖表座標。</summary>
        private void UpdateRowChartPitch()
        {
            if (_settings == null) return;
            double lineRateHz = _settings.Acquisition.CameraLineRateHz[0]; // CAM1 master
            _rowChartLiveHelper?.SetRowPitchFromSpeed(
                _settings.AniloxRollSpeedMPerMin, lineRateHz);
            _muraChartHorizontalHelper?.SetRowPitchFromSpeed(
                _settings.AniloxRollSpeedMPerMin, lineRateHz);
        }

        private void btnCameraFree_Click(object sender, EventArgs e)
        {
            ClearBackgroundPreview();
            _liveCameraManager.FreeCameras();
            _telemetryPresenter?.ResetAll();
            UpdateGrabButton(false);
        }

        /// <summary>
        /// 取得背景：啟動 grab → 採集 N 秒 → 多幀平均 column mean → 存 MCBF bin。
        /// </summary>
        private async void btnGetBackground_Click(object sender, EventArgs e)
        {
            if (_settings.Recipe.Algorithm != BackgroundAlgorithm.StandardBgSub)
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
            btnCameraFree.Enabled = false;

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
            if (_settings.Recipe.Algorithm != BackgroundAlgorithm.StandardBgSub)
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
            if (_settings.Recipe.Algorithm != BackgroundAlgorithm.StandardBgSub)
            {
                // 非 StandardBgSub：正常解鎖
                btnCameraGrab.Enabled = true;
                btnCameraFree.Enabled = true;
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
            btnCameraFree.Enabled = allConnectedHaveBin;
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

        /// <summary>SmartCanvas 滑鼠移動時更新 lblPixelInfo。</summary>
        private void BgPreviewCanvas_StatusChanged(CanvasInfo info)
        {
            if (lblPixelInfo == null) return;

            int camId = _bgPreviewSelectedCamIndex + 1;
            string text;
            if (info.ImageX < 0 || info.ImageY < 0 ||
                _bgPreviewMainCanvas?.Image == null ||
                info.ImageX >= _bgPreviewMainCanvas.Image.Width ||
                info.ImageY >= _bgPreviewMainCanvas.Image.Height)
            {
                text = $"背景預覽 [CAM {camId}] | 游標超出影像範圍";
            }
            else
            {
                int gray = info.PixelColor.R;  // 8bpp grayscale: R=G=B
                text = $"背景預覽 [CAM {camId}] | X: {info.ImageX}, Y: {info.ImageY} | 灰階值: {gray} | 縮放: {info.Zoom:F2}x";
            }

            if (InvokeRequired)
                BeginInvoke(new Action(() => lblPixelInfo.Text = text));
            else
                lblPixelInfo.Text = text;
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
            if (isGrabbing)
            {
                lblStatusGrab.Text      = "● 相機抓取中";
                lblStatusGrab.BackColor = Color.FromArgb(56, 142, 60);   // IEC 綠：運轉中
                lblStatusGrab.ForeColor = Color.White;
            }
            else
            {
                lblStatusGrab.Text      = "● 待機";
                lblStatusGrab.BackColor = Color.FromArgb(117, 117, 117); // IEC 白/灰：中性待機
                lblStatusGrab.ForeColor = Color.White;
                UpdateStandardBgSubLockState(); // 停止後依 bin 狀態重新檢查
            }
        }

        private void checkBoxEnableImageProcessing_CheckedChanged(object sender, EventArgs e)
        {
            bool enabled = checkBoxEnableImageProcessing.Checked;
            _liveCameraManager?.SetImageProcessingEnabled(enabled);
            UserSessionState.SetLastEnableImageProcessing(enabled);
            UserSessionState.Save();

            // 同步 chart 背景色：取消勾選時清除高亮
            UpdateLiveDirectionVisual(enabled ? _liveDisplayDirection : null);
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
            _interactionHelper.HandleSettingsChanged();
            _liveCameraManager?.SetCaptureSettings(_settings);
            _muraChartHelper?.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            _muraChartLiveHelper?.SetOps(_settings.Cam1_Ops);
            _muraChartLiveHelper?.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            _liveOverviewHelper?.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            _rowChartLiveHelper?.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            _muraChartHorizontalHelper?.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            UpdateRowChartPitch();

            // 抓圖進行中設定變更 → 立刻在 CSV 插入 #CFG
            if (_liveCameraManager?.IsLiveGrabbing == true)
                _inspectionLogService?.ForceWriteConfig(CsvConfigSnapshot.FromSettings(_settings));

            string changedPropertyName = e?.ChangedItem?.PropertyDescriptor?.Name ?? string.Empty;

            // 圖表設定變更 → 立刻套用
            if (changedPropertyName == nameof(InspectionSettings.ChartScaleMode))
            {
                ApplyChartScaleFromSettings();
            }
            else if (changedPropertyName == nameof(InspectionSettings.ChartYearlyYMax))
                ApplyFixedScale(chartYearly, _settings.Chart.YearlyYMax);
            else if (changedPropertyName == nameof(InspectionSettings.ChartMonthlyYMax))
                ApplyFixedScale(chartMonthly, _settings.Chart.MonthlyYMax);
            else if (changedPropertyName == nameof(InspectionSettings.ChartDailyYMax))
                ApplyFixedScale(chartDaily, _settings.Chart.DailyYMax);

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
                _lastReviewProcessedMode = true;
                _syncingProcessedCheckbox = true;
                try { checkBoxShowProcessed.Checked = true; }
                finally { _syncingProcessedCheckbox = false; }
                ClearStitchedMode();
                await _presenter.LoadImagesWithPeriodLockAsync(true, _interactionHelper.LoadImages);
                UpdateOverviewChartFromRepository();
                _interactionHelper.RefreshCurrentCanvasResult();
            }
        }

        private async void btnSelectFolder_Click(object sender, EventArgs e)
        {
            _interactionHelper.SelectAndLoadFolder();
            _presenter.UpdatePeriodNavigationState();
            _lastReviewProcessedMode = false;
            _syncingProcessedCheckbox = true;
            try { checkBoxShowProcessed.Checked = false; }
            finally { _syncingProcessedCheckbox = false; }

            // 同步載入序號清單並填充所有序號 ComboBox（Review + Data）
            if (_imageRepository.FileCount > 0)
            {
                var reviewPath = UserSessionState.LastDataPath;
                if (!string.IsNullOrWhiteSpace(reviewPath))
                {
                    _statsDataRootPath  = reviewPath;
                    _statAvailableTimes = InspectionStatisticsService.LoadAvailableTimes(reviewPath);
                    _grabIdInfos        = InspectionStatisticsService.LoadGrabIdInfosDescending(reviewPath);

                    _statComboUpdating = true;
                    try
                    {
                        cbReviewGrabId.Items.Clear();
                        cbGrabIdStart.Items.Clear();
                        cbGrabIdEnd.Items.Clear();
                        cbDataGrabId.Items.Clear();
                        foreach (var info in _grabIdInfos)
                        {
                            cbReviewGrabId.Items.Add(info.GrabId);
                            cbGrabIdStart.Items.Add(info.GrabId);
                            cbGrabIdEnd.Items.Add(info.GrabId);
                            cbDataGrabId.Items.Add(info.GrabId);
                        }
                        SyncGrabIdFromTimeCombos();
                        UpdateGrabIdNavState();
                        if (cbGrabIdStart.Items.Count > 0)
                        {
                            cbGrabIdStart.SelectedIndex = cbGrabIdStart.Items.Count - 1;
                            cbGrabIdEnd.SelectedIndex = 0;
                        }
                    }
                    finally { _statComboUpdating = false; }

                    // 填充日期/時間 ComboBox（全範圍）
                    if (_statAvailableTimes.Count > 0)
                        PopulateStatDateCombos(_statAvailableTimes.Min, _statAvailableTimes.Max);

                    PopulateChartNavigators(_statAvailableTimes.Count > 0
                        ? (DateTime?)_statAvailableTimes.Max : null);
                    RefreshStats();
                }
            }

            ClearStitchedMode();
            SetGroupBoxActive(grpReviewTimePeriod, true);
            SetGroupBoxActive(grpReviewGrabNav, false);
            await _presenter.LoadImagesWithPeriodLockAsync(false, _interactionHelper.LoadImages);
            UpdateOverviewChartFromRepository();
        }

        private async void checkBoxShowProcessed_CheckedChanged(object sender, EventArgs e)
        {
            if (_syncingProcessedCheckbox) return;
            bool enableProcess = checkBoxShowProcessed.Checked;
            UpdateRidgeDirectionVisual(enableProcess ? _activeRidgeDirection : null);
            if (_stitchedImages != null)
            {
                await ReloadCurrentStitchedView(enableProcess);
                return;
            }
            _lastReviewProcessedMode = enableProcess;
            ClearStitchedMode();
            await _presenter.LoadImagesWithPeriodLockAsync(enableProcess, _interactionHelper.LoadImages);
            UpdateOverviewChartFromRepository();
        }

        private async Task ReloadCurrentStitchedView(bool enableProcess)
        {
            int idx = cbReviewGrabId.SelectedIndex;
            if (idx < 0 || idx >= _grabIdInfos.Count) return;
            _interactionHelper.SaveCanvasView();
            var info = _grabIdInfos[idx];
            await LoadGrabStitchedViewAsync(info.GrabId, info.Earliest, info.Latest, enableProcess);
        }

        private async void btnPeriodPrev_Click(object sender, EventArgs e)
        { _interactionHelper.SaveCanvasView(); ClearStitchedMode(); await _presenter.MovePeriodAsync(-1, _lastReviewProcessedMode, _interactionHelper.LoadImages); UpdateOverviewChartFromRepository(); }

        private async void btnPeriodNext_Click(object sender, EventArgs e)
        { _interactionHelper.SaveCanvasView(); ClearStitchedMode(); await _presenter.MovePeriodAsync(+1, _lastReviewProcessedMode, _interactionHelper.LoadImages); UpdateOverviewChartFromRepository(); }

        /// <summary>cbDate/cbTime 手動滾動時載入對應圖片（同 btnPeriodPrev/Next）。
        /// _syncingGrabIdNav 時跳過（由 OnReviewGrabIdChanged 等程式碼觸發的 NavigateToDateTime）。</summary>
        private async void OnPeriodComboChanged()
        {
            if (_syncingGrabIdNav) return;
            if (_imageRepository.FileCount == 0) return;
            _interactionHelper.SaveCanvasView();
            ClearStitchedMode();
            SetGroupBoxActive(grpReviewGrabNav, false);
            SetGroupBoxActive(grpReviewTimePeriod, true);
            await _presenter.LoadImagesWithPeriodLockAsync(_lastReviewProcessedMode, _interactionHelper.LoadImages);
            UpdateOverviewChartFromRepository();
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
            const int TickFreq  =  1000;

            // ── 7 台相機控制項陣列（存為 Form 欄位，供 SyncFromCamera 存取）────
            var acq = _settings.Acquisition;
            _expBars = new[] { trackBarExpCam1, trackBarExpCam2, trackBarExpCam3, trackBarExpCam4, trackBarExpCam5, trackBarExpCam6, trackBarExpCam7 };
            _expNums = new[] { numExpCam1,      numExpCam2,      numExpCam3,      numExpCam4,      numExpCam5,      numExpCam6,      numExpCam7      };
            _lrBars  = new[] { trackBarLrCam1,  trackBarLrCam2,  trackBarLrCam3,  trackBarLrCam4,  trackBarLrCam5,  trackBarLrCam6,  trackBarLrCam7  };
            _lrNums  = new[] { numLrCam1,       numLrCam2,       numLrCam3,       numLrCam4,       numLrCam5,       numLrCam6,       numLrCam7       };
            _htBars  = new[] { trackBarHtCam1,  trackBarHtCam2,  trackBarHtCam3,  trackBarHtCam4,  trackBarHtCam5,  trackBarHtCam6,  trackBarHtCam7  };
            _htNums  = new[] { numHtCam1,       numHtCam2,       numHtCam3,       numHtCam4,       numHtCam5,       numHtCam6,       numHtCam7       };

            for (int i = 0; i < 7; i++)
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
                int expVal = (int)Math.Max(ExpMin, Math.Min(expMax, acq.CameraExposureTimeUs[idx]));
                _expBars[idx].Minimum = ExpMin; _expBars[idx].Maximum = expMax; _expBars[idx].TickFrequency = TickFreq;
                _expNums[idx].Minimum = ExpMin; _expNums[idx].Maximum = expMax;
                _expBars[idx].Value   = expVal; _expNums[idx].Value   = expVal;

                bool syncExp = false;
                _expBars[idx].MouseDown  += (s, e) => _dragging.Add(_expBars[idx]);
                _expBars[idx].MouseUp    += (s, e) =>
                {
                    _dragging.Remove(_expBars[idx]);
                    // 拖曳結束：補送一次硬體寫入（拖曳中被抑制）
                    _liveCameraManager?.SetExposureForCamera(camId, _expBars[idx].Value);
                    _liveCameraManager?.SwitchToCamera(camId);
                };
                _expBars[idx].ValueChanged += (s, e) =>
                {
                    if (syncExp || _syncingFromHw) return; syncExp = true;
                    _expNums[idx].Value = _expBars[idx].Value;
                    acq.CameraExposureTimeUs[idx] = _expBars[idx].Value;
                    ConfigManager.SaveAcquisitionSettings(acq);
                    if (!_dragging.Contains(_expBars[idx]))
                        _liveCameraManager?.SetExposureForCamera(camId, _expBars[idx].Value);
                    syncExp = false;
                };
                _expNums[idx].ValueChanged += (s, e) =>
                {
                    if (syncExp || _syncingFromHw) return; syncExp = true;
                    int v = (int)_expNums[idx].Value;
                    _expBars[idx].Value = Math.Max(ExpMin, Math.Min(_expBars[idx].Maximum, v));
                    acq.CameraExposureTimeUs[idx] = v;
                    ConfigManager.SaveAcquisitionSettings(acq);
                    _liveCameraManager?.SetExposureForCamera(camId, v);
                    syncExp = false;
                };

                // ── 線掃速率 ────────────────────────────────────────────
                int lrVal = (int)Math.Max(LrMin, Math.Min(LrMax, acq.CameraLineRateHz[idx]));
                _lrBars[idx].Minimum = LrMin; _lrBars[idx].Maximum = LrMax; _lrBars[idx].TickFrequency = TickFreq;
                _lrNums[idx].Minimum = LrMin; _lrNums[idx].Maximum = LrMax;
                _lrBars[idx].Value   = lrVal; _lrNums[idx].Value   = lrVal;

                bool syncLr = false;
                _lrBars[idx].MouseDown += (s, e) => _dragging.Add(_lrBars[idx]);
                _lrBars[idx].MouseUp   += (s, e) =>
                {
                    _dragging.Remove(_lrBars[idx]);
                    _liveCameraManager?.SetLineRateForCamera(camId, _lrBars[idx].Value);
                    _liveCameraManager?.SwitchToCamera(camId);
                };
                _lrBars[idx].ValueChanged += (s, e) =>
                {
                    if (syncLr || _syncingFromHw) return; syncLr = true;
                    _lrNums[idx].Value = _lrBars[idx].Value;
                    acq.CameraLineRateHz[idx] = _lrBars[idx].Value;
                    ConfigManager.SaveAcquisitionSettings(acq);
                    if (!_dragging.Contains(_lrBars[idx]))
                        _liveCameraManager?.SetLineRateForCamera(camId, _lrBars[idx].Value);
                    UpdateExpMaxAndClampColor(idx, CalcExpMax());
                    if (idx == 0) UpdateRowChartPitch();
                    syncLr = false;
                };
                _lrNums[idx].ValueChanged += (s, e) =>
                {
                    if (syncLr || _syncingFromHw) return; syncLr = true;
                    int v = (int)_lrNums[idx].Value;
                    _lrBars[idx].Value = Math.Max(LrMin, Math.Min(LrMax, v));
                    acq.CameraLineRateHz[idx] = v;
                    ConfigManager.SaveAcquisitionSettings(acq);
                    _liveCameraManager?.SetLineRateForCamera(camId, v);
                    UpdateExpMaxAndClampColor(idx, CalcExpMax());
                    if (idx == 0) UpdateRowChartPitch();
                    syncLr = false;
                };

                // ── 擷取高度 ────────────────────────────────────────────
                int htVal = Math.Max(HtMin, Math.Min(HtMax, acq.CameraGrabHeight[idx]));
                _htBars[idx].Minimum = HtMin; _htBars[idx].Maximum = HtMax; _htBars[idx].TickFrequency = TickFreq;
                _htBars[idx].SmallChange = 64; _htBars[idx].LargeChange = 512;
                _htNums[idx].Minimum = HtMin; _htNums[idx].Maximum = HtMax;
                _htBars[idx].Value   = htVal; _htNums[idx].Value   = htVal;

                bool syncHt = false;
                _htBars[idx].MouseDown += (s, e) => _dragging.Add(_htBars[idx]);
                _htBars[idx].MouseUp   += (s, e) =>
                {
                    _dragging.Remove(_htBars[idx]);
                    // SetGrabHeight 代價高（重分配 Buffer），拖曳結束才執行一次
                    _liveCameraManager?.SetGrabHeightForCamera(camId, _htBars[idx].Value);
                    _liveCameraManager?.SwitchToCamera(camId);
                };
                _htBars[idx].ValueChanged += (s, e) =>
                {
                    if (syncHt || _syncingFromHw) return; syncHt = true;
                    _htNums[idx].Value = _htBars[idx].Value;
                    acq.CameraGrabHeight[idx] = _htBars[idx].Value;
                    ConfigManager.SaveAcquisitionSettings(acq);
                    if (!_dragging.Contains(_htBars[idx]))
                        _liveCameraManager?.SetGrabHeightForCamera(camId, _htBars[idx].Value);
                    syncHt = false;
                };
                _htNums[idx].ValueChanged += (s, e) =>
                {
                    if (syncHt || _syncingFromHw) return; syncHt = true;
                    int v = (int)_htNums[idx].Value;
                    _htBars[idx].Value = Math.Max(HtMin, Math.Min(HtMax, v));
                    acq.CameraGrabHeight[idx] = v;
                    ConfigManager.SaveAcquisitionSettings(acq);
                    _liveCameraManager?.SetGrabHeightForCamera(camId, v);
                    syncHt = false;
                };
            }

            // 滾輪每格移動 1（攔截原生 3 格行為）
            RegisterWheelInterceptors(_expBars);
            RegisterWheelInterceptors(_lrBars);
            RegisterWheelInterceptors(_htBars);
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
            AutoFitListViewColumns(listViewEngine);

            // ── 圖表引擎常數 ────────────────────────────────────────────────
            listViewChartConst.Columns.Add("參數", 160);
            listViewChartConst.Columns.Add("值",    90);
            listViewChartConst.Items.Add(new ListViewItem(new[] { "MaxOverviewPoints", MaxOverviewPoints.ToString() }));
            listViewChartConst.Items.Add(new ListViewItem(new[] { "TelemetryInterval", "500 ms" }));
            listViewChartConst.Items.Add(new ListViewItem(new[] { "OverviewRefresh",   "FPS-sync" }));
            listViewChartConst.Items.Add(new ListViewItem(new[] { "DownsampleMode",    "Max-Window" }));
            listViewChartConst.Items.Add(new ListViewItem(new[] { "OverlapMean",       "Average" }));
            listViewChartConst.Items.Add(new ListViewItem(new[] { "OverlapMax",        "Maximum" }));
            AutoFitListViewColumns(listViewChartConst);

            // ── 硬體參數（螢幕 + 未來 PLC）──────────────────────────────────
            listViewHardware.Columns.Add("參數", 120);
            listViewHardware.Columns.Add("值",   120);
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
            }
            catch { /* 非關鍵資訊，忽略 */ }
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

        }

        private void LiveOverviewTimer_Tick(object sender, EventArgs e)
        {
            if (_liveCameraManager == null || _liveCameraManager.IsReleasing) return;
            if (!_liveOverviewDirty || _liveOverviewHelper == null || _settings == null) return;
            _liveOverviewDirty = false;
            UpdateOverviewChart(_liveCurveMean, _liveCurveMax,
                _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray(),
                _settings.ErrorValueMean, _settings.ErrorValueMax, _liveOverviewHelper);
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
            _statsPresenter = new InspectionStatsPresenter(
                listViewStats,
                new[] { panelStatCam1, panelStatCam2, panelStatCam3, panelStatCam4,
                        panelStatCam5, panelStatCam6, panelStatCam7 });
            _statsPresenter.Initialize();

            // 預設時間：開始 = 今天 00:00，結束 = 今天 23:59
            DateTime today = DateTime.Today;
            PopulateStatDateCombos(today.AddDays(-7), today);

            // 預設資料夾與 CaptureRootPath 相同
            _statsDataRootPath = _settings?.CaptureRootPath ?? string.Empty;

            btnSelectDataFolder.Click += BtnSelectDataFolder_Click;
            btnShowFail.Click         += BtnShowFail_Click;
            WireStatDateCombos();
            InitGrabDetailListView();
            InitPeriodCharts();
            cbChartYear.SelectedIndexChanged  += (s, e) => { if (!_chartNavUpdating) OnChartYearIndexChanged();  };
            cbChartMonth.SelectedIndexChanged += (s, e) => { if (!_chartNavUpdating) OnChartMonthIndexChanged(); };
            cbChartDay.SelectedIndexChanged   += (s, e) => { if (!_chartNavUpdating) OnChartDayIndexChanged();   };
            // 滾輪上滾 = 數值增加（反轉 ComboBox 預設行為）——僅用於升序排列的 ComboBox
            // cbDate/cbTime/cbStart*/cbEnd*/cbGrabId* 為降序（newest first），使用預設方向（上滾=newer）
            foreach (var cb in new[] { cbChartYear, cbChartMonth, cbChartDay })
                _wheelInterceptors.Add(new ComboBoxWheelReverser(cb));

            cbGrabIdStart.SelectedIndexChanged  += (s, e) => OnGrabIdComboChanged(isStart: true);
            cbGrabIdEnd.SelectedIndexChanged    += (s, e) => OnGrabIdComboChanged(isStart: false);
            cbDataGrabId.SelectedIndexChanged   += (s, e) => OnSingleSheetComboChanged();
            cbReviewGrabId.SelectedIndexChanged += (s, e) => OnReviewGrabIdChanged();
            btnGrabIdPrev.Click             += (s, e) => StepReviewGrabId(-1);
            btnGrabIdNext.Click             += (s, e) => StepReviewGrabId(+1);
            btnGrabIdDataPrev.Click         += (s, e) => StepDataGrabId(-1);
            btnGrabIdDataNext.Click         += (s, e) => StepDataGrabId(+1);
        }

        private void PopulateStatDateCombos(DateTime start, DateTime end)
        {
            var dates = GetAvailableDateStrings();
            string startDateStr = start.ToString("yyyy-MM-dd");
            string endDateStr   = end.ToString("yyyy-MM-dd");
            string startTimeStr = start.ToString("HH:mm:ss");
            string endTimeStr   = end.ToString("HH:mm:ss");

            // Start
            cbStartDate.Items.Clear();
            cbStartDate.Items.AddRange(dates.ToArray());
            int si = dates.IndexOf(startDateStr);
            cbStartDate.SelectedIndex = si >= 0 ? si : (dates.Count > 0 ? dates.Count - 1 : -1);
            RefreshStatTimeCombo(cbStartDate, cbStartTime, startTimeStr);

            // End（降序：第一筆 = 最新）
            cbEndDate.Items.Clear();
            cbEndDate.Items.AddRange(dates.ToArray());
            int ei = dates.IndexOf(endDateStr);
            cbEndDate.SelectedIndex = ei >= 0 ? ei : (dates.Count > 0 ? 0 : -1);
            RefreshStatTimeCombo(cbEndDate, cbEndTime, endTimeStr);
        }

        private void RefreshStatTimeCombo(ComboBox dateCb, ComboBox timeCb, string preferred)
        {
            var times = GetAvailableTimeStrings(dateCb.Text);
            timeCb.Items.Clear();
            timeCb.Items.AddRange(times.ToArray());
            if (times.Count == 0) return;
            int idx = times.IndexOf(preferred);
            timeCb.SelectedIndex = idx >= 0 ? idx : (times.Count > 0 ? 0 : -1);
        }

        private void BtnSelectDataFolder_Click(object sender, EventArgs e)
        {
            using (var dlg = new FolderBrowserDialog())
            {
                dlg.Description       = "選擇 AniloxCaptures 根目錄";
                dlg.SelectedPath      = string.IsNullOrWhiteSpace(_statsDataRootPath)
                    ? (_settings?.CaptureRootPath ?? string.Empty)
                    : _statsDataRootPath;
                dlg.ShowNewFolderButton = false;

                if (dlg.ShowDialog() == DialogResult.OK)
                {
                    _statsDataRootPath  = dlg.SelectedPath;
                    _statAvailableTimes = InspectionStatisticsService.LoadAvailableTimes(_statsDataRootPath);
                    _grabIdInfos        = InspectionStatisticsService.LoadGrabIdInfosDescending(_statsDataRootPath);

                    // 同步 Review tab：載入圖片索引 + 時間導航
                    UserSessionState.SetLastDataPath(_statsDataRootPath);
                    UserSessionState.Save();
                    _interactionHelper.LoadDirectoryAndInitNavigator(_statsDataRootPath);
                    _presenter.UpdatePeriodNavigationState();

                    // 填充序號 ComboBox
                    _statComboUpdating = true;
                    try
                    {
                        cbGrabIdStart.Items.Clear();
                        cbGrabIdEnd.Items.Clear();
                        cbDataGrabId.Items.Clear();
                        cbReviewGrabId.Items.Clear();
                        foreach (var info in _grabIdInfos)
                        {
                            cbGrabIdStart.Items.Add(info.GrabId);
                            cbGrabIdEnd.Items.Add(info.GrabId);
                            cbDataGrabId.Items.Add(info.GrabId);
                            cbReviewGrabId.Items.Add(info.GrabId);
                        }
                        SyncGrabIdFromTimeCombos();
                        UpdateGrabIdNavState();
                        if (cbGrabIdStart.Items.Count > 0)
                        {
                            cbGrabIdStart.SelectedIndex = cbGrabIdStart.Items.Count - 1;
                            cbGrabIdEnd.SelectedIndex = 0;
                            cbDataGrabId.SelectedIndex = cbGrabIdStart.SelectedIndex;
                        }

                    }
                    finally { _statComboUpdating = false; }

                    // 填充日期/時間 ComboBox（全範圍）
                    if (_statAvailableTimes.Count > 0)
                        PopulateStatDateCombos(_statAvailableTimes.Min, _statAvailableTimes.Max);

                    SetActiveStatGroupBox(groupBoxGrabIdRange);
                    PopulateChartNavigators(_statAvailableTimes.Count > 0
                        ? (DateTime?)_statAvailableTimes.Max : null);
                    RefreshStats();
                }
            }
        }


        private bool TryParseStatDateTime(out DateTime start, out DateTime end)
        {
            start = end = DateTime.MinValue;
            if (!TryBuildDateTimeFromCombos(cbStartDate, cbStartTime, out start)) return false;
            if (!TryBuildDateTimeFromCombos(cbEndDate,   cbEndTime,   out end))   return false;
            // 若無毫秒精度，將 end 推至該秒末尾以涵蓋所有毫秒
            if (end.Millisecond == 0) end = end.AddMilliseconds(999);
            return start <= end;
        }

        private static bool TryBuildDateTimeFromCombos(ComboBox dateCb, ComboBox timeCb, out DateTime result)
        {
            result = DateTime.MinValue;
            string dateText = dateCb.Text ?? "";
            string timeText = timeCb.Text ?? "";
            string combined = dateText + " " + timeText;
            // 嘗試 "yyyy-MM-dd HH:mm:ss.fff" 或 "yyyy-MM-dd HH:mm:ss"
            if (DateTime.TryParseExact(combined, new[] { "yyyy-MM-dd HH:mm:ss.fff", "yyyy-MM-dd HH:mm:ss" },
                    System.Globalization.CultureInfo.InvariantCulture,
                    System.Globalization.DateTimeStyles.None, out result))
                return true;
            return false;
        }

        // ==========================================
        // --- 統計 Tab：Cascading ComboBox 邏輯 ---
        // ==========================================

        private void WireStatDateCombos()
        {
            cbStartDate.SelectedIndexChanged += (s, e) => OnStartComboChanged(1);
            cbStartTime.SelectedIndexChanged += (s, e) => OnStartComboChanged(2);
            cbEndDate.SelectedIndexChanged   += (s, e) => OnEndComboChanged(1);
            cbEndTime.SelectedIndexChanged   += (s, e) => OnEndComboChanged(2);
        }

        private void OnStartComboChanged(int fromLevel)
        {
            if (_statComboUpdating) return;
            SetActiveStatGroupBox(groupBoxTimeRange);
            if (_statAvailableTimes.Count > 0)
            {
                _statComboUpdating = true;
                try
                {
                    if (fromLevel <= 1) RefreshStatTimeCombo(cbStartDate, cbStartTime, cbStartTime.Text);
                    ClampEndToStart();
                }
                finally { _statComboUpdating = false; }
            }
            RefreshStats();
        }

        private void OnEndComboChanged(int fromLevel)
        {
            if (_statComboUpdating) return;
            SetActiveStatGroupBox(groupBoxTimeRange);
            if (_statAvailableTimes.Count > 0)
            {
                _statComboUpdating = true;
                try
                {
                    if (fromLevel <= 1) RefreshStatTimeCombo(cbEndDate, cbEndTime, cbEndTime.Text);
                    ClampStartToEnd();
                }
                finally { _statComboUpdating = false; }
            }
            RefreshStats();
        }

        private void SetCombosToDateTime(bool isStart, DateTime dt)
        {
            string dateStr = dt.ToString("yyyy-MM-dd");
            string timeStr = dt.ToString("HH:mm:ss");
            if (isStart)
            {
                if (cbStartDate.Items.Contains(dateStr)) cbStartDate.SelectedItem = dateStr;
                else cbStartDate.Text = dateStr;
                RefreshStatTimeCombo(cbStartDate, cbStartTime, timeStr);
            }
            else
            {
                if (cbEndDate.Items.Contains(dateStr)) cbEndDate.SelectedItem = dateStr;
                else cbEndDate.Text = dateStr;
                RefreshStatTimeCombo(cbEndDate, cbEndTime, timeStr);
            }
        }

        /// <summary>若 start > end，將 end 推至最近的可用時間 ≥ start。</summary>
        private void ClampEndToStart()
        {
            if (!TryBuildDateTimeFromCombos(cbStartDate, cbStartTime, out DateTime start)) return;
            if (!TryBuildDateTimeFromCombos(cbEndDate,   cbEndTime,   out DateTime end))   return;
            if (start <= end) return;
            var view = _statAvailableTimes.GetViewBetween(start, DateTime.MaxValue);
            DateTime newEnd = view.Count > 0 ? view.Min : _statAvailableTimes.Max;
            SetCombosToDateTime(false, newEnd);
        }

        /// <summary>若 end < start，將 start 推至最近的可用時間 ≤ end。</summary>
        private void ClampStartToEnd()
        {
            if (!TryBuildDateTimeFromCombos(cbStartDate, cbStartTime, out DateTime start)) return;
            if (!TryBuildDateTimeFromCombos(cbEndDate,   cbEndTime,   out DateTime end))   return;
            if (start <= end) return;
            var view = _statAvailableTimes.GetViewBetween(DateTime.MinValue, end);
            DateTime newStart = view.Count > 0 ? view.Max : _statAvailableTimes.Min;
            SetCombosToDateTime(true, newStart);
        }

        /// <summary>
        /// cbGrabIdStart（序號起）或 cbGrabIdEnd（序號迄）變更時：
        /// 強制 start ≤ end、更新 cbStart/cbEnd 時間、重新統計。
        /// </summary>
        private void OnGrabIdComboChanged(bool isStart)
        {
            if (_statComboUpdating || _grabIdInfos.Count == 0) return;
            SetActiveStatGroupBox(groupBoxGrabIdRange);

            int idx1 = cbGrabIdStart.SelectedIndex;
            int idx2 = cbGrabIdEnd.SelectedIndex;
            if (idx1 < 0 || idx2 < 0) return;

            // 強制 cbGrabIdStart（舊）≥ cbGrabIdEnd（新）in descending index
            _statComboUpdating = true;
            try
            {
                if (isStart && idx1 < idx2)
                    cbGrabIdEnd.SelectedIndex = idx1;
                else if (!isStart && idx2 > idx1)
                    cbGrabIdStart.SelectedIndex = idx2;

                // 更新 cbStart/cbEnd 時間
                var startInfo = _grabIdInfos[cbGrabIdStart.SelectedIndex];
                var endInfo   = _grabIdInfos[cbGrabIdEnd.SelectedIndex];
                SetCombosToDateTime(true,  startInfo.Earliest);
                SetCombosToDateTime(false, endInfo.Latest);
            }
            finally { _statComboUpdating = false; }

            RefreshStats();
        }

        private async void OnSingleSheetComboChanged()
        {
            UpdateDataGrabIdNavState();
            if (_statComboUpdating || _grabIdInfos.Count == 0) return;
            if (_syncingGrabIdCross) return;
            SetActiveStatGroupBox(grpDataSingleSheet);
            int idx = cbDataGrabId.SelectedIndex;
            if (idx < 0) return;

            _statComboUpdating = true;
            try
            {
                cbGrabIdStart.SelectedIndex = idx;
                cbGrabIdEnd.SelectedIndex   = idx;
                var info = _grabIdInfos[idx];
                SetCombosToDateTime(true,  info.Earliest);
                SetCombosToDateTime(false, info.Latest);
            }
            finally { _statComboUpdating = false; }

            RefreshStats();

            // 同步 cbReviewGrabId（影像回顧）+ 拼接顯示
            if (!_syncingGrabIdCross && cbReviewGrabId.Items.Count > 0 && idx < cbReviewGrabId.Items.Count)
            {
                _syncingGrabIdCross = true;
                try
                {
                    var info = _grabIdInfos[idx];
                    _syncingGrabIdNav = true;
                    try
                    {
                        cbReviewGrabId.SelectedIndex = idx;
                        _interactionHelper.NavigateToDateTime(info.Earliest);
                    }
                    finally { _syncingGrabIdNav = false; }
                    UpdateGrabIdNavState();
                    SetGroupBoxActive(grpReviewGrabNav, true);
                    SetGroupBoxActive(grpReviewTimePeriod, false);
                    await LoadGrabStitchedViewAsync(info.GrabId, info.Earliest, info.Latest);
                }
                finally { _syncingGrabIdCross = false; }
            }
        }

        // ── 影像回顧 序號跳轉 ─────────────────────────────────────────────

        private async void OnReviewGrabIdChanged()
        {
            UpdateGrabIdNavState();
            if (_syncingGrabIdNav) return;
            if (_syncingGrabIdCross) return;
            if (_grabIdInfos.Count == 0) return;
            int idx = cbReviewGrabId.SelectedIndex;
            if (idx < 0 || idx >= _grabIdInfos.Count) return;

            _interactionHelper.SaveCanvasView();
            var info = _grabIdInfos[idx];
            _syncingGrabIdNav = true;
            try { _interactionHelper.NavigateToDateTime(info.Earliest); }
            finally { _syncingGrabIdNav = false; }

            await LoadGrabStitchedViewAsync(info.GrabId, info.Earliest, info.Latest);

            // 同步 cbDataGrabId（單片資訊）+ 統計
            if (!_syncingGrabIdCross && cbDataGrabId.Items.Count > 0 && idx < cbDataGrabId.Items.Count)
            {
                _syncingGrabIdCross = true;
                try
                {
                    cbDataGrabId.SelectedIndex = idx;
                    // 同步序號範圍 + 時間 + 統計
                    _statComboUpdating = true;
                    try
                    {
                        cbGrabIdStart.SelectedIndex = idx;
                        cbGrabIdEnd.SelectedIndex   = idx;
                        SetCombosToDateTime(true,  info.Earliest);
                        SetCombosToDateTime(false, info.Latest);
                    }
                    finally { _statComboUpdating = false; }
                    RefreshStats();
                    SetActiveStatGroupBox(grpDataSingleSheet);
                }
                finally { _syncingGrabIdCross = false; }
            }
        }

        private Task LoadGrabStitchedViewAsync(string grabId, DateTime hintFrom, DateTime hintTo)
            => LoadGrabStitchedViewAsync(grabId, hintFrom, hintTo, _lastReviewProcessedMode);

        private async Task LoadGrabStitchedViewAsync(string grabId, DateTime hintFrom, DateTime hintTo,
            bool enableProcess)
        {
            string root = !string.IsNullOrWhiteSpace(UserSessionState.LastDataPath)
                          ? UserSessionState.LastDataPath : _statsDataRootPath;
            if (string.IsNullOrWhiteSpace(root)) return;

            _interactionHelper.SetUiLoadingState(true);
            _lastReviewProcessedMode = enableProcess;
            _syncingProcessedCheckbox = true;
            try { checkBoxShowProcessed.Checked = enableProcess; }
            finally { _syncingProcessedCheckbox = false; }
            var swTotal = Stopwatch.StartNew();
            try
            {
                long csvMs = 0, stitchMs = 0;
                string ridgeDir = _activeRidgeDirection; // 快照 UI 狀態
                float[][] newCurveMean    = new float[7][];
                float[][] newCurveMax     = new float[7][];
                float[][] newRowCurveMean = new float[7][];
                float[][] newRowCurveMax  = new float[7][];
                CsvConfigSnapshot grabCfg = null;
                var newImages = await Task.Run(() =>
                {
                    var swCsv = Stopwatch.StartNew();
                    var grouped = InspectionStatisticsService.LoadImagePathsForGrabId(
                        root, grabId, hintFrom, hintTo);
                    grabCfg = InspectionStatisticsService.LoadConfigForGrabId(
                        root, grabId, hintFrom, hintTo);
                    csvMs = swCsv.ElapsedMilliseconds;

                    var swStitch = Stopwatch.StartNew();
                    int scale = InspectionEngineConfig.DefaultSaveResizeScale;
                    var imgs = new Bitmap[7];
                    for (int i = 0; i < 7; i++)
                    {
                        int camId = i + 1;
                        if (grouped.TryGetValue(camId, out var paths) && paths.Count > 0)
                        {
                            try
                            {
                                bool isBmp = paths[0].EndsWith(".bmp", StringComparison.OrdinalIgnoreCase);

                                if (enableProcess && isBmp && _inspectionService != null)
                                {
                                    // BMP 處理模式：逐張 GPU pipeline + resize，再拼接
                                    // ProcessBmpAtScale 回傳 V ridge 並同時存 _proc_v.jpg + _proc_h.jpg
                                    string capturedDir = ridgeDir;
                                    Func<string, Bitmap> procLoader = (p) =>
                                    {
                                        var bmp = _inspectionService.ProcessBmpAtScale(p, scale,
                                            out float[] m, out float[] x);
                                        if (capturedDir == "h" && bmp != null)
                                        {
                                            // H 方向：ProcessBmpAtScale 已存好 _proc_h.jpg，從磁碟載入
                                            string baseName = System.IO.Path.Combine(
                                                System.IO.Path.GetDirectoryName(p),
                                                System.IO.Path.GetFileNameWithoutExtension(p));
                                            string procH = baseName + "_proc_h.jpg";
                                            if (System.IO.File.Exists(procH))
                                            {
                                                bmp.Dispose();
                                                byte[] bytes = System.IO.File.ReadAllBytes(procH);
                                                using (var ms = new System.IO.MemoryStream(bytes))
                                                    return new Bitmap(ms);
                                            }
                                        }
                                        return bmp;
                                    };
                                    imgs[i] = GrabImageStitcher.StitchCamera(paths, scale, procLoader,
                                        ridgeDirection: ridgeDir);
                                }
                                else
                                {
                                    // JPEG 路徑（含 _proc_v/h.jpg 切換）或 BMP 原圖路徑
                                    Func<string, Bitmap> bmpLoader = _inspectionService != null
                                        ? (Func<string, Bitmap>)(p => _inspectionService.LoadBmpAtScale(p, scale))
                                        : null;
                                    imgs[i] = GrabImageStitcher.StitchCamera(paths, scale, bmpLoader,
                                        useProcessed: enableProcess, ridgeDirection: ridgeDir);
                                }
                                MergeCurves(paths, out newCurveMean[i], out newCurveMax[i]);
                                MergeRowCurves(paths, out newRowCurveMean[i], out newRowCurveMax[i]);
                            }
                            catch (Exception ex)
                            {
                                System.Diagnostics.Trace.WriteLine(
                                    $"[StitchView] CAM{camId}: {ex.GetType().Name}: {ex.Message}");
                            }
                        }
                    }
                    stitchMs = swStitch.ElapsedMilliseconds;
                    return imgs;
                });

                ClearStitchedMode();
                _stitchedImages    = newImages;
                _stitchedCurveMean    = newCurveMean;
                _stitchedCurveMax     = newCurveMax;
                _stitchedRowCurveMean = newRowCurveMean;
                _stitchedRowCurveMax  = newRowCurveMax;
                _currentGrabConfig = grabCfg;
                SetGroupBoxActive(grpReviewGrabNav, true); SetGroupBoxActive(grpReviewTimePeriod, false);
                _galleryManager.SetImages(_stitchedImages);
                ShowStitchedCameraInCanvas(_galleryManager.SelectedIndex);
                UpdateStitchedOverviewChart();

                Trace.WriteLine($"[StitchView] {grabId} proc={enableProcess} | CSV={csvMs}ms | Stitch={stitchMs}ms | Total={swTotal.ElapsedMilliseconds}ms");
            }
            finally
            {
                _interactionHelper.SetUiLoadingState(false);
            }
        }

        private void ClearStitchedMode()
        {
            if (_stitchedImages == null) return;
            canvasMain.Image = null;
            _galleryManager.ClearImages();
            foreach (var bmp in _stitchedImages) bmp?.Dispose();
            _stitchedImages = null;
            _stitchedCurveMean    = null;
            _stitchedCurveMax     = null;
            _stitchedRowCurveMean = null;
            _stitchedRowCurveMax  = null;
            _currentGrabConfig = null;
            // 恢復 chart 為當前設定（stitch mode 可能改用了歷史 #CFG 的 Ops/閾值）
            _muraChartHelper?.SetOps(_settings.Cam1_Ops);
            _muraChartHelper?.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            // 清除全覽圖
            if (_stitchedOverviewHelper != null && chartOverview.ChartAreas.Count > 0)
            {
                chartOverview.Series["Mean"].Points.Clear();
                chartOverview.Series["Max"].Points.Clear();
            }
            SetGroupBoxActive(grpReviewGrabNav, false); SetGroupBoxActive(grpReviewTimePeriod, true);
        }

        // ── GroupBox 綠色高亮指示 ───────────────────────────────────────────

        private static readonly Color _activeGrpFill   = Color.FromArgb(220, 248, 225);
        private static readonly Color _activeGrpBorder = Color.FromArgb(0, 140, 60);

        private void SetGroupBoxActive(GroupBox box, bool active)
        {
            if (active)
            {
                box.Paint -= ActiveGroupBox_Paint;
                box.Paint += ActiveGroupBox_Paint;
            }
            else
            {
                box.Paint -= ActiveGroupBox_Paint;
            }
            box.Invalidate();
        }

        private void SetActiveStatGroupBox(GroupBox active)
        {
            _activeStatMode = active;
            foreach (var box in new[] { groupBoxGrabIdRange, grpDataSingleSheet, groupBoxTimeRange })
                SetGroupBoxActive(box, box == active);
        }

        private static void ActiveGroupBox_Paint(object sender, PaintEventArgs e)
        {
            var g = e.Graphics;
            var box = (GroupBox)sender;
            int textH = (int)g.MeasureString(box.Text, box.Font).Height;
            int midY = textH / 2;

            using (var brush = new SolidBrush(_activeGrpFill))
                g.FillRectangle(brush, 0, midY, box.Width, box.Height - midY);
            using (var pen = new Pen(_activeGrpBorder, 1.5f))
                g.DrawRectangle(pen, 0, midY, box.Width - 1, box.Height - midY - 1);

            var textSize = g.MeasureString(box.Text, box.Font);
            using (var bgBrush = new SolidBrush(_activeGrpFill))
                g.FillRectangle(bgBrush, 6, 0, textSize.Width + 2, textH);
            using (var textBrush = new SolidBrush(_activeGrpBorder))
                g.DrawString(box.Text, box.Font, textBrush, 8, 0);
        }

        /// <summary>
        /// 切換 Live 顯示的 V/H 處理圖方向，點選 muraChartVerticalLive/HorizontalLive 觸發。
        /// 三態邏輯同 Review tab 的 SwitchRidgeDirection：
        /// 未勾選 → 自動勾選 + 設方向；同方向 → 取消勾選；不同方向 → 切換。
        /// </summary>
        private void SwitchLiveDisplayDirection(string dir)
        {
            if (!checkBoxEnableImageProcessing.Checked)
            {
                // 未勾選 → 自動勾選 + 設方向
                _liveDisplayDirection = dir;
                _liveCameraManager?.SetLiveDisplayDirection(dir);
                UpdateLiveDirectionVisual(dir);
                checkBoxEnableImageProcessing.Checked = true; // 觸發 CheckedChanged
                return;
            }

            if (dir == _liveDisplayDirection)
            {
                // 同方向再點一次 → 取消勾選（回原圖）
                UpdateLiveDirectionVisual(null);
                checkBoxEnableImageProcessing.Checked = false; // 觸發 CheckedChanged
                return;
            }

            // 不同方向 → 切換（不改 checkbox）
            _liveDisplayDirection = dir;
            _liveCameraManager?.SetLiveDisplayDirection(dir);
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
            if (!_lastReviewProcessedMode)
            {
                // 未勾選 → 自動勾選 + 設方向
                _activeRidgeDirection = dir;
                _interactionHelper.SetRidgeDirection(dir);
                UpdateRidgeDirectionVisual(dir);
                checkBoxShowProcessed.Checked = true; // 觸發 CheckedChanged → 載入處理圖
                return;
            }

            if (dir == _activeRidgeDirection)
            {
                // 同方向再點一次 → 取消勾選（回原圖）
                UpdateRidgeDirectionVisual(null);
                checkBoxShowProcessed.Checked = false; // 觸發 CheckedChanged → 載入原圖
                return;
            }

            // 不同方向 → 切換（重新載入處理圖）
            _activeRidgeDirection = dir;
            _interactionHelper.SetRidgeDirection(dir);
            UpdateRidgeDirectionVisual(dir);
            _interactionHelper.SaveCanvasView();

            if (_stitchedImages != null)
            {
                // cbReviewGrabId.Items 為 string，用 _grabIdInfos[idx] 取得完整資訊
                int idx = cbReviewGrabId.SelectedIndex;
                if (idx >= 0 && idx < _grabIdInfos.Count)
                {
                    var info = _grabIdInfos[idx];
                    await LoadGrabStitchedViewAsync(info.GrabId, info.Earliest, info.Latest, true);
                }
            }
            else
            {
                // 非合圖路徑：重新載入所有處理圖（方向已更新）
                ClearStitchedMode();
                await _presenter.LoadImagesWithPeriodLockAsync(true, _interactionHelper.LoadImages);
                UpdateOverviewChartFromRepository();
            }
        }

        private void UpdateRidgeDirectionVisual(string dir)
        {
            chartMuraVertical.BackColor = (dir == "v")
                ? System.Drawing.Color.FromArgb(230, 240, 255) : System.Drawing.SystemColors.Control;
            chartMuraHorizontal.BackColor = (dir == "h")
                ? System.Drawing.Color.FromArgb(230, 240, 255) : System.Drawing.SystemColors.Control;
        }

        private void ShowStitchedCameraInCanvas(int idx)
        {
            if (_stitchedImages == null) return;
            var bmp = (idx >= 0 && idx < _stitchedImages.Length) ? _stitchedImages[idx] : null;

            // 設定 scaleFactor 和 cameraIndex，FitToScreen 觸發 StatusChanged 時 mm 換算才正確
            _interactionHelper.SetCanvasScaleAndCamera(
                InspectionEngineConfig.DefaultSaveResizeScale, idx);

            canvasMain.Image = bmp;
            if (bmp != null) _interactionHelper.RestoreCanvasViewOrFit();

            // 更新 MuraChart（含 X 軸範圍，與 Period 模式一致）
            // 若有 #CFG 設定快照，用抓圖當時的 Ops/Pos/閾值；否則 fallback 到當前 _settings
            if (_muraChartHelper != null && _settings != null)
            {
                float[] mean = (_stitchedCurveMean != null && idx >= 0 && idx < _stitchedCurveMean.Length)
                    ? _stitchedCurveMean[idx] : null;
                float[] max = (_stitchedCurveMax != null && idx >= 0 && idx < _stitchedCurveMax.Length)
                    ? _stitchedCurveMax[idx] : null;

                double[] posArr;
                if (_currentGrabConfig != null)
                {
                    double opsUm = (idx >= 0 && idx < _currentGrabConfig.CamOps.Length)
                        ? _currentGrabConfig.CamOps[idx] : _settings.Cam1_Ops;
                    _muraChartHelper.SetOps(opsUm);
                    _muraChartHelper.SetThresholds(
                        _currentGrabConfig.ErrorValueMean, _currentGrabConfig.ErrorValueMax);
                    posArr = _currentGrabConfig.CamPos;
                }
                else
                {
                    posArr = _settings.GetCameraStartPositionMmArray();
                }

                double startPos = (idx >= 0 && idx < posArr.Length) ? posArr[idx] : 0;
                _interactionHelper.TryComputeCurrentViewRange(idx, out double leftMm, out double rightMm);
                _muraChartHelper.UpdateDataAndView(mean, max, startPos, leftMm, rightMm);
            }

            // 更新法向（水平）Mura 曲線圖
            if (_muraChartHorizontalHelper != null)
            {
                float[] rowMean = (_stitchedRowCurveMean != null && idx >= 0 && idx < _stitchedRowCurveMean.Length)
                    ? _stitchedRowCurveMean[idx] : null;
                float[] rowMax = (_stitchedRowCurveMax != null && idx >= 0 && idx < _stitchedRowCurveMax.Length)
                    ? _stitchedRowCurveMax[idx] : null;
                if (rowMean != null)
                {
                    _muraChartHorizontalHelper.UpdateData(rowMean, rowMax);
                    _interactionHelper.RefreshRowChartRange();
                }
            }
        }

        /// <summary>
        /// 合圖路徑：用 _stitchedCurveMean/Max 更新 chart1 全覽圖。
        /// </summary>
        private void UpdateStitchedOverviewChart()
        {
            if (_stitchedCurveMean == null) return;

            double[] opsArr, posArr;
            float errMean, errMax;
            if (_currentGrabConfig != null)
            {
                opsArr  = _currentGrabConfig.CamOps;
                posArr  = _currentGrabConfig.CamPos;
                errMean = _currentGrabConfig.ErrorValueMean;
                errMax  = _currentGrabConfig.ErrorValueMax;
            }
            else
            {
                opsArr  = _settings.GetCameraOpsUmArray();
                posArr  = _settings.GetCameraStartPositionMmArray();
                errMean = _settings.ErrorValueMean;
                errMax  = _settings.ErrorValueMax;
            }

            UpdateOverviewChart(_stitchedCurveMean, _stitchedCurveMax, opsArr, posArr, errMean, errMax);
        }

        /// <summary>
        /// 原圖路徑：從當前 Repository 時間點讀取 7 台 .bin 曲線更新 chart1 全覽圖。
        /// </summary>
        private void UpdateOverviewChartFromRepository()
        {
            if (_stitchedOverviewHelper == null || _stitchedImages != null) return;

            var images = _imageRepository.GetImages(
                _dateTimeNavigator.GetCurrentYear(),
                _dateTimeNavigator.GetCurrentMonth(),
                _dateTimeNavigator.GetCurrentDay(),
                _dateTimeNavigator.GetCurrentHour(),
                _dateTimeNavigator.GetCurrentMin(),
                _dateTimeNavigator.GetCurrentSec());

            if (images == null || images.Count == 0)
            {
                chartOverview.Series["Mean"].Points.Clear();
                chartOverview.Series["Max"].Points.Clear();
                return;
            }

            var curveMean = new float[7][];
            var curveMax  = new float[7][];
            for (int i = 0; i < 7; i++)
            {
                if (!images.TryGetValue(i + 1, out string path)) continue;
                string basePath;
                if (path.EndsWith("_raw.jpg", StringComparison.OrdinalIgnoreCase))
                    basePath = path.Substring(0, path.Length - "_raw.jpg".Length);
                else
                    basePath = System.IO.Path.Combine(
                        System.IO.Path.GetDirectoryName(path),
                        System.IO.Path.GetFileNameWithoutExtension(path));
                curveMean[i] = InspectionEngine.LoadCurveBin(basePath + "_mean_v.bin")
                            ?? InspectionEngine.LoadCurveBin(basePath + "_mean.bin");
                curveMax[i]  = InspectionEngine.LoadCurveBin(basePath + "_max_v.bin")
                            ?? InspectionEngine.LoadCurveBin(basePath + "_max.bin");
            }

            UpdateOverviewChart(curveMean, curveMax,
                _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray(),
                _settings.ErrorValueMean, _settings.ErrorValueMax);
        }

        /// <summary>
        /// 將 7 台相機的曲線依機台布局位置合併到全覽圖。
        /// 重疊區域：Mean 取平均、Max 取最大值。target 預設 chart1（回顧），可指定 chartLiveOverview（即時）。
        /// </summary>
        private void UpdateOverviewChart(float[][] allMean, float[][] allMax,
            double[] opsArr, double[] posArr, float errMean, float errMax,
            MuraChartHelper target = null)
        {
            target = target ?? _stitchedOverviewHelper;
            if (target == null || allMean == null) return;

            // 全域範圍：涵蓋全部 7 台位置，缺圖用現有影像寬度平均類推
            double sumWidthMm = 0;
            int widthCount = 0;
            double minOpsUm = double.MaxValue;
            for (int i = 0; i < 7; i++)
            {
                if (opsArr[i] > 0 && opsArr[i] < minOpsUm) minOpsUm = opsArr[i];
                var curve = allMean[i];
                if (curve != null && curve.Length > 0)
                {
                    sumWidthMm += curve.Length * (opsArr[i] / 1000.0);
                    widthCount++;
                }
            }
            if (minOpsUm <= 0 || minOpsUm == double.MaxValue) minOpsUm = 33.0;
            double avgWidthMm = widthCount > 0 ? sumWidthMm / widthCount : 400.0;

            double globalMin = double.MaxValue, globalMax = double.MinValue;
            for (int i = 0; i < 7; i++)
            {
                double camStart = posArr[i];
                var curve = allMean[i];
                double camEnd = (curve != null && curve.Length > 0)
                    ? camStart + curve.Length * (opsArr[i] / 1000.0)
                    : camStart + avgWidthMm;
                if (camStart < globalMin) globalMin = camStart;
                if (camEnd   > globalMax) globalMax = camEnd;
            }
            if (globalMin >= globalMax) return;

            // 格點間距：至少 OPS 精度，但上限 MaxOverviewPoints 點
            double gridMm = Math.Max(minOpsUm / 1000.0, (globalMax - globalMin) / MaxOverviewPoints);

            int totalLen = (int)Math.Ceiling((globalMax - globalMin) / gridMm);
            if (totalLen <= 0 || totalLen > MaxOverviewPoints + 1) return;

            // 兩層合併：
            // 1) bin 內降解析（同一台相機多點 → 1 bin）→ max-window 保峰值
            // 2) 相機重疊（多台相機同一 bin）→ Mean 取平均、Max 取最大值
            // 先逐台 max-window 到暫存，再跨台合併
            var mergedMean   = new float[totalLen];
            var mergedMax    = new float[totalLen];
            var overlapCount = new int[totalLen];

            for (int i = 0; i < 7; i++)
            {
                var curveMean = allMean[i];
                if (curveMean == null || curveMean.Length == 0) continue;
                var curveMax = (allMax != null && i < allMax.Length) ? allMax[i] : null;

                double camOpsMm = opsArr[i] / 1000.0;
                double camStart = posArr[i];

                // 逐台 max-window：同一台相機多個原始點落入同一 bin 時取最大值
                var camBinMean = new float[totalLen];
                var camBinMax  = new float[totalLen];
                var camBinHit  = new bool[totalLen];

                for (int j = 0; j < curveMean.Length; j++)
                {
                    int idx = (int)((camStart + j * camOpsMm - globalMin) / gridMm);
                    if (idx < 0 || idx >= totalLen) continue;

                    if (!camBinHit[idx] || curveMean[j] > camBinMean[idx])
                        camBinMean[idx] = curveMean[j];

                    float mv = (curveMax != null && j < curveMax.Length) ? curveMax[j] : 0;
                    if (!camBinHit[idx] || mv > camBinMax[idx])
                        camBinMax[idx] = mv;

                    camBinHit[idx] = true;
                }

                // 跨台合併：Mean 累加（後面除 count）、Max 取最大值
                for (int k = 0; k < totalLen; k++)
                {
                    if (!camBinHit[k]) continue;
                    mergedMean[k] += camBinMean[k];
                    overlapCount[k] += 1;
                    if (camBinMax[k] > mergedMax[k]) mergedMax[k] = camBinMax[k];
                }
            }

            // 重疊區域 Mean 取平均
            for (int i = 0; i < totalLen; i++)
                if (overlapCount[i] > 1) mergedMean[i] /= overlapCount[i];

            // 降解析後每點間距 = gridMm，轉回 μm 給 MuraChartHelper
            target.SetOps(gridMm * 1000.0);
            target.SetThresholds(errMean, errMax);
            target.UpdateData(mergedMean, mergedMax, globalMin);
        }

        /// <summary>
        /// 載入多張影像的 .bin 曲線，Mean 取平均、Max 取最大值。
        /// 曲線保持全解析度（mm 座標由 MuraChart 映射，不需與圖片 pixel 對齊）。
        /// </summary>
        private static void MergeCurves(IList<string> imagePaths,
            out float[] mergedMean, out float[] mergedMax)
        {
            mergedMean = null;
            mergedMax  = null;

            var allMean = new List<float[]>();
            var allMax  = new List<float[]>();
            int curveLen = 0;

            foreach (string path in imagePaths)
            {
                string basePath;
                if (path.EndsWith("_raw.jpg", StringComparison.OrdinalIgnoreCase))
                    basePath = path.Substring(0, path.Length - "_raw.jpg".Length);
                else
                    basePath = System.IO.Path.Combine(
                        System.IO.Path.GetDirectoryName(path),
                        System.IO.Path.GetFileNameWithoutExtension(path));

                var mean = InspectionEngine.LoadCurveBin(basePath + "_mean_v.bin")
                        ?? InspectionEngine.LoadCurveBin(basePath + "_mean.bin");
                var max  = InspectionEngine.LoadCurveBin(basePath + "_max_v.bin")
                        ?? InspectionEngine.LoadCurveBin(basePath + "_max.bin");
                if (mean != null && max != null && mean.Length > 0)
                {
                    allMean.Add(mean);
                    allMax.Add(max);
                    if (curveLen == 0) curveLen = mean.Length;
                }
            }

            if (allMean.Count == 0 || curveLen == 0) return;

            // 合併：Mean 取全部平均，Max 取全部最大值
            mergedMean = new float[curveLen];
            mergedMax  = new float[curveLen];
            for (int x = 0; x < curveLen; x++)
            {
                float sumMean = 0;
                float maxVal  = float.MinValue;
                int count = 0;
                for (int j = 0; j < allMean.Count; j++)
                {
                    if (x < allMean[j].Length) { sumMean += allMean[j][x]; count++; }
                    if (x < allMax[j].Length && allMax[j][x] > maxVal) maxVal = allMax[j][x];
                }
                mergedMean[x] = count > 0 ? sumMean / count : 0;
                mergedMax[x]  = maxVal > float.MinValue ? maxVal : 0;
            }
        }

        /// <summary>
        /// Row 曲線合併：多張影像的 row curves 依時間順序串接（非 per-index 平均），
        /// 因為每張圖代表不同的法向（axial）位置。
        /// </summary>
        private static void MergeRowCurves(IList<string> imagePaths,
            out float[] mergedMean, out float[] mergedMax)
        {
            mergedMean = null;
            mergedMax  = null;

            var allMean = new List<float[]>();
            var allMax  = new List<float[]>();

            foreach (string path in imagePaths)
            {
                string basePath;
                if (path.EndsWith("_raw.jpg", StringComparison.OrdinalIgnoreCase))
                    basePath = path.Substring(0, path.Length - "_raw.jpg".Length);
                else
                    basePath = System.IO.Path.Combine(
                        System.IO.Path.GetDirectoryName(path),
                        System.IO.Path.GetFileNameWithoutExtension(path));

                var mean = InspectionEngine.LoadCurveBin(basePath + "_mean_h.bin")
                        ?? InspectionEngine.LoadCurveBin(basePath + "_row_mean.bin");
                var max  = InspectionEngine.LoadCurveBin(basePath + "_max_h.bin")
                        ?? InspectionEngine.LoadCurveBin(basePath + "_row_max.bin");
                if (mean != null && max != null && mean.Length > 0)
                {
                    allMean.Add(mean);
                    allMax.Add(max);
                }
            }

            if (allMean.Count == 0) return;

            // 串接：每張圖的 row curves 依序接起來（對應 GrabImageStitcher 的垂直拼接）
            int totalLen = 0;
            foreach (var a in allMean) totalLen += a.Length;

            mergedMean = new float[totalLen];
            mergedMax  = new float[totalLen];
            int offset = 0;
            for (int j = 0; j < allMean.Count; j++)
            {
                Array.Copy(allMean[j], 0, mergedMean, offset, allMean[j].Length);
                Array.Copy(allMax[j],  0, mergedMax,  offset, allMax[j].Length);
                offset += allMean[j].Length;
            }
        }

        private void StepReviewGrabId(int delta)
        {
            if (_grabIdInfos.Count == 0) return;
            int next = cbReviewGrabId.SelectedIndex + delta;
            if (next >= 0 && next < cbReviewGrabId.Items.Count)
                cbReviewGrabId.SelectedIndex = next;   // triggers OnReviewGrabIdChanged
        }

        private void StepDataGrabId(int delta)
        {
            if (_grabIdInfos.Count == 0) return;
            int next = cbDataGrabId.SelectedIndex + delta;
            if (next >= 0 && next < cbDataGrabId.Items.Count)
                cbDataGrabId.SelectedIndex = next;   // triggers OnSingleSheetComboChanged
        }

        private void UpdateGrabIdNavState()
        {
            int idx   = cbReviewGrabId.SelectedIndex;
            int count = cbReviewGrabId.Items.Count;
            btnGrabIdPrev.Enabled = idx > 0;
            btnGrabIdNext.Enabled = idx >= 0 && idx < count - 1;
            UpdateDataGrabIdNavState();
        }

        private void UpdateDataGrabIdNavState()
        {
            int idx   = cbDataGrabId.SelectedIndex;
            int count = cbDataGrabId.Items.Count;
            btnGrabIdDataPrev.Enabled = idx > 0;
            btnGrabIdDataNext.Enabled = count > 0 && idx < count - 1;
        }

        /// <summary>
        /// 時間 ComboBox 變更時，同步 cbReviewGrabId 到包含該時間的序號。
        /// </summary>
        private void SyncGrabIdFromTimeCombos()
        {
            if (_syncingGrabIdNav || _grabIdInfos.Count == 0) return;
            DateTime current = _dateTimeNavigator.GetCurrentPeriodOrDefault(DateTime.MinValue);
            if (current == DateTime.MinValue) return;

            // 找包含 current 的 grab ID（Earliest ≤ current ≤ Latest），
            // 若無精確匹配則找 Earliest 最接近且 ≤ current 的
            int bestIdx = -1;
            long bestDiff = long.MaxValue;
            for (int i = 0; i < _grabIdInfos.Count; i++)
            {
                var info = _grabIdInfos[i];
                if (current >= info.Earliest && current <= info.Latest)
                {
                    bestIdx = i;
                    break;
                }
                long diff = Math.Abs(current.Ticks - info.Earliest.Ticks);
                if (diff < bestDiff)
                {
                    bestDiff = diff;
                    bestIdx = i;
                }
            }

            if (bestIdx >= 0 && bestIdx < cbReviewGrabId.Items.Count
                && bestIdx != cbReviewGrabId.SelectedIndex)
            {
                _syncingGrabIdNav = true;
                try { cbReviewGrabId.SelectedIndex = bestIdx; }
                finally { _syncingGrabIdNav = false; }
            }
        }

        private void RefreshStats()
        {
            if (string.IsNullOrWhiteSpace(_statsDataRootPath)) return;

            // 序號模式（groupBoxGrabIdRange 或 grpDataSingleSheet 活動中）
            if (_activeStatMode != groupBoxTimeRange
                && cbGrabIdStart.SelectedIndex >= 0 && cbGrabIdEnd.SelectedIndex >= 0
                && _grabIdInfos.Count > 0)
            {
                var startInfo = _grabIdInfos[cbGrabIdStart.SelectedIndex];
                var endInfo   = _grabIdInfos[cbGrabIdEnd.SelectedIndex];
                int startNum  = startInfo.GrabNum;
                int endNum    = endInfo.GrabNum;

                var stats   = InspectionStatisticsService.ComputeByGrabIdRange(
                    _statsDataRootPath, startNum, endNum);
                var details = InspectionStatisticsService.ComputeDetailedByGrabIdRange(
                    _statsDataRootPath, startNum, endNum);

                _statsPresenter.Update(stats);
                _currentDetails = details;
                ApplyFailFilter();
                return;
            }

            // 時間模式
            if (!TryParseStatDateTime(out DateTime start, out DateTime end)) return;

            // 找出時間範圍內的序號，用序號邏輯統計（同一序號同一相機一票否決）
            var grabInfosInRange = _grabIdInfos
                .Where(g => g.Earliest <= end && g.Latest >= start).ToList();

            if (grabInfosInRange.Count > 0)
            {
                int startNum = grabInfosInRange.Min(g => g.GrabNum);
                int endNum   = grabInfosInRange.Max(g => g.GrabNum);

                var stats   = InspectionStatisticsService.ComputeByGrabIdRange(
                    _statsDataRootPath, startNum, endNum);
                var details = InspectionStatisticsService.ComputeDetailedByGrabIdRange(
                    _statsDataRootPath, startNum, endNum);

                _statsPresenter.Update(stats);
                _currentDetails = details;
            }
            else
            {
                var statsTime = InspectionStatisticsService.Compute(_statsDataRootPath, start, end);
                _statsPresenter.Update(statsTime);
                _currentDetails = new List<GrabDetail>();
            }
            ApplyFailFilter();
        }

        private void InitGrabDetailListView()
        {
            listViewGrabDetail.View          = View.Details;
            listViewGrabDetail.FullRowSelect = true;
            listViewGrabDetail.GridLines     = true;
            listViewGrabDetail.Columns.Clear();
            listViewGrabDetail.Items.Clear();

            listViewGrabDetail.Columns.Add("料件序號", -1, HorizontalAlignment.Center);
            for (int i = 1; i <= 7; i++)
                listViewGrabDetail.Columns.Add($"{i}", -1, HorizontalAlignment.Center);
            FitListViewColumnsProportional(listViewGrabDetail);
        }

        private static readonly System.Drawing.Color _detailPass  = System.Drawing.Color.FromArgb(232, 245, 233);
        private static readonly System.Drawing.Color _detailFail  = System.Drawing.Color.FromArgb(255, 235, 238);
        private static readonly System.Drawing.Color _detailEmpty = SystemColors.Window;

        private void UpdateGrabDetailListView(List<GrabDetail> details)
        {
            listViewGrabDetail.BeginUpdate();
            listViewGrabDetail.Items.Clear();

            foreach (var d in details)
            {
                var item = new ListViewItem(d.GrabId);
                bool rowHasFail = false;

                for (int i = 0; i < 7; i++)
                {
                    if (d.CamResult[i] == null)
                    {
                        item.SubItems.Add("—");
                    }
                    else if (d.CamResult[i] == false)
                    {
                        item.SubItems.Add("Pass");
                    }
                    else
                    {
                        item.SubItems.Add("Fail");
                        rowHasFail = true;
                    }
                }

                item.BackColor = rowHasFail ? _detailFail : _detailPass;
                listViewGrabDetail.Items.Add(item);
            }

            listViewGrabDetail.EndUpdate();
        }

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

        /// <summary>
        /// 依欄位標題文字長度按比例分配 ListView 欄寬，填滿控制項寬度（不出現水平捲軸）。
        /// </summary>
        private static void FitListViewColumnsProportional(ListView lv)
        {
            if (lv.Columns.Count == 0) return;
            int available = lv.ClientSize.Width - SystemInformation.VerticalScrollBarWidth;
            if (available <= 0) return;

            using (var g = lv.CreateGraphics())
            {
                var weights = new float[lv.Columns.Count];
                float totalWeight = 0;
                for (int i = 0; i < lv.Columns.Count; i++)
                {
                    float w = g.MeasureString(lv.Columns[i].Text + "WW", lv.Font).Width;
                    weights[i] = w;
                    totalWeight += w;
                }
                if (totalWeight <= 0) return;

                int assigned = 0;
                for (int i = 0; i < lv.Columns.Count; i++)
                {
                    int colW = (i < lv.Columns.Count - 1)
                        ? (int)(available * weights[i] / totalWeight)
                        : available - assigned;
                    lv.Columns[i].Width = Math.Max(20, colW);
                    assigned += lv.Columns[i].Width;
                }
            }
        }

        // ── 異常篩選 ─────────────────────────────────────────────────────

        private void BtnShowFail_Click(object sender, EventArgs e)
        {
            _showFailOnly = !_showFailOnly;
            btnShowFail.Text      = _showFailOnly ? "顯示全部" : "篩選異常";
            btnShowFail.BackColor = _showFailOnly
                ? System.Drawing.Color.FromArgb(255, 235, 238)
                : SystemColors.Control;
            ApplyFailFilter();
        }

        private void ApplyFailFilter()
        {
            var toShow = _showFailOnly
                ? _currentDetails.Where(d => d.CamResult.Any(r => r == true)).ToList()
                : _currentDetails;
            UpdateGrabDetailListView(toShow);
        }

        // ── 趨勢圖（年 / 月 / 日）────────────────────────────────────────

        private void InitPeriodCharts()
        {
            var cs = _settings.Chart;
            InitOneChart(chartYearly,  yDefault: cs.YearlyYMax,  xCount: 12, xStart: 1);  // 月份 1-12
            InitOneChart(chartMonthly, yDefault: cs.MonthlyYMax, xCount: 31, xStart: 1);  // 日期 1-31
            InitOneChart(chartDaily,   yDefault: cs.DailyYMax,   xCount: 24, xStart: 0);  // 小時 0-23

            // ScaleMode=Auto 時，初始即套用自動範圍（空資料 → 預設 niceMax=5）
            if (cs.ScaleMode == ChartScaleMode.Auto)
            {
                var empty = new List<PeriodStats>();
                ApplyAutoScale(chartYearly,  empty);
                ApplyAutoScale(chartMonthly, empty);
                ApplyAutoScale(chartDaily,   empty);
            }

            chartYearly.MouseClick  -= PeriodChart_ToggleAutoScale;
            chartMonthly.MouseClick -= PeriodChart_ToggleAutoScale;
            chartDaily.MouseClick   -= PeriodChart_ToggleAutoScale;
            chartYearly.MouseClick  += PeriodChart_ToggleAutoScale;
            chartMonthly.MouseClick += PeriodChart_ToggleAutoScale;
            chartDaily.MouseClick   += PeriodChart_ToggleAutoScale;
        }

        private void PeriodChart_ToggleAutoScale(object sender, System.Windows.Forms.MouseEventArgs e)
        {
            var chart = (System.Windows.Forms.DataVisualization.Charting.Chart)sender;
            if (chart.ChartAreas.Count == 0) return;

            bool isAuto = "auto".Equals(chart.Tag)
                       || _settings.Chart.ScaleMode == ChartScaleMode.Auto;

            if (isAuto)
            {
                int fixedMax = chart == chartYearly  ? _settings.Chart.YearlyYMax
                             : chart == chartMonthly ? _settings.Chart.MonthlyYMax
                             :                         _settings.Chart.DailyYMax;
                ApplyFixedScale(chart, fixedMax);
            }
            else
            {
                chart.Tag = "auto";
                // 從目前資料重新計算自動範圍
                var data = new List<PeriodStats>();
                var sPass = chart.Series["合格"];
                var sFail = chart.Series["異常"];
                for (int i = 0; i < sPass.Points.Count; i++)
                    data.Add(new PeriodStats { Label = "", Pass = (int)sPass.Points[i].YValues[0], Fail = (int)sFail.Points[i].YValues[0] });
                ApplyAutoScale(chart, data);
            }
        }

        private static void InitOneChart(
            System.Windows.Forms.DataVisualization.Charting.Chart chart,
            int xLabelAngle = 0,
            int yDefault    = 10,
            int xCount      = 0,
            int xStart      = 1)
        {
            chart.ChartAreas.Clear();
            chart.Series.Clear();
            chart.Legends.Clear();
            chart.Titles.Clear();

            var area = new System.Windows.Forms.DataVisualization.Charting.ChartArea("Main");
            // X 軸：格線（垂直虛線）、刻度、每格顯示標籤、小字型
            area.AxisX.MajorGrid.Enabled        = true;
            area.AxisX.MajorGrid.LineColor      = System.Drawing.Color.FromArgb(220, 220, 220);
            area.AxisX.MajorGrid.LineDashStyle  = System.Windows.Forms.DataVisualization.Charting.ChartDashStyle.Dot;
            area.AxisX.MajorTickMark.Enabled    = true;
            area.AxisX.MajorTickMark.LineColor  = System.Drawing.Color.FromArgb(120, 120, 120);
            area.AxisX.IsMarginVisible          = false;
            area.AxisX.Interval                 = 1;
            area.AxisX.LabelStyle.Angle         = xLabelAngle;
            area.AxisX.LabelStyle.Font          = new System.Drawing.Font("Arial", 5f);
            // Y 軸（左）：完全隱藏
            area.AxisY.LineColor                = System.Drawing.Color.Transparent;
            area.AxisY.MajorGrid.Enabled        = false;
            area.AxisY.MajorTickMark.Enabled    = false;
            area.AxisY.MinorTickMark.Enabled    = false;
            area.AxisY.LabelStyle.Enabled       = false;
            area.AxisY.Minimum                  = 0;
            // Y 軸（left/Primary）：格線由此軸驅動，但標籤隱藏
            // 與 chartMura 相同策略：Primary 軸提供格線，Secondary 軸提供右側標籤
            area.AxisY.Interval              = yDefault / 5.0;
            area.AxisY.Maximum               = yDefault;
            area.AxisY.MajorGrid.Enabled     = true;
            area.AxisY.MajorGrid.LineColor   = System.Drawing.Color.FromArgb(220, 220, 220);
            area.AxisY.MajorGrid.LineDashStyle =
                System.Windows.Forms.DataVisualization.Charting.ChartDashStyle.Dot;
            // Y2 軸（right/Secondary）：顯示右側標籤；格線由 AxisY 驅動故此處關閉
            area.AxisY2.Enabled                 = System.Windows.Forms.DataVisualization.Charting.AxisEnabled.True;
            area.AxisY2.MajorGrid.Enabled       = false;
            area.AxisY2.MajorTickMark.Enabled   = true;
            area.AxisY2.MajorTickMark.LineColor = System.Drawing.Color.FromArgb(120, 120, 120);
            area.AxisY2.LabelStyle.Font         = new System.Drawing.Font("Arial", 5f);
            area.AxisY2.Minimum                 = 0;
            area.AxisY2.Maximum                 = yDefault;
            area.AxisY2.Interval                = yDefault / 5.0;
            area.AxisY2.LabelStyle.Interval     = yDefault;
            // 縮小 InnerPlotPosition，左邊界最小化（Y 軸在右側不佔左邊空間）
            area.InnerPlotPosition.Auto     = false;
            area.InnerPlotPosition.X        = 0f;
            area.InnerPlotPosition.Y        = 12f;
            area.InnerPlotPosition.Width    = 93f;
            area.InnerPlotPosition.Height   = 66f;
            chart.ChartAreas.Add(area);

            // Legend 放在圖表內右上角，透明背景
            var legend = new System.Windows.Forms.DataVisualization.Charting.Legend("L");
            legend.IsDockedInsideChartArea  = true;
            legend.DockedToChartArea        = "Main";
            legend.Docking                  = System.Windows.Forms.DataVisualization.Charting.Docking.Top;
            legend.Alignment                = System.Drawing.StringAlignment.Far;
            legend.Font                     = new System.Drawing.Font("Arial", 6.5f);
            legend.BackColor                = System.Drawing.Color.Transparent;
            legend.BorderColor              = System.Drawing.Color.Transparent;
            chart.Legends.Add(legend);

            var sPass = new System.Windows.Forms.DataVisualization.Charting.Series("合格");
            sPass.ChartType  = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.StackedColumn;
            sPass.Color      = System.Drawing.Color.FromArgb(102, 187, 106);
            sPass.ChartArea  = "Main";
            sPass.Legend     = "L";
            sPass.YAxisType  = System.Windows.Forms.DataVisualization.Charting.AxisType.Secondary;
            chart.Series.Add(sPass);

            var sFail = new System.Windows.Forms.DataVisualization.Charting.Series("異常");
            sFail.ChartType  = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.StackedColumn;
            sFail.Color      = System.Drawing.Color.FromArgb(239, 83, 80);
            sFail.ChartArea  = "Main";
            sFail.Legend     = "L";
            sFail.YAxisType  = System.Windows.Forms.DataVisualization.Charting.AxisType.Secondary;
            chart.Series.Add(sFail);

            // 預填 0 值佔位：建立 X category 軸，讓 grid/tick/axis 在空圖時就能渲染
            // FillPeriodChart 載入真實資料前會先 Points.Clear()，不影響顯示
            if (xCount > 0)
            {
                for (int i = 0; i < xCount; i++)
                {
                    sPass.Points.AddXY((xStart + i).ToString(), 0);
                    sFail.Points.AddXY((xStart + i).ToString(), 0);
                }
            }
            // 無標題
        }

        private void UpdatePeriodCharts(DateTime start, DateTime end)
        {
            var byMonth = InspectionStatisticsService.ComputeGroupedByMonthOfYear(_statsDataRootPath, start, end);
            var byDay   = InspectionStatisticsService.ComputeGroupedByDayOfMonth(_statsDataRootPath,  start, end);
            var byHour  = InspectionStatisticsService.ComputeGroupedByHourOfDay(_statsDataRootPath,   start, end);

            FillPeriodChart(chartYearly,  byMonth);  // 月
            FillPeriodChart(chartMonthly, byDay);    // 日
            FillPeriodChart(chartDaily,   byHour);   // 時
        }

        private void FillPeriodChart(
            System.Windows.Forms.DataVisualization.Charting.Chart chart,
            List<PeriodStats> data)
        {
            var sPass = chart.Series["合格"];
            var sFail = chart.Series["異常"];
            sPass.Points.Clear();
            sFail.Points.Clear();
            foreach (var p in data)
            {
                sPass.Points.AddXY(p.Label, p.Pass);
                sFail.Points.AddXY(p.Label, p.Fail);
            }

            // 自動範圍：全域設定為 Auto 或單圖被點擊切換為 auto
            if (_settings.Chart.ScaleMode == ChartScaleMode.Auto || "auto".Equals(chart.Tag))
                ApplyAutoScale(chart, data);
        }

        private static void ApplyAutoScale(
            System.Windows.Forms.DataVisualization.Charting.Chart chart,
            List<PeriodStats> data)
        {
            int maxTotal = 0;
            foreach (var p in data)
                maxTotal = Math.Max(maxTotal, p.Pass + p.Fail);
            int niceMax = Math.Max(5, (int)(Math.Ceiling(maxTotal / 5.0) * 5));
            var area     = chart.ChartAreas["Main"];
            double yMax  = niceMax * 1.05;
            double yStep = niceMax / 5.0;
            area.AxisY.Maximum               = yMax;
            area.AxisY.Interval              = yStep;
            area.AxisY.MajorGrid.Interval    = yStep;
            area.AxisY2.Maximum              = yMax;
            area.AxisY2.Interval             = yStep;
            area.AxisY2.MajorGrid.Interval   = yStep;
            area.AxisY2.LabelStyle.Interval  = niceMax;
        }

        private void ApplyFixedScale(
            System.Windows.Forms.DataVisualization.Charting.Chart chart, int fixedMax)
        {
            chart.Tag = null;
            var area = chart.ChartAreas["Main"];
            double yStep = fixedMax / 5.0;
            area.AxisY.Maximum               = fixedMax;
            area.AxisY.Interval              = yStep;
            area.AxisY.MajorGrid.Interval    = yStep;
            area.AxisY2.Maximum              = fixedMax;
            area.AxisY2.Interval             = yStep;
            area.AxisY2.MajorGrid.Interval   = yStep;
            area.AxisY2.LabelStyle.Interval  = fixedMax;
        }

        private void ApplyChartScaleFromSettings()
        {
            if (_settings.Chart.ScaleMode == ChartScaleMode.Fixed)
            {
                ApplyFixedScale(chartYearly,  _settings.Chart.YearlyYMax);
                ApplyFixedScale(chartMonthly, _settings.Chart.MonthlyYMax);
                ApplyFixedScale(chartDaily,   _settings.Chart.DailyYMax);
            }
            else
            {
                foreach (var chart in new[] { chartYearly, chartMonthly, chartDaily })
                {
                    if (chart.ChartAreas.Count == 0) continue;
                    var sPass = chart.Series["合格"];
                    var sFail = chart.Series["異常"];
                    var data = new List<PeriodStats>();
                    for (int i = 0; i < sPass.Points.Count; i++)
                        data.Add(new PeriodStats { Label = "", Pass = (int)sPass.Points[i].YValues[0], Fail = (int)sFail.Points[i].YValues[0] });
                    ApplyAutoScale(chart, data);
                }
            }
        }

        // ── 圖表導航列（◄ 年/月/日 ►）────────────────────────────────────

        /// <summary>
        /// 將 values 填入 cb（帶 _chartNavUpdating guard 防 cascade），選取最後一筆。
        /// </summary>
        private void RefillChartComboBox(ComboBox cb, List<int> values, int preferred = -1)
        {
            _chartNavUpdating = true;
            cb.Items.Clear();
            foreach (var v in values) cb.Items.Add(v.ToString());
            if (preferred >= 0)
            {
                int idx = values.IndexOf(preferred);
                cb.SelectedIndex = idx >= 0 ? idx : (values.Count > 0 ? values.Count - 1 : -1);
            }
            else
            {
                cb.SelectedIndex = values.Count > 0 ? values.Count - 1 : -1;
            }
            _chartNavUpdating = false;
        }

        /// <summary>資料夾載入後，以 CSV 中實際存在的年份初始化三列導航。</summary>
        private void PopulateChartNavigators() => PopulateChartNavigators(null);

        private void PopulateChartNavigators(DateTime? hintDate)
        {
            _chartYears = GetAvailableYears();
            RefillChartComboBox(cbChartYear, _chartYears, hintDate?.Year ?? -1);
            OnChartYearIndexChanged(hintDate);
        }

        private void OnChartYearIndexChanged() => OnChartYearIndexChanged(null);
        private void OnChartYearIndexChanged(DateTime? hint)
        {
            int idx = cbChartYear.SelectedIndex;
            bool ok = idx >= 0 && idx < _chartYears.Count;

            _chartMonths = ok ? GetAvailableMonths(_chartYears[idx]) : new List<int>();
            RefillChartComboBox(cbChartMonth, _chartMonths, hint?.Month ?? -1);

            if (!ok) return;
            int year = _chartYears[idx];
            FillPeriodChart(chartYearly,
                InspectionStatisticsService.ComputeGroupedByMonthOfYear(_statsDataRootPath,
                    new DateTime(year, 1, 1), new DateTime(year, 12, 31, 23, 59, 59)));

            OnChartMonthIndexChanged(hint);
        }

        private void OnChartMonthIndexChanged() => OnChartMonthIndexChanged(null);
        private void OnChartMonthIndexChanged(DateTime? hint)
        {
            int idx  = cbChartMonth.SelectedIndex;
            int yIdx = cbChartYear.SelectedIndex;
            bool ok  = idx >= 0 && idx < _chartMonths.Count && yIdx >= 0;

            _chartDays = ok ? GetAvailableDays(_chartYears[yIdx], _chartMonths[idx]) : new List<int>();
            RefillChartComboBox(cbChartDay, _chartDays, hint?.Day ?? -1);

            if (!ok) return;
            int year    = _chartYears[yIdx];
            int month   = _chartMonths[idx];
            int lastDay = DateTime.DaysInMonth(year, month);
            FillPeriodChart(chartMonthly,
                InspectionStatisticsService.ComputeGroupedByDayOfMonth(_statsDataRootPath,
                    new DateTime(year, month, 1), new DateTime(year, month, lastDay, 23, 59, 59)));

            OnChartDayIndexChanged();
        }

        private void OnChartDayIndexChanged()
        {
            int dIdx = cbChartDay.SelectedIndex;
            int mIdx = cbChartMonth.SelectedIndex;
            int yIdx = cbChartYear.SelectedIndex;
            bool ok  = dIdx >= 0 && mIdx >= 0 && yIdx >= 0
                    && dIdx < _chartDays.Count && mIdx < _chartMonths.Count && yIdx < _chartYears.Count;

            if (!ok) return;
            int year  = _chartYears[yIdx];
            int month = _chartMonths[mIdx];
            int day   = _chartDays[dIdx];
            FillPeriodChart(chartDaily,
                InspectionStatisticsService.ComputeGroupedByHourOfDay(_statsDataRootPath,
                    new DateTime(year, month, day), new DateTime(year, month, day, 23, 59, 59)));
        }

        // ── Available date/time string helpers (stat 2-combo) ─────────────

        private List<string> GetAvailableDateStrings() =>
            _statAvailableTimes.Select(t => t.ToString("yyyy-MM-dd")).Distinct()
                .OrderByDescending(x => x).ToList();

        private List<string> GetAvailableTimeStrings(string dateStr) =>
            _statAvailableTimes
                .Where(t => t.ToString("yyyy-MM-dd") == dateStr)
                .Select(t => t.ToString("HH:mm:ss"))
                .Distinct().OrderByDescending(x => x).ToList();

        // ── Available values helpers (period charts) ─────────────────────

        private List<int> GetAvailableYears() =>
            _statAvailableTimes.Select(t => t.Year).Distinct().ToList();

        private List<int> GetAvailableMonths(int y) =>
            _statAvailableTimes.Where(t => t.Year == y)
                               .Select(t => t.Month).Distinct().ToList();

        private List<int> GetAvailableDays(int y, int mo) =>
            _statAvailableTimes.Where(t => t.Year == y && t.Month == mo)
                               .Select(t => t.Day).Distinct().ToList();

        private List<int> GetAvailableHours(int y, int mo, int d) =>
            _statAvailableTimes.Where(t => t.Year == y && t.Month == mo && t.Day == d)
                               .Select(t => t.Hour).Distinct().ToList();

        private List<int> GetAvailableMinutes(int y, int mo, int d, int h) =>
            _statAvailableTimes.Where(t => t.Year == y && t.Month == mo && t.Day == d && t.Hour == h)
                               .Select(t => t.Minute).Distinct().ToList();

        private List<int> GetAvailableSeconds(int y, int mo, int d, int h, int mi) =>
            _statAvailableTimes.Where(t => t.Year == y && t.Month == mo && t.Day == d
                                        && t.Hour == h && t.Minute == mi)
                               .Select(t => t.Second).Distinct().ToList();

        // ── TrackBar 滾輪：每格僅移動 1 ──────────────────────────────────
        private void RegisterWheelInterceptors(TrackBar[] bars)
        {
            foreach (var bar in bars)
                _wheelInterceptors.Add(new TrackBarWheelInterceptor(bar));
        }

        /// <summary>
        /// 攔截原生 WM_MOUSEWHEEL：Windows TRACKBAR 每個滾輪 notch 會送出 3 個
        /// TB_LINEUP/TB_LINEDOWN（等同 3 × SmallChange），此攔截器改為每格僅移動 1。
        /// </summary>
        private sealed class TrackBarWheelInterceptor : NativeWindow
        {
            private const int WM_MOUSEWHEEL = 0x020A;
            private readonly TrackBar _bar;

            public TrackBarWheelInterceptor(TrackBar bar)
            {
                _bar = bar;
                AssignHandle(bar.Handle);
                bar.HandleCreated   += (s, e) => AssignHandle(_bar.Handle);
                bar.HandleDestroyed += (s, e) => ReleaseHandle();
            }

            protected override void WndProc(ref Message m)
            {
                if (m.Msg == WM_MOUSEWHEEL)
                {
                    int delta = (short)(((long)m.WParam >> 16) & 0xFFFF);
                    _bar.Value = Math.Max(_bar.Minimum, Math.Min(_bar.Maximum, _bar.Value + Math.Sign(delta)));
                    return; // 跳過原生 3 格行為
                }
                base.WndProc(ref m);
            }
        }

        /// <summary>
        /// 反轉 ComboBox 滾輪方向：上滾 (delta &gt; 0) → SelectedIndex 增加（數值變大）。
        /// 預設 ComboBox 行為是上滾減少 index，此攔截器直接處理後 return，略過原生訊息。
        /// </summary>
        private sealed class ComboBoxWheelReverser : NativeWindow
        {
            private const int WM_MOUSEWHEEL = 0x020A;
            private readonly ComboBox _cb;

            public ComboBoxWheelReverser(ComboBox cb)
            {
                _cb = cb;
                AssignHandle(cb.Handle);
                cb.HandleCreated   += (s, e) => AssignHandle(_cb.Handle);
                cb.HandleDestroyed += (s, e) => ReleaseHandle();
            }

            protected override void WndProc(ref Message m)
            {
                if (m.Msg == WM_MOUSEWHEEL)
                {
                    int delta  = (short)(((long)m.WParam >> 16) & 0xFFFF);
                    int newIdx = _cb.SelectedIndex + Math.Sign(delta); // 正 delta = 上滾 = index++
                    if (newIdx >= 0 && newIdx < _cb.Items.Count)
                        _cb.SelectedIndex = newIdx;
                    return; // 跳過原生行為（原生是上滾 index--）
                }
                base.WndProc(ref m);
            }
        }

        private void SyncCameraParamsFromHardware()
        {
            if (_expBars == null || _lrBars == null) return;

            var cameras = _liveCameraManager.Cameras;
            var acq     = _settings?.Acquisition;
            if (acq == null) return;

            for (int idx = 0; idx < 7; idx++)
            {
                int camId = idx + 1;

                // 依 CameraId 找相機
                Core.Camera.AniloxCamera cam = null;
                for (int k = 0; k < cameras.Count; k++)
                    if (cameras[k].CameraId == camId) { cam = cameras[k]; break; }
                if (cam == null) continue;

                // Sync 曝光
                if (!_dragging.Contains(_expBars[idx]))
                {
                    double hwExp = cam.GetMeasuredExposureUs();
                    if (hwExp > 0)
                    {
                        int clamped = Math.Max(_expBars[idx].Minimum, Math.Min(_expBars[idx].Maximum, (int)hwExp));
                        double diff = Math.Abs(clamped - _expBars[idx].Value) / (double)Math.Max(1, _expBars[idx].Value);
                        if (diff > 0.05)
                        {
                            _syncingFromHw = true;
                            _expBars[idx].Value = clamped;
                            _expNums[idx].Value = clamped;
                            acq.CameraExposureTimeUs[idx] = clamped;
                            _syncingFromHw = false;
                        }
                    }
                }

                // Sync 線掃速率
                if (!_dragging.Contains(_lrBars[idx]))
                {
                    double hwLr = cam.GetLineRateHz();
                    if (hwLr > 0)
                    {
                        int clamped = Math.Max(_lrBars[idx].Minimum, Math.Min(_lrBars[idx].Maximum, (int)hwLr));
                        double diff = Math.Abs(clamped - _lrBars[idx].Value) / (double)Math.Max(1, _lrBars[idx].Value);
                        if (diff > 0.05)
                        {
                            _syncingFromHw = true;
                            _lrBars[idx].Value = clamped;
                            _lrNums[idx].Value = clamped;
                            acq.CameraLineRateHz[idx] = clamped;
                            _syncingFromHw = false;
                        }
                    }
                }
            }
        }
    }
}
