using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Data;
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
        // --- 核心服務 ---
        private readonly ImageRepository _imageRepository = new ImageRepository();
        private BatchInspectionService _inspectionService;

        // --- UI Helpers ---
        private DateTimeNavigator _dateTimeNavigator;
        private ThumbnailGridPresenter _galleryManager;
        private AniloxRollPresenter _presenter;
        private FormInteractionHelper _interactionHelper;
        private MuraChartHelper _muraChartHelper;
        private LiveCameraManager _liveCameraManager;

        // --- 相機參數控制項陣列（供 SyncFromCamera 存取）---
        private TrackBar[]      _expBars;
        private NumericUpDown[] _expNums;
        private TrackBar[]      _lrBars;
        private NumericUpDown[] _lrNums;
        private TrackBar[]      _htBars;
        private NumericUpDown[] _htNums;

        // --- 拖曳偵測：拖曳中時抑制硬體寫入 ---
        private readonly HashSet<TrackBar> _dragging = new HashSet<TrackBar>();

        // --- Hardware → UI 同步：防止 SyncFromHardware 觸發 ValueChanged 再回寫硬體 ---
        private bool _syncingFromHw = false;

        // --- Telemetry ---
        private LiveTelemetryPresenter _telemetryPresenter;
        private System.Windows.Forms.Timer _telemetryTimer;

        // --- 檢測日誌 ---
        private InspectionLogService _inspectionLogService;
        private string _currentGrabId;

        // --- 統計 ---
        private InspectionStatsPresenter    _statsPresenter;
        private string                      _statsDataRootPath   = string.Empty;
        private SortedSet<DateTime>         _statAvailableTimes  = new SortedSet<DateTime>();
        private List<GrabIdInfo>            _grabIdInfos         = new List<GrabIdInfo>();
        private bool                        _statComboUpdating;

        // --- 資料緩存 ---
        private readonly List<Image> _thumbnailCache = new List<Image>();
        private InspectionSettings _settings;
        private bool _lastReviewProcessedMode = false;


        public AniloxRollForm()
        {
            InitializeComponent();
            InitializeSystem();
        }

        private void InitializeSystem()
        {
            if (_settings == null) _settings = ConfigManager.LoadInspectionSettings();

            _inspectionService = new BatchInspectionService();

            _dateTimeNavigator = new DateTimeNavigator(
                _imageRepository, cbYear, cbMonth, cbDay, cbHour, cbMin, cbSec);

            _galleryManager = new ThumbnailGridPresenter();
            _galleryManager.Initialize(new PictureBox[] {
                pbCam1, pbCam2, pbCam3, pbCam4, pbCam5, pbCam6, pbCam7
            });

            _presenter = new AniloxRollPresenter(
                _imageRepository, _inspectionService, _dateTimeNavigator, _galleryManager);

            _muraChartHelper = new MuraChartHelper(this.chartMura);
            _muraChartHelper.SetOps(_settings.Cam1_Ops);
            _muraChartHelper.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);

            checkBoxEnableImageProcessing.Checked = UserSessionState.GetLastEnableImageProcessing(checkBoxEnableImageProcessing.Checked);

            propertyGrid1.SelectedObject = _settings;
            propertyGrid1.ToolbarVisible = false;
            propertyGrid1.PropertyValueChanged -= _propertyGrid_PropertyValueChanged;
            propertyGrid1.PropertyValueChanged += _propertyGrid_PropertyValueChanged;

            _interactionHelper = new FormInteractionHelper(new FormInteractionContext
            {
                Form = this,
                Canvas = canvasMain,
                ButtonsToLock = new Button[] { btnShowOriginal, btnShowProcessed, btnSelectFolder },
                ThumbnailCache = _thumbnailCache,
                Presenter = _presenter,
                InspectionService = _inspectionService,
                ImageRepository = _imageRepository,
                TimeNavigator = _dateTimeNavigator,
                GalleryManager = _galleryManager,
                MuraChartHelper = _muraChartHelper,
                Settings = _settings,
                StatusLabel = lblPixelInfo,
                CameraPanels = new[] { pbCam1, pbCam2, pbCam3, pbCam4, pbCam5, pbCam6, pbCam7 }
            });

            _inspectionLogService = new InspectionLogService(
                () => _settings?.CaptureRootPath ?? string.Empty,
                UserSessionState.LastGrabIdNum);

            _interactionHelper.ApplySettingsToService();

            _presenter.BusyStateChanged += _interactionHelper.SetUiLoadingState;
            _presenter.LogReported += OnPresenterLogReported;
            _galleryManager.SelectionChanged += _interactionHelper.OnGallerySelectionChanged;

            _dateTimeNavigator.PeriodSelectionChanged += _presenter.UpdatePeriodNavigationState;
            _presenter.PeriodNavigationStateChanged += (canLast, canNext) =>
            {
                btnLastPeriod.Enabled = canLast;
                btnNextPeriod.Enabled = canNext;
            };
            _presenter.UpdatePeriodNavigationState();

            canvasMain.StatusChanged += _interactionHelper.UpdateCanvasInfo;
            canvasMain.EdgeReached += _interactionHelper.NavigateCamera;

            _liveCameraManager = new LiveCameraManager(
                this,
                new[] { panelLiveCam1, panelLiveCam2, panelLiveCam3, panelLiveCam4, panelLiveCam5, panelLiveCam6, panelLiveCam7 },
                panelMainDisplay,
                pixelText => { if (lblPixelInfo != null) lblPixelInfo.Text = pixelText; }
            );
            _liveCameraManager.SetCaptureSettings(_settings);
            _liveCameraManager.OnInspectionResult += OnCameraInspectionResult;

            FormClosed += (_, __) => _liveCameraManager.FreeCameras();

            InitializeRightPanelControls();
            SetupDataTab();
        }


        private void OnPresenterLogReported(string log)
        {
            Debug.WriteLine(log);

            if (lblPixelInfo == null || string.IsNullOrWhiteSpace(log)) return;

            if (InvokeRequired)
            {
                BeginInvoke(new Action<string>(OnPresenterLogReported), log);
                return;
            }

            lblPixelInfo.Text = log.Replace(Environment.NewLine, " | ");
        }

        // ==========================================
        // --- 相機按鈕事件 ---
        // ==========================================

        private void btnCameraGrab_Click(object sender, EventArgs e)
        {
            bool wasGrabbing = _liveCameraManager.IsLiveGrabbing;

            if (!_liveCameraManager.IsAllocated)
            {
                try
                {
                    _liveCameraManager.EnsureAllocatedAndToggleGrab(checkBoxEnableImageProcessing.Checked);
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

            btnCameraGrab.Text = _liveCameraManager.IsLiveGrabbing ? "停止抓取" : "開始抓取";
        }

        /// <summary>
        /// 相機存檔後回呼（MIL 執行緒，非 UI 執行緒）。
        /// EnableAutoCapture=true 且抓取中時才會觸發。
        /// </summary>
        private void OnCameraInspectionResult(int camId, string fileNameNoExt, float meanPeak, float maxPeak)
        {
            if (string.IsNullOrEmpty(_currentGrabId)) return;
            _inspectionLogService?.AppendRecord(
                _currentGrabId,
                fileNameNoExt,
                meanPeak,
                maxPeak,
                _settings.ErrorValueMean,
                _settings.ErrorValueMax);
        }

        private void btnCameraFree_Click(object sender, EventArgs e)
        {
            _liveCameraManager.FreeCameras();
            _telemetryPresenter?.ResetAll();
            btnCameraGrab.Text = "開始抓取";
        }

        private void checkBoxEnableImageProcessing_CheckedChanged(object sender, EventArgs e)
        {
            _liveCameraManager.SetImageProcessingEnabled(checkBoxEnableImageProcessing.Checked);
            UserSessionState.SetLastEnableImageProcessing(checkBoxEnableImageProcessing.Checked);
            UserSessionState.Save();
        }

        private async void _propertyGrid_PropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            _interactionHelper.HandleSettingsChanged();
            _liveCameraManager?.SetCaptureSettings(_settings);
            _muraChartHelper?.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);

            // 任何 Recipe 參數（HessianMaxFactor / ErrorValueMean / ErrorValueMax）變更都觸發重載
            string changedPropertyName = e?.ChangedItem?.PropertyDescriptor?.Name ?? string.Empty;
            bool isRecipeChange =
                string.Equals(changedPropertyName, nameof(InspectionRecipe.HessianMaxFactor), StringComparison.Ordinal) ||
                string.Equals(changedPropertyName, "Hessian Max Factor",                      StringComparison.Ordinal) ||
                string.Equals(changedPropertyName, nameof(InspectionRecipe.ErrorValueMean),   StringComparison.Ordinal) ||
                string.Equals(changedPropertyName, "Error Value Mean",                        StringComparison.Ordinal) ||
                string.Equals(changedPropertyName, nameof(InspectionRecipe.ErrorValueMax),    StringComparison.Ordinal) ||
                string.Equals(changedPropertyName, "Error Value Max",                         StringComparison.Ordinal);

            // 有影像且為配方參數變更 → 重載（始終用 processed 模式，因為配方只影響演算法輸出）
            if (isRecipeChange && _imageRepository.FileCount > 0)
            {
                _lastReviewProcessedMode = true;
                await _presenter.LoadImagesWithPeriodLockAsync(true, _interactionHelper.LoadImages);
                _interactionHelper.RefreshCurrentCanvasResult();
            }
        }

        private void btnSelectFolder_Click(object sender, EventArgs e)
        {
            _interactionHelper.SelectAndLoadFolder();
            _presenter.UpdatePeriodNavigationState();
        }

        private async void btnShowOriginal_Click(object sender, EventArgs e)
        {
            _lastReviewProcessedMode = false;
            await _presenter.LoadImagesWithPeriodLockAsync(false, _interactionHelper.LoadImages);
        }

        private async void btnShowProcessed_Click(object sender, EventArgs e)
        {
            _lastReviewProcessedMode = true;
            await _presenter.LoadImagesWithPeriodLockAsync(true, _interactionHelper.LoadImages);
        }

        private async void btnLastPeriod_Click(object sender, EventArgs e)
            => await _presenter.MovePeriodAsync(-1, _lastReviewProcessedMode, _interactionHelper.LoadImages);

        private async void btnNextPeriod_Click(object sender, EventArgs e)
            => await _presenter.MovePeriodAsync(+1, _lastReviewProcessedMode, _interactionHelper.LoadImages);

        // ==========================================
        // --- 右側面板：初始化 ---
        // ==========================================

        private void InitializeRightPanelControls()
        {
            SetupCameraTab();
            SetupSystemTab();
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
                    ConfigManager.SaveAcquisitionSettings(acq);
                    _liveCameraManager?.SwitchToCamera(camId);
                };
                _expBars[idx].ValueChanged += (s, e) =>
                {
                    if (syncExp || _syncingFromHw) return; syncExp = true;
                    _expNums[idx].Value = _expBars[idx].Value;
                    acq.CameraExposureTimeUs[idx] = _expBars[idx].Value;
                    if (!_dragging.Contains(_expBars[idx]))
                    {
                        _liveCameraManager?.SetExposureForCamera(camId, _expBars[idx].Value);
                        ConfigManager.SaveAcquisitionSettings(acq);
                    }
                    syncExp = false;
                };
                _expNums[idx].ValueChanged += (s, e) =>
                {
                    if (syncExp || _syncingFromHw) return; syncExp = true;
                    int v = (int)_expNums[idx].Value;
                    _expBars[idx].Value = Math.Max(ExpMin, Math.Min(_expBars[idx].Maximum, v));
                    acq.CameraExposureTimeUs[idx] = v;
                    _liveCameraManager?.SetExposureForCamera(camId, v);
                    ConfigManager.SaveAcquisitionSettings(acq);
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
                    ConfigManager.SaveAcquisitionSettings(acq);
                    _liveCameraManager?.SwitchToCamera(camId);
                };
                _lrBars[idx].ValueChanged += (s, e) =>
                {
                    if (syncLr || _syncingFromHw) return; syncLr = true;
                    _lrNums[idx].Value = _lrBars[idx].Value;
                    acq.CameraLineRateHz[idx] = _lrBars[idx].Value;
                    if (!_dragging.Contains(_lrBars[idx]))
                    {
                        _liveCameraManager?.SetLineRateForCamera(camId, _lrBars[idx].Value);
                        ConfigManager.SaveAcquisitionSettings(acq);
                    }
                    UpdateExpMaxAndClampColor(idx, CalcExpMax());
                    syncLr = false;
                };
                _lrNums[idx].ValueChanged += (s, e) =>
                {
                    if (syncLr || _syncingFromHw) return; syncLr = true;
                    int v = (int)_lrNums[idx].Value;
                    _lrBars[idx].Value = Math.Max(LrMin, Math.Min(LrMax, v));
                    acq.CameraLineRateHz[idx] = v;
                    _liveCameraManager?.SetLineRateForCamera(camId, v);
                    ConfigManager.SaveAcquisitionSettings(acq);
                    UpdateExpMaxAndClampColor(idx, CalcExpMax());
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
                    ConfigManager.SaveAcquisitionSettings(acq);
                    _liveCameraManager?.SwitchToCamera(camId);
                };
                _htBars[idx].ValueChanged += (s, e) =>
                {
                    if (syncHt || _syncingFromHw) return; syncHt = true;
                    _htNums[idx].Value = _htBars[idx].Value;
                    acq.CameraGrabHeight[idx] = _htBars[idx].Value;
                    if (!_dragging.Contains(_htBars[idx]))
                    {
                        _liveCameraManager?.SetGrabHeightForCamera(camId, _htBars[idx].Value);
                        ConfigManager.SaveAcquisitionSettings(acq);
                    }
                    syncHt = false;
                };
                _htNums[idx].ValueChanged += (s, e) =>
                {
                    if (syncHt || _syncingFromHw) return; syncHt = true;
                    int v = (int)_htNums[idx].Value;
                    _htBars[idx].Value = Math.Max(HtMin, Math.Min(HtMax, v));
                    acq.CameraGrabHeight[idx] = v;
                    _liveCameraManager?.SetGrabHeightForCamera(camId, v);
                    ConfigManager.SaveAcquisitionSettings(acq);
                    syncHt = false;
                };
            }
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

            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxWidth",          InspectionEngineConfig.MaxWidth.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxHeight",         InspectionEngineConfig.MaxHeight.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxThumbnailSide",  InspectionEngineConfig.MaxThumbnailSide.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultBgSigma",    InspectionEngineConfig.DefaultBgSigma.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultRidgeSigma", InspectionEngineConfig.DefaultRidgeSigma.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultHessianMax", InspectionEngineConfig.DefaultHessianMaxFactor.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultRidgeMode",  InspectionEngineConfig.DefaultRidgeMode }));

            // ── Telemetry Timer（每 500ms 更新 ListView + SyncFromHardware）─
            _telemetryTimer = new System.Windows.Forms.Timer { Interval = 500 };
            _telemetryTimer.Tick += TelemetryTimer_Tick;
            _telemetryTimer.Start();
        }

        // ==========================================
        // --- Telemetry Timer ---
        // ==========================================

        private void TelemetryTimer_Tick(object sender, EventArgs e)
        {
            if (_liveCameraManager == null || _liveCameraManager.IsReleasing) return;

            _telemetryPresenter?.Update(_liveCameraManager.Cameras);

            if (_liveCameraManager.IsAllocated)
                SyncCameraParamsFromHardware();
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
            btnQueryStats.Click       += BtnQueryStats_Click;
            WireStatDateCombos();

            comboBox1.SelectedIndexChanged += (s, e) => OnGrabIdComboChanged(isStart: true);
            comboBox2.SelectedIndexChanged += (s, e) => OnGrabIdComboChanged(isStart: false);
        }

        private void PopulateStatDateCombos(DateTime start, DateTime end)
        {
            // Year
            int curYear = DateTime.Today.Year;
            cbStartYear.Items.Clear(); cbEndYear.Items.Clear();
            for (int y = curYear - 5; y <= curYear + 1; y++)
            {
                cbStartYear.Items.Add(y.ToString());
                cbEndYear.Items.Add(y.ToString());
            }
            cbStartYear.Text = start.Year.ToString();
            cbEndYear.Text   = end.Year.ToString();

            // Month
            cbStartMonth.Items.Clear(); cbEndMonth.Items.Clear();
            for (int m = 1; m <= 12; m++)
            {
                cbStartMonth.Items.Add(m.ToString("D2"));
                cbEndMonth.Items.Add(m.ToString("D2"));
            }
            cbStartMonth.Text = start.Month.ToString("D2");
            cbEndMonth.Text   = end.Month.ToString("D2");

            // Day
            cbStartDay.Items.Clear(); cbEndDay.Items.Clear();
            for (int d = 1; d <= 31; d++)
            {
                cbStartDay.Items.Add(d.ToString("D2"));
                cbEndDay.Items.Add(d.ToString("D2"));
            }
            cbStartDay.Text = start.Day.ToString("D2");
            cbEndDay.Text   = end.Day.ToString("D2");

            // Hour
            cbStartHour.Items.Clear(); cbEndHour.Items.Clear();
            for (int h = 0; h <= 23; h++)
            {
                cbStartHour.Items.Add(h.ToString("D2"));
                cbEndHour.Items.Add(h.ToString("D2"));
            }
            cbStartHour.Text = start.Hour.ToString("D2");
            cbEndHour.Text   = end.Hour.ToString("D2");

            // Min
            cbStartMin.Items.Clear(); cbEndMin.Items.Clear();
            for (int mn = 0; mn <= 59; mn++)
            {
                cbStartMin.Items.Add(mn.ToString("D2"));
                cbEndMin.Items.Add(mn.ToString("D2"));
            }
            cbStartMin.Text = start.Minute.ToString("D2");
            cbEndMin.Text   = end.Minute.ToString("D2");

            // Sec
            cbStartSec.Items.Clear(); cbEndSec.Items.Clear();
            for (int s = 0; s <= 59; s++)
            {
                cbStartSec.Items.Add(s.ToString("D2"));
                cbEndSec.Items.Add(s.ToString("D2"));
            }
            cbStartSec.Text = start.Second.ToString("D2");
            cbEndSec.Text   = end.Second.ToString("D2");
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
                    _grabIdInfos        = InspectionStatisticsService.LoadGrabIdInfos(_statsDataRootPath);

                    // 填充序號 ComboBox
                    _statComboUpdating = true;
                    try
                    {
                        comboBox1.Items.Clear();
                        comboBox2.Items.Clear();
                        foreach (var info in _grabIdInfos)
                        {
                            comboBox1.Items.Add(info.GrabId);
                            comboBox2.Items.Add(info.GrabId);
                        }
                        if (comboBox1.Items.Count > 0)
                        {
                            comboBox1.SelectedIndex = 0;
                            comboBox2.SelectedIndex = comboBox2.Items.Count - 1;
                        }

                        // 時間 ComboBox 設為全範圍
                        if (_statAvailableTimes.Count > 0)
                        {
                            SetCombosToDateTime(true,  _statAvailableTimes.Min);
                            SetCombosToDateTime(false, _statAvailableTimes.Max);
                            DoRefreshCombos(true,  0);
                            DoRefreshCombos(false, 0);
                        }
                    }
                    finally { _statComboUpdating = false; }
                    RefreshStats();
                }
            }
        }

        private void BtnQueryStats_Click(object sender, EventArgs e) => RefreshStats();

        private bool TryParseStatDateTime(out DateTime start, out DateTime end)
        {
            start = end = DateTime.MinValue;
            if (!TryBuildDateTime(cbStartYear, cbStartMonth, cbStartDay,
                                  cbStartHour, cbStartMin, cbStartSec, out start)) return false;
            if (!TryBuildDateTime(cbEndYear,   cbEndMonth,   cbEndDay,
                                  cbEndHour,   cbEndMin,     cbEndSec,   out end))   return false;
            return start <= end;
        }

        private static bool TryBuildDateTime(
            ComboBox year, ComboBox month, ComboBox day,
            ComboBox hour, ComboBox min,  ComboBox sec,
            out DateTime result)
        {
            result = DateTime.MinValue;
            if (!int.TryParse(year.Text,  out int y)) return false;
            if (!int.TryParse(month.Text, out int mo)) return false;
            if (!int.TryParse(day.Text,   out int d))  return false;
            if (!int.TryParse(hour.Text,  out int h))  return false;
            if (!int.TryParse(min.Text,   out int mn)) return false;
            if (!int.TryParse(sec.Text,   out int s))  return false;
            try { result = new DateTime(y, mo, d, h, mn, s); return true; }
            catch { return false; }
        }

        // ==========================================
        // --- 統計 Tab：Cascading ComboBox 邏輯 ---
        // ==========================================

        private void WireStatDateCombos()
        {
            cbStartYear.SelectedIndexChanged  += (s, e) => OnStartComboChanged(1);
            cbStartMonth.SelectedIndexChanged += (s, e) => OnStartComboChanged(2);
            cbStartDay.SelectedIndexChanged   += (s, e) => OnStartComboChanged(3);
            cbStartHour.SelectedIndexChanged  += (s, e) => OnStartComboChanged(4);
            cbStartMin.SelectedIndexChanged   += (s, e) => OnStartComboChanged(5);
            cbStartSec.SelectedIndexChanged   += (s, e) => OnStartComboChanged(6);

            cbEndYear.SelectedIndexChanged    += (s, e) => OnEndComboChanged(1);
            cbEndMonth.SelectedIndexChanged   += (s, e) => OnEndComboChanged(2);
            cbEndDay.SelectedIndexChanged     += (s, e) => OnEndComboChanged(3);
            cbEndHour.SelectedIndexChanged    += (s, e) => OnEndComboChanged(4);
            cbEndMin.SelectedIndexChanged     += (s, e) => OnEndComboChanged(5);
            cbEndSec.SelectedIndexChanged     += (s, e) => OnEndComboChanged(6);
        }

        private void OnStartComboChanged(int fromLevel)
        {
            if (_statComboUpdating) return;
            if (_statAvailableTimes.Count > 0)
            {
                _statComboUpdating = true;
                try { DoRefreshCombos(true, fromLevel); ClampEndToStart(); }
                finally { _statComboUpdating = false; }
            }
            RefreshStats();
        }

        private void OnEndComboChanged(int fromLevel)
        {
            if (_statComboUpdating) return;
            if (_statAvailableTimes.Count > 0)
            {
                _statComboUpdating = true;
                try { DoRefreshCombos(false, fromLevel); ClampStartToEnd(); }
                finally { _statComboUpdating = false; }
            }
            RefreshStats();
        }

        /// <summary>
        /// 從 fromLevel 開始（含）向下重建 isStart 側的 ComboBox Items，
        /// 只保留 _statAvailableTimes 中實際存在的選項。
        /// </summary>
        private void DoRefreshCombos(bool isStart, int fromLevel)
        {
            var cbs = isStart
                ? new[] { cbStartYear, cbStartMonth, cbStartDay, cbStartHour, cbStartMin, cbStartSec }
                : new[] { cbEndYear,   cbEndMonth,   cbEndDay,   cbEndHour,   cbEndMin,   cbEndSec   };

            // 先讀取目前文字（作為偏好值）
            int[] cur = new int[6];
            for (int i = 0; i < 6; i++) int.TryParse(cbs[i].Text, out cur[i]);

            if (fromLevel <= 0) RefillCombo(cbs[0], GetAvailableYears(),                            cur[0], "");
            if (!int.TryParse(cbs[0].Text, out int y))  return;

            if (fromLevel <= 1) RefillCombo(cbs[1], GetAvailableMonths(y),                         cur[1], "D2");
            if (!int.TryParse(cbs[1].Text, out int mo)) return;

            if (fromLevel <= 2) RefillCombo(cbs[2], GetAvailableDays(y, mo),                       cur[2], "D2");
            if (!int.TryParse(cbs[2].Text, out int d))  return;

            if (fromLevel <= 3) RefillCombo(cbs[3], GetAvailableHours(y, mo, d),                   cur[3], "D2");
            if (!int.TryParse(cbs[3].Text, out int h))  return;

            if (fromLevel <= 4) RefillCombo(cbs[4], GetAvailableMinutes(y, mo, d, h),              cur[4], "D2");
            if (!int.TryParse(cbs[4].Text, out int mi)) return;

            if (fromLevel <= 5) RefillCombo(cbs[5], GetAvailableSeconds(y, mo, d, h, mi),          cur[5], "D2");
        }

        private static void RefillCombo(ComboBox cb, List<int> values, int preferred, string fmt)
        {
            string target = preferred > 0 ? preferred.ToString(fmt) : cb.Text;
            cb.Items.Clear();
            foreach (int v in values) cb.Items.Add(v.ToString(fmt));
            int idx = cb.Items.IndexOf(target);
            cb.SelectedIndex = idx >= 0 ? idx : (cb.Items.Count > 0 ? 0 : -1);
        }

        private void SetCombosToDateTime(bool isStart, DateTime dt)
        {
            if (isStart)
            {
                cbStartYear.Text  = dt.Year.ToString();
                cbStartMonth.Text = dt.Month.ToString("D2");
                cbStartDay.Text   = dt.Day.ToString("D2");
                cbStartHour.Text  = dt.Hour.ToString("D2");
                cbStartMin.Text   = dt.Minute.ToString("D2");
                cbStartSec.Text   = dt.Second.ToString("D2");
            }
            else
            {
                cbEndYear.Text  = dt.Year.ToString();
                cbEndMonth.Text = dt.Month.ToString("D2");
                cbEndDay.Text   = dt.Day.ToString("D2");
                cbEndHour.Text  = dt.Hour.ToString("D2");
                cbEndMin.Text   = dt.Minute.ToString("D2");
                cbEndSec.Text   = dt.Second.ToString("D2");
            }
        }

        /// <summary>若 start > end，將 end 推至最近的可用時間 ≥ start。</summary>
        private void ClampEndToStart()
        {
            if (!TryBuildDateTime(cbStartYear, cbStartMonth, cbStartDay,
                                  cbStartHour, cbStartMin, cbStartSec, out DateTime start)) return;
            if (!TryBuildDateTime(cbEndYear, cbEndMonth, cbEndDay,
                                  cbEndHour, cbEndMin, cbEndSec, out DateTime end)) return;
            if (start <= end) return;

            var view = _statAvailableTimes.GetViewBetween(start, DateTime.MaxValue);
            DateTime newEnd = view.Count > 0 ? view.Min : _statAvailableTimes.Max;
            SetCombosToDateTime(false, newEnd);
            DoRefreshCombos(false, 0);
        }

        /// <summary>若 end < start，將 start 推至最近的可用時間 ≤ end。</summary>
        private void ClampStartToEnd()
        {
            if (!TryBuildDateTime(cbStartYear, cbStartMonth, cbStartDay,
                                  cbStartHour, cbStartMin, cbStartSec, out DateTime start)) return;
            if (!TryBuildDateTime(cbEndYear, cbEndMonth, cbEndDay,
                                  cbEndHour, cbEndMin, cbEndSec, out DateTime end)) return;
            if (start <= end) return;

            var view = _statAvailableTimes.GetViewBetween(DateTime.MinValue, end);
            DateTime newStart = view.Count > 0 ? view.Max : _statAvailableTimes.Min;
            SetCombosToDateTime(true, newStart);
            DoRefreshCombos(true, 0);
        }

        /// <summary>
        /// comboBox1（序號起）或 comboBox2（序號迄）變更時：
        /// 強制 start ≤ end、更新 cbStart/cbEnd 時間、重新統計。
        /// </summary>
        private void OnGrabIdComboChanged(bool isStart)
        {
            if (_statComboUpdating || _grabIdInfos.Count == 0) return;

            int idx1 = comboBox1.SelectedIndex;
            int idx2 = comboBox2.SelectedIndex;
            if (idx1 < 0 || idx2 < 0) return;

            // 強制 comboBox1 ≤ comboBox2
            _statComboUpdating = true;
            try
            {
                if (isStart && idx1 > idx2)
                    comboBox2.SelectedIndex = idx1;
                else if (!isStart && idx2 < idx1)
                    comboBox1.SelectedIndex = idx2;

                // 更新 cbStart/cbEnd 時間
                var startInfo = _grabIdInfos[comboBox1.SelectedIndex];
                var endInfo   = _grabIdInfos[comboBox2.SelectedIndex];
                SetCombosToDateTime(true,  startInfo.Earliest);
                SetCombosToDateTime(false, endInfo.Latest);
                if (_statAvailableTimes.Count > 0)
                {
                    DoRefreshCombos(true,  0);
                    DoRefreshCombos(false, 0);
                }
            }
            finally { _statComboUpdating = false; }

            RefreshStats();
        }

        private void RefreshStats()
        {
            if (string.IsNullOrWhiteSpace(_statsDataRootPath)) return;

            // 序號模式（comboBox1/2 已設定）
            if (comboBox1.SelectedIndex >= 0 && comboBox2.SelectedIndex >= 0
                && _grabIdInfos.Count > 0)
            {
                var startInfo = _grabIdInfos[comboBox1.SelectedIndex];
                var endInfo   = _grabIdInfos[comboBox2.SelectedIndex];
                var stats = InspectionStatisticsService.ComputeByGrabIdRange(
                    _statsDataRootPath, startInfo.GrabNum, endInfo.GrabNum);
                _statsPresenter.Update(stats);
                return;
            }

            // 時間模式（fallback）
            if (!TryParseStatDateTime(out DateTime start, out DateTime end)) return;
            var statsTime = InspectionStatisticsService.Compute(_statsDataRootPath, start, end);
            _statsPresenter.Update(statsTime);
        }

        // ── Available values helpers ──────────────────────────────────────

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
