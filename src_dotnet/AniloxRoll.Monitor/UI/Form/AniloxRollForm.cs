using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Managers;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Presenters;
using AniloxRoll.Monitor.UI.Widgets;
using AniloxRoll.Monitor.UI.State;

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

            FormClosed += (_, __) => _liveCameraManager.FreeCameras();

            InitializeRightPanelControls();
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

                btnCameraGrab.Text = _liveCameraManager.IsLiveGrabbing ? "停止抓取" : "開始抓取";
                return;
            }

            _liveCameraManager.ToggleGrab();
            btnCameraGrab.Text = _liveCameraManager.IsLiveGrabbing ? "停止抓取" : "開始抓取";
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
