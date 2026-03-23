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
        private MuraChartHelper _muraChartLiveHelper;
        private MuraChartHelper _stitchedOverviewHelper;
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

        // --- 檢測日誌 ---
        private InspectionLogService _inspectionLogService;
        private string _currentGrabId;

        // --- 統計 ---
        private InspectionStatsPresenter    _statsPresenter;
        private string                      _statsDataRootPath   = string.Empty;
        private SortedSet<DateTime>         _statAvailableTimes  = new SortedSet<DateTime>();
        private List<GrabIdInfo>            _grabIdInfos         = new List<GrabIdInfo>();
        private bool                        _statComboUpdating;
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

        // --- Grab ID 拼接模式（null = 一般模式）---
        private Bitmap[] _stitchedImages;
        private float[][] _stitchedCurveMean;
        private float[][] _stitchedCurveMax;
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

            _muraChartLiveHelper = new MuraChartHelper(this.muraChartLive);
            _muraChartLiveHelper.SetOps(_settings.Cam1_Ops);
            _muraChartLiveHelper.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);

            _stitchedOverviewHelper = new MuraChartHelper(this.chart1);
            _stitchedOverviewHelper.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            if (chart1.ChartAreas.Count > 0)
                chart1.ChartAreas[0].AxisX.ScaleView.Zoomable = false;

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
                ButtonsToLock    = new Button[] { btnShowOriginal, btnShowProcessed, btnSelectFolder },
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
                ImageFormatLabel = lblImageFormat,
                ImageScaleLabel  = lblImageScale
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
            _presenter.PeriodNavigationStateChanged   += (canLast, canNext) =>
            {
                btnPeriodPrev.Enabled = canLast;
                btnPeriodNext.Enabled = canNext;
            };
            _presenter.UpdatePeriodNavigationState();

            canvasMain.StatusChanged += _interactionHelper.UpdateCanvasInfo;
            canvasMain.EdgeReached   += _interactionHelper.NavigateCamera;
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
            _liveCameraManager.OnLiveCurveData   += OnLiveCurveData;

            FormClosed += (_, __) => _liveCameraManager.FreeCameras();
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
            // 只顯示目前主畫面相機的曲線
            if (camId != _liveCameraManager.SelectedMainCameraId) return;

            if (InvokeRequired)
            {
                BeginInvoke(new Action<int, float[], float[]>(OnLiveCurveData), camId, meanArr, maxArr);
                return;
            }

            if (_muraChartLiveHelper == null || _settings == null) return;

            int cameraIndex = camId - 1;
            double[] startPositions = _settings.GetCameraStartPositionMmArray();
            double startPos = (cameraIndex >= 0 && cameraIndex < startPositions.Length)
                ? startPositions[cameraIndex] : 0;

            // Live 模式顯示完整範圍（viewLeft/viewRight 設 NaN 表示 FitToScreen）
            _muraChartLiveHelper.UpdateDataAndView(meanArr, maxArr,
                startPos, double.NaN, double.NaN);
        }

        private void btnCameraFree_Click(object sender, EventArgs e)
        {
            _liveCameraManager.FreeCameras();
            _telemetryPresenter?.ResetAll();
            UpdateGrabButton(false);
        }

        private void UpdateGrabButton(bool isGrabbing)
        {
            btnCameraGrab.Text = isGrabbing ? "停止抓取" : "開始抓取";
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
            }
        }

        private void checkBoxEnableImageProcessing_CheckedChanged(object sender, EventArgs e)
        {
            _liveCameraManager.SetImageProcessingEnabled(checkBoxEnableImageProcessing.Checked);
            UserSessionState.SetLastEnableImageProcessing(checkBoxEnableImageProcessing.Checked);
            UserSessionState.Save();
        }

        // PropertyGrid 回傳的 ChangedItem.PropertyDescriptor.Name 可能是 MemberName 或 DisplayName 其中之一，
        // 因此兩種形式都放入集合，避免版本差異導致漏判。
        private static readonly HashSet<string> RecipePropertyNames = new HashSet<string>(StringComparer.Ordinal)
        {
            nameof(InspectionRecipe.HessianMaxFactor), "Hessian Max Factor",
            nameof(InspectionRecipe.ErrorValueMean),   "Error Value Mean",
            nameof(InspectionRecipe.ErrorValueMax),    "Error Value Max",
        };

        private async void _propertyGrid_PropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            _interactionHelper.HandleSettingsChanged();
            _liveCameraManager?.SetCaptureSettings(_settings);
            _muraChartHelper?.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            _muraChartLiveHelper?.SetOps(_settings.Cam1_Ops);
            _muraChartLiveHelper?.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);

            // 抓圖進行中設定變更 → 立刻在 CSV 插入 #CFG
            if (_liveCameraManager?.IsLiveGrabbing == true)
                _inspectionLogService?.ForceWriteConfig(CsvConfigSnapshot.FromSettings(_settings));

            string changedPropertyName = e?.ChangedItem?.PropertyDescriptor?.Name ?? string.Empty;
            bool isRecipeChange = RecipePropertyNames.Contains(changedPropertyName);

            // 有影像且為配方參數變更 → 重載（始終用 processed 模式，因為配方只影響演算法輸出）
            if (isRecipeChange && _imageRepository.FileCount > 0)
            {
                _lastReviewProcessedMode = true;
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

            // 同步載入序號清單並填充所有序號 ComboBox（Review + Data）
            if (_imageRepository.FileCount > 0)
            {
                var reviewPath = UserSessionState.LastDataPath;
                if (!string.IsNullOrWhiteSpace(reviewPath))
                {
                    _statsDataRootPath  = reviewPath;
                    _statAvailableTimes = InspectionStatisticsService.LoadAvailableTimes(reviewPath);
                    _grabIdInfos        = InspectionStatisticsService.LoadGrabIdInfos(reviewPath);

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
                            cbGrabIdStart.SelectedIndex = 0;
                            cbGrabIdEnd.SelectedIndex = cbGrabIdEnd.Items.Count - 1;
                        }
                        if (_statAvailableTimes.Count > 0)
                        {
                            SetCombosToDateTime(true,  _statAvailableTimes.Min);
                            SetCombosToDateTime(false, _statAvailableTimes.Max);
                            DoRefreshCombos(true,  0);
                            DoRefreshCombos(false, 0);
                        }
                    }
                    finally { _statComboUpdating = false; }
                    PopulateChartNavigators();
                    RefreshStats();
                }
            }

            ClearStitchedMode();
            SetGroupBoxActive(grpReviewTimePeriod, true);
            SetGroupBoxActive(grpReviewGrabNav, false);
            await _presenter.LoadImagesWithPeriodLockAsync(false, _interactionHelper.LoadImages);
            UpdateOverviewChartFromRepository();
        }

        private async void btnShowOriginal_Click(object sender, EventArgs e)
        {
            if (_stitchedImages != null)
            {
                await ReloadCurrentStitchedView(false);
                return;
            }
            _lastReviewProcessedMode = false;
            ClearStitchedMode();
            await _presenter.LoadImagesWithPeriodLockAsync(false, _interactionHelper.LoadImages);
            UpdateOverviewChartFromRepository();
        }

        private async void btnShowProcessed_Click(object sender, EventArgs e)
        {
            if (_stitchedImages != null)
            {
                await ReloadCurrentStitchedView(true);
                return;
            }
            _lastReviewProcessedMode = true;
            ClearStitchedMode();
            await _presenter.LoadImagesWithPeriodLockAsync(true, _interactionHelper.LoadImages);
            UpdateOverviewChartFromRepository();
        }

        private async Task ReloadCurrentStitchedView(bool enableProcess)
        {
            int idx = cbReviewGrabId.SelectedIndex;
            if (idx < 0 || idx >= _grabIdInfos.Count) return;
            var info = _grabIdInfos[idx];
            await LoadGrabStitchedViewAsync(info.GrabId, info.Earliest, info.Latest, enableProcess);
        }

        private async void btnPeriodPrev_Click(object sender, EventArgs e)
        { ClearStitchedMode(); await _presenter.MovePeriodAsync(-1, _lastReviewProcessedMode, _interactionHelper.LoadImages); UpdateOverviewChartFromRepository(); }

        private async void btnPeriodNext_Click(object sender, EventArgs e)
        { ClearStitchedMode(); await _presenter.MovePeriodAsync(+1, _lastReviewProcessedMode, _interactionHelper.LoadImages); UpdateOverviewChartFromRepository(); }

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

            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxWidth",          InspectionEngineConfig.MaxWidth.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxHeight",         InspectionEngineConfig.MaxHeight.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxThumbnailSide",  InspectionEngineConfig.MaxThumbnailSide.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultBgSigma",    InspectionEngineConfig.DefaultBgSigma.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultRidgeSigma", InspectionEngineConfig.DefaultRidgeSigma.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultHessianMax", InspectionEngineConfig.DefaultHessianMaxFactor.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultRidgeMode",  InspectionEngineConfig.DefaultRidgeMode }));
            AutoFitListViewColumns(listViewEngine);

            // ── Telemetry Timer（每 500ms 更新 ListView + SyncFromHardware）─
            _telemetryTimer = new System.Windows.Forms.Timer { Interval = 500 };
            _telemetryTimer.Tick += TelemetryTimer_Tick;
            _telemetryTimer.Start();
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
            btnShowFail.Click         += BtnShowFail_Click;
            WireStatDateCombos();
            InitGrabDetailListView();
            InitPeriodCharts();
            cbChartYear.SelectedIndexChanged  += (s, e) => { if (!_chartNavUpdating) OnChartYearIndexChanged();  };
            cbChartMonth.SelectedIndexChanged += (s, e) => { if (!_chartNavUpdating) OnChartMonthIndexChanged(); };
            cbChartDay.SelectedIndexChanged   += (s, e) => { if (!_chartNavUpdating) OnChartDayIndexChanged();   };
            // 滾輪上滾 = 數值增加（反轉 ComboBox 預設行為）
            foreach (var cb in new[] {
                cbChartYear, cbChartMonth, cbChartDay,
                cbStartYear, cbStartMonth, cbStartDay, cbStartHour, cbStartMin, cbStartSec,
                cbEndYear,   cbEndMonth,   cbEndDay,   cbEndHour,   cbEndMin,   cbEndSec,
                cbGrabIdStart, cbGrabIdEnd, cbDataGrabId, cbReviewGrabId })
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
                            cbGrabIdStart.SelectedIndex = 0;
                            cbGrabIdEnd.SelectedIndex = cbGrabIdEnd.Items.Count - 1;
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
                    SetActiveStatGroupBox(groupBoxGrabIdRange);
                    PopulateChartNavigators();
                    RefreshStats();
                }
            }
        }


        private bool TryParseStatDateTime(out DateTime start, out DateTime end)
        {
            start = end = DateTime.MinValue;
            if (!TryBuildDateTime(cbStartYear, cbStartMonth, cbStartDay,
                                  cbStartHour, cbStartMin, cbStartSec, out start)) return false;
            if (!TryBuildDateTime(cbEndYear,   cbEndMonth,   cbEndDay,
                                  cbEndHour,   cbEndMin,     cbEndSec,   out end))   return false;
            // 秒級 ComboBox 無毫秒，將 end 推至該秒末尾以涵蓋所有毫秒
            end = end.AddMilliseconds(999);
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
            SetActiveStatGroupBox(groupBoxTimeRange);
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
            SetActiveStatGroupBox(groupBoxTimeRange);
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

            // 強制 cbGrabIdStart ≤ cbGrabIdEnd
            _statComboUpdating = true;
            try
            {
                if (isStart && idx1 > idx2)
                    cbGrabIdEnd.SelectedIndex = idx1;
                else if (!isStart && idx2 < idx1)
                    cbGrabIdStart.SelectedIndex = idx2;

                // 更新 cbStart/cbEnd 時間
                var startInfo = _grabIdInfos[cbGrabIdStart.SelectedIndex];
                var endInfo   = _grabIdInfos[cbGrabIdEnd.SelectedIndex];
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
                if (_statAvailableTimes.Count > 0)
                {
                    DoRefreshCombos(true,  0);
                    DoRefreshCombos(false, 0);
                }
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
                        if (_statAvailableTimes.Count > 0)
                        {
                            DoRefreshCombos(true,  0);
                            DoRefreshCombos(false, 0);
                        }
                    }
                    finally { _statComboUpdating = false; }
                    RefreshStats();
                    SetActiveStatGroupBox(grpDataSingleSheet);
                }
                finally { _syncingGrabIdCross = false; }
            }
        }

        private Task LoadGrabStitchedViewAsync(string grabId, DateTime hintFrom, DateTime hintTo)
            => LoadGrabStitchedViewAsync(grabId, hintFrom, hintTo, false);

        private async Task LoadGrabStitchedViewAsync(string grabId, DateTime hintFrom, DateTime hintTo,
            bool enableProcess)
        {
            string root = !string.IsNullOrWhiteSpace(UserSessionState.LastDataPath)
                          ? UserSessionState.LastDataPath : _statsDataRootPath;
            if (string.IsNullOrWhiteSpace(root)) return;

            _interactionHelper.SetUiLoadingState(true);
            _lastReviewProcessedMode = enableProcess;
            var swTotal = Stopwatch.StartNew();
            try
            {
                long csvMs = 0, stitchMs = 0;
                float[][] newCurveMean = new float[7][];
                float[][] newCurveMax  = new float[7][];
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
                                    Func<string, Bitmap> procLoader = (p) =>
                                    {
                                        var bmp = _inspectionService.ProcessBmpAtScale(p, scale,
                                            out float[] m, out float[] x);
                                        // 曲線在 MergeCurves 統一處理（.bin 已在 ProcessBmpAtScale 存好）
                                        return bmp;
                                    };
                                    imgs[i] = GrabImageStitcher.StitchCamera(paths, scale, procLoader);
                                }
                                else
                                {
                                    // JPEG 路徑（含 _proc.jpg 切換）或 BMP 原圖路徑
                                    Func<string, Bitmap> bmpLoader = _inspectionService != null
                                        ? (Func<string, Bitmap>)(p => _inspectionService.LoadBmpAtScale(p, scale))
                                        : null;
                                    imgs[i] = GrabImageStitcher.StitchCamera(paths, scale, bmpLoader,
                                        useProcessed: enableProcess);
                                }
                                MergeCurves(paths, out newCurveMean[i], out newCurveMax[i]);
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
                _stitchedCurveMean = newCurveMean;
                _stitchedCurveMax  = newCurveMax;
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
            _stitchedCurveMean = null;
            _stitchedCurveMax = null;
            _currentGrabConfig = null;
            // 恢復 chart 為當前設定（stitch mode 可能改用了歷史 #CFG 的 Ops/閾值）
            _muraChartHelper?.SetOps(_settings.Cam1_Ops);
            _muraChartHelper?.SetThresholds(_settings.ErrorValueMean, _settings.ErrorValueMax);
            // 清除全覽圖
            if (_stitchedOverviewHelper != null && chart1.ChartAreas.Count > 0)
            {
                chart1.Series["Mean"].Points.Clear();
                chart1.Series["Max"].Points.Clear();
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

        private void ShowStitchedCameraInCanvas(int idx)
        {
            if (_stitchedImages == null) return;
            var bmp = (idx >= 0 && idx < _stitchedImages.Length) ? _stitchedImages[idx] : null;

            // 設定 scaleFactor 和 cameraIndex，FitToScreen 觸發 StatusChanged 時 mm 換算才正確
            _interactionHelper.SetCanvasScaleAndCamera(
                InspectionEngineConfig.DefaultSaveResizeScale, idx);

            canvasMain.Image = bmp;
            if (bmp != null) canvasMain.FitToScreen();

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
                chart1.Series["Mean"].Points.Clear();
                chart1.Series["Max"].Points.Clear();
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
                curveMean[i] = InspectionEngine.LoadCurveBin(basePath + "_mean.bin");
                curveMax[i]  = InspectionEngine.LoadCurveBin(basePath + "_max.bin");
            }

            UpdateOverviewChart(curveMean, curveMax,
                _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray(),
                _settings.ErrorValueMean, _settings.ErrorValueMax);
        }

        /// <summary>
        /// 將 7 台相機的曲線依機台布局位置合併到 chart1（全覽圖）。
        /// 重疊區域：Mean 取平均、Max 取最大值。
        /// </summary>
        private void UpdateOverviewChart(float[][] allMean, float[][] allMax,
            double[] opsArr, double[] posArr, float errMean, float errMax)
        {
            if (_stitchedOverviewHelper == null || allMean == null) return;

            // 最細 OPS 作為統一格點
            double minOpsUm = double.MaxValue;
            for (int i = 0; i < 7; i++)
                if (opsArr[i] > 0 && opsArr[i] < minOpsUm) minOpsUm = opsArr[i];
            if (minOpsUm <= 0 || minOpsUm == double.MaxValue) minOpsUm = 33.0;
            double gridMm = minOpsUm / 1000.0;

            // 全域範圍
            double globalMin = double.MaxValue, globalMax = double.MinValue;
            for (int i = 0; i < 7; i++)
            {
                var curve = allMean[i];
                if (curve == null || curve.Length == 0) continue;
                double camStart = posArr[i];
                double camEnd   = camStart + curve.Length * (opsArr[i] / 1000.0);
                if (camStart < globalMin) globalMin = camStart;
                if (camEnd   > globalMax) globalMax = camEnd;
            }
            if (globalMin >= globalMax) return;

            int totalLen = (int)Math.Ceiling((globalMax - globalMin) / gridMm);
            if (totalLen <= 0 || totalLen > 2000000) return;

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

                for (int j = 0; j < curveMean.Length; j++)
                {
                    int idx = (int)((camStart + j * camOpsMm - globalMin) / gridMm);
                    if (idx < 0 || idx >= totalLen) continue;

                    mergedMean[idx]   += curveMean[j];
                    overlapCount[idx] += 1;

                    float mv = (curveMax != null && j < curveMax.Length) ? curveMax[j] : 0;
                    if (mv > mergedMax[idx]) mergedMax[idx] = mv;
                }
            }

            // 重疊區域 Mean 取平均
            for (int i = 0; i < totalLen; i++)
                if (overlapCount[i] > 1) mergedMean[i] /= overlapCount[i];

            _stitchedOverviewHelper.SetOps(minOpsUm);
            _stitchedOverviewHelper.SetThresholds(errMean, errMax);
            _stitchedOverviewHelper.UpdateData(mergedMean, mergedMax, globalMin);
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

                var mean = InspectionEngine.LoadCurveBin(basePath + "_mean.bin");
                var max  = InspectionEngine.LoadCurveBin(basePath + "_max.bin");
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

            // 序號模式（cbGrabIdStart/2 已設定）
            if (cbGrabIdStart.SelectedIndex >= 0 && cbGrabIdEnd.SelectedIndex >= 0
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

            // 時間模式（fallback）
            if (!TryParseStatDateTime(out DateTime start, out DateTime end)) return;
            var statsTime = InspectionStatisticsService.Compute(_statsDataRootPath, start, end);
            _statsPresenter.Update(statsTime);
            _currentDetails = new List<GrabDetail>();
            ApplyFailFilter();
        }

        private void InitGrabDetailListView()
        {
            listViewGrabDetail.View          = View.Details;
            listViewGrabDetail.FullRowSelect = true;
            listViewGrabDetail.GridLines     = true;
            listViewGrabDetail.Columns.Clear();
            listViewGrabDetail.Items.Clear();

            listViewGrabDetail.Columns.Add("序號");
            for (int i = 1; i <= 7; i++)
                listViewGrabDetail.Columns.Add($"CAM{i}");
            AutoFitListViewColumns(listViewGrabDetail);
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
            InitOneChart(chartYearly,  yDefault: 60000, xCount: 12, xStart: 1);  // 月份 1-12
            InitOneChart(chartMonthly, yDefault: 2000,  xCount: 31, xStart: 1);  // 日期 1-31
            InitOneChart(chartDaily,   yDefault: 300,   xCount: 24, xStart: 0);  // 小時 0-23
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

        private static void FillPeriodChart(
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

            // 動態計算 Y2 軸：5 等分格線，只顯示 0 和 max 兩個標籤
            // Maximum 比 niceMax 多 5%，讓最頂格線上方留約半個字元的空白
            int maxTotal = 0;
            foreach (var p in data)
                maxTotal = Math.Max(maxTotal, p.Pass + p.Fail);
            int niceMax = Math.Max(5, (int)(Math.Ceiling(maxTotal / 5.0) * 5));
            var area     = chart.ChartAreas["Main"];
            double yMax  = niceMax * 1.05;
            double yStep = niceMax / 5.0;
            // Primary 軸（AxisY）驅動格線：scale 與 Y2 保持同步
            area.AxisY.Maximum               = yMax;
            area.AxisY.Interval              = yStep;
            area.AxisY.MajorGrid.Interval    = yStep;
            // Secondary 軸（AxisY2）顯示右側標籤
            area.AxisY2.Maximum              = yMax;
            area.AxisY2.Interval             = yStep;
            area.AxisY2.MajorGrid.Interval   = yStep;
            area.AxisY2.LabelStyle.Interval  = niceMax;
        }

        // ── 圖表導航列（◄ 年/月/日 ►）────────────────────────────────────

        /// <summary>
        /// 將 values 填入 cb（帶 _chartNavUpdating guard 防 cascade），選取最後一筆。
        /// </summary>
        private void RefillChartComboBox(ComboBox cb, List<int> values)
        {
            _chartNavUpdating = true;
            cb.Items.Clear();
            foreach (var v in values) cb.Items.Add(v.ToString());
            cb.SelectedIndex = values.Count > 0 ? values.Count - 1 : -1;
            _chartNavUpdating = false;
        }

        /// <summary>資料夾載入後，以 CSV 中實際存在的年份初始化三列導航。</summary>
        private void PopulateChartNavigators()
        {
            _chartYears = GetAvailableYears();
            RefillChartComboBox(cbChartYear, _chartYears);
            OnChartYearIndexChanged();
        }

        private void OnChartYearIndexChanged()
        {
            int idx = cbChartYear.SelectedIndex;
            bool ok = idx >= 0 && idx < _chartYears.Count;

            _chartMonths = ok ? GetAvailableMonths(_chartYears[idx]) : new List<int>();
            RefillChartComboBox(cbChartMonth, _chartMonths);

            if (!ok) return;
            int year = _chartYears[idx];
            FillPeriodChart(chartYearly,
                InspectionStatisticsService.ComputeGroupedByMonthOfYear(_statsDataRootPath,
                    new DateTime(year, 1, 1), new DateTime(year, 12, 31, 23, 59, 59)));

            OnChartMonthIndexChanged();
        }

        private void OnChartMonthIndexChanged()
        {
            int idx  = cbChartMonth.SelectedIndex;
            int yIdx = cbChartYear.SelectedIndex;
            bool ok  = idx >= 0 && idx < _chartMonths.Count && yIdx >= 0;

            _chartDays = ok ? GetAvailableDays(_chartYears[yIdx], _chartMonths[idx]) : new List<int>();
            RefillChartComboBox(cbChartDay, _chartDays);

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
