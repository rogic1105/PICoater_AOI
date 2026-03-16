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
        private LiveCameraManager _liveCameraManager; // [新增] 負責管理所有相機硬體與畫面顯示

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
            // [Settings 模組] 載入檢測參數與相機擷取設定（供 PropertyGrid、流程與相機初始化共用）。
            if (_settings == null) _settings = ConfigManager.LoadInspectionSettings();

            // [ImageProcessing 模組] 建立批次檢測服務，負責縮圖/全尺寸檢測流程與演算法參數套用。
            _inspectionService = new BatchInspectionService();

            // [ImageCatalog 模組] 管理時間條件（年/月/日/時/分/秒）與影像索引查詢。
            _dateTimeNavigator = new DateTimeNavigator(
                _imageRepository, cbYear, cbMonth, cbDay, cbHour, cbMin, cbSec);

            // [UI 顯示模組] 管理多相機縮圖牆的初始化、選取狀態與同步更新。
            _galleryManager = new ThumbnailGridPresenter();
            _galleryManager.Initialize(new PictureBox[] {
                pbCam1, pbCam2, pbCam3, pbCam4, pbCam5, pbCam6, pbCam7
            });

            // [Workflow 協調模組] 串接資料存取、檢測服務與縮圖選取，提供表單層統一操作入口。
            _presenter = new AniloxRollPresenter(
                _imageRepository, _inspectionService, _dateTimeNavigator, _galleryManager);

            // [視覺化模組] 管理 Mura 曲線圖顯示與 Ops 套用。
            _muraChartHelper = new MuraChartHelper(this.chartMura);
            _muraChartHelper.SetOps(_settings.Cam1_Ops);

            checkBoxEnableImageProcessing.Checked = UserSessionState.GetLastEnableImageProcessing(checkBoxEnableImageProcessing.Checked);

            propertyGrid1.SelectedObject = _settings;
            propertyGrid1.ToolbarVisible = false;
            propertyGrid1.PropertyValueChanged -= _propertyGrid_PropertyValueChanged;
            propertyGrid1.PropertyValueChanged += _propertyGrid_PropertyValueChanged;

            // [UI 互動模組] 封裝按鈕事件、畫面切換、縮圖選取與主畫布資訊更新等表單互動流程。
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

            // [Acquisition 模組] 管理 MIL 相機硬體生命週期（配置、連續抓圖、釋放）與即時畫面輸出。
            _liveCameraManager = new LiveCameraManager(
                this,
                new[] { panelLiveCam1, panelLiveCam2, panelLiveCam3, panelLiveCam4, panelLiveCam5, panelLiveCam6, panelLiveCam7 },
                panelMainDisplay,
                pixelText => { if (lblPixelInfo != null) lblPixelInfo.Text = pixelText; }
            );
            _liveCameraManager.SetCaptureSettings(_settings);

            // 關閉視窗時確保釋放硬體
            FormClosed += (_, __) => _liveCameraManager.FreeCameras();

            // [右側面板] 初始化相機 TrackBar 與系統 ListView
            InitializeRightPanelControls();
        }


        private void OnPresenterLogReported(string log)
        {
            Debug.WriteLine(log);

            if (lblPixelInfo == null || string.IsNullOrWhiteSpace(log))
            {
                return;
            }

            if (InvokeRequired)
            {
                BeginInvoke(new Action<string>(OnPresenterLogReported), log);
                return;
            }

            lblPixelInfo.Text = log.Replace(Environment.NewLine, " | ");
        }

        // ==========================================
        // --- 相機按鈕事件：呼叫 Manager 執行 ---
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

            string changedPropertyName = e?.ChangedItem?.PropertyDescriptor?.Name;
            bool isHessianMaxFactorChanged =
                string.Equals(changedPropertyName, nameof(InspectionRecipe.HessianMaxFactor), StringComparison.Ordinal) ||
                string.Equals(changedPropertyName, "Hessian Max Factor", StringComparison.Ordinal);

            if (isHessianMaxFactorChanged)
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

            // ── 7 台相機 TrackBar 陣列，索引 0 = CAM1 ─────────────────
            var acq     = _settings.Acquisition;
            var expBars = new[] { trackBarExpCam1, trackBarExpCam2, trackBarExpCam3, trackBarExpCam4, trackBarExpCam5, trackBarExpCam6, trackBarExpCam7 };
            var expNums = new[] { numExpCam1,      numExpCam2,      numExpCam3,      numExpCam4,      numExpCam5,      numExpCam6,      numExpCam7      };
            var lrBars  = new[] { trackBarLrCam1,  trackBarLrCam2,  trackBarLrCam3,  trackBarLrCam4,  trackBarLrCam5,  trackBarLrCam6,  trackBarLrCam7  };
            var lrNums  = new[] { numLrCam1,       numLrCam2,       numLrCam3,       numLrCam4,       numLrCam5,       numLrCam6,       numLrCam7       };
            var htBars  = new[] { trackBarHtCam1,  trackBarHtCam2,  trackBarHtCam3,  trackBarHtCam4,  trackBarHtCam5,  trackBarHtCam6,  trackBarHtCam7  };
            var htNums  = new[] { numHtCam1,       numHtCam2,       numHtCam3,       numHtCam4,       numHtCam5,       numHtCam6,       numHtCam7       };

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
                expBars[idx].Minimum = ExpMin; expBars[idx].Maximum = expMax; expBars[idx].TickFrequency = TickFreq;
                expNums[idx].Minimum = ExpMin; expNums[idx].Maximum = expMax;
                expBars[idx].Value   = expVal; expNums[idx].Value   = expVal;

                bool syncExp = false;
                expBars[idx].ValueChanged += (s, e) =>
                {
                    if (syncExp) return; syncExp = true;
                    expNums[idx].Value = expBars[idx].Value;
                    acq.CameraExposureTimeUs[idx] = expBars[idx].Value;
                    _liveCameraManager?.SetExposureForCamera(camId, expBars[idx].Value);
                    ConfigManager.SaveAcquisitionSettings(acq);
                    syncExp = false;
                };
                expNums[idx].ValueChanged += (s, e) =>
                {
                    if (syncExp) return; syncExp = true;
                    int v = (int)expNums[idx].Value;
                    expBars[idx].Value = Math.Max(ExpMin, Math.Min(expBars[idx].Maximum, v));
                    acq.CameraExposureTimeUs[idx] = v;
                    _liveCameraManager?.SetExposureForCamera(camId, v);
                    ConfigManager.SaveAcquisitionSettings(acq);
                    syncExp = false;
                };
                expBars[idx].MouseUp += (s, e) => _liveCameraManager?.SwitchToCamera(camId);

                // ── 線掃速率 ────────────────────────────────────────────
                int lrVal = (int)Math.Max(LrMin, Math.Min(LrMax, acq.CameraLineRateHz[idx]));
                lrBars[idx].Minimum = LrMin; lrBars[idx].Maximum = LrMax; lrBars[idx].TickFrequency = TickFreq;
                lrNums[idx].Minimum = LrMin; lrNums[idx].Maximum = LrMax;
                lrBars[idx].Value   = lrVal; lrNums[idx].Value   = lrVal;

                bool syncLr = false;
                lrBars[idx].ValueChanged += (s, e) =>
                {
                    if (syncLr) return; syncLr = true;
                    lrNums[idx].Value = lrBars[idx].Value;
                    acq.CameraLineRateHz[idx] = lrBars[idx].Value;
                    _liveCameraManager?.SetLineRateForCamera(camId, lrBars[idx].Value);
                    int newMax = CalcExpMax();
                    expBars[idx].Maximum = newMax; expNums[idx].Maximum = newMax;
                    if (expBars[idx].Value > newMax) { expBars[idx].Value = newMax; expNums[idx].Value = newMax; }
                    ConfigManager.SaveAcquisitionSettings(acq);
                    syncLr = false;
                };
                lrNums[idx].ValueChanged += (s, e) =>
                {
                    if (syncLr) return; syncLr = true;
                    int v = (int)lrNums[idx].Value;
                    lrBars[idx].Value = Math.Max(LrMin, Math.Min(LrMax, v));
                    acq.CameraLineRateHz[idx] = v;
                    _liveCameraManager?.SetLineRateForCamera(camId, v);
                    int newMax = CalcExpMax();
                    expBars[idx].Maximum = newMax; expNums[idx].Maximum = newMax;
                    if (expBars[idx].Value > newMax) { expBars[idx].Value = newMax; expNums[idx].Value = newMax; }
                    ConfigManager.SaveAcquisitionSettings(acq);
                    syncLr = false;
                };
                lrBars[idx].MouseUp += (s, e) => _liveCameraManager?.SwitchToCamera(camId);

                // ── 擷取高度 ────────────────────────────────────────────
                int htVal = Math.Max(HtMin, Math.Min(HtMax, acq.CameraGrabHeight[idx]));
                htBars[idx].Minimum = HtMin; htBars[idx].Maximum = HtMax; htBars[idx].TickFrequency = TickFreq;
                htBars[idx].SmallChange = 64; htBars[idx].LargeChange = 512;
                htNums[idx].Minimum = HtMin; htNums[idx].Maximum = HtMax;
                htBars[idx].Value   = htVal; htNums[idx].Value   = htVal;

                bool syncHt = false;
                htBars[idx].ValueChanged += (s, e) =>
                {
                    if (syncHt) return; syncHt = true;
                    htNums[idx].Value = htBars[idx].Value;
                    acq.CameraGrabHeight[idx] = htBars[idx].Value;
                    _liveCameraManager?.SetGrabHeightForCamera(camId, htBars[idx].Value);
                    ConfigManager.SaveAcquisitionSettings(acq);
                    syncHt = false;
                };
                htNums[idx].ValueChanged += (s, e) =>
                {
                    if (syncHt) return; syncHt = true;
                    int v = (int)htNums[idx].Value;
                    htBars[idx].Value = Math.Max(HtMin, Math.Min(HtMax, v));
                    acq.CameraGrabHeight[idx] = v;
                    _liveCameraManager?.SetGrabHeightForCamera(camId, v);
                    ConfigManager.SaveAcquisitionSettings(acq);
                    syncHt = false;
                };
                htBars[idx].MouseUp += (s, e) => _liveCameraManager?.SwitchToCamera(camId);
            }
        }

        private void SetupSystemTab()
        {
            // ── 相機硬體設定 ──────────────────────────────
            listViewCameras.Columns.Add("Cam",      38);
            listViewCameras.Columns.Add("System",   80);
            listViewCameras.Columns.Add("Sys#",     38);
            listViewCameras.Columns.Add("Dev#",     38);
            listViewCameras.Columns.Add("DCF Path", 200);

            var sysSettings = SystemSettings.CreateDefault();
            foreach (var cam in sysSettings.CameraDevices)
            {
                var item = new ListViewItem(cam.Id.ToString());
                item.SubItems.Add(cam.SystemDescriptor ?? "");
                item.SubItems.Add(cam.SystemNum.ToString());
                item.SubItems.Add(cam.DevNum.ToString());
                item.SubItems.Add(cam.DcfPath ?? "");
                listViewCameras.Items.Add(item);
            }

            // ── 影像引擎常數 ──────────────────────────────
            listViewEngine.Columns.Add("參數", 160);
            listViewEngine.Columns.Add("值",    90);

            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxWidth",          InspectionEngineConfig.MaxWidth.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxHeight",         InspectionEngineConfig.MaxHeight.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxThumbnailSide",  InspectionEngineConfig.MaxThumbnailSide.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultBgSigma",    InspectionEngineConfig.DefaultBgSigma.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultRidgeSigma", InspectionEngineConfig.DefaultRidgeSigma.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultHessianMax", InspectionEngineConfig.DefaultHessianMaxFactor.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultRidgeMode",  InspectionEngineConfig.DefaultRidgeMode }));
        }

    }
}
