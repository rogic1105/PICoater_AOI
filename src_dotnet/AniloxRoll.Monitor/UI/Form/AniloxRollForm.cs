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
        private bool _isApplyingCameraReinit = false;
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

            string changedPropertyName = e?.ChangedItem?.PropertyDescriptor?.Name;
            bool isHessianMaxFactorChanged =
                string.Equals(changedPropertyName, nameof(InspectionRecipe.HessianMaxFactor), StringComparison.Ordinal) ||
                string.Equals(changedPropertyName, "Hessian Max Factor", StringComparison.Ordinal);

            // 任何設定改變，只要相機已配置就重新初始化，避免舊參數在抓圖中導致當機。
            if (_liveCameraManager != null && _liveCameraManager.IsAllocated && !_isApplyingCameraReinit)
            {
                _isApplyingCameraReinit = true;
                try
                {
                    _liveCameraManager.ReinitializeForAcquisitionSettings(checkBoxEnableImageProcessing.Checked, _settings);
                    btnCameraGrab.Text = _liveCameraManager.IsLiveGrabbing ? "停止抓取" : "開始抓取";
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"重設相機失敗: {ex.Message}", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    _isApplyingCameraReinit = false;
                }
            }
            else
            {
                // 相機尚未配置時，只更新緩存設定供下次配置使用。
                _liveCameraManager?.SetCaptureSettings(_settings);
            }

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
            const int HtDefault =  2048;   // px
            const int TickFreq  =  1000;

            // ── 曝光 CAM2–7 陣列（ApplyExpMax 需在宣告後使用）────────
            var expBarsRef = new[] { trackBarExpCam2, trackBarExpCam3, trackBarExpCam4, trackBarExpCam5, trackBarExpCam6, trackBarExpCam7 };
            var expNumsRef = new[] { numExpCam2, numExpCam3, numExpCam4, numExpCam5, numExpCam6, numExpCam7 };
            var expCamIds  = new[] { 2, 3, 4, 5, 6, 7 };

            // ── 動態計算曝光上限：0.9 / lrHz * 1e6 μs ──────────────
            int CalcExpMax(int lrHz)
            {
                if (lrHz <= 0) return ExpMaxCap;
                int dyn = (int)(900000.0 / lrHz);
                return Math.Max(ExpMin, Math.Min(ExpMaxCap, dyn));
            }

            void ApplyExpMax(int expMax)
            {
                trackBarExpCam1.Maximum = expMax;
                numExpCam1.Maximum      = expMax;
                if (trackBarExpCam1.Value > expMax)
                {
                    trackBarExpCam1.Value = expMax;
                    numExpCam1.Value      = expMax;
                }
                foreach (var b in expBarsRef) b.Maximum = expMax;
                foreach (var n in expNumsRef) { n.Maximum = expMax; if (n.Value > expMax) n.Value = expMax; }
            }

            // ── 曝光時間 (tabPageExposure) ────────────────────────
            int lrInit  = (int)Math.Max(LrMin, Math.Min(LrMax, _settings.Acquisition.CameraLineRateHz));
            int expMax0 = CalcExpMax(lrInit);
            int expVal  = (int)Math.Max(ExpMin, Math.Min(expMax0, _settings.Acquisition.CameraExposureTimeUs));

            trackBarExpCam1.Minimum       = ExpMin;
            trackBarExpCam1.Maximum       = expMax0;
            trackBarExpCam1.TickFrequency = TickFreq;
            numExpCam1.Minimum            = ExpMin;
            numExpCam1.Maximum            = expMax0;
            trackBarExpCam1.Value         = expVal;
            numExpCam1.Value              = expVal;
            for (int i = 0; i < expBarsRef.Length; i++)
            {
                expBarsRef[i].Minimum       = ExpMin;
                expBarsRef[i].Maximum       = expMax0;
                expBarsRef[i].TickFrequency = TickFreq;
                expBarsRef[i].Value         = expVal;
                expNumsRef[i].Minimum       = ExpMin;
                expNumsRef[i].Maximum       = expMax0;
                expNumsRef[i].Value         = expVal;
            }

            bool syncingExp = false;
            trackBarExpCam1.ValueChanged += (s, e) =>
            {
                if (syncingExp) return;
                syncingExp = true;
                numExpCam1.Value = trackBarExpCam1.Value;
                _settings.Acquisition.CameraExposureTimeUs = trackBarExpCam1.Value;
                _liveCameraManager?.SetExposureForAll(trackBarExpCam1.Value);
                syncingExp = false;
            };
            numExpCam1.ValueChanged += (s, e) =>
            {
                if (syncingExp) return;
                syncingExp = true;
                int v = (int)numExpCam1.Value;
                trackBarExpCam1.Value = Math.Max(ExpMin, Math.Min(trackBarExpCam1.Maximum, v));
                _settings.Acquisition.CameraExposureTimeUs = v;
                _liveCameraManager?.SetExposureForAll(v);
                syncingExp = false;
            };

            for (int i = 0; i < expBarsRef.Length; i++)
            {
                int camId    = expCamIds[i];
                var bar      = expBarsRef[i];
                var num      = expNumsRef[i];
                bool syncing = false;
                bar.ValueChanged += (s, e) =>
                {
                    if (syncing) return;
                    syncing = true;
                    num.Value = bar.Value;
                    _liveCameraManager?.SetExposureForCamera(camId, bar.Value);
                    syncing = false;
                };
                num.ValueChanged += (s, e) =>
                {
                    if (syncing) return;
                    syncing = true;
                    int v = (int)num.Value;
                    bar.Value = Math.Max(ExpMin, Math.Min(bar.Maximum, v));
                    _liveCameraManager?.SetExposureForCamera(camId, v);
                    syncing = false;
                };
            }

            // ── 線掃速率 (tabPageLineRate) ────────────────────────
            // min=100 Hz；調整 LR 後動態更新曝光上限（ExpMax = 0.9/LR*1e6 μs）
            int lrVal = lrInit;

            trackBarLrCam1.Minimum       = LrMin;
            trackBarLrCam1.Maximum       = LrMax;
            trackBarLrCam1.TickFrequency = TickFreq;
            numLrCam1.Minimum            = LrMin;
            numLrCam1.Maximum            = LrMax;
            trackBarLrCam1.Value         = lrVal;
            numLrCam1.Value              = lrVal;

            // tabPageLineRate 面板順序：panel15=CAM7 … panel20=CAM2
            var lrBars   = new[] { trackBarLrCam7, trackBarLrCam6, trackBarLrCam5,
                                   trackBarLrCam4, trackBarLrCam3, trackBarLrCam2 };
            var lrNums   = new[] { numLrCam7, numLrCam6, numLrCam5,
                                   numLrCam4, numLrCam3, numLrCam2 };
            var lrCamIds = new[] { 7, 6, 5, 4, 3, 2 };
            for (int i = 0; i < lrBars.Length; i++)
            {
                lrBars[i].Minimum       = LrMin;
                lrBars[i].Maximum       = LrMax;
                lrBars[i].TickFrequency = TickFreq;
                lrBars[i].Value         = lrVal;
                lrNums[i].Minimum       = LrMin;
                lrNums[i].Maximum       = LrMax;
                lrNums[i].Value         = lrVal;
            }

            bool syncingLr = false;
            trackBarLrCam1.ValueChanged += (s, e) =>
            {
                if (syncingLr) return;
                syncingLr = true;
                numLrCam1.Value = trackBarLrCam1.Value;
                _settings.Acquisition.CameraLineRateHz = trackBarLrCam1.Value;
                _liveCameraManager?.SetLineRateForAll(trackBarLrCam1.Value);
                ApplyExpMax(CalcExpMax(trackBarLrCam1.Value));
                syncingLr = false;
            };
            numLrCam1.ValueChanged += (s, e) =>
            {
                if (syncingLr) return;
                syncingLr = true;
                int v = (int)numLrCam1.Value;
                trackBarLrCam1.Value = Math.Max(LrMin, Math.Min(LrMax, v));
                _settings.Acquisition.CameraLineRateHz = v;
                _liveCameraManager?.SetLineRateForAll(v);
                ApplyExpMax(CalcExpMax(v));
                syncingLr = false;
            };

            for (int i = 0; i < lrBars.Length; i++)
            {
                int camId    = lrCamIds[i];
                var bar      = lrBars[i];
                var num      = lrNums[i];
                bool syncing = false;
                bar.ValueChanged += (s, e) =>
                {
                    if (syncing) return;
                    syncing = true;
                    num.Value = bar.Value;
                    _liveCameraManager?.SetLineRateForCamera(camId, bar.Value);
                    syncing = false;
                };
                num.ValueChanged += (s, e) =>
                {
                    if (syncing) return;
                    syncing = true;
                    int v = (int)num.Value;
                    bar.Value = Math.Max(LrMin, Math.Min(LrMax, v));
                    _liveCameraManager?.SetLineRateForCamera(camId, v);
                    syncing = false;
                };
            }

            // ── 擷取高度 (tabPageGrabHeight) ────────────────────────
            // Grab Height 需走 Stop→Free→Realloc→Restart 完整流程，不可 live apply。
            // tabPageGrabHeight 面板順序：panel21=CAM7 … panel27=CAM1
            int htVal = Math.Max(HtMin, Math.Min(HtMax, _settings.Acquisition.CameraGrabHeight));
            if (htVal == 0) htVal = HtDefault;

            trackBarHtCam1.Minimum       = HtMin;
            trackBarHtCam1.Maximum       = HtMax;
            trackBarHtCam1.TickFrequency = TickFreq;
            trackBarHtCam1.SmallChange   = 64;
            trackBarHtCam1.LargeChange   = 512;
            numHtCam1.Minimum            = HtMin;
            numHtCam1.Maximum            = HtMax;
            trackBarHtCam1.Value         = htVal;
            numHtCam1.Value              = htVal;

            var htBars   = new[] { trackBarHtCam7, trackBarHtCam6, trackBarHtCam5,
                                   trackBarHtCam4, trackBarHtCam3, trackBarHtCam2 };
            var htNums   = new[] { numHtCam7, numHtCam6, numHtCam5,
                                   numHtCam4, numHtCam3, numHtCam2 };
            var htCamIds = new[] { 7, 6, 5, 4, 3, 2 };
            for (int i = 0; i < htBars.Length; i++)
            {
                htBars[i].Minimum       = HtMin;
                htBars[i].Maximum       = HtMax;
                htBars[i].TickFrequency = TickFreq;
                htBars[i].SmallChange   = 64;
                htBars[i].LargeChange   = 512;
                htBars[i].Value         = htVal;
                htNums[i].Minimum       = HtMin;
                htNums[i].Maximum       = HtMax;
                htNums[i].Value         = htVal;
            }

            // 拖動結束後重新套用主顯示，確保畫面更新到 panelMainDisplay
            void RefreshOnMouseUp(object s, MouseEventArgs e)
                => _liveCameraManager?.RefreshMainDisplay();

            trackBarHtCam1.MouseUp += RefreshOnMouseUp;
            foreach (var bar in htBars)
                bar.MouseUp += RefreshOnMouseUp;

            bool syncingHt = false;
            trackBarHtCam1.ValueChanged += (s, e) =>
            {
                if (syncingHt) return;
                syncingHt = true;
                numHtCam1.Value = trackBarHtCam1.Value;
                _settings.Acquisition.CameraGrabHeight = trackBarHtCam1.Value;
                _liveCameraManager?.SetGrabHeightForAll(trackBarHtCam1.Value);
                syncingHt = false;
            };
            numHtCam1.ValueChanged += (s, e) =>
            {
                if (syncingHt) return;
                syncingHt = true;
                int v = (int)numHtCam1.Value;
                trackBarHtCam1.Value = Math.Max(HtMin, Math.Min(HtMax, v));
                _settings.Acquisition.CameraGrabHeight = v;
                _liveCameraManager?.SetGrabHeightForAll(v);
                syncingHt = false;
            };

            for (int i = 0; i < htBars.Length; i++)
            {
                int camId    = htCamIds[i];
                var bar      = htBars[i];
                var num      = htNums[i];
                bool syncing = false;
                bar.ValueChanged += (s, e) =>
                {
                    if (syncing) return;
                    syncing = true;
                    num.Value = bar.Value;
                    _liveCameraManager?.SetGrabHeightForCamera(camId, bar.Value);
                    syncing = false;
                };
                num.ValueChanged += (s, e) =>
                {
                    if (syncing) return;
                    syncing = true;
                    int v = (int)num.Value;
                    bar.Value = Math.Max(HtMin, Math.Min(HtMax, v));
                    _liveCameraManager?.SetGrabHeightForCamera(camId, v);
                    syncing = false;
                };
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
