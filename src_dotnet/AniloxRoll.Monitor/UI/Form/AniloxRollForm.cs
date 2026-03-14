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
                new[] { panel1, panel2, panel3, panel4, panel5, panel6, panel7 },
                panel8,
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
            const int ExpMin      =     1;
            const int ExpMax      = 10000;
            const int LrMin       =     1;
            const int LrMax       = 10000;
            const int HtMin       =     1;
            const int HtMax       = 10000;
            const int HtDefault   =  2048;
            const int TickFreq    =  1000;

            // ── 曝光時間 (tabPageExposure) ────────────────────────
            // trackBarExposure / numExposure = CAM1 master → SetExposureForAll
            int expVal = (int)Math.Max(ExpMin, Math.Min(ExpMax, _settings.Acquisition.CameraExposureTimeUs));

            // 設定 CAM1 master 範圍
            trackBarExposure.Minimum       = ExpMin;
            trackBarExposure.Maximum       = ExpMax;
            trackBarExposure.TickFrequency = TickFreq;
            numExposure.Minimum            = ExpMin;
            numExposure.Maximum            = ExpMax;
            trackBarExposure.Value         = expVal;
            numExposure.Value              = expVal;

            // 設定 CAM2–7 個別曝光控制項範圍
            var expBars = new[] { trackBar2, trackBar3, trackBar4, trackBar5, trackBar6, trackBar7 };
            var expNums = new[] { numericUpDown2, numericUpDown3, numericUpDown4,
                                  numericUpDown5, numericUpDown6, numericUpDown7 };
            // CAM ID：trackBar2=CAM2 … trackBar7=CAM7
            var expCamIds = new[] { 2, 3, 4, 5, 6, 7 };
            for (int i = 0; i < expBars.Length; i++)
            {
                expBars[i].Minimum       = ExpMin;
                expBars[i].Maximum       = ExpMax;
                expBars[i].TickFrequency = TickFreq;
                expBars[i].Value         = expVal;
                expNums[i].Minimum       = ExpMin;
                expNums[i].Maximum       = ExpMax;
                expNums[i].Value         = expVal;
            }

            // 繫結事件：CAM1 master（所有相機）
            bool syncingExp = false;
            trackBarExposure.ValueChanged += (s, e) =>
            {
                if (syncingExp) return;
                syncingExp = true;
                numExposure.Value = trackBarExposure.Value;
                _settings.Acquisition.CameraExposureTimeUs = trackBarExposure.Value;
                _liveCameraManager?.SetExposureForAll(trackBarExposure.Value);
                syncingExp = false;
            };
            numExposure.ValueChanged += (s, e) =>
            {
                if (syncingExp) return;
                syncingExp = true;
                int v = (int)numExposure.Value;
                trackBarExposure.Value = Math.Max(ExpMin, Math.Min(ExpMax, v));
                _settings.Acquisition.CameraExposureTimeUs = v;
                _liveCameraManager?.SetExposureForAll(v);
                syncingExp = false;
            };

            // 繫結事件：CAM2–7 個別曝光
            for (int i = 0; i < expBars.Length; i++)
            {
                int camId   = expCamIds[i];
                var bar     = expBars[i];
                var num     = expNums[i];
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
                    bar.Value = Math.Max(ExpMin, Math.Min(ExpMax, v));
                    _liveCameraManager?.SetExposureForCamera(camId, v);
                    syncing = false;
                };
            }

            // ── 線掃速率 (tabPageLineRate) ────────────────────────
            // trackBarGrabHeight / numGrabHeight = CAM1 master → SetLineRateForAll
            // 注意：控制項名稱沿用舊名，實際功能為線掃速率。
            int lrVal = (int)Math.Max(LrMin, Math.Min(LrMax, _settings.Acquisition.CameraLineRateHz));

            trackBarGrabHeight.Minimum       = LrMin;
            trackBarGrabHeight.Maximum       = LrMax;
            trackBarGrabHeight.TickFrequency = TickFreq;
            numGrabHeight.Minimum            = LrMin;
            numGrabHeight.Maximum            = LrMax;
            trackBarGrabHeight.Value         = lrVal;
            numGrabHeight.Value              = lrVal;

            // 設定 CAM2–7 個別線掃速率控制項範圍
            // tabPageLineRate 面板順序：panel15=CAM7, panel16=CAM6, panel17=CAM5,
            //                          panel18=CAM4, panel19=CAM3, panel20=CAM2
            var lrBars   = new[] { trackBar8,  trackBar9,  trackBar10,
                                   trackBar11, trackBar12, trackBar13 };
            var lrNums   = new[] { numericUpDown8,  numericUpDown9,  numericUpDown10,
                                   numericUpDown11, numericUpDown12, numericUpDown13 };
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

            // 繫結事件：CAM1 master 線掃速率（所有相機）
            bool syncingLr = false;
            trackBarGrabHeight.ValueChanged += (s, e) =>
            {
                if (syncingLr) return;
                syncingLr = true;
                numGrabHeight.Value = trackBarGrabHeight.Value;
                _settings.Acquisition.CameraLineRateHz = trackBarGrabHeight.Value;
                _liveCameraManager?.SetLineRateForAll(trackBarGrabHeight.Value);
                syncingLr = false;
            };
            numGrabHeight.ValueChanged += (s, e) =>
            {
                if (syncingLr) return;
                syncingLr = true;
                int v = (int)numGrabHeight.Value;
                trackBarGrabHeight.Value = Math.Max(LrMin, Math.Min(LrMax, v));
                _settings.Acquisition.CameraLineRateHz = v;
                _liveCameraManager?.SetLineRateForAll(v);
                syncingLr = false;
            };

            // 繫結事件：CAM2–7 個別線掃速率
            for (int i = 0; i < lrBars.Length; i++)
            {
                int camId   = lrCamIds[i];
                var bar     = lrBars[i];
                var num     = lrNums[i];
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
            // tabPageGrabHeight 面板順序：panel21=CAM7, panel22=CAM6, panel23=CAM5,
            //                             panel24=CAM4, panel25=CAM3, panel26=CAM2, panel27=CAM1
            // trackBar19 / numericUpDown19 = CAM1 master → SetGrabHeightForAll
            int htVal = Math.Max(HtMin, Math.Min(HtMax, _settings.Acquisition.CameraGrabHeight));
            if (htVal == 0) htVal = HtDefault;

            trackBar19.Minimum       = HtMin;
            trackBar19.Maximum       = HtMax;
            trackBar19.TickFrequency = TickFreq;
            trackBar19.SmallChange   = 64;
            trackBar19.LargeChange   = 512;
            numericUpDown19.Minimum  = HtMin;
            numericUpDown19.Maximum  = HtMax;
            trackBar19.Value         = htVal;
            numericUpDown19.Value    = htVal;

            // 設定 CAM2–7 個別擷取高度控制項範圍
            var htBars   = new[] { trackBar1,  trackBar14, trackBar15,
                                   trackBar16, trackBar17, trackBar18 };
            var htNums   = new[] { numericUpDown1,  numericUpDown14, numericUpDown15,
                                   numericUpDown16, numericUpDown17, numericUpDown18 };
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

            // 繫結事件：CAM1 master 擷取高度（所有相機）
            bool syncingHt = false;
            trackBar19.ValueChanged += (s, e) =>
            {
                if (syncingHt) return;
                syncingHt = true;
                numericUpDown19.Value = trackBar19.Value;
                _settings.Acquisition.CameraGrabHeight = trackBar19.Value;
                _liveCameraManager?.SetGrabHeightForAll(trackBar19.Value);
                syncingHt = false;
            };
            numericUpDown19.ValueChanged += (s, e) =>
            {
                if (syncingHt) return;
                syncingHt = true;
                int v = (int)numericUpDown19.Value;
                trackBar19.Value = Math.Max(HtMin, Math.Min(HtMax, v));
                _settings.Acquisition.CameraGrabHeight = v;
                _liveCameraManager?.SetGrabHeightForAll(v);
                syncingHt = false;
            };

            // 繫結事件：CAM2–7 個別擷取高度
            for (int i = 0; i < htBars.Length; i++)
            {
                int camId   = htCamIds[i];
                var bar     = htBars[i];
                var num     = htNums[i];
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
