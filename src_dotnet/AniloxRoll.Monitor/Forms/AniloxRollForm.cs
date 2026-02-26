using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.Forms.Helpers;

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
        private bool _isPeriodNavigationBusy = false;
        private int _periodNavigationBusyCount = 0;

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

            checkBoxEnableImageProcessing.Checked = UserSettingsService.LastEnableImageProcessing;

            propertyGrid1.SelectedObject = _settings;
            propertyGrid1.ToolbarVisible = false;
            propertyGrid1.PropertyValueChanged -= _propertyGrid_PropertyValueChanged;
            propertyGrid1.PropertyValueChanged += _propertyGrid_PropertyValueChanged;

            // [UI 互動模組] 封裝按鈕事件、畫面切換、縮圖選取與主畫布資訊更新等表單互動流程。
            _interactionHelper = new FormInteractionHelper(
                this, canvasMain, new Button[] { btnShowOriginal, btnShowProcessed, btnSelectFolder },
                _thumbnailCache, _presenter, _inspectionService, _imageRepository,
                _dateTimeNavigator, _galleryManager, _muraChartHelper, _settings, lblPixelInfo
            );

            _interactionHelper.ApplySettingsToService();

            _presenter.BusyStateChanged += _interactionHelper.SetUiLoadingState;
            _presenter.LogReported += log => Console.WriteLine(log);
            _galleryManager.SelectionChanged += _interactionHelper.OnGallerySelectionChanged;

            BindPeriodEvents();
            UpdatePeriodNavigationState();

            canvasMain.StatusChanged += OnCanvasStatusChanged;
            canvasMain.EdgeReached += OnCanvasEdgeReached;

            // [Acquisition 模組] 管理 MIL 相機硬體生命週期（配置、連續抓圖、釋放）與即時畫面輸出。
            _liveCameraManager = new LiveCameraManager(
                this,
                panel1,
                panel2,
                panel3,
                panel4,
                panel5,
                panel6,
                panel7,
                panel8,
                pixelText => { if (lblPixelInfo != null) lblPixelInfo.Text = pixelText; }
            );
            _liveCameraManager.SetCaptureSettings(_settings);

            // 關閉視窗時確保釋放硬體
            FormClosed += (_, __) => _liveCameraManager.FreeCameras();
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
                    _liveCameraManager.AllocateCameras(checkBoxEnableImageProcessing.Checked);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"相機配置失敗: {ex.Message}", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
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
            UserSettingsService.SetLastEnableImageProcessing(checkBoxEnableImageProcessing.Checked);
            UserSettingsService.Save();
        }

        // ==========================================
        // --- 原本的委派事件 ---
        // ==========================================
        private void OnCanvasStatusChanged(AOI.SDK.UI.CanvasInfo info)
            => _interactionHelper.UpdateCanvasInfo(info);

        private void OnCanvasEdgeReached(int direction)
            => _interactionHelper.NavigateCamera(direction);

        private void _propertyGrid_PropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            _interactionHelper.HandleSettingsChanged();
            _liveCameraManager?.SetCaptureSettings(_settings);

            bool isCameraAcqParam =
                e?.ChangedItem?.PropertyDescriptor?.Name == nameof(InspectionSettings.CameraGrabHeight) ||
                e?.ChangedItem?.PropertyDescriptor?.Name == nameof(InspectionSettings.CameraExposureTimeUs);

            if (isCameraAcqParam && _liveCameraManager != null && _liveCameraManager.IsAllocated && !_isApplyingCameraReinit)
            {
                _isApplyingCameraReinit = true;
                try
                {
                    bool wasLive = _liveCameraManager.IsLiveGrabbing;
                    if (wasLive)
                    {
                        _liveCameraManager.StopGrab();
                    }

                    _liveCameraManager.FreeCameras();
                    btnCameraGrab.Text = "開始抓取";

                    _liveCameraManager.AllocateCameras(checkBoxEnableImageProcessing.Checked);
                    _liveCameraManager.SetCaptureSettings(_settings);

                    if (wasLive)
                    {
                        _liveCameraManager.StartGrab();
                        btnCameraGrab.Text = "停止抓取";
                    }
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
        }

        private void btnSelectFolder_Click(object sender, EventArgs e)
        {
            _interactionHelper.SelectAndLoadFolder();
            UpdatePeriodNavigationState();
        }

        private async void btnShowOriginal_Click(object sender, EventArgs e)
        {
            if (_isPeriodNavigationBusy) return;

            _lastReviewProcessedMode = false;
            await LoadImagesWithPeriodLockAsync(false);
        }

        private async void btnShowProcessed_Click(object sender, EventArgs e)
        {
            if (_isPeriodNavigationBusy) return;

            _lastReviewProcessedMode = true;
            await LoadImagesWithPeriodLockAsync(true);
        }

        private async void btnLastPeriod_Click(object sender, EventArgs e)
            => await MovePeriodAsync(-1);

        private async void btnNextPeriod_Click(object sender, EventArgs e)
            => await MovePeriodAsync(+1);

        private async Task MovePeriodAsync(int step)
        {
            if (_isPeriodNavigationBusy) return;

            var periods = _imageRepository.GetAvailablePeriods();
            if (periods.Count == 0)
            {
                UpdatePeriodNavigationState();
                return;
            }

            DateTime current = GetCurrentPeriodOrDefault(periods[0]);
            int idx = periods.FindIndex(x => x == current);
            if (idx < 0)
            {
                idx = periods.FindLastIndex(x => x <= current);
                if (idx < 0) idx = 0;
            }

            int target = Math.Max(0, Math.Min(periods.Count - 1, idx + step));
            if (target == idx)
            {
                UpdatePeriodNavigationState();
                return;
            }

            SetPeriodToCombo(periods[target]);
            await LoadImagesWithPeriodLockAsync(_lastReviewProcessedMode);
        }

        private DateTime GetCurrentPeriodOrDefault(DateTime fallback)
        {
            if (int.TryParse(cbYear.Text, out int y) && int.TryParse(cbMonth.Text, out int m) && int.TryParse(cbDay.Text, out int d) &&
                int.TryParse(cbHour.Text, out int h) && int.TryParse(cbMin.Text, out int min) && int.TryParse(cbSec.Text, out int s))
            {
                try { return new DateTime(y, m, d, h, min, s); }
                catch { }
            }
            return fallback;
        }

        private void SetPeriodToCombo(DateTime dt)
        {
            cbYear.Text = dt.ToString("yyyy");
            cbMonth.Text = dt.ToString("MM");
            cbDay.Text = dt.ToString("dd");
            cbHour.Text = dt.ToString("HH");
            cbMin.Text = dt.ToString("mm");
            cbSec.Text = dt.ToString("ss");
        }

        private void BindPeriodEvents()
        {
            cbYear.SelectedIndexChanged += OnPeriodSelectionChanged;
            cbMonth.SelectedIndexChanged += OnPeriodSelectionChanged;
            cbDay.SelectedIndexChanged += OnPeriodSelectionChanged;
            cbHour.SelectedIndexChanged += OnPeriodSelectionChanged;
            cbMin.SelectedIndexChanged += OnPeriodSelectionChanged;
            cbSec.SelectedIndexChanged += OnPeriodSelectionChanged;
        }

        private void OnPeriodSelectionChanged(object sender, EventArgs e)
            => UpdatePeriodNavigationState();

        private async Task LoadImagesWithPeriodLockAsync(bool isProcessedMode)
        {
            BeginPeriodNavigationBusy();
            try
            {
                await _interactionHelper.LoadImages(isProcessedMode);
            }
            finally
            {
                EndPeriodNavigationBusy();
            }
        }

        private void BeginPeriodNavigationBusy()
        {
            _periodNavigationBusyCount++;
            _isPeriodNavigationBusy = _periodNavigationBusyCount > 0;
            UpdatePeriodNavigationState();
        }

        private void EndPeriodNavigationBusy()
        {
            _periodNavigationBusyCount = Math.Max(0, _periodNavigationBusyCount - 1);
            _isPeriodNavigationBusy = _periodNavigationBusyCount > 0;
            UpdatePeriodNavigationState();
        }

        private void UpdatePeriodNavigationState()
        {
            var periods = _imageRepository.GetAvailablePeriods();
            if (periods.Count == 0)
            {
                btnLastPeriod.Enabled = false;
                btnNextPeriod.Enabled = false;
                return;
            }

            DateTime current = GetCurrentPeriodOrDefault(periods[0]);
            int idx = periods.FindIndex(x => x == current);
            if (idx < 0)
            {
                idx = periods.FindLastIndex(x => x <= current);
                if (idx < 0) idx = 0;
            }

            bool canOperate = !_isPeriodNavigationBusy;
            btnLastPeriod.Enabled = canOperate && idx > 0;
            btnNextPeriod.Enabled = canOperate && idx < periods.Count - 1;
        }

    }
}
