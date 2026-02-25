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
        private DateTimeNavigator _timeSelectionManager;
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

        public AniloxRollForm()
        {
            InitializeComponent();
            InitializeSystem();
        }

        private void InitializeSystem()
        {
            if (_settings == null) _settings = InspectionSettingsStore.Load();

            _inspectionService = new BatchInspectionService();

            _timeSelectionManager = new DateTimeNavigator(
                _imageRepository, cbYear, cbMonth, cbDay, cbHour, cbMin, cbSec);

            _galleryManager = new ThumbnailGridPresenter();
            _galleryManager.Initialize(new PictureBox[] {
                pbCam1, pbCam2, pbCam3, pbCam4, pbCam5, pbCam6, pbCam7
            });

            _presenter = new AniloxRollPresenter(
                _imageRepository, _inspectionService, _timeSelectionManager, _galleryManager);

            _muraChartHelper = new MuraChartHelper(this.chartMura);
            _muraChartHelper.SetOps(_settings.Cam1_Ops);

            checkBoxEnableImageProcessing.Checked = UserSettingsService.LastEnableImageProcessing;

            propertyGrid1.SelectedObject = _settings;
            propertyGrid1.ToolbarVisible = false;
            propertyGrid1.PropertyValueChanged -= _propertyGrid_PropertyValueChanged;
            propertyGrid1.PropertyValueChanged += _propertyGrid_PropertyValueChanged;

            _interactionHelper = new FormInteractionHelper(
                this, canvasMain, new Button[] { btnShowOriginal, btnShowProcessed, btnSelectFolder },
                _thumbnailCache, _presenter, _inspectionService, _imageRepository,
                _timeSelectionManager, _galleryManager, _muraChartHelper, _settings, lblPixelInfo
            );

            _interactionHelper.ApplySettingsToService();

            _presenter.BusyStateChanged += _interactionHelper.SetUiLoadingState;
            _presenter.LogReported += log => Console.WriteLine(log);
            _galleryManager.SelectionChanged += _interactionHelper.OnGallerySelectionChanged;

            cbYear.SelectedIndexChanged += (_, __) => UpdatePeriodNavigationState();
            cbMonth.SelectedIndexChanged += (_, __) => UpdatePeriodNavigationState();
            cbDay.SelectedIndexChanged += (_, __) => UpdatePeriodNavigationState();
            cbHour.SelectedIndexChanged += (_, __) => UpdatePeriodNavigationState();
            cbMin.SelectedIndexChanged += (_, __) => UpdatePeriodNavigationState();
            cbSec.SelectedIndexChanged += (_, __) => UpdatePeriodNavigationState();

            canvasMain.StatusChanged += OnCanvasStatusChanged;
            canvasMain.EdgeReached += OnCanvasEdgeReached;

            // [新增] 初始化 LiveCameraManager
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

            UpdatePeriodNavigationState();
        }

        // ==========================================
        // --- 相機按鈕事件：呼叫 Manager 執行 ---
        // ==========================================
        private void btnCameraAllocation_Click(object sender, EventArgs e)
        {
            try
            {
                _liveCameraManager.AllocateCameras(checkBoxEnableImageProcessing.Checked);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"相機配置失敗: {ex.Message}", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

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

            SetPeriodNavigationBusy(true);

            var periods = _imageRepository.GetAvailablePeriods();
            if (periods.Count == 0)
            {
                SetPeriodNavigationBusy(false);
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
                SetPeriodNavigationBusy(false);
                return;
            }

            SetPeriodToCombo(periods[target]);
            try
            {
                await _interactionHelper.LoadImages(_lastReviewProcessedMode);
            }
            finally
            {
                SetPeriodNavigationBusy(false);
            }
        }

        private async Task LoadImagesWithPeriodLockAsync(bool isProcessedMode)
        {
            SetPeriodNavigationBusy(true);

            try
            {
                await _interactionHelper.LoadImages(isProcessedMode);
            }
            finally
            {
                SetPeriodNavigationBusy(false);
            }
        }

        private void SetPeriodNavigationBusy(bool isBusy)
        {
            _isPeriodNavigationBusy = isBusy;
            UpdatePeriodNavigationState();
        }

        private void UpdatePeriodNavigationState()
        {
            if (_isPeriodNavigationBusy)
            {
                btnLastPeriod.Enabled = false;
                btnNextPeriod.Enabled = false;
                return;
            }

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

            btnLastPeriod.Enabled = idx > 0;
            btnNextPeriod.Enabled = idx < periods.Count - 1;
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

    }
}
