using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
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

        // --- 資料緩存 ---
        private readonly List<Image> _thumbnailCache = new List<Image>();

        // --- 參數設定 (核心) ---
        private InspectionSettings _settings;

        // --- 狀態變數 ---
        private int _currentCameraIndex = 0;
        private int _currentViewLeftX = 0;
        private int _currentViewRightX = 0;

        public AniloxRollForm()
        {
            InitializeComponent();

            InitializeSystem();
        }

        private void InitializeSystem()
        {

            if (_settings == null) _settings = InspectionSettings.LoadFromSettings();
            // 2. 初始化服務
            _inspectionService = new BatchInspectionService();
            // 立即將參數套用到 Service
            ApplySettingsToService();

            _timeSelectionManager = new DateTimeNavigator(
                _imageRepository, cbYear, cbMonth, cbDay, cbHour, cbMin, cbSec);

            _galleryManager = new ThumbnailGridPresenter();
            _galleryManager.Initialize(new PictureBox[] {
                pbCam1, pbCam2, pbCam3, pbCam4, pbCam5, pbCam6, pbCam7
            });

            _presenter = new AniloxRollPresenter(
                _imageRepository,
                _inspectionService,
                _timeSelectionManager,
                _galleryManager
            );

            // 3. 初始化 Chart
            _muraChartHelper = new MuraChartHelper(this.chartMura);
            // 套用第一支相機的 OPS
            _muraChartHelper.SetOps(_settings.Cam1_Ops);

            // 4. 設定 PropertyGrid
            propertyGrid1.SelectedObject = _settings;
            propertyGrid1.ToolbarVisible = false;

            // [關鍵] 先移除事件 (防止重複)，再綁定
            // 並且確保這行是在 _inspectionService 初始化之後
            propertyGrid1.PropertyValueChanged -= _propertyGrid_PropertyValueChanged;
            propertyGrid1.PropertyValueChanged += _propertyGrid_PropertyValueChanged;

            // 5. 初始化 InteractionHelper
            _interactionHelper = new FormInteractionHelper(
                this,
                canvasMain,
                new Button[] { btnShowOriginal, btnShowProcessed, btnSelectFolder },
                _thumbnailCache,
                _presenter,
                _inspectionService,
                _imageRepository,
                _timeSelectionManager,
                _galleryManager,
                _muraChartHelper
            );

            // 6. 綁定事件
            _presenter.BusyStateChanged += _interactionHelper.SetUiLoadingState;
            _presenter.LogReported += log => Console.WriteLine(log);

            _galleryManager.SelectionChanged += (idx) =>
            {
                if (idx >= 0 && idx < 7) _currentCameraIndex = idx;
            };

            _galleryManager.SelectionChanged += _interactionHelper.OnGallerySelectionChanged;

            canvasMain.StatusChanged += OnCanvasStatusChanged;
            canvasMain.EdgeReached += OnCanvasEdgeReached;
        }

        private void OnCanvasStatusChanged(AOI.SDK.UI.CanvasInfo info)
        {
            // 從 Settings 動態取得陣列
            double[] opsArray = _settings.GetOpsArray();
            double[] posArray = _settings.GetPosArray();

            double ops = opsArray[_currentCameraIndex];
            double startPos = posArray[_currentCameraIndex];

            // 計算物理座標
            double physicalX = startPos + (info.ImageX * (ops / 1000.0));

            // 計算視野範圍
            _currentViewLeftX = (int)((0 - info.PanOffset.X) / info.Zoom);
            _currentViewRightX = (int)((canvasMain.Width - info.PanOffset.X) / info.Zoom);

            // 更新狀態列
            lblPixelInfo.Text =
                $"位置:{physicalX:F2} mm | " +
                $"座標: ({info.ImageX}, {info.ImageY}) | " +
                $"亮度: {info.PixelColor.R} | " +
                $"倍率:{info.Zoom:F2}x | " +
                $"平移:({info.PanOffset.X:F0}, {info.PanOffset.Y:F0})";
        }

        private void OnCanvasEdgeReached(int direction)
        {
            int nextIndex = _currentCameraIndex + direction;
            if (nextIndex >= 0 && nextIndex < 7)
            {
                _galleryManager.Select(nextIndex);
            }
        }

        // [關鍵] 參數變更時的邏輯
        private void _propertyGrid_PropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            // 防呆 1: Settings 是否存在
            if (_settings == null) return;

            // 1. 儲存設定
            _settings.SaveToSettings();

            // 2. 更新服務 (內部有檢查 _inspectionService)
            ApplySettingsToService();

            // 3. 更新圖表
            if (_muraChartHelper != null)
            {
                _muraChartHelper.SetOps(_settings.Cam1_Ops);
            }

            // 4. 刷新 UI
            if (canvasMain != null) canvasMain.Invalidate();
        }

        // 抽取出來的共用函式：將設定套用到 Service
        private void ApplySettingsToService()
        {
            // 必須嚴格檢查所有可能為 null 的物件
            if (_inspectionService == null || _settings == null) return;

            _inspectionService.UpdateAlgorithmParams(
                _settings.HessianMaxFactor,
                _settings.ErrorValueMean,
                _settings.ErrorValueMax
            );
        }

        // --- 按鈕事件 ---
        private void btnSelectFolder_Click(object sender, EventArgs e)
            => _interactionHelper.SelectAndLoadFolder();

        private async void btnShowOriginal_Click(object sender, EventArgs e)
            => await _interactionHelper.LoadImages(false);

        private async void btnShowProcessed_Click(object sender, EventArgs e)
            => await _interactionHelper.LoadImages(true);
    }
}