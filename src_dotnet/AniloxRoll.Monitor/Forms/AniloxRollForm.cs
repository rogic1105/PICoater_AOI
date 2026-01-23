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

        // [移除] 狀態變數已移至 FormInteractionHelper
        // private int _currentCameraIndex = 0;
        // private int _currentViewLeftX = 0;
        // private int _currentViewRightX = 0;

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
            // 參數套用邏輯稍後透過 Helper 執行，或在此手動呼叫一次，
            // 但現在 Helper 尚未建立，為了避免依賴順序問題，我們在 Helper 建立後呼叫一次。

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
            _muraChartHelper.SetOps(_settings.Cam1_Ops);

            // 4. 設定 PropertyGrid
            propertyGrid1.SelectedObject = _settings;
            propertyGrid1.ToolbarVisible = false;

            // 先移除事件 (防止重複)
            propertyGrid1.PropertyValueChanged -= _propertyGrid_PropertyValueChanged;
            propertyGrid1.PropertyValueChanged += _propertyGrid_PropertyValueChanged;

            // 5. 初始化 InteractionHelper
            // [關鍵] 傳入 _settings 與 lblPixelInfo (假設其類型為 ToolStripStatusLabel)
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
                _muraChartHelper,
                _settings,      // 新增參數
                lblPixelInfo    // 新增參數
            );

            // [新增] 立即套用參數 (取代原有的 ApplySettingsToService() 呼叫)
            _interactionHelper.ApplySettingsToService();

            // 6. 綁定事件
            _presenter.BusyStateChanged += _interactionHelper.SetUiLoadingState;
            _presenter.LogReported += log => Console.WriteLine(log);

            // [修改] 移除這裡對 _currentCameraIndex 的直接操作，Helper 內部會處理
            // _galleryManager.SelectionChanged += (idx) => ... [移除]

            _galleryManager.SelectionChanged += _interactionHelper.OnGallerySelectionChanged;

            canvasMain.StatusChanged += OnCanvasStatusChanged;
            canvasMain.EdgeReached += OnCanvasEdgeReached;
        }

        // [修改] 委派給 Helper
        private void OnCanvasStatusChanged(AOI.SDK.UI.CanvasInfo info)
        {
            _interactionHelper.UpdateCanvasInfo(info);
        }

        // [修改] 委派給 Helper
        private void OnCanvasEdgeReached(int direction)
        {
            _interactionHelper.NavigateCamera(direction);
        }

        // [修改] 委派給 Helper
        private void _propertyGrid_PropertyValueChanged(object s, PropertyValueChangedEventArgs e)
        {
            _interactionHelper.HandleSettingsChanged();
        }

        // [移除] 此方法已移至 Helper
        // private void ApplySettingsToService() { ... }

        // --- 按鈕事件 ---
        private void btnSelectFolder_Click(object sender, EventArgs e)
            => _interactionHelper.SelectAndLoadFolder();

        private async void btnShowOriginal_Click(object sender, EventArgs e)
            => await _interactionHelper.LoadImages(false);

        private async void btnShowProcessed_Click(object sender, EventArgs e)
            => await _interactionHelper.LoadImages(true);
    }
}