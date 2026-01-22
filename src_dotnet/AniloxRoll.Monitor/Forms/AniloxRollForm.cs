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
        // 核心服務
        private readonly ImageRepository _imageRepository = new ImageRepository();
        private BatchInspectionService _inspectionService;

        // UI Helpers
        private DateTimeNavigator _timeSelectionManager;
        private ThumbnailGridPresenter _galleryManager;
        private AniloxRollPresenter _presenter;
        private FormInteractionHelper _interactionHelper;

        // 資源管理 (傳給 Helper 管理)
        private readonly List<Image> _thumbnailCache = new List<Image>();

        public AniloxRollForm()
        {
            InitializeComponent();
            InitializeSystem();
        }

        private void InitializeSystem()
        {
            // 1. 建立 Service
            _inspectionService = new BatchInspectionService();

            // 2. 建立 Presenter 與 Navigator
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

            // 3. 建立 InteractionHelper (傳入 Form 上需要被控制的元件)
            _interactionHelper = new FormInteractionHelper(
                this,
                canvasMain,
                new Button[] { btnShowOriginal, btnShowProcessed, btnSelectFolder },
                _thumbnailCache,
                _presenter,
                _inspectionService,
                // [新增] 傳入這三個依賴
                _imageRepository,
                _timeSelectionManager,
                _galleryManager
            );

            // 4. 綁定事件
            _presenter.BusyStateChanged += _interactionHelper.SetUiLoadingState;
            _presenter.LogReported += log => Console.WriteLine(log);
            _galleryManager.SelectionChanged += _interactionHelper.OnGallerySelectionChanged;

            // 1. 滑鼠移動 (PixelHovered) -> 更新變數 -> 呼叫統一更新
            canvasMain.StatusChanged += (info) =>
            {
                lblPixelInfo.Text =
                    $"座標: ({info.ImageX}, {info.ImageY}) | " +
                    $"亮度: {info.PixelColor.R} | " +
                    $"倍率: {info.Zoom:F5}x | " +
                    $"平移: ({info.PanOffset.X:F0}, {info.PanOffset.Y:F0})";
            };
        }


        // =========================================================
        // button
        // =========================================================

        private void btnSelectFolder_Click(object sender, EventArgs e)
                    => _interactionHelper.SelectAndLoadFolder();

        private async void btnShowOriginal_Click(object sender, EventArgs e)
            => await _interactionHelper.LoadImages(false);

        private async void btnShowProcessed_Click(object sender, EventArgs e)
            => await _interactionHelper.LoadImages(true);



    }
}