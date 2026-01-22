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
        private readonly ImageRepository _imageRepository = new ImageRepository();
        private BatchInspectionService _inspectionService;
        private DateTimeNavigator _timeSelectionManager;
        private ThumbnailGridPresenter _galleryManager;
        private AniloxRollPresenter _presenter;
        private FormInteractionHelper _interactionHelper;
        private readonly List<Image> _thumbnailCache = new List<Image>();

        private double[] _cameraOps = new double[] { 33, 33, 33, 33, 33, 33, 33 };
        private double[] _cameraStartPos = new double[] { 0, 400, 800, 1200, 1600, 2000, 2400 };
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
            _inspectionService = new BatchInspectionService();

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

            _interactionHelper = new FormInteractionHelper(
                this,
                canvasMain,
                new Button[] { btnShowOriginal, btnShowProcessed, btnSelectFolder },
                _thumbnailCache,
                _presenter,
                _inspectionService,
                _imageRepository,
                _timeSelectionManager,
                _galleryManager
            );

            _presenter.BusyStateChanged += _interactionHelper.SetUiLoadingState;
            _presenter.LogReported += log => Console.WriteLine(log);

            _galleryManager.SelectionChanged += (idx) =>
            {
                if (idx >= 0 && idx < 7) _currentCameraIndex = idx;
            };

            _galleryManager.SelectionChanged += _interactionHelper.OnGallerySelectionChanged;

            // ================================================================
            // [修正重點] 這裡只使用 StatusChanged，請刪除舊的 PixelHovered
            // ================================================================
            canvasMain.StatusChanged += (info) =>
            {
                double ops = _cameraOps[_currentCameraIndex];
                double startPos = _cameraStartPos[_currentCameraIndex];

                // 計算物理座標
                double physicalX = startPos + (info.ImageX * (ops / 1000.0));

                // 計算視野範圍
                _currentViewLeftX = (int)((0 - info.PanOffset.X) / info.Zoom);
                _currentViewRightX = (int)((canvasMain.Width - info.PanOffset.X) / info.Zoom);

                // 更新狀態列
                lblPixelInfo.Text =
                    $"位置:{physicalX:F2} mm| " +
                    $"座標: ({info.ImageX}, {info.ImageY}) | " +
                    $"亮度: {info.PixelColor.R} | " +
                    $"倍率:{info.Zoom:F2}x | " +
                    $"平移:({info.PanOffset.X:F0}, {info.PanOffset.Y:F0})";
            };

            // 綁定邊緣觸發事件 (自動跳轉)
            canvasMain.EdgeReached += (direction) =>
            {
                int nextIndex = _currentCameraIndex + direction;
                if (nextIndex >= 0 && nextIndex < 7)
                {
                    _galleryManager.Select(nextIndex);
                }
            };
        }
        private void btnSelectFolder_Click(object sender, EventArgs e)
            => _interactionHelper.SelectAndLoadFolder();

        private async void btnShowOriginal_Click(object sender, EventArgs e)
            => await _interactionHelper.LoadImages(false);

        private async void btnShowProcessed_Click(object sender, EventArgs e)
            => await _interactionHelper.LoadImages(true);
    }
}