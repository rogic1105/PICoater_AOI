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

        // --- 即時取像 (監控頁 Panel1 / Panel5) ---
        private readonly Dictionary<int, PictureBox> _liveViewBoxes = new Dictionary<int, PictureBox>();
        private readonly Dictionary<int, Label> _cameraStatusLabels = new Dictionary<int, Label>();
        private readonly Dictionary<int, Bitmap> _latestLiveFrames = new Dictionary<int, Bitmap>();
        private readonly Timer _liveGrabTimer = new Timer();
        private bool _milAllocated;
        private bool _isLiveGrabbing;
        private int _frameIndex;

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

            InitializeLiveGrabPanels();
        }

        private void InitializeLiveGrabPanels()
        {
            SetupLivePanel(panel1, 1);
            SetupLivePanel(panel5, 5);

            _liveGrabTimer.Interval = 120;
            _liveGrabTimer.Tick += LiveGrabTimer_Tick;

            button1.Click += button1_Click;
            button2.Click += button2_Click;
            button3.Click += button3_Click;

            FormClosed += (_, __) =>
            {
                _liveGrabTimer.Stop();
                _liveGrabTimer.Tick -= LiveGrabTimer_Tick;
                ClearLiveFrames();
            };

            UpdateCameraStatus("未配置 (MIL Not Allocated)");
        }

        private void SetupLivePanel(Panel panel, int cameraIndex)
        {
            panel.BackColor = Color.Black;

            var preview = new PictureBox
            {
                Dock = DockStyle.Fill,
                SizeMode = PictureBoxSizeMode.Zoom,
                BackColor = Color.Black
            };

            var status = new Label
            {
                Dock = DockStyle.Bottom,
                Height = 18,
                ForeColor = Color.Lime,
                BackColor = Color.FromArgb(32, 32, 32),
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font("Segoe UI", 7.5f, FontStyle.Bold)
            };

            panel.Controls.Add(preview);
            panel.Controls.Add(status);

            _liveViewBoxes[cameraIndex] = preview;
            _cameraStatusLabels[cameraIndex] = status;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            _milAllocated = true;
            UpdateCameraStatus("已配置 (Allocated)");
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (!_milAllocated)
            {
                UpdateCameraStatus("抓取失敗: 尚未配置");
                return;
            }

            if (_isLiveGrabbing)
            {
                UpdateCameraStatus("抓取中 (Live)");
                return;
            }

            _isLiveGrabbing = true;
            _liveGrabTimer.Start();
            UpdateCameraStatus("抓取中 (Live)");
        }

        private void button3_Click(object sender, EventArgs e)
        {
            _liveGrabTimer.Stop();
            _isLiveGrabbing = false;
            _milAllocated = false;
            UpdateCameraStatus("已釋放 (Freed)");
            ClearLiveFrames();
        }

        private void LiveGrabTimer_Tick(object sender, EventArgs e)
        {
            _frameIndex++;
            UpdateLiveFrame(1);
            UpdateLiveFrame(5);
        }

        private void UpdateLiveFrame(int cameraIndex)
        {
            if (!_liveViewBoxes.TryGetValue(cameraIndex, out var box)) return;

            int width = Math.Max(box.Width, 148);
            int height = Math.Max(box.Height - 18, 93);
            var bmp = BuildDemoGrabFrame(cameraIndex, width, height, _frameIndex);

            if (_latestLiveFrames.TryGetValue(cameraIndex, out var oldBmp))
            {
                oldBmp.Dispose();
            }

            _latestLiveFrames[cameraIndex] = bmp;
            box.Image = bmp;
        }

        private static Bitmap BuildDemoGrabFrame(int cameraIndex, int width, int height, int frameIndex)
        {
            var bmp = new Bitmap(width, height);
            using (var g = Graphics.FromImage(bmp))
            {
                g.Clear(Color.Black);

                int markerX = (frameIndex * 7) % Math.Max(1, width - 30);
                int markerY = (frameIndex * 5) % Math.Max(1, height - 30);

                var gradientRect = new Rectangle(0, 0, width, height);
                using (var brush = new System.Drawing.Drawing2D.LinearGradientBrush(
                    gradientRect,
                    cameraIndex == 1 ? Color.DarkBlue : Color.DarkRed,
                    cameraIndex == 1 ? Color.Cyan : Color.Orange,
                    35f))
                {
                    g.FillRectangle(brush, gradientRect);
                }

                using (var pen = new Pen(Color.Lime, 2f))
                {
                    g.DrawRectangle(pen, markerX, markerY, 28, 28);
                }

                using (var font = new Font("Consolas", 10f, FontStyle.Bold))
                using (var brush = new SolidBrush(Color.White))
                {
                    string text = $"CAM{cameraIndex} Live\nFrame: {frameIndex}\n{DateTime.Now:HH:mm:ss.fff}";
                    g.DrawString(text, font, brush, new PointF(8, 8));
                }
            }

            return bmp;
        }

        private void UpdateCameraStatus(string statusText)
        {
            foreach (var pair in _cameraStatusLabels)
            {
                pair.Value.Text = $"CAM{pair.Key}: {statusText}";
            }
        }

        private void ClearLiveFrames()
        {
            foreach (var pair in _liveViewBoxes)
            {
                pair.Value.Image = null;
            }

            foreach (var bmp in _latestLiveFrames.Values)
            {
                bmp.Dispose();
            }

            _latestLiveFrames.Clear();
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
