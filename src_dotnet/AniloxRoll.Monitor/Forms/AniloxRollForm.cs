using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.Forms.Helpers;
using AniloxRoll.Monitor.Core.Camera; // 確保引入你的相機命名空間
using Matrox.MatroxImagingLibrary;

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

        private int _selectedMainCameraId = 1;

        // ==========================================
        // --- 即時取像 (監控頁 Panel1 / Panel5) ---
        // ==========================================
        private class CameraConfig
        {
            public int Id { get; set; }
            public string SystemDescriptor { get; set; }
            public int SystemNum { get; set; }
            public MIL_INT DevNum { get; set; }
            public string DcfPath { get; set; }

            // 【修改1】把 PictureBox 改成 Panel
            public Panel DisplayPanel { get; set; }
            public Label StatusLabel { get; set; }
        }

        private List<AniloxCamera> _cameras = new List<AniloxCamera>();
        private List<CameraConfig> _cameraConfigs;
        private Dictionary<int, MIL_ID> _allocatedSystems = new Dictionary<int, MIL_ID>();

        // 【修改2】把 PictureBox 字典改成 Panel 字典
        private readonly Dictionary<int, Panel> _liveViewPanels = new Dictionary<int, Panel>();
        private readonly Dictionary<int, Label> _cameraStatusLabels = new Dictionary<int, Label>();

        private Timer _cameraStatusTimer;
        private bool _milAllocated = false;
        private bool _isLiveGrabbing = false;

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

            _timeSelectionManager = new DateTimeNavigator(
                _imageRepository, cbYear, cbMonth, cbDay, cbHour, cbMin, cbSec);

            _galleryManager = new ThumbnailGridPresenter();
            _galleryManager.Initialize(new PictureBox[] {
                pbCam1, pbCam2, pbCam3, pbCam4, pbCam5, pbCam6, pbCam7
            });

            _presenter = new AniloxRollPresenter(
                _imageRepository, _inspectionService, _timeSelectionManager, _galleryManager);

            // 3. 初始化 Chart
            _muraChartHelper = new MuraChartHelper(this.chartMura);
            _muraChartHelper.SetOps(_settings.Cam1_Ops);

            // 4. 設定 PropertyGrid
            propertyGrid1.SelectedObject = _settings;
            propertyGrid1.ToolbarVisible = false;
            propertyGrid1.PropertyValueChanged -= _propertyGrid_PropertyValueChanged;
            propertyGrid1.PropertyValueChanged += _propertyGrid_PropertyValueChanged;

            // 5. 初始化 InteractionHelper
            _interactionHelper = new FormInteractionHelper(
                this, canvasMain, new Button[] { btnShowOriginal, btnShowProcessed, btnSelectFolder },
                _thumbnailCache, _presenter, _inspectionService, _imageRepository,
                _timeSelectionManager, _galleryManager, _muraChartHelper, _settings, lblPixelInfo
            );

            _interactionHelper.ApplySettingsToService();

            // 6. 綁定事件
            _presenter.BusyStateChanged += _interactionHelper.SetUiLoadingState;
            _presenter.LogReported += log => Console.WriteLine(log);
            _galleryManager.SelectionChanged += _interactionHelper.OnGallerySelectionChanged;

            canvasMain.StatusChanged += OnCanvasStatusChanged;
            canvasMain.EdgeReached += OnCanvasEdgeReached;

            // 7. 初始化即時相機面板
            InitializeLiveGrabPanels();
        }

        // ==========================================
        // --- 相機硬體初始化與綁定 ---
        // ==========================================
        private void InitializeLiveGrabPanels()
        {
            SetupLivePanel(panel1, 1);
            SetupLivePanel(panel5, 5);

            _cameraConfigs = new List<CameraConfig>
            {
                new CameraConfig
                {
                    Id = 1,
                    SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL,
                    SystemNum = 0,
                    DevNum = MIL.M_DEV0,
                    DcfPath = @"C:\Users\User\Downloads\dcf\Radient_Config.dcf",
                    DisplayPanel = _liveViewPanels[1], // 【修改4】綁定 Panel
                    StatusLabel = _cameraStatusLabels[1]
                },
                new CameraConfig
                {
                    Id = 5,
                    SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL,
                    SystemNum = 1,
                    DevNum = MIL.M_DEV0,
                    DcfPath = @"C:\Users\User\Downloads\dcf\Radient_Config.dcf",
                    DisplayPanel = _liveViewPanels[5], // 【修改4】綁定 Panel
                    StatusLabel = _cameraStatusLabels[5]
                }
            };

            _cameraStatusTimer = new Timer { Interval = 500 };
            _cameraStatusTimer.Tick += CameraStatusTimer_Tick;

            UpdateCameraStatus("未配置 (MIL Not Allocated)", Color.Gray);

            FormClosed += (_, __) => { FreeCameras(); };
        }
        private void SetupLivePanel(Panel panel, int cameraIndex)
        {
            panel.BackColor = Color.Black;
            panel.Controls.Clear();

            var displayPanel = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = Color.Black
            };

            var status = new Label
            {
                Dock = DockStyle.Bottom,
                Height = 18,
                ForeColor = Color.White,
                BackColor = Color.FromArgb(32, 32, 32),
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font("Segoe UI", 7.5f, FontStyle.Bold)
            };

            // [新增] 綁定點擊事件，當點擊小面板時，切換主畫面
            displayPanel.MouseClick += (s, e) => SwitchMainDisplay(cameraIndex);
            status.MouseClick += (s, e) => SwitchMainDisplay(cameraIndex);

            panel.Controls.Add(displayPanel);
            panel.Controls.Add(status);
            displayPanel.BringToFront();

            _liveViewPanels[cameraIndex] = displayPanel;
            _cameraStatusLabels[cameraIndex] = status;
        }

        private void SwitchMainDisplay(int cameraIndex)
        {
            _selectedMainCameraId = cameraIndex;

            // 1. 更新 UI 標籤背景顏色 (選中的變深藍色，其餘恢復)
            foreach (var kvp in _cameraStatusLabels)
            {
                if (kvp.Key == cameraIndex)
                    kvp.Value.BackColor = Color.DarkBlue;
                else
                    kvp.Value.BackColor = Color.FromArgb(32, 32, 32);
            }

            // 2. 切換 MIL 的大畫面顯示控制代碼
            foreach (var cam in _cameras)
            {
                if (cam.CameraId == cameraIndex)
                {
                    // 被選中的相機，將第二畫面投影到 panel8
                    cam.SetSecondaryDisplay(panel8.Handle);
                }
                else
                {
                    // 其他相機，取消第二畫面投影
                    cam.SetSecondaryDisplay(IntPtr.Zero);
                }
            }
        }

        // ==========================================
        // --- 按鈕事件：配置 (Allocate) ---
        // ==========================================
        private void btnCameraAllocation_Click(object sender, EventArgs e)
        {
            if (_milAllocated) return;

            try
            {
                CameraSystemManager.Initialize();

                foreach (var cfg in _cameraConfigs)
                {
                    MIL_ID currentSysId = MIL.M_NULL;

                    if (_allocatedSystems.ContainsKey(cfg.SystemNum))
                    {
                        currentSysId = _allocatedSystems[cfg.SystemNum];
                    }
                    else
                    {
                        currentSysId = CameraSystemManager.AllocateSystem(cfg.SystemDescriptor, cfg.SystemNum);
                        if (currentSysId != MIL.M_NULL)
                        {
                            _allocatedSystems.Add(cfg.SystemNum, currentSysId);
                        }
                        else
                        {
                            UpdateSingleCameraStatus(cfg.Id, "分配 System 失敗", Color.Red);
                            continue;
                        }
                    }

                    // 實例化 AniloxCamera (將 PictureBox.Handle 交給 MIL 處理顯示)
                    var cam = new AniloxCamera(
                        currentSysId,
                        cfg.Id,
                        cfg.DevNum,
                        cfg.DcfPath,
                        cfg.DisplayPanel.Handle, // 【確認】這裡是傳入 Panel 的 Handle
                        checkBoxEnableImageProcessing.Checked // 【修改這裡】讀取 CheckBox 目前的狀態
                    );

                    cam.Initialize();
                    _cameras.Add(cam);
                }

                _milAllocated = true;
                _cameraStatusTimer.Start();
                UpdateCameraStatus("已配置 (Ready)", Color.Yellow);
                SwitchMainDisplay(_selectedMainCameraId);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"相機配置失敗: {ex.Message}", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        // ==========================================
        // --- 按鈕事件：抓取/停止 (Toggle Grab) ---
        // ==========================================
        private void btnCameraGrab_Click(object sender, EventArgs e)
        {
            if (!_milAllocated)
            {
                MessageBox.Show("請先點擊「相機配置」!", "提示");
                return;
            }

            _isLiveGrabbing = !_isLiveGrabbing;

            foreach (var cam in _cameras)
            {
                cam.SetUserGrabIntent(_isLiveGrabbing);
            }

            btnCameraGrab.Text = _isLiveGrabbing ? "停止抓取" : "開始抓取";
        }

        // ==========================================
        // --- 按鈕事件：釋放 (Free) ---
        // ==========================================
        private void btnCameraFree_Click(object sender, EventArgs e)
        {
            FreeCameras();
            btnCameraGrab.Text = "開始抓取";
        }

        private void FreeCameras()
        {
            _cameraStatusTimer?.Stop();
            _isLiveGrabbing = false;

            // 1. 釋放相機與 Buffers
            foreach (var cam in _cameras)
            {
                cam.Free();
            }
            _cameras.Clear();

            // 2. 釋放擷取卡系統
            foreach (var kvp in _allocatedSystems)
            {
                CameraSystemManager.FreeSystem(kvp.Value);
            }
            _allocatedSystems.Clear();

            // 3. 釋放 MIL Application
            CameraSystemManager.FreeApplication();

            _milAllocated = false;
            UpdateCameraStatus("已釋放 (Freed)", Color.Gray);
        }

        // ==========================================
        // --- 狀態更新 (FPS & 連線狀態) ---
        // ==========================================
        private void CameraStatusTimer_Tick(object sender, EventArgs e)
        {
            foreach (var cam in _cameras)
            {
                bool isConnected = cam.CheckPresence();
                string fpsText = cam.IsLive ? $" | FPS: {cam.CurrentFps:F1}" : "";

                string statusText = isConnected
                    ? (cam.IsLive ? $"Live{fpsText}" : "Ready")
                    : "Offline";

                Color color = isConnected
                    ? (cam.IsLive ? Color.Lime : Color.Yellow)
                    : Color.Red;

                UpdateSingleCameraStatus(cam.CameraId, statusText, color);
            }
        }

        private void UpdateCameraStatus(string statusText, Color color)
        {
            foreach (var pair in _cameraStatusLabels)
            {
                pair.Value.Text = $"CAM{pair.Key}: {statusText}";
                pair.Value.ForeColor = color;
            }
        }

        private void UpdateSingleCameraStatus(int cameraIndex, string statusText, Color color)
        {
            if (_cameraStatusLabels.TryGetValue(cameraIndex, out var label))
            {
                label.Text = $"CAM{cameraIndex}: {statusText}";
                label.ForeColor = color;
            }
        }

        // ==========================================
        // --- 原本的委派事件 ---
        // ==========================================
        private void OnCanvasStatusChanged(AOI.SDK.UI.CanvasInfo info)
            => _interactionHelper.UpdateCanvasInfo(info);

        private void OnCanvasEdgeReached(int direction)
            => _interactionHelper.NavigateCamera(direction);

        private void _propertyGrid_PropertyValueChanged(object s, PropertyValueChangedEventArgs e)
            => _interactionHelper.HandleSettingsChanged();

        private void btnSelectFolder_Click(object sender, EventArgs e)
            => _interactionHelper.SelectAndLoadFolder();

        private async void btnShowOriginal_Click(object sender, EventArgs e)
            => await _interactionHelper.LoadImages(false);

        private async void btnShowProcessed_Click(object sender, EventArgs e)
            => await _interactionHelper.LoadImages(true);

        private void checkBoxEnableImageProcessing_CheckedChanged(object sender, EventArgs e)
        {
            // 取得當前 CheckBox 的狀態
            bool enableImageProcessing = checkBoxEnableImageProcessing.Checked;

            // 遍歷所有已開啟的相機，即時更新它們的內部屬性
            foreach (var cam in _cameras)
            {
                cam.EnableImageProcessing = enableImageProcessing;
            }
        }
    }
}