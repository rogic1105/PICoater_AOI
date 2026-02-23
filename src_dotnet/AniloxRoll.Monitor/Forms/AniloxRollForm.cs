using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
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
        private readonly LiveGrabAdapter _liveGrabAdapter = new LiveGrabAdapter();
        private bool _milAllocated;
        private bool _isLiveGrabbing;

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
                _liveGrabAdapter.Free();
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
            if (_liveGrabAdapter.TryAllocate(out string status))
            {
                _milAllocated = true;
                UpdateCameraStatus(status);
                return;
            }

            _milAllocated = false;
            UpdateCameraStatus(status);
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

            if (!_liveGrabAdapter.TryStartGrab(out string status))
            {
                UpdateCameraStatus(status);
                return;
            }

            _isLiveGrabbing = true;
            _liveGrabTimer.Start();
            UpdateCameraStatus(status);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            _liveGrabTimer.Stop();
            _isLiveGrabbing = false;
            _milAllocated = false;
            ClearLiveFrames();

            _liveGrabAdapter.Free();
            UpdateCameraStatus("已釋放 (Freed)");
        }

        private void LiveGrabTimer_Tick(object sender, EventArgs e)
        {
            UpdateLiveFrame(1);
            UpdateLiveFrame(5);
        }

        private void UpdateLiveFrame(int cameraIndex)
        {
            if (!_liveViewBoxes.TryGetValue(cameraIndex, out var box)) return;

            if (_liveGrabAdapter.TryGrabFrame(cameraIndex, out Bitmap bitmap, out string status))
            {
                if (_latestLiveFrames.TryGetValue(cameraIndex, out var oldBmp)) oldBmp.Dispose();
                _latestLiveFrames[cameraIndex] = bitmap;
                box.Image = bitmap;
                UpdateSingleCameraStatus(cameraIndex, status);
                return;
            }

            UpdateSingleCameraStatus(cameraIndex, status);
        }

        private void UpdateCameraStatus(string statusText)
        {
            foreach (var pair in _cameraStatusLabels)
            {
                pair.Value.Text = $"CAM{pair.Key}: {statusText}";
            }
        }

        private void UpdateSingleCameraStatus(int cameraIndex, string statusText)
        {
            if (_cameraStatusLabels.TryGetValue(cameraIndex, out var label))
            {
                label.Text = $"CAM{cameraIndex}: {statusText}";
            }
        }

        private void ClearLiveFrames()
        {
            foreach (var pair in _liveViewBoxes) pair.Value.Image = null;

            foreach (var bmp in _latestLiveFrames.Values) bmp.Dispose();

            _latestLiveFrames.Clear();
        }

        private sealed class LiveGrabAdapter
        {
            private object _grabber;
            private MethodInfo _allocMethod;
            private MethodInfo _startMethod;
            private MethodInfo _freeMethod;
            private MethodInfo _grabMethod;
            private MethodInfo _statusMethod;

            public bool TryAllocate(out string status)
            {
                status = string.Empty;
                if (!EnsureBound(out status)) return false;

                if (!InvokeBool(_allocMethod, out status, "AOI_SDK: MIL 配置失敗")) return false;

                status = "已配置 (Allocated)";
                return true;
            }

            public bool TryStartGrab(out string status)
            {
                status = string.Empty;
                if (!EnsureBound(out status)) return false;

                if (_startMethod == null)
                {
                    status = "抓取中 (Live)";
                    return true;
                }

                if (!InvokeBool(_startMethod, out status, "AOI_SDK: 啟動抓取失敗")) return false;

                status = "抓取中 (Live)";
                return true;
            }

            public bool TryGrabFrame(int cameraIndex, out Bitmap bitmap, out string status)
            {
                bitmap = null;
                status = "未抓到影像";
                if (!EnsureBound(out status) || _grabMethod == null) return false;

                object result;
                try
                {
                    var pars = _grabMethod.GetParameters();
                    result = pars.Length == 0
                        ? _grabMethod.Invoke(_grabber, null)
                        : _grabMethod.Invoke(_grabber, new object[] { cameraIndex });
                }
                catch (Exception ex)
                {
                    status = $"抓取異常: {ex.GetBaseException().Message}";
                    return false;
                }

                if (result is Bitmap bmp)
                {
                    bitmap = (Bitmap)bmp.Clone();
                    status = GetCameraStatus(cameraIndex) ?? "抓取中";
                    return true;
                }

                if (result is Image image)
                {
                    bitmap = new Bitmap(image);
                    status = GetCameraStatus(cameraIndex) ?? "抓取中";
                    return true;
                }

                status = "AOI_SDK 回傳型別非 Bitmap";
                return false;
            }

            public void Free()
            {
                if (_grabber == null || _freeMethod == null) return;
                try { _freeMethod.Invoke(_grabber, null); } catch { }
            }

            private string GetCameraStatus(int cameraIndex)
            {
                if (_statusMethod == null || _grabber == null) return null;

                try
                {
                    var pars = _statusMethod.GetParameters();
                    object result = pars.Length == 0
                        ? _statusMethod.Invoke(_grabber, null)
                        : _statusMethod.Invoke(_grabber, new object[] { cameraIndex });
                    return result?.ToString();
                }
                catch
                {
                    return null;
                }
            }

            private bool EnsureBound(out string status)
            {
                status = string.Empty;
                if (_grabber != null) return true;

                foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
                {
                    Type[] types;
                    try { types = asm.GetTypes(); }
                    catch (ReflectionTypeLoadException ex) { types = ex.Types.Where(t => t != null).ToArray(); }
                    catch { continue; }

                    var candidate = types.FirstOrDefault(t =>
                        t != null &&
                        t.IsClass &&
                        !t.IsAbstract &&
                        (t.Name.IndexOf("Mdig", StringComparison.OrdinalIgnoreCase) >= 0 ||
                         t.Name.IndexOf("MilGrab", StringComparison.OrdinalIgnoreCase) >= 0 ||
                         t.Name.IndexOf("Grab", StringComparison.OrdinalIgnoreCase) >= 0));

                    if (candidate == null) continue;

                    object instance = ResolveInstance(candidate);
                    if (instance == null) continue;

                    _grabber = instance;
                    BindMethods(candidate);

                    if (_allocMethod == null || _grabMethod == null)
                    {
                        status = $"AOI_SDK 類別 {candidate.FullName} 缺少 Allocate/Grab 方法";
                        continue;
                    }

                    return true;
                }

                status = "找不到 AOI_SDK 的 MdigGrab 類別 (請確認 AOI_SDK 已載入)";
                return false;
            }

            private void BindMethods(Type t)
            {
                _allocMethod = FindMethod(t, "Alloc", "Allocate", "Initialize");
                _startMethod = FindMethod(t, "Start", "GrabStart", "Run", "GrabContinuous");
                _freeMethod = FindMethod(t, "Free", "Release", "Dispose", "Uninitialize");
                _grabMethod = FindMethod(t, "Grab", "GetImage", "GetBitmap", "GrabFrame", "Capture");
                _statusMethod = FindMethod(t, "Status", "GetStatus", "CameraStatus");
            }

            private static MethodInfo FindMethod(Type t, params string[] keys)
            {
                var methods = t.GetMethods(BindingFlags.Public | BindingFlags.Instance)
                    .Where(m => !m.IsSpecialName).ToArray();

                foreach (var key in keys)
                {
                    var hit = methods.FirstOrDefault(m =>
                        m.Name.IndexOf(key, StringComparison.OrdinalIgnoreCase) >= 0 &&
                        (m.GetParameters().Length == 0 || m.GetParameters().Length == 1));
                    if (hit != null) return hit;
                }

                return null;
            }

            private static object ResolveInstance(Type t)
            {
                var instanceProp = t.GetProperty("Instance", BindingFlags.Public | BindingFlags.Static);
                if (instanceProp != null) return instanceProp.GetValue(null, null);

                var defaultCtor = t.GetConstructor(Type.EmptyTypes);
                return defaultCtor != null ? Activator.CreateInstance(t) : null;
            }

            private bool InvokeBool(MethodInfo method, out string status, string failText)
            {
                status = failText;
                if (method == null) return true;

                try
                {
                    object result = method.Invoke(_grabber, null);
                    if (result is bool ok) return ok;
                    if (result is int code) return code == 0;
                    return true;
                }
                catch (Exception ex)
                {
                    status = $"{failText}: {ex.GetBaseException().Message}";
                    return false;
                }
            }
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
