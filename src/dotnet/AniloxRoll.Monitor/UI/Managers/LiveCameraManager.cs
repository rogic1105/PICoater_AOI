using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;
using MilGrabber.Core;
using TanukiCv.Core; // PixelMmMapper（已收進 sdk 唯一來源）
using TanukiCv.Controls; // LiveDisplayView（共用多相機監控顯示元件）
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.Core.Interop; // NativeMethods（LOD GPU resize；P/Invoke 宣告唯一點）

namespace AniloxRoll.Monitor.UI.Managers
{
    public class LiveCameraManager
    {
        private readonly Form _mainForm;
        private readonly Panel _mainDisplayPanel;
        private Panel[] _cameraPanels;                 // camLive1~7（SmartCanvas 模式 thumbnail 用）
        private LiveDisplayView _smartDisplay;        // 共用多相機監控顯示（sdk TanukiCv.Controls；he_MainDisplay==SmartCanvas）
        private bool SmartCanvasMode => _inspectionSettings != null
            && _inspectionSettings.he_MainDisplay == AniloxRoll.Monitor.Core.Data.MainDisplayMode.SmartCanvas;
        private readonly Action<string> _updatePixelInfoCallback;

        // 動態 LOD（he_LiveLod）：GPU provider 專用 pinned（只裁可見小區，非每幀全幀）；CPU 走 GrayResizeCpu 無 pinned。
        private IntPtr _lodSrcPinned, _lodDstPinned;
        private int _lodSrcCap, _lodDstCap;
        private readonly object _lodBufLock = new object();
        private volatile bool _lodReleased;

        /// <summary>法向(Y) mm/影像列（row pitch）：form 從速度+線掃算好餵入 → SetLayout → 法向曲線圖 Y 對齊。
        /// 0 時 LiveDisplayView 退回用 X 的 ops（Y 不對齊）。</summary>
        public double RowPitchMm { get; set; }

        private List<AniloxCamera> _cameras = new List<AniloxCamera>();
        private List<CameraHardwareConfig> _cameraHardwareConfigs;
        private Dictionary<int, MIL_ID> _allocatedSystems = new Dictionary<int, MIL_ID>();

        private readonly Dictionary<int, Panel> _liveViewPanels    = new Dictionary<int, Panel>();
        private readonly Dictionary<int, Panel> _liveParentPanels  = new Dictionary<int, Panel>();
        private readonly Dictionary<int, Label> _cameraStatusLabels = new Dictionary<int, Label>();

        private Timer _cameraStatusTimer;
        private bool _enableAutoCapture;
        private bool _saveOriginalBmp = false;
        private string _captureRootPath = string.Empty;
        private int[]    _cameraGrabHeight    = new int[7];
        private double[] _cameraExposureTimeUs = new double[7];
        private double[] _cameraLineRateHz     = new double[7];
        private int _saveResizeScale = InspectionEngineConfig.DefaultSaveResizeScale;
        private int _saveJpgQuality  = InspectionEngineConfig.DefaultSaveJpgQuality;
        private float _hessianMaxFactor = InspectionEngineConfig.DefaultHessianMaxFactor;
        private string _ridgeMode = InspectionEngineConfig.DefaultRidgeMode;
        private string _dcfPath = string.Empty;
        private readonly CaptureTimestampCoordinator _timestampCoordinator = new CaptureTimestampCoordinator();

        public bool IsAllocated    { get; private set; } = false;
        public bool IsLiveGrabbing { get; private set; } = false;

        /// <summary>目前已初始化的相機清單（唯讀），供 LiveTelemetryPresenter 查詢 Telemetry。</summary>
        public IReadOnlyList<AniloxCamera> Cameras => _cameras.AsReadOnly();

        /// <summary>預期相機數量（由硬體設定決定）。</summary>
        public int ExpectedCameraCount => _cameraHardwareConfigs?.Count ?? 0;

        /// <summary>目前已連線的相機數量（每 500ms 更新）。</summary>
        public int ConnectedCameraCount { get; private set; }

        /// <summary>連線數變更時通知 UI。參數：(connected, expected)。</summary>
        public event Action<int, int> OnCameraCountChanged;

        /// <summary>所有相機 CLProtocol 初始化完成（曝光/線掃已套，可安全 grab）時一次性觸發，供 UI 解鎖
        /// 「開始抓取」鈕。CLProtocol 逾時/失敗也算完成（fallback legacy 參數），故按鈕不會永久卡死。</summary>
        public event Action OnHwReady;
        private bool _hwReadyRaised;

        /// <summary>所有相機 CLProtocol 是否就緒（已套曝光/線掃）。未就緒時上層應禁用「開始抓取」。</summary>
        public bool AreCamerasHwReady
        {
            get
            {
                if (!IsAllocated || _cameras.Count == 0) return false;
                foreach (var cam in _cameras)
                    if (!cam.IsHwParamsStable) return false;
                return true;
            }
        }

        /// <summary>每台相機存檔並完成 inspection 後觸發。
        /// 參數：(cameraId, fileNameWithoutExt, meanPeak_0to1, maxPeak_0to1)</summary>
        public event Action<int, string, float, float> OnInspectionResult;

        /// <summary>每幀 GPU pipeline 完成後觸發（MIL 回呼執行緒）。
        /// 參數：(cameraId, curveMean_raw255, curveMax_raw255)</summary>
        public event Action<int, float[], float[]> OnLiveCurveData;

        /// <summary>每幀 GPU pipeline 完成後觸發（MIL 回呼執行緒）。
        /// 參數：(cameraId, rowCurveMean_raw255, rowCurveMax_raw255)</summary>
        public event Action<int, float[], float[]> OnLiveRowCurveData;

        /// <summary>存檔完成回呼：傳入已儲存的檔案路徑陣列（供遠端複製佇列）。</summary>
        public Action<string[]> OnFilesSaved { get; set; }

        /// <summary>Vertical 模式滾輪縮放後立即觸發，讓 chart 不必等下一幀 callback 就同步視野。</summary>
        public Action OnAfterVerticalZoom { get; set; }

        /// <summary>
        /// 正在執行釋放流程時為 true，防止 Timer Tick 在資源已釋放後繼續存取相機。
        /// 同 CameraSession.IsReleasing。
        /// </summary>
        public volatile bool IsReleasing = false;

        private int _selectedMainCameraId = 1;
        public int SelectedMainCameraId => _selectedMainCameraId;

        /// <summary>使用者明確點選的相機 ID（不受 Global 模式視野中心 timer 影響）。</summary>
        private int _userSelectedMainCameraId = 1;

        // --- Global merge（即時合圖）---
        // 合圖的「拼」（佈局 + 合併 buffer + 每台 merge target）委派給 MultiCameraMerger 工頭（sdk/MIL）。
        // 本類別只負責「秀」：MdispSelectWindow 顯示、33ms 防閃爍刷新、滑鼠 hook、overview 聯動。
        private MultiCameraMerger _merger;
        private MIL_ID _mergedDisplay = MIL.M_NULL;
        public bool IsGlobalMergeActive { get; private set; }
        // 座標欄位為工頭值的本地鏡像（值來源 = 工頭），EnableGlobalMerge/RefreshGlobalMergeLayout 後同步。
        private double _mergedMinStartMm;   // 合併座標系原點（mm）
        private double _mergedRefOpsMm;     // 合併像素尺寸（mm/px）
        private int    _mergedTotalW;       // 合併 buffer 寬度（px）
        private int    _mergedTotalH;       // 合併 buffer 高度（px）
        private double[] _mergedSlotStartsMm;  // 7 槽位起始 mm（含空缺）
        private double[] _mergedSlotEndsMm;    // 7 槽位結束 mm（含空缺）
        private MIL_DISP_HOOK_FUNCTION_PTR _mergedMouseDelegate;
        private Timer _mergedDisplayTimer;  // 定時刷新合圖 display（取代 MIL 自動刷新，避免多相機非同步閃爍）

        private InspectionSettings _inspectionSettings;
        private double _screenMmPerPx;
        private WheelZoomFilter _wheelFilter;

        public LiveCameraManager(
            Form mainForm,
            Panel[] cameraPanels,
            Panel mainDisplayPanel,
            Action<string> updatePixelInfoCallback)
        {
            if (cameraPanels == null)
                throw new ArgumentNullException(nameof(cameraPanels));
            if (cameraPanels.Length < 7)
                throw new ArgumentException("cameraPanels must contain at least 7 panels.", nameof(cameraPanels));

            _mainForm = mainForm;
            _mainDisplayPanel = mainDisplayPanel;
            _cameraPanels = cameraPanels;
            _updatePixelInfoCallback = updatePixelInfoCallback;
            _mainDisplayPanel.BackColor = Color.Black;

            _wheelFilter = new WheelZoomFilter(this);
            Application.AddMessageFilter(_wheelFilter);

            for (int i = 0; i < 7; i++)
                SetupLivePanel(cameraPanels[i], i + 1);

            _cameraHardwareConfigs = SystemSettings.CreateDefault().CameraDevices;

            _cameraStatusTimer = new Timer { Interval = 500 };
            _cameraStatusTimer.Tick += CameraStatusTimer_Tick;

            UpdateCameraStatus("未配置", Color.Gray);
        }

        private void SetupLivePanel(Panel parentPanel, int cameraIndex)
        {
            parentPanel.BackColor = Color.Black;
            parentPanel.Padding   = new Padding(2);
            parentPanel.Controls.Clear();

            var displayPanel = new Panel
            {
                Dock      = DockStyle.Fill,
                BackColor = Color.Black
            };

            var status = new Label
            {
                Dock      = DockStyle.Bottom,
                Height    = 18,
                ForeColor = Color.DarkGray,
                BackColor = Color.FromArgb(32, 32, 32),
                TextAlign = ContentAlignment.MiddleCenter,
                // 動態建立、未被 ProportionalScaler 記錄縮放 → 直接用較小字級對齊全域 0.85（DPI 感知下 10f 偏大）
                Font      = new Font("Segoe UI", 8.5f, FontStyle.Regular)
            };

            displayPanel.MouseClick += (s, e) => SwitchMainDisplay(cameraIndex);
            status.MouseClick       += (s, e) => SwitchMainDisplay(cameraIndex);
            parentPanel.Paint       += (s, e) => OnLivePanelPaint(s, e, cameraIndex);

            parentPanel.Controls.Add(displayPanel);
            parentPanel.Controls.Add(status);
            displayPanel.BringToFront();

            _liveViewPanels[cameraIndex]     = displayPanel;
            _liveParentPanels[cameraIndex]   = parentPanel;
            _cameraStatusLabels[cameraIndex] = status;
        }

        // ==================== Allocate ====================

        public void AllocateCameras(bool enableImageProcessing)
        {
            if (IsAllocated) return;
            IsReleasing = false;

            CameraSystemManager.Initialize();

            foreach (var cfg in _cameraHardwareConfigs)
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

                if (!_liveViewPanels.TryGetValue(cfg.Id, out Panel displayPanel) ||
                    !_cameraStatusLabels.ContainsKey(cfg.Id))
                    continue;

                string dcf = DcfPathHelper.Resolve(!string.IsNullOrEmpty(_dcfPath) ? _dcfPath : cfg.DcfPath);
                var cam = new AniloxCamera(
                    currentSysId,
                    cfg.Id,
                    cfg.DevNum,   // 固定 device 位置（絕對值，直接傳）；MIL 轉換收斂在 MilCamera ctor
                    dcf,
                    displayPanel.Handle,
                    enableImageProcessing);

                int camIdx = cfg.Id - 1; // cfg.Id 為 1–7，轉為 0–6 陣列索引
                cam.EnableAutoCapture    = _enableAutoCapture;
                cam.SaveOriginalBmp = _saveOriginalBmp;
                cam.CaptureRootPath      = _captureRootPath;
                cam.CameraGrabHeight     = _cameraGrabHeight[camIdx];
                cam.CameraExposureTimeUs = _cameraExposureTimeUs[camIdx]; // Initialize() 會呼叫 SetExposureUs 套用
                cam.SetLineRateHz(_cameraLineRateHz[camIdx]);  // 記錄 _appliedLineRateHz（CLProtocol 就緒後自動重套）
                cam.HessianSigma         = InspectionEngineConfig.DefaultRidgeSigma;
                cam.HessianFixedMax      = _hessianMaxFactor;
                cam.RidgeMode            = _ridgeMode;
                cam.SaveResizeScale      = _saveResizeScale;
                cam.SaveJpgQuality       = _saveJpgQuality;
                cam.TimestampCoordinator = _timestampCoordinator;

                cam.OnMouseDataChanged   += HandleMouseDataChanged;
                cam.OnCameraClicked      += SwitchMainDisplay;
                cam.OnInspectionResult   += (camId, fn, mp, xp) =>
                    OnInspectionResult?.Invoke(camId, fn, mp, xp);
                cam.OnLiveCurveData      += (camId, mean, max) =>
                    OnLiveCurveData?.Invoke(camId, mean, max);
                cam.OnLiveRowCurveData   += (camId, mean, max) =>
                    OnLiveRowCurveData?.Invoke(camId, mean, max);
                cam.OnFilesSaved = OnFilesSaved;
                cam.Initialize();
                _cameras.Add(cam);
            }

            // CLProtocol 啟用移到「所有相機 buffer 分配完成後」的背景階段：不與 MbufAlloc/MdispAlloc 競爭
            // MIL 內部鎖，也不在 grab 期間 enable + 重套線掃（會掉幀，cam1 最明顯）。利用「分配 → 使用者點抓取」
            // 空檔跑完 2-5s/台；完成前 AreCamerasHwReady=false，上層把「開始抓取」鈕維持灰色。
            // 只對「在線」相機啟用 CLProtocol：對斷線相機 enable 會卡住 MIL 內部鎖（全斷線時 7 台全卡 →
            // 10s 逾時旗標翻 true 後 timer 恢復、CheckPresence 跟還卡著的背景 MIL 搶鎖 → 整個 UI 凍死）。
            // 斷線相機 _clProtocolInitStarted=false → IsHwParamsStable=true（不擋 AreCamerasHwReady）；
            // 若之後才連上，走 legacy 參數路徑（與導入 CLProtocol 前行為相同）。順帶：正常 2/7 時 init
            // 不再空等 5 台死相機逾時 → 從 ~10s 縮到 ~2-4s。
            _hwReadyRaised = false;
            foreach (var cam in _cameras)
                if (cam.CheckPresence())
                    cam.BeginCLProtocolInit();

            IsAllocated = true;
            _cameraStatusTimer.Start();
            UpdateCameraStatus("已配置", Color.White);

            EnsureSmartDisplay(); // SmartCanvas 模式：在 camLiveMain 疊 SmartCanvas + 訂閱各相機每幀 bytes

            SwitchMainDisplay(_selectedMainCameraId);

            // 初始化後立即發布相機數量（分配成功不代表已連線，Timer 會持續更新）
            ConnectedCameraCount = _cameras.Count;
            OnCameraCountChanged?.Invoke(_cameras.Count, ExpectedCameraCount);
        }

        // ==================== Grab Control ====================

        public void ToggleGrab()
        {
            if (!IsAllocated) return;
            if (IsLiveGrabbing) StopGrab();
            else StartGrab();
        }

        public void EnsureAllocatedAndToggleGrab(bool enableImageProcessing)
        {
            if (!IsAllocated)
                AllocateCameras(enableImageProcessing);
            ToggleGrab();
        }

        public void StartGrab()
        {
            if (!IsAllocated || IsLiveGrabbing) return;
            IsLiveGrabbing = true;
            // 切「主畫面顯示」設定後重開抓取即生效：SmartCanvas 模式建立、MilDirect 模式拆除
            if (SmartCanvasMode) EnsureSmartDisplay();
            else TeardownSmartDisplay();
            foreach (var cam in _cameras)
                cam.SetUserGrabIntent(true);
        }

        public void StopGrab()
        {
            if (!IsAllocated || !IsLiveGrabbing) return;
            IsLiveGrabbing = false;
            foreach (var cam in _cameras)
                cam.SetUserGrabIntent(false);

        }

        // ==================== Release ====================

        public void FreeCameras()
        {
            IsReleasing = true;
            _cameraStatusTimer.Stop();
            IsLiveGrabbing = false;
            _hwReadyRaised = false;

            // 先停止 global merge（必須在 cam.Free 之前）：DisableGlobalMerge 會先清各相機 merge target
            // 再由工頭 MbufFree 合併 buffer，避免 grab hook 把幀複製進已釋放的 buffer。
            DisableGlobalMerge();

            // SmartCanvas 顯示：解訂閱 + dispose（移除 camLiveMain 上的 SmartCanvas/thumbnail）
            if (_smartDisplay != null)
            {
                foreach (var cam in _cameras) cam.OnDisplayFrame -= OnCameraDisplayFrame;
                _smartDisplay.Dispose();
                _smartDisplay = null;
            }

            foreach (var cam in _cameras)
                cam.Free();
            _cameras.Clear();

            foreach (var kvp in _allocatedSystems)
                CameraSystemManager.FreeSystem(kvp.Value);
            _allocatedSystems.Clear();

            CameraSystemManager.FreeApplication();

            IsAllocated = false;
            UpdateCameraStatus("已釋放 (Freed)", Color.Gray);
        }

        /// <summary>
        /// 非同步釋放所有 MIL 資源，避免阻塞 UI 執行緒。
        /// 先在呼叫端執行緒停止 Timer，防止 background thread 釋放相機時
        /// UI thread 的 Tick 仍在存取 _cameras（可能 InvalidOperationException 或 MdigInquire on freed digitizer）。
        /// 同 CameraSession.ReleaseAsync()。
        /// </summary>
        public async Task ReleaseAsync()
        {
            // 在交給 background thread 之前，先於呼叫端執行緒停止 Timer。
            // WinForms Timer.Tick 在 UI thread 執行，若 FreeCameras 在 background thread 執行，
            // Stop() 必須先呼叫，否則 Tick 可能在 cam.Free() 期間存取同一台相機資源。
            _cameraStatusTimer.Stop();
            IsReleasing = true;
            await Task.Run(() => FreeCameras());
        }

        // ==================== Settings ====================

        public void SetImageProcessingEnabled(bool enable)
        {
            foreach (var cam in _cameras)
                cam.EnableImageProcessing = enable;
        }

        /// <summary>即時顯示方向（"v"/"h"）套到所有相機，控制 grab hook 顯示 V 或 H ridge。</summary>
        public void SetLiveDisplayDirection(string dir)
        {
            foreach (var cam in _cameras)
                cam.LiveDisplayDirection = dir;
        }

        public void SetScreenMmPerPixel(double mmPerPx) => _screenMmPerPx = mmPerPx;

        public void SetCaptureSettings(InspectionSettings settings)
        {
            if (settings == null) return;
            _inspectionSettings = settings;

            UpdateCaptureSettingsCache(settings);

            foreach (var cam in _cameras)
            {
                int camIdx = cam.CameraId - 1;
                cam.EnableAutoCapture    = _enableAutoCapture;
                cam.SaveOriginalBmp = _saveOriginalBmp;
                cam.CaptureRootPath      = _captureRootPath;
                cam.CameraGrabHeight     = _cameraGrabHeight[camIdx];
                cam.HessianSigma         = InspectionEngineConfig.DefaultRidgeSigma;
                cam.HessianFixedMax      = _hessianMaxFactor;
                cam.RidgeMode            = _ridgeMode;
                cam.SaveResizeScale      = _saveResizeScale;
                cam.SaveJpgQuality       = _saveJpgQuality;
                cam.TimestampCoordinator = _timestampCoordinator;

                // 曝光：走 CLProtocol-aware SetExposureUs（CLProtocol 未就緒時記錄，就緒後自動重套）
                cam.SetExposureUs(_cameraExposureTimeUs[camIdx]);
                // 線掃速率：同上，CLProtocol 未就緒時記錄，就緒後自動重套
                cam.SetLineRateHz(_cameraLineRateHz[camIdx]);
            }
        }

        /// <summary>
        /// 對指定相機設定曝光時間（μs）。CLProtocol 就緒後走 Feature API，否則 fallback MdigControl。
        /// 同 CameraSession.SetExposureForCamera()。
        /// </summary>
        public void SetExposureForCamera(int camId, double exposureUs)
        {
            FindCamera(camId)?.SetExposureUs(exposureUs);
        }

        /// <summary>
        /// 對指定相機設定 Line Rate（Hz），需 CLProtocol 已啟用。
        /// 同 CameraSession.SetLineRateForCamera()。
        /// </summary>
        public void SetLineRateForCamera(int camId, double hz)
        {
            FindCamera(camId)?.SetLineRateHz(hz);
        }

        /// <summary>
        /// 對指定相機變更 Grab 高度（px）。
        /// 內部走 Stop → Free → Realloc → Restart 完整流程（由 AniloxCamera.SetGrabHeight 處理）。
        /// 同 CameraSession.SetGrabHeightForCamera()。
        /// </summary>
        public void SetGrabHeightForCamera(int camId, int height)
        {
            FindCamera(camId)?.SetGrabHeight(height);
        }

        private AniloxCamera FindCamera(int camId)
        {
            for (int i = 0; i < _cameras.Count; i++)
                if (_cameras[i].CameraId == camId) return _cameras[i];
            return null;
        }

        private void UpdateCaptureSettingsCache(InspectionSettings settings)
        {
            if (settings == null) return;
            _enableAutoCapture    = settings.EnableAutoCapture;
            _saveOriginalBmp = settings.Storage?.SaveOriginalBmp ?? false;
            _captureRootPath      = settings.CaptureRootPath ?? string.Empty;
            _cameraGrabHeight     = settings.Acquisition.CameraGrabHeight;
            _cameraExposureTimeUs = settings.Acquisition.CameraExposureTimeUs;
            _cameraLineRateHz     = settings.Acquisition.CameraLineRateHz;
            _saveResizeScale      = settings.Recipe?.SaveResizeScale ?? InspectionEngineConfig.DefaultSaveResizeScale;
            _saveJpgQuality       = settings.Recipe?.SaveJpgQuality  ?? InspectionEngineConfig.DefaultSaveJpgQuality;
            // capture-time HM 用 V（baked 進 .bin）；H 為 view-time only，不送進 native
            _hessianMaxFactor     = settings.HessianMaxFactorV > 0
                ? settings.HessianMaxFactorV
                : InspectionEngineConfig.DefaultHessianMaxFactor;
            _ridgeMode            = InspectionRecipe.RidgeDirectionToNative(settings.RidgeDir);
            _dcfPath              = settings.DcfPath ?? string.Empty;
        }

        // ==================== Display Switching ====================

        /// <summary>
        /// <summary>
        /// 重新套用目前選定相機的主顯示，用於 SetGrabHeight 後重新綁定畫面。
        /// </summary>
        public void RefreshMainDisplay()
        {
            SwitchMainDisplay(_selectedMainCameraId);
        }

        /// <summary>重置主顯示器（MIL secondary display）的縮放/平移為 fit-to-window。</summary>
        public void ResetMainDisplayView()
        {
            if (IsGlobalMergeActive && _mergedDisplay != MIL.M_NULL)
            {
                try
                {
                    MIL.MdispControl(_mergedDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                    MIL.MdispControl(_mergedDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
                }
                catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.ResetView] {ex.GetType().Name}: {ex.Message}"); }
                return;
            }
            var cam = _cameras.Find(c => c.CameraId == _selectedMainCameraId);
            cam?.ResetSecondaryDisplayView();
        }

        private void OnLivePanelPaint(object sender, PaintEventArgs e, int cameraIndex)
        {
            if (!(sender is Panel panel)) return;
            bool isSelected = cameraIndex == _selectedMainCameraId;
            Color borderColor = isSelected ? Color.Orange : Color.FromArgb(60, 60, 60);
            int   borderWidth = isSelected ? 3 : 1;
            ControlPaint.DrawBorder(e.Graphics, panel.ClientRectangle,
                borderColor, borderWidth, ButtonBorderStyle.Solid,
                borderColor, borderWidth, ButtonBorderStyle.Solid,
                borderColor, borderWidth, ButtonBorderStyle.Solid,
                borderColor, borderWidth, ButtonBorderStyle.Solid);
        }

        // ── SmartCanvas 顯示路徑橋接 ──
        /// <summary>SmartCanvas 模式且尚未建立 → 在 camLiveMain 疊 SmartCanvas + 訂閱各相機每幀 bytes（冪等）。
        /// 在「相機配置」與「開始抓取」都呼叫 → 切設定後重開抓取即生效，不必重啟程式。</summary>
        private void EnsureSmartDisplay()
        {
            if (!SmartCanvasMode || _smartDisplay != null) return;
            if (_mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) return;
            _smartDisplay = new LiveDisplayView(_mainDisplayPanel, _cameraPanels, _screenMmPerPx);
            _smartDisplay.SelectRequested  += SmartSelectCamera;
            // 反向連動（合圖視野移動 → sdk 已自動高亮縮圖）：只同步 app 選中狀態，不走 SwitchMainDisplay（防重載/遞迴）
            _smartDisplay.SelectedCamChanged += camId => _selectedMainCameraId = camId;
            _smartDisplay.ViewRangeMmChanged += OnSmartViewRange;
            _smartDisplay.SetSelected(_selectedMainCameraId);
            if (IsGlobalMergeActive && _merger != null)
            {
                var ops = new double[_merger.SlotStartsMm?.Length ?? 0];
                for (int i = 0; i < ops.Length; i++) ops[i] = _merger.RefOpsMm * 1000.0; // 均勻 ops（µm）
                _smartDisplay.SetLayout(_merger.SlotStartsMm, ops, 1, RowPitchMm); // 主程式餵全解析度顯示 bytes → feedScale=1
            }
            _smartDisplay.MergeAll = IsGlobalMergeActive;     // 全域＝合圖全部（含無畫面相機黑占位）
            _smartDisplay.SetMergeMode(IsGlobalMergeActive);
            foreach (var cam in _cameras) cam.OnDisplayFrame += OnCameraDisplayFrame;
            if (_inspectionSettings != null) SetLodMode(_inspectionSettings.LiveLod); // 套目前 LOD 設定
        }

        /// <summary>套用動態 LOD 模式到 LiveDisplayView（he_LiveLod 變更 / 顯示建立時呼叫）。</summary>
        public void SetLodMode(LiveLodMode mode)
        {
            if (_smartDisplay == null) return;
            switch (mode)
            {
                case LiveLodMode.GPU: _lodReleased = false; _smartDisplay.EnableLod(LodResizeGpu); break;
                case LiveLodMode.CPU: _smartDisplay.EnableLod(GrayResizeCpu.Resize); break;
                default:              _smartDisplay.DisableLod(); break;
            }
        }

        /// <summary>GPU LOD resize 委派（LiveDisplayView 背景執行緒呼叫；只縮「可見區」一塊）。</summary>
        private byte[] LodResizeGpu(byte[] src, int sw, int sh, int dw, int dh)
        {
            int srcPix = sw * sh, dstPix = dw * dh;
            byte[] dst;
            lock (_lodBufLock)
            {
                if (_lodReleased) return null;
                if (_lodSrcCap < srcPix)
                {
                    if (_lodSrcPinned != IntPtr.Zero) NativeMethods.TanukiCv_FreePinned(_lodSrcPinned);
                    _lodSrcPinned = NativeMethods.TanukiCv_AllocPinned((ulong)srcPix); _lodSrcCap = srcPix;
                }
                if (_lodDstCap < dstPix)
                {
                    if (_lodDstPinned != IntPtr.Zero) NativeMethods.TanukiCv_FreePinned(_lodDstPinned);
                    _lodDstPinned = NativeMethods.TanukiCv_AllocPinned((ulong)dstPix); _lodDstCap = dstPix;
                }
                if (_lodSrcPinned == IntPtr.Zero || _lodDstPinned == IntPtr.Zero) return null;
                System.Runtime.InteropServices.Marshal.Copy(src, 0, _lodSrcPinned, srcPix);
                NativeMethods.TanukiCv_Resize_GPU(_lodSrcPinned, sw, sh, _lodDstPinned, dw, dh);
                dst = new byte[dstPix];
                System.Runtime.InteropServices.Marshal.Copy(_lodDstPinned, dst, 0, dstPix);
            }
            return dst;
        }

        /// <summary>切回 MIL 模式（he_MainDisplay==MilDirect）→ 解訂閱 + dispose SmartCanvas，露出底層 MIL。</summary>
        private void TeardownSmartDisplay()
        {
            if (_smartDisplay == null) return;
            foreach (var cam in _cameras) cam.OnDisplayFrame -= OnCameraDisplayFrame;
            _smartDisplay.Dispose();
            _smartDisplay = null;
            // LOD pinned 釋放（鎖內 + 旗標，等背景 provider 用完防 use-after-free）
            lock (_lodBufLock)
            {
                _lodReleased = true;
                if (_lodSrcPinned != IntPtr.Zero) { NativeMethods.TanukiCv_FreePinned(_lodSrcPinned); _lodSrcPinned = IntPtr.Zero; _lodSrcCap = 0; }
                if (_lodDstPinned != IntPtr.Zero) { NativeMethods.TanukiCv_FreePinned(_lodDstPinned); _lodDstPinned = IntPtr.Zero; _lodDstCap = 0; }
            }
        }

        private void OnCameraDisplayFrame(int camId, byte[] bytes, int w, int h) => _smartDisplay?.PushFrame(camId, bytes, w, h);
        private void SmartSelectCamera(int camId) => SwitchMainDisplay(camId);
        /// <summary>監控主畫面（LiveDisplayView）縮放/平移 → 把可見範圍轉給 form 連動 live 曲線圖
        /// （切向/overview 用 X 範圍、法向用 Y 範圍）。bin↔主畫面對齊。</summary>
        private void OnSmartViewRange(double leftMm, double rightMm, double topMm, double botMm)
            => OnLiveViewRange?.Invoke(leftMm, rightMm, topMm, botMm);

        /// <summary>監控主畫面可見範圍變更（leftX, rightX, topY, botY mm）→ form 訂閱、連動 live 曲線圖 zoom。</summary>
        public event Action<double, double, double, double> OnLiveViewRange;

        private void SwitchMainDisplay(int cameraIndex)
        {
            // 關閉/釋放期間 form 或 panel 可能已 dispose；存取 .Handle 會觸發 CreateHandle()
            // 而拋 ObjectDisposedException（FreeCameras→DisableGlobalMerge→此處的崩潰路徑）。
            if (_mainForm == null || _mainForm.IsDisposed
                || _mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) return;

            if (_mainForm.InvokeRequired)
            {
                try { _mainForm.BeginInvoke(new Action(() => SwitchMainDisplay(cameraIndex))); }
                catch (InvalidOperationException) { /* ObjectDisposedException 亦繼承自此 */ }
                return;
            }

            _selectedMainCameraId = cameraIndex;
            _userSelectedMainCameraId = cameraIndex;
            _smartDisplay?.SetSelected(cameraIndex);

            foreach (var kvp in _liveParentPanels)
                kvp.Value.Invalidate();

            // Global merge 時主畫面由合併 display 控制，不切換單台；但 pan 到相機中心
            if (IsGlobalMergeActive)
            {
                PanMergedDisplayToCameraCenter(cameraIndex);
                return;
            }

            // SmartCanvas 模式：主畫面由 LiveDisplayView 顯示，不綁 MIL secondary display 到 camLiveMain
            // （MIL display 的 M_MOUSE_USE 會攔截滾輪，疊在上面的 SmartCanvas 無法縮放）。一律卸成 IntPtr.Zero。
            foreach (var cam in _cameras)
            {
                if (!SmartCanvasMode && cam.CameraId == cameraIndex)
                    cam.SetSecondaryDisplay(_mainDisplayPanel.Handle);
                else
                    cam.SetSecondaryDisplay(IntPtr.Zero);
            }
        }

        // ==================== Global Merge ====================

        /// <summary>啟用即時全域合圖：工頭算佈局 + 分配合併 buffer + 設每台 merge target；本類別綁 display 顯示。</summary>
        public void EnableGlobalMerge(double[] opsUm, double[] startPosMm)
        {
            if (IsGlobalMergeActive || _cameras.Count == 0) return;

            // 「拼」委派工頭：傳入底層 MilCamera 清單（空缺槽以 MaxWidth 作為標準寬度算全域範圍）
            var mils = new List<MilCamera>(_cameras.Count);
            foreach (var cam in _cameras) mils.Add(cam.Mil);

            _merger = new MultiCameraMerger(mils);
            if (!_merger.EnableMerge(opsUm, startPosMm, InspectionEngineConfig.MaxWidth))
            {
                _merger = null;
                return;
            }

            MIL_ID sysId = _cameras[0].OwnerSystemId;
            if (sysId == MIL.M_NULL) { _merger.DisableMerge(); _merger = null; return; }

            // 解除所有相機的 secondary display，改用合併 display
            foreach (var cam in _cameras)
                cam.SetSecondaryDisplay(IntPtr.Zero);

            // 從工頭同步座標系參數（供滑鼠回呼 + overview 計算）
            SyncCoordsFromMerger();

            // panel 已 dispose（關閉/釋放期）→ 不碰 .Handle（會觸發 CreateHandle/ObjectDisposedException）
            if (_mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) { _merger.DisableMerge(); _merger = null; return; }

            // SmartCanvas 模式：合圖由 LiveDisplayView CPU 拼，不需 MIL 合圖 display。
            // 關鍵：不把 MIL display 綁到 camLiveMain（否則 MIL display 的 M_MOUSE_USE 會攔截滾輪，
            // 疊在上面的 SmartCanvas 收不到 → 無法縮放）。MIL 直繪模式才走下面整套。
            if (!SmartCanvasMode)
            {
                MIL.MdispAlloc(sysId, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref _mergedDisplay);
                // MdispAlloc 失敗(多 board/資源不足)→ M_NULL，後續 MdispControl/SelectWindow 對 M_NULL 會 MIL 報錯
                if (_mergedDisplay == MIL.M_NULL)
                {
                    System.Diagnostics.Trace.TraceWarning("[LiveCameraManager.EnableGlobalMerge] MdispAlloc 失敗（合圖 display）");
                    _merger.DisableMerge(); _merger = null; return;
                }

                // 先關自動刷新「再」select window：避免 select 瞬間把 grab hook 尚未貼滿的合併 buffer
                // 顯示出來（半貼狀態 → 橫條殘影閃一下）。改由 33ms timer 手動刷新，確保上螢幕時已較完整。
                MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_DISABLE);
                MIL.MdispSelectWindow(_mergedDisplay, _merger.MergedBuffer, _mainDisplayPanel.Handle);
                MIL.MdispControl(_mergedDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                MIL.MdispControl(_mergedDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
                MIL.MdispControl(_mergedDisplay, MIL.M_MOUSE_USE, MIL.M_ENABLE);

                // 改用定時器手動刷新（~30fps），確保每次顯示的是所有相機的最新合成結果
                _mergedDisplayTimer = new Timer { Interval = 33 };
                _mergedDisplayTimer.Tick += MergedDisplayTimer_Tick;
                _mergedDisplayTimer.Start();

                // Hook 滑鼠移動 → 更新 lblPixelInfo
                _mergedMouseDelegate = new MIL_DISP_HOOK_FUNCTION_PTR(MergedMouseStatusHandler);
                MIL.MdispHookFunction(_mergedDisplay, MIL.M_MOUSE_MOVE, _mergedMouseDelegate, IntPtr.Zero);
            }

            IsGlobalMergeActive = true;

            // SmartCanvas 合圖：用工頭佈局(各台 start/ops) CPU 拼（feedScale=1：主程式餵全解析度）
            if (SmartCanvasMode && _smartDisplay != null)
            {
                _smartDisplay.SetLayout(startPosMm, opsUm, 1, RowPitchMm);
                _smartDisplay.MergeAll = true;   // 全域＝合圖全部（含無畫面相機黑占位）
                _smartDisplay.SetMergeMode(true);
            }
        }

        /// <summary>從工頭同步座標系參數到本地鏡像欄位（值來源 = 工頭）。</summary>
        private void SyncCoordsFromMerger()
        {
            if (_merger == null) return;
            _mergedMinStartMm   = _merger.MinStartMm;
            _mergedRefOpsMm     = _merger.RefOpsMm;
            _mergedTotalW       = _merger.TotalW;
            _mergedTotalH       = _merger.TotalH;
            _mergedSlotStartsMm = _merger.SlotStartsMm;
            _mergedSlotEndsMm   = _merger.SlotEndsMm;
        }

        /// <summary>停用即時全域合圖：本類別釋放 display，工頭釋放合併 buffer + 清各相機 merge target。</summary>
        public void DisableGlobalMerge()
        {
            if (!IsGlobalMergeActive) return;

            // 停止定時刷新（顯示職責，留本類別）
            if (_mergedDisplayTimer != null)
            {
                _mergedDisplayTimer.Stop();
                _mergedDisplayTimer.Dispose();
                _mergedDisplayTimer = null;
            }

            // Unhook 滑鼠 + 解除 display 綁定（必須在工頭 MbufFree 合併 buffer 之前）
            if (_mergedDisplay != MIL.M_NULL)
            {
                if (_mergedMouseDelegate != null)
                    MIL.MdispHookFunction(_mergedDisplay, MIL.M_MOUSE_MOVE + MIL.M_UNHOOK,
                        _mergedMouseDelegate, IntPtr.Zero);
                MIL.MdispSelectWindow(_mergedDisplay, MIL.M_NULL, IntPtr.Zero);
                MIL.MdispFree(_mergedDisplay);
                _mergedDisplay = MIL.M_NULL;
            }
            _mergedMouseDelegate = null;

            // 「拆」委派工頭：清各相機 merge target + 釋放合併 buffer
            _merger?.DisableMerge();
            _merger = null;

            _smartDisplay?.SetMergeMode(false); // SmartCanvas 回單相機

            IsGlobalMergeActive = false;
            _mergedSlotStartsMm = null;
            _mergedSlotEndsMm   = null;

            // 恢復使用者明確點選的相機 secondary display（_selectedMainCameraId 可能已被視野中心 timer 改寫）
            SwitchMainDisplay(_userSelectedMainCameraId);
        }

        /// <summary>OPS/Start 變更時，重新計算全域合圖佈局（下一幀生效）。運算委派工頭，顯示重綁留本類別。</summary>
        public void RefreshGlobalMergeLayout(double[] opsUm, double[] startPosMm)
        {
            if (!IsGlobalMergeActive || _merger == null || _cameras.Count == 0) return;
            if (_mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) return; // 關閉期不碰 .Handle

            // 「拼」委派工頭（暫停合併 → 重算佈局 → 視需要重分配 buffer → 重設 merge target）
            // 回傳 true 表示合併 buffer 已重新分配，display 需重綁。
            bool reallocated = _merger.RefreshLayout(opsUm, startPosMm, InspectionEngineConfig.MaxWidth);

            // 「秀」：buffer 重分配時，本類別重新 MdispSelectWindow 綁定新 buffer handle
            if (reallocated && _mergedDisplay != MIL.M_NULL)
            {
                MIL.MdispSelectWindow(_mergedDisplay, MIL.M_NULL, IntPtr.Zero);
                MIL.MdispSelectWindow(_mergedDisplay, _merger.MergedBuffer, _mainDisplayPanel.Handle);
                MIL.MdispControl(_mergedDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
            }

            // 從工頭同步座標系參數
            SyncCoordsFromMerger();

            // SmartCanvas 合圖佈局同步（feedScale=1：主程式餵全解析度顯示 bytes）
            if (SmartCanvasMode && _smartDisplay != null)
                _smartDisplay.SetLayout(startPosMm, opsUm, 1, RowPitchMm);
        }

        // ==================== Merged Display Refresh ====================

        private void MergedDisplayTimer_Tick(object sender, EventArgs e)
        {
            if (_mergedDisplay == MIL.M_NULL) return;
            try { MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_NOW); }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.MergedDisplayTimer] {ex.GetType().Name}: {ex.Message}"); }
            UpdateSelectedCameraFromViewCenter();
        }

        private void PanMergedDisplayToCameraCenter(int camIdx)
        {
            if (_mergedDisplay == MIL.M_NULL || _mergedSlotStartsMm == null) return;
            int i = camIdx - 1;
            if (i < 0 || i >= _mergedSlotStartsMm.Length) return;
            try
            {
                double centerMm = (_mergedSlotStartsMm[i] + _mergedSlotEndsMm[i]) / 2.0;
                double centerPx = PixelMmMapper.MmToPixel(centerMm, _mergedMinStartMm, _mergedRefOpsMm);
                double zoomX = 0, panY = 0;
                MIL.MdispInquire(_mergedDisplay, MIL.M_ZOOM_FACTOR_X, ref zoomX);
                MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_Y, ref panY);
                if (zoomX <= 0) return;
                double viewW  = _mainDisplayPanel.Width / zoomX;
                double newPanX = Math.Max(0, Math.Min(_mergedTotalW - viewW, centerPx - viewW / 2.0));
                MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_DISABLE);
                MIL.MdispControl(_mergedDisplay, MIL.M_CENTER_DISPLAY, MIL.M_DISABLE);
                MIL.MdispPan(_mergedDisplay, newPanX, panY);
                MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_ENABLE);
            }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.PanToCenter] {ex.GetType().Name}: {ex.Message}"); }
        }

        private void UpdateSelectedCameraFromViewCenter()
        {
            if (_mergedSlotStartsMm == null) return;
            if (!TryGetMergedViewRange(out double leftMm, out double rightMm)) return;
            double centerMm = (leftMm + rightMm) / 2.0;
            int bestIdx = 0;
            double bestDist = double.MaxValue;
            for (int i = 0; i < _mergedSlotStartsMm.Length; i++)
            {
                double dist = Math.Abs(centerMm - (_mergedSlotStartsMm[i] + _mergedSlotEndsMm[i]) / 2.0);
                if (dist < bestDist) { bestDist = dist; bestIdx = i; }
            }
            int newId = bestIdx + 1;
            if (newId == _selectedMainCameraId) return;
            _selectedMainCameraId = newId;
            foreach (var kvp in _liveParentPanels)
                kvp.Value.Invalidate();
        }

        // ==================== Merged Display Mouse ====================

        private MIL_INT MergedMouseStatusHandler(MIL_INT HookType, MIL_ID EventId, IntPtr UserPtr)
        {
            MIL_ID mergedBuffer = _merger?.MergedBuffer ?? MIL.M_NULL;
            if (mergedBuffer == MIL.M_NULL) return MIL.M_NULL;

            double posX = 0, posY = 0;
            MIL.MdispGetHookInfo(EventId, MIL.M_MOUSE_POSITION_BUFFER_X, ref posX);
            MIL.MdispGetHookInfo(EventId, MIL.M_MOUSE_POSITION_BUFFER_Y, ref posY);

            int x = (int)posX;
            int y = (int)posY;
            int pixelValue = -1;

            if (x >= 0 && x < _mergedTotalW && y >= 0)
            {
                try
                {
                    byte[] data = new byte[1];
                    MIL.MbufGet2d(mergedBuffer, x, y, 1, 1, data);
                    pixelValue = data[0];
                }
                catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.MergedMouseStatus] {ex.GetType().Name}: {ex.Message}"); }
            }

            HandleMergedMouseData(x, y, pixelValue);
            return MIL.M_NULL;
        }

        private void HandleMergedMouseData(int x, int y, int pixelValue)
        {
            // MIL display hook 執行緒回 UI；關閉/釋放期 form 已 dispose → 守 guard 防 InvalidOperationException
            if (IsReleasing || _mainForm == null || _mainForm.IsDisposed || !_mainForm.IsHandleCreated) return;
            if (_mainForm.InvokeRequired)
            {
                try { _mainForm.BeginInvoke(new Action(() => HandleMergedMouseData(x, y, pixelValue))); }
                catch (InvalidOperationException) { /* ObjectDisposedException 亦繼承自此 */ }
                return;
            }

            string infoText;
            if (pixelValue == -1)
            {
                infoText = "即時影像 [全域合圖] | 游標超出影像範圍";
            }
            else
            {
                double physicalX = PixelMmMapper.PixelToMm(x, _mergedMinStartMm, _mergedRefOpsMm);

                var s = _inspectionSettings;
                double lineRateHz = (_cameraLineRateHz.Length > 0) ? _cameraLineRateHz[0] : 0;
                double speedMPerMin = s?.AniloxRollSpeedMPerMin ?? 0;
                double rowPitchMm = (speedMPerMin > 0 && lineRateHz > 0)
                    ? (speedMPerMin / 60.0 * 1000.0) / lineRateHz : 0;
                double physicalY = y * rowPitchMm;

                // 合併 display zoom/pan → 視野範圍
                string rangeStr = "";
                string magStr = "-";
                if (TryGetMergedViewRange(out double viewLeftMm, out double viewRightMm))
                {
                    rangeStr = $"X範圍:{viewLeftMm:F1}~{viewRightMm:F1} mm | ";

                    double zoomX = 0;
                    try { MIL.MdispInquire(_mergedDisplay, MIL.M_ZOOM_FACTOR_X, ref zoomX); }
                    catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.MergedMouseStatus.ZoomInquire] {ex.GetType().Name}: {ex.Message}"); }
                    if (zoomX > 0 && rowPitchMm > 0)
                    {
                        double panOffY = 0;
                        try { MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_Y, ref panOffY); }
                        catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.MergedMouseStatus.PanInquire] {ex.GetType().Name}: {ex.Message}"); }
                        double viewTopMm = panOffY * rowPitchMm;
                        double viewBotMm = (panOffY + _mainDisplayPanel.Height / zoomX) * rowPitchMm;
                        rangeStr += $"Y範圍:{viewTopMm:F1}~{viewBotMm:F1} mm | ";
                    }

                    if (zoomX > 0 && _screenMmPerPx > 0 && _mergedRefOpsMm > 0)
                    {
                        double physicalMag = PixelMmMapper.PhysicalMagnification(zoomX, _screenMmPerPx, _mergedRefOpsMm);
                        magStr = $"{physicalMag:F2}x";
                    }
                }

                infoText = $"即時影像 [全域合圖] | " +
                           $"位置:({physicalX:F2}, {physicalY:F2}) mm | " +
                           rangeStr +
                           $"座標: ({x}, {y}) | " +
                           $"亮度: {pixelValue} | " +
                           $"實體倍率:{magStr}";
            }

            _updatePixelInfoCallback?.Invoke(infoText);
        }

        /// <summary>取得合併 display 的 X 視野範圍（mm），供 overview chart 聯動。</summary>
        public bool TryGetMergedViewRange(out double leftMm, out double rightMm)
        {
            leftMm = rightMm = 0;
            if (!IsGlobalMergeActive || _mergedDisplay == MIL.M_NULL) return false;
            try
            {
                double zoomX = 0, panX = 0;
                MIL.MdispInquire(_mergedDisplay, MIL.M_ZOOM_FACTOR_X, ref zoomX);
                MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_X, ref panX);
                if (zoomX <= 0) return false;

                double pixelLeft  = panX;
                double pixelRight = panX + _mainDisplayPanel.Width / zoomX;
                leftMm  = PixelMmMapper.PixelToMm(pixelLeft,  _mergedMinStartMm, _mergedRefOpsMm);
                rightMm = PixelMmMapper.PixelToMm(pixelRight, _mergedMinStartMm, _mergedRefOpsMm);
                return true;
            }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.TryGetMergedViewRange] {ex.GetType().Name}: {ex.Message}"); return false; }
        }

        /// <summary>取得合併 display 的 Y 視野範圍（pixel），供法向曲線圖聯動。</summary>
        public bool TryGetMergedViewRangeY(out double topPixel, out double botPixel)
        {
            topPixel = botPixel = 0;
            if (!IsGlobalMergeActive || _mergedDisplay == MIL.M_NULL) return false;
            try
            {
                double zoomY = 0, panY = 0;
                MIL.MdispInquire(_mergedDisplay, MIL.M_ZOOM_FACTOR_Y, ref zoomY);
                MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_Y, ref panY);
                if (zoomY <= 0) return false;

                topPixel = panY;
                botPixel = panY + _mainDisplayPanel.Height / zoomY;
                return true;
            }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.TryGetMergedViewRangeY] {ex.GetType().Name}: {ex.Message}"); return false; }
        }

        // ==================== Mouse Data ====================

        private void HandleMouseDataChanged(int camId, int x, int y, int pixelValue)
        {
            // MIL display hook 執行緒回 UI；關閉/釋放期 form 已 dispose → 守 guard 防 InvalidOperationException
            if (IsReleasing || _mainForm == null || _mainForm.IsDisposed || !_mainForm.IsHandleCreated) return;
            if (_mainForm.InvokeRequired)
            {
                try { _mainForm.BeginInvoke(new Action(() => HandleMouseDataChanged(camId, x, y, pixelValue))); }
                catch (InvalidOperationException) { /* ObjectDisposedException 亦繼承自此 */ }
                return;
            }

            string infoText;
            if (pixelValue == -1)
            {
                infoText = $"即時影像 [CAM {camId}] | 游標超出影像範圍";
            }
            else
            {
                int camIdx = camId - 1;
                var s = _inspectionSettings;
                double[] opsUmArr  = s?.GetCameraOpsUmArray();
                double[] startMmArr = s?.GetCameraStartPositionMmArray();

                if (opsUmArr == null || camIdx < 0 || camIdx >= opsUmArr.Length)
                {
                    infoText = $"即時影像 [CAM {camId}] | 座標: ({x}, {y}) | 亮度: {pixelValue}";
                }
                else
                {
                    double opsInMm    = opsUmArr[camIdx] / 1000.0;
                    double startPosMm = startMmArr[camIdx];
                    double physicalX  = PixelMmMapper.PixelToMm(x, startPosMm, opsInMm);
                    double lineRateHz = (camIdx < _cameraLineRateHz.Length) ? _cameraLineRateHz[camIdx] : 0;
                    double speedMPerMin = s.AniloxRollSpeedMPerMin;
                    double rowPitchMm = (speedMPerMin > 0 && lineRateHz > 0)
                        ? (speedMPerMin / 60.0 * 1000.0) / lineRateHz : 0;
                    double physicalY  = y * rowPitchMm;

                    // MIL display zoom/pan → 視野範圍
                    string rangeStr = "";
                    string magStr = "-";
                    var cam = _cameras.Find(c => c.CameraId == camId);
                    if (cam != null && cam.TryGetSecondaryDisplayGeometry(
                            out double zoomX, out _, out double panOffX, out double panOffY))
                    {
                        double panelW = _mainDisplayPanel.Width;
                        double panelH = _mainDisplayPanel.Height;
                        double viewLeftMm  = PixelMmMapper.PixelToMm(panOffX, startPosMm, opsInMm);
                        double viewRightMm = PixelMmMapper.PixelToMm(panOffX + panelW / zoomX, startPosMm, opsInMm);
                        rangeStr = $"X範圍:{viewLeftMm:F1}~{viewRightMm:F1} mm | ";

                        if (rowPitchMm > 0)
                        {
                            double viewTopMm = panOffY * rowPitchMm;
                            double viewBotMm = (panOffY + panelH / zoomX) * rowPitchMm;
                            rangeStr += $"Y範圍:{viewTopMm:F1}~{viewBotMm:F1} mm | ";
                        }

                        if (_screenMmPerPx > 0 && opsInMm > 0)
                        {
                            double physicalMag = PixelMmMapper.PhysicalMagnification(zoomX, _screenMmPerPx, opsInMm);
                            magStr = $"{physicalMag:F2}x";
                        }
                    }

                    infoText = $"即時影像 [CAM {camId}] | " +
                               $"位置:({physicalX:F2}, {physicalY:F2}) mm | " +
                               rangeStr +
                               $"座標: ({x}, {y}) | " +
                               $"亮度: {pixelValue} | " +
                               $"實體倍率:{magStr}";
                }
            }

            _updatePixelInfoCallback?.Invoke(infoText);
        }

        // ==================== Status Timer ====================

        /// <summary>
        /// 每 500ms 輪詢相機連線狀態並自動重啟抓圖，同 CameraSession.UpdatePresence()。
        /// IsReleasing = true 時提早返回，防止存取已釋放的相機資源。
        /// 使用快照（ToArray）避免 background FreeCameras 呼叫 _cameras.Clear() 時導致 InvalidOperationException。
        /// </summary>
        private void CameraStatusTimer_Tick(object sender, EventArgs e)
        {
            if (IsReleasing) return;

            // CLProtocol 背景初始化期間（分配後 ~2-10s）：UI 執行緒不可呼叫 CheckPresence（MdigInquire），
            // 否則與背景 CLProtocol enable（MdigControl）搶 MIL 內部鎖 → UI 執行緒卡在 tick 裡 →
            // 整個視窗凍結（拖不動）+ 顯示誤導的暫態連線數。就緒前完全跳過輪詢，UI 維持「初始化中」。
            // AreCamerasHwReady 只讀 _clProtocolInitDone 旗標（非 MIL 呼叫），不造成競爭。
            if (!AreCamerasHwReady) return;

            // 先拍快照：防止 ReleaseAsync 在 background thread 執行 _cameras.Clear() 時，
            // foreach 拋出 InvalidOperationException 或存取已釋放的相機物件。
            AniloxCamera[] snapshot;
            try { snapshot = _cameras.ToArray(); }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.CameraStatusTimer] {ex.GetType().Name}: {ex.Message}"); return; }

            foreach (var cam in snapshot)
            {
                if (IsReleasing) return; // 釋放流程已開始，立即中止

                bool isConnected = cam.CheckPresence();

                // 連線恢復且使用者希望抓圖時，自動重啟（同 CameraSession.UpdatePresence）
                if (isConnected && cam.UserWantsGrab && !cam.IsLive)
                    cam.ApplyGrabState();

                string statusText = isConnected
                    ? (cam.IsLive ? $"FPS: {cam.CurrentFps:F1}" : "就緒")
                    : "斷線";
                Color color = isConnected
                    ? (cam.IsLive ? Color.LightGreen : Color.Yellow)
                    : Color.Pink;

                UpdateSingleCameraStatus(cam.CameraId, statusText, color);
            }

            // 彙總連線數，變化時通知 UI
            int connected = 0;
            foreach (var cam in snapshot)
                if (cam.IsConnected) connected++;
            if (connected != ConnectedCameraCount)
            {
                ConnectedCameraCount = connected;
                OnCameraCountChanged?.Invoke(connected, ExpectedCameraCount);
            }

            // CLProtocol 全就緒 → 一次性通知 UI 解鎖「開始抓取」鈕
            if (!_hwReadyRaised && AreCamerasHwReady)
            {
                _hwReadyRaised = true;
                OnHwReady?.Invoke();
            }
        }

        // ==================== UI Helpers ====================

        private void UpdateCameraStatus(string statusText, Color color)
        {
            foreach (var pair in _cameraStatusLabels)
            {
                pair.Value.Text      = $"{pair.Key}: {statusText}";
                pair.Value.ForeColor = color;
            }
        }

        private void UpdateSingleCameraStatus(int cameraIndex, string statusText, Color color)
        {
            if (_cameraStatusLabels.TryGetValue(cameraIndex, out var label))
            {
                label.Text      = $"{cameraIndex}: {statusText}";
                label.ForeColor = color;
            }
        }

        // ==================== Physical Magnification 1x ====================

        /// <summary>設定主顯示 zoom 使實體倍率 = 1x（螢幕 1mm = 實際 1mm）。</summary>
        public void SetPhysicalMagnification1x()
        {
            if (!IsLiveGrabbing || _screenMmPerPx <= 0) return;

            if (IsGlobalMergeActive && _mergedDisplay != MIL.M_NULL)
            {
                if (_mergedRefOpsMm <= 0) return;
                double zoom1x = PixelMmMapper.OneToOneZoom(_mergedRefOpsMm, _screenMmPerPx);

                double cx = _mainDisplayPanel.Width / 2.0;
                double cy = _mainDisplayPanel.Height / 2.0;

                try
                {
                    double curZoom = 0, curPanX = 0, curPanY = 0;
                    MIL.MdispInquire(_mergedDisplay, MIL.M_ZOOM_FACTOR_X, ref curZoom);
                    MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_X, ref curPanX);
                    MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_Y, ref curPanY);
                    if (curZoom <= 0) curZoom = 1.0;

                    double imgCx = curPanX + cx / curZoom;
                    double imgCy = curPanY + cy / curZoom;
                    double newPanX = imgCx - cx / zoom1x;
                    double newPanY = imgCy - cy / zoom1x;

                    MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_DISABLE);
                    MIL.MdispControl(_mergedDisplay, MIL.M_CENTER_DISPLAY, MIL.M_DISABLE);
                    MIL.MdispZoom(_mergedDisplay, zoom1x, zoom1x);
                    MIL.MdispPan(_mergedDisplay, newPanX, newPanY);
                    MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_ENABLE);
                }
                catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.Set1xZoom.Merged] {ex.GetType().Name}: {ex.Message}"); }
                return;
            }

            int camIdx = _selectedMainCameraId - 1;
            var s = _inspectionSettings;
            double[] opsUmArr = s?.GetCameraOpsUmArray();
            if (opsUmArr == null || camIdx < 0 || camIdx >= opsUmArr.Length) return;

            double opsInMm = opsUmArr[camIdx] / 1000.0;
            if (opsInMm <= 0) return;

            // physicalMag = zoom * screenMmPerPx / opsInMm = 1  →  zoom = opsInMm / screenMmPerPx
            double zoom1xCam = PixelMmMapper.OneToOneZoom(opsInMm, _screenMmPerPx);

            var cam = _cameras.Find(c => c.CameraId == _selectedMainCameraId);
            if (cam == null) return;

            // 以面板中心為基準
            double cxCam = _mainDisplayPanel.Width / 2.0;
            double cyCam = _mainDisplayPanel.Height / 2.0;

            if (cam.TryGetSecondaryDisplayGeometry(out double curZoomCam, out _, out double curPanXCam, out double curPanYCam) && curZoomCam > 0)
            {
                double imgCx = curPanXCam + cxCam / curZoomCam;
                double imgCy = curPanYCam + cyCam / curZoomCam;
                double newPanX = imgCx - cxCam / zoom1xCam;
                double newPanY = imgCy - cyCam / zoom1xCam;
                cam.SetSecondaryDisplayZoom(zoom1xCam, newPanX, newPanY);
            }
            else
            {
                cam.SetSecondaryDisplayZoom(zoom1xCam, 0, 0);
            }
        }

        // ==================== Custom Wheel Zoom ====================

        internal void ApplyCustomZoom(int wheelDelta)
        {
            if (!IsLiveGrabbing) return;

            double zoomX, panX, panY;

            if (IsGlobalMergeActive && _mergedDisplay != MIL.M_NULL)
            {
                // Global merge 模式：zoom/pan 合併 display
                try
                {
                    zoomX = panX = panY = 0;
                    MIL.MdispInquire(_mergedDisplay, MIL.M_ZOOM_FACTOR_X, ref zoomX);
                    MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_X, ref panX);
                    MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_Y, ref panY);
                }
                catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.ApplyCustomZoom.Inquire] {ex.GetType().Name}: {ex.Message}"); return; }
                if (zoomX <= 0) zoomX = 1.0;

                double factor = wheelDelta > 0 ? 1.1 : (1.0 / 1.1);
                double newZoom = zoomX * factor;
                if (newZoom < 0.05) newZoom = 0.05;
                if (newZoom > 32.0) newZoom = 32.0;

                double cx = _mainDisplayPanel.Width / 2.0;
                double cy = _mainDisplayPanel.Height / 2.0;
                double imgX = panX + cx / zoomX;
                double imgY = panY + cy / zoomX;
                double newPanX = imgX - cx / newZoom;
                double newPanY = imgY - cy / newZoom;

                try
                {
                    MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_DISABLE);
                    MIL.MdispControl(_mergedDisplay, MIL.M_CENTER_DISPLAY, MIL.M_DISABLE);
                    MIL.MdispZoom(_mergedDisplay, newZoom, newZoom);
                    MIL.MdispPan(_mergedDisplay, newPanX, newPanY);
                    MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_ENABLE);
                }
                catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.ApplyCustomZoom.Apply] {ex.GetType().Name}: {ex.Message}"); }
                return;
            }

            var cam = _cameras.Find(c => c.CameraId == _selectedMainCameraId);
            if (cam == null) return;
            if (!cam.TryGetSecondaryDisplayGeometry(out zoomX, out _, out panX, out panY))
                return;

            double factor2 = wheelDelta > 0 ? 1.1 : (1.0 / 1.1);
            double newZoom2 = zoomX * factor2;
            if (newZoom2 < 0.05) newZoom2 = 0.05;
            if (newZoom2 > 32.0) newZoom2 = 32.0;

            // 以面板中心為縮放基準點
            double cx2 = _mainDisplayPanel.Width / 2.0;
            double cy2 = _mainDisplayPanel.Height / 2.0;
            double imgX2 = panX + cx2 / zoomX;
            double imgY2 = panY + cy2 / zoomX;
            double newPanX2 = imgX2 - cx2 / newZoom2;
            double newPanY2 = imgY2 - cy2 / newZoom2;

            cam.SetSecondaryDisplayZoom(newZoom2, newPanX2, newPanY2);
            OnAfterVerticalZoom?.Invoke();
        }

        /// <summary>攔截 camLiveMain 上的 WM_MOUSEWHEEL，用 1.1x 步長取代 MIL 預設的整數倍跳躍。</summary>
        private class WheelZoomFilter : IMessageFilter
        {
            private const int WM_MOUSEWHEEL = 0x020A;
            private readonly LiveCameraManager _mgr;

            public WheelZoomFilter(LiveCameraManager mgr) => _mgr = mgr;

            public bool PreFilterMessage(ref Message m)
            {
                if (m.Msg != WM_MOUSEWHEEL) return false;
                // SmartCanvas 模式：主畫面由 SmartCanvas 自己處理滾輪 zoom（無 MIL 巨圖 display）。
                // filter 不可攔截，否則滾輪被吃掉 → SmartCanvas 收不到 → camLiveMain「縮不動」
                // （雙三擊是點擊事件、不走此 filter，故一直有反應）。此 filter 只服務 MIL 直繪合圖縮放。
                if (_mgr.SmartCanvasMode) return false;
                if (!_mgr.IsLiveGrabbing) return false;

                var panel = _mgr._mainDisplayPanel;
                var screenPt = Cursor.Position;
                if (!panel.RectangleToScreen(panel.ClientRectangle).Contains(screenPt))
                    return false;

                int delta = (short)(m.WParam.ToInt64() >> 16);
                _mgr.ApplyCustomZoom(delta);
                return true; // 攔截訊息，不讓 MIL 處理
            }
        }
    }
}
