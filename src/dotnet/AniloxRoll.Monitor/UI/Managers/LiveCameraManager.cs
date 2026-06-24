using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
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
    public partial class LiveCameraManager
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
        private int[]    _cameraGrabHeight    = new int[7];   // 已 clamp 到 MaxGrabHeightPx（防超單 path 板載 stall）
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

            // 每板（SystemNum）台數＝同板共用板載的相機數，供 autoMax 計算（依拓樸，不管在線）。
            var boardCounts = new Dictionary<int, int>();
            foreach (var c in _cameraHardwareConfigs)
                boardCounts[c.SystemNum] = (boardCounts.TryGetValue(c.SystemNum, out var n) ? n : 0) + 1;

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

                // grab 高度走 json（_cameraGrabHeight 已在 UpdateCaptureSettingsCache clamp 到 MaxGrabHeightPx=12000）。
                cam.CameraGrabHeight = _cameraGrabHeight[camIdx];

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
                cam.Initialize();   // 拿已 clamp 的 CameraGrabHeight 配 buffer（與可行版本同路徑，不 stall）
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

            ApplyMainDisplayMode(); // 依 he_MainDisplay 套用：SmartCanvas / MilDirect / Waterfall（三選一互斥）

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
            // 切「主畫面顯示」設定後重開抓取即生效：SmartCanvas / MilDirect / Waterfall 三選一互斥
            ApplyMainDisplayMode();
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
                cam.CameraGrabHeight     = _cameraGrabHeight[camIdx]; // 已在 UpdateCaptureSettingsCache clamp 到 MaxGrabHeightPx
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
            // grab 中拉大到 ~12062 會 stall → 一律 cap 在 MaxGrabHeightPx(12000) 以下（per-camera 固定，不分台數）。
            if (height > AcquisitionDefaults.MaxGrabHeightPx) height = AcquisitionDefaults.MaxGrabHeightPx;
            FindCamera(camId)?.SetGrabHeight(height);
        }

        private int _coordDepth;

        /// <summary>協調式套用「單一相機」參數（#3 修正）：只停**該台**→寫→重啟該台，其他相機完全不碰。
        /// 相位已用 phaselog 證明不重要（free-run 2-3 條線就夠好）→ 不再全部一起停/開，
        /// 避免沒被改的相機（最常見 cam2）被反覆 stop/start 連累而 stall。
        /// 重入 / 非抓取中 / 該台未在抓 → 直接寫。</summary>
        public void ApplyParamCoordinated(int camId, Action write)
        {
            if (write == null) return;
            var cam = FindCamera(camId);
            if (cam == null || !IsLiveGrabbing || _coordDepth > 0 || !cam.IsLive) { write?.Invoke(); return; }

            _coordDepth++;
            try
            {
                cam.SetUserGrabIntent(false);   // 只停這一台
                write();
            }
            finally
            {
                if (cam.IsConnected) cam.SetUserGrabIntent(true);   // 只重啟這一台
                // 不在此同步等出幀：同步 Thread.Sleep 會凍 UI（stall 時必卡滿）→ Windows 變灰 Not Responding →
                // 拉滑桿被排隊 replay → 解凍後「跳到空拉位置」再觸發一次套用＝暴力漏洞。stop→write→start 本身
                // 同步序列化（MouseUp handler 跑完才處理下一個事件），不需凍結等待。stall 偵測交給 status timer。
                _coordDepth--;
            }
        }

        /// <summary>協調式套用參數（#3）：grab 中改任何相機參數時，**全部相機一起停 → 寫 → 一起重啟**，
        /// 讓重啟後相位 offset 一致重建（測「停→寫→開」對相位的效果；高度也順帶安全＝停著重配）。
        /// 重入保護：巢狀呼叫（如 All 一次套 7 台）只寫、不重複停/開。非抓取中 → 直接寫。
        /// 寫入時全部相機停著：SetGrabHeight 見 wasLive=false → 只重配 buffer 不自行重啟；曝光/線掃 live 寫即可。</summary>
        public void ApplyParamCoordinated(Action write)
        {
            if (write == null) return;
            if (!IsLiveGrabbing || _coordDepth > 0) { write(); return; }

            _coordDepth++;
            var paused = new List<AniloxCamera>();
            try
            {
                foreach (var cam in _cameras)
                    if (cam != null && cam.IsLive) { cam.SetUserGrabIntent(false); paused.Add(cam); }
                write();
            }
            finally
            {
                // 協調重啟：back-to-back 一起 M_START（All 一次套 7 台才用此版；單台走 camId overload）
                foreach (var cam in paused)
                    if (cam.IsConnected) cam.SetUserGrabIntent(true);

                // 不同步等出幀（會凍 UI → 變灰 Not Responding → 拉滑桿排隊 replay 跳值＝暴力漏洞）。
                // stall 偵測交給 status timer（FPS≈0 判據）。
                _coordDepth--;
            }
        }

        private AniloxCamera FindCamera(int camId)
        {
            for (int i = 0; i < _cameras.Count; i++)
                if (_cameras[i].CameraId == camId) return _cameras[i];
            return null;
        }

        /// <summary>改參數窗口：暫停/恢復「全部相機」存檔（不影響 grab/檢測/顯示）。
        /// 套用參數時設 true → 等全部相機恢復同步（UI 解鎖時）設 false → 存出的序列不含重啟空檔、各台齊全。</summary>
        public void SetCaptureSuppressed(bool suppressed)
        {
            foreach (var cam in _cameras)
                if (cam != null) cam.SuppressCapture = suppressed;
        }

        /// <summary>所有在抓相機中「最大的幀週期」(ms)＝FrameHeight/AppliedLineRateHz。
        /// 供改參數後的參數鎖：至少鎖住一個完整幀週期，確保改完的相機跑完一張乾淨幀才放行下一次改（防連改太快 stall）。</summary>
        public int GetMaxFramePeriodMs()
        {
            int maxMs = 0;
            foreach (var cam in _cameras)
            {
                if (cam == null || !cam.IsLive) continue;
                double lr = cam.AppliedLineRateHz;
                int h = cam.FrameHeight;
                if (lr > 0 && h > 0)
                {
                    int ms = (int)System.Math.Ceiling(h / lr * 1000.0);
                    if (ms > maxMs) maxMs = ms;
                }
            }
            return maxMs;
        }

        /// <summary>快照所有「正在抓」相機的 FrameCount（camId→count）；供 UI 參數鎖判「重啟後是否已恢復出幀」。</summary>
        public Dictionary<int, long> SnapshotLiveFrameCounts()
        {
            var d = new Dictionary<int, long>();
            foreach (var cam in _cameras)
                if (cam != null && cam.IsLive) d[cam.CameraId] = cam.GetFrameCount();
            return d;
        }

        /// <summary>baseline 內每台是否都已出至少一張新幀（FrameCount 前進）＝重啟落定。
        /// stalled 相機永不前進 → 回 false，呼叫端用逾時兜底解鎖。</summary>
        public bool AllAdvancedSince(Dictionary<int, long> baseCounts)
        {
            if (baseCounts == null) return true;
            foreach (var kv in baseCounts)
            {
                var cam = FindCamera(kv.Key);
                if (cam != null && cam.GetFrameCount() <= kv.Value) return false;
            }
            return true;
        }

        private void UpdateCaptureSettingsCache(InspectionSettings settings)
        {
            if (settings == null) return;
            _enableAutoCapture    = settings.EnableAutoCapture;
            _saveOriginalBmp = settings.Storage?.SaveOriginalBmp ?? false;
            _captureRootPath      = settings.CaptureRootPath ?? string.Empty;
            // grab 高度 clamp 到硬上限 MaxGrabHeightPx（grab 中拉大超過 ~12062 會 stall；per-camera 固定不分台數）。
            // clamp 後是純數字、走與 json 同路徑設給 cam.CameraGrabHeight（不碰 MIL init 查詢）→ 不會 stall。
            int maxH = AcquisitionDefaults.MaxGrabHeightPx;
            var srcH = settings.Acquisition.CameraGrabHeight;
            _cameraGrabHeight = new int[srcH.Length];
            for (int i = 0; i < srcH.Length; i++)
            {
                int h = srcH[i] > 0 ? srcH[i] : AcquisitionDefaults.GrabHeight;
                _cameraGrabHeight[i] = h > maxH ? maxH : h;
            }
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
    }
}
