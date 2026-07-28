using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;
using MilGrabber.Core;
using TanukiCv.Core; // PixelMmMapper（已收進 sdk 唯一來源）
using TanukiCv.Controls; // ImageDisplayView（共用多相機監控顯示元件）
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.Core.Interop; // NativeMethods（LOD GPU resize；P/Invoke 宣告唯一點）

namespace AniloxRoll.Monitor.UI.Managers
{
    public partial class LiveCameraManager
    {
        private List<AniloxCamera> _cameras = new List<AniloxCamera>();
        private List<CameraHardwareConfig> _cameraHardwareConfigs;
        private Dictionary<int, MIL_ID> _allocatedSystems = new Dictionary<int, MIL_ID>();

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
        private float _ridgeSigma = InspectionEngineConfig.DefaultRidgeSigma;   // 細線濾除（ridge_sigma）；設定改 → 下次 grab 生效
        private string _ridgeMode = InspectionEngineConfig.DefaultRidgeMode;
        private string _liveDisplayDirection = "v";
        private string _dcfPath = string.Empty;
        private readonly CaptureTimestampCoordinator _timestampCoordinator = new CaptureTimestampCoordinator();
        private readonly System.Threading.SemaphoreSlim _allocationGate =
            new System.Threading.SemaphoreSlim(1, 1);
        private volatile bool _isAllocating;
        private volatile bool _isParameterReconfiguring;
        private volatile bool _isCaptureSynchronizing;

        private const int CaptureSynchronizationMaxAttempts = 3;
        private const double CapturePhaseToleranceMs = 5.0;

        private sealed class CapturePhaseSample
        {
            public int CameraId;
            public int SystemNum;
            public long FrameStartTicks;
            public long ClockFrequencyHz;
            public int FrameHeight;
            public double AppliedLineRateHz;
        }

        private sealed class AcquisitionSyncResult
        {
            public bool Succeeded;
            public bool Canceled;
            public string Error;
            public List<CapturePhaseSample> Samples =
                new List<CapturePhaseSample>();
        }

        public bool IsAllocated    { get; private set; } = false;
        public bool IsAllocating => _isAllocating;
        public bool IsLiveGrabbing { get; private set; } = false;
        private volatile bool _captureGateOpen;

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
        /// <summary>
        /// Raised after reconfigured cameras have produced a new raw frame and immediately before
        /// the product-frame gate reopens. UI subscribers clear any curve state from the old
        /// parameter generation here.
        /// </summary>
        public event Action OnCaptureSequenceReset;

        public event Action OnMainContentPresented
        {
            add { _display.MainContentPresented += value; }
            remove { _display.MainContentPresented -= value; }
        }

        private bool _hwReadyRaised;

        /// <summary>所有相機 CLProtocol 是否就緒（已套曝光/線掃）。未就緒時上層應禁用「開始抓取」。</summary>
        public bool AreCamerasHwReady
        {
            get
            {
                if (!AreCameraParametersReady) return false;
                bool hasConnectedCamera = false;
                foreach (var cam in _cameras)
                {
                    if (!cam.IsConnected) continue;
                    hasConnectedCamera = true;
                    if (!cam.IsAcquisitionWarm) return false;
                }
                return hasConnectedCamera;
            }
        }

        private bool AreCameraParametersReady
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
        /// 參數：(cameraId, fileNameWithoutExt, meanPeak_0to1, maxPeak_0to1,
        /// maxCMean_0to1, meanRPeak_0to1, maxRPeak_0to1)</summary>
        public event Action<string, int, string, float, float, float, float, float> OnInspectionResult;

        /// <summary>每幀 GPU pipeline 完成後觸發（MIL 回呼執行緒）。
        /// 參數：(cameraId, curveMean_raw255, curveMax_raw255)</summary>
        public event Action<int, float[], float[]> OnLiveCurveData;

        /// <summary>每幀 GPU pipeline 完成後觸發（MIL 回呼執行緒）。
        /// 參數：(cameraId, rowCurveMean_raw255, rowCurveMax_raw255)</summary>
        public event Action<int, float[], float[]> OnLiveRowCurveData;

        /// <summary>
        /// Raised when every connected capture camera has completed at least the reported
        /// number of rows in the current product capture.
        /// </summary>
        public event Action<int> OnCaptureCommonRowsCompleted;

        /// <summary>存檔完成回呼：(cameraId, 已儲存檔案路徑陣列)。</summary>
        public Action<int, string[]> OnFilesSaved { get; set; }
        public Action<int, string> OnCaptureSaveFailed { get; set; }

        /// <summary>
        /// 正在執行釋放流程時為 true，防止 Timer Tick 在資源已釋放後繼續存取相機。
        /// 同 CameraSession.IsReleasing。
        /// </summary>
        public volatile bool IsReleasing = false;

        // --- Global merge（即時合圖）---
        // 「拼」（佈局 + 合併 buffer + 每台 merge target）委派 MultiCameraMerger 工頭（sdk/MIL），
        // 生命週期由 GlobalMergeCoordinator 擁有；「秀」一律 CPU（ImageDisplayView / WaterfallView 讀工頭佈局）。
        // 本類別只留編排 + forwarder。
        private readonly GlobalMergeCoordinator _globalMerge;
        private readonly LiveDisplayCoordinator _display;
        public bool IsGlobalMergeActive => _globalMerge.IsActive;

        private InspectionSettings _inspectionSettings;

        public LiveCameraManager(
            Form mainForm,
            Panel[] cameraPanels,
            Panel mainDisplayPanel)
        {
            if (cameraPanels == null)
                throw new ArgumentNullException(nameof(cameraPanels));
            if (cameraPanels.Length < 7)
                throw new ArgumentException("cameraPanels must contain at least 7 panels.", nameof(cameraPanels));

            _globalMerge = new GlobalMergeCoordinator();

            _display = new LiveDisplayCoordinator(
                mainForm, cameraPanels, mainDisplayPanel,
                _globalMerge,
                () => _cameras.AsReadOnly(),
                () => _inspectionSettings,
                () => _cameraLineRateHz,
                () => IsLiveGrabbing);

            _cameraHardwareConfigs = SystemSettings.CreateDefault().CameraDevices;

            _cameraStatusTimer = new Timer { Interval = 500 };
            _cameraStatusTimer.Tick += CameraStatusTimer_Tick;

            _display.UpdateCameraStatus("未配置", Color.Gray);
        }

        // ==================== Allocate ====================

        public async Task AllocateCamerasAsync(bool enableImageProcessing)
        {
            _display.SetWaterfallDisplayLayer(ToWaterfallLayer(enableImageProcessing, _liveDisplayDirection));
            await _allocationGate.WaitAsync();
            try
            {
                if (IsAllocated || IsReleasing) return;
                _isAllocating = true;
                try
                {
                    await AllocateCamerasCoreAsync(enableImageProcessing);
                }
                catch
                {
                    FreeCamerasCore();
                    IsReleasing = false;
                    throw;
                }
            }
            finally
            {
                _isAllocating = false;
                _allocationGate.Release();
            }
        }

        private async Task AllocateCamerasCoreAsync(bool enableImageProcessing)
        {
            if (IsAllocated) return;
            _captureGateOpen = false;
            ClearCaptureBoundary();
            InvalidateCapturePhase("allocate");
            FlowTrace.Log($"AllocateCameras begin（expect {_cameraHardwareConfigs.Count} cams）");
            IsReleasing = false;
            var totalSw = Stopwatch.StartNew();
            var acquisitionSw = Stopwatch.StartNew();

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
                        _display.UpdateSingleCameraStatus(cfg.Id, "分配 System 失敗", Color.Red);
                        continue;
                    }
                }

                if (!_display.TryGetDisplayPanel(cfg.Id, out Panel displayPanel) ||
                    !_display.HasCameraStatusLabel(cfg.Id))
                    continue;

                string dcf = DcfPathHelper.Resolve(!string.IsNullOrEmpty(_dcfPath) ? _dcfPath : cfg.DcfPath);
                var cam = new AniloxCamera(
                    currentSysId,
                    cfg.Id,
                    cfg.DevNum,   // 固定 device 位置（絕對值，直接傳）；MIL 轉換收斂在 MilCamera ctor
                    dcf,
                    IntPtr.Zero,  // 顯示鐵則2：app 顯示一律 CPU（ImageDisplayView/ThumbStrip/WaterfallView），
                                  // 不 attach 任何原生顯示視窗（headless）。panel 留給 ThumbStrip 用。
                    enableImageProcessing);
                cam.LiveDisplayDirection = _liveDisplayDirection;

                int camIdx = cfg.Id - 1; // cfg.Id 為 1–7，轉為 0–6 陣列索引
                cam.EnableAutoCapture    = _enableAutoCapture;
                cam.SaveOriginalBmp = _saveOriginalBmp;
                cam.CaptureRootPath      = _captureRootPath;

                // grab 高度走 json（_cameraGrabHeight 已在 UpdateCaptureSettingsCache clamp 到 MaxGrabHeightPx=12000）。
                cam.CameraGrabHeight = _cameraGrabHeight[camIdx];

                cam.CameraExposureTimeUs = _cameraExposureTimeUs[camIdx]; // InitializeAcquisition() 會呼叫 SetExposureUs 套用
                cam.SetLineRateHz(_cameraLineRateHz[camIdx]);  // 記錄 _appliedLineRateHz（CLProtocol 就緒後自動重套）
                cam.HessianSigma         = _ridgeSigma;   // 細線濾除（設定值，非硬編常數）
                cam.HessianFixedMax      = _hessianMaxFactor;
                cam.RidgeMode            = _ridgeMode;
                cam.SaveResizeScale      = _saveResizeScale;
                cam.SaveJpgQuality       = _saveJpgQuality;
                cam.TimestampCoordinator = _timestampCoordinator;
                cam.CaptureFrameAccepted = IsCaptureFrameAccepted;
                cam.CaptureFrameCompleted = NotifyCaptureFrameCompleted;

                cam.OnInspectionResult += (grabId, camId, fn, mp, xp, maxCMean, meanRPeak, maxRPeak) =>
                    OnInspectionResult?.Invoke(
                        grabId, camId, fn, mp, xp, maxCMean, meanRPeak, maxRPeak);
                cam.OnLiveCurveData      += (camId, mean, max) =>
                    OnLiveCurveData?.Invoke(camId, mean, max);
                cam.OnLiveRowCurveData   += (camId, mean, max) =>
                    OnLiveRowCurveData?.Invoke(camId, mean, max);
                int captureCameraId = cam.CameraId;
                cam.OnFilesSaved = files =>
                    OnFilesSaved?.Invoke(captureCameraId, files);
                cam.OnCaptureSaveFailed = OnCaptureSaveFailed;
                cam.InitializeAcquisition();
                _cameras.Add(cam);
            }

            acquisitionSw.Stop();
            FlowTrace.Log(
                $"camera init phase=acquisition done cams={_cameras.Count} " +
                $"ms={acquisitionSw.ElapsedMilliseconds}");

            var processingSw = Stopwatch.StartNew();
            FlowTrace.Log($"camera init phase=processing begin cams={_cameras.Count}");
            AniloxCamera[] processingCameras = _cameras.ToArray();
            await Task.Run(() =>
            {
                foreach (var cam in processingCameras)
                {
                    if (IsReleasing) return;
                    cam.InitializeProcessingResources();
                }
            });
            processingSw.Stop();
            FlowTrace.Log(
                $"camera init phase=processing done cams={processingCameras.Length} " +
                $"ms={processingSw.ElapsedMilliseconds}");
            if (IsReleasing) return;

            // CLProtocol 啟用移到「所有相機 buffer 分配完成後」的背景階段：不與 MbufAlloc/MdispAlloc 競爭
            // MIL 內部鎖，也不在 grab 期間 enable + 重套線掃（會掉幀，cam1 最明顯）。利用「分配 → 使用者點抓取」
            // 空檔跑完 2-5s/台；之後還要由 hot standby 實測每台第一幀，兩階段完成前
            // AreCamerasHwReady=false，上層把「開始抓取」鈕維持灰色。
            // 只對「在線」相機啟用 CLProtocol：對斷線相機 enable 會卡住 MIL 內部鎖（全斷線時 7 台全卡 →
            // 10s 逾時旗標翻 true 後 timer 恢復、CheckPresence 跟還卡著的背景 MIL 搶鎖 → 整個 UI 凍死）。
            // 斷線相機 _clProtocolInitStarted=false → IsHwParamsStable=true（不擋參數就緒判定）；
            // 若之後才連上，走 legacy 參數路徑（與導入 CLProtocol 前行為相同）。順帶：正常 2/7 時 init
            // 不再空等 5 台死相機逾時 → 從 ~10s 縮到 ~2-4s。
            _hwReadyRaised = false;
            foreach (var cam in _cameras)
                if (cam.CheckPresence())
                    cam.BeginCLProtocolInit();

            IsAllocated = true;
            _cameraStatusTimer.Start();
            _display.UpdateCameraStatus("已配置", Color.White);

            // 顯示 view 訂閱各 cam.OnDisplayFrame，且 Enable* 冪等（view 已存在就早退）→ 若 view 在本批相機
            // 建立前就存在，會殘留空/舊訂閱、收不到新相機的幀。先 teardown 再 Apply 重建 → 一定訂閱「這批」相機
            //（與 ReleaseAsync 的 teardown 對稱）。
            _display.TeardownImageDisplay();
            _display.TeardownWaterfallDisplay();
            ApplyMainDisplayMode(); // 依 he_MainDisplay 套用：即時 / 瀑布

            _display.SwitchMainDisplay(_display.SelectedMainCameraId);

            // 初始化後立即發布「實際在線」數（上面 CheckPresence 的結果）。配置數≠在線數：
            // quad 卡空通道也配得起來（2026-07-07 盲測：只接 2 台卻報 4，hwReady gate 開了才修正
            // ＝幽靈相機數＋假「相機離線 4→2」）。Timer 之後持續更新。
            int present = 0;
            foreach (var cam in _cameras) if (cam.IsConnected) present++;
            ConnectedCameraCount = present;
            OnCameraCountChanged?.Invoke(present, ExpectedCameraCount);
            FlowTrace.Log($"AllocateCameras done（配置 {_cameras.Count}、在線 {present}/{ExpectedCameraCount}）");
            totalSw.Stop();
            FlowTrace.Log(
                $"camera init summary cams={_cameras.Count} totalMs={totalSw.ElapsedMilliseconds} " +
                $"acquisitionMs={acquisitionSw.ElapsedMilliseconds} processingMs={processingSw.ElapsedMilliseconds}");
        }

        // ==================== Grab Control ====================

        public async Task<bool> ToggleGrabAsync(
            bool deferCaptureGate = false,
            bool requireVerifiedStandby = false)
        {
            if (!IsAllocated) return false;
            if (IsLiveGrabbing)
            {
                StopGrab();
                return true;
            }
            return await StartGrabAsync(
                deferCaptureGate,
                requireVerifiedStandby);
        }

        public async Task<bool> EnsureAllocatedAndToggleGrabAsync(
            bool enableImageProcessing, bool deferCaptureGate = false)
        {
            if (!IsAllocated)
                await AllocateCamerasAsync(enableImageProcessing);
            if (IsAllocated && !AreCamerasHwReady)
                await WaitForCamerasReadyAsync();
            if (IsAllocated && AreCamerasHwReady)
                return await ToggleGrabAsync(deferCaptureGate);
            return false;
        }

        private async Task<bool> WaitForCamerasReadyAsync()
        {
            if (AreCamerasHwReady) return true;

            var ready = new TaskCompletionSource<bool>();
            Action handler = null;
            handler = () => ready.TrySetResult(true);
            OnHwReady += handler;
            try
            {
                if (AreCamerasHwReady) return true;
                Task completed = await Task.WhenAny(
                    ready.Task, Task.Delay(TimeSpan.FromSeconds(15)));
                bool ok = completed == ready.Task && AreCamerasHwReady;
                if (!ok)
                    FlowTrace.Log("acquisition standby timeout 15s");
                return ok;
            }
            finally
            {
                OnHwReady -= handler;
            }
        }

        private async Task<bool> StartGrabAsync(
            bool deferCaptureGate,
            bool requireVerifiedStandby)
        {
            await _allocationGate.WaitAsync();
            try
            {
                if (!IsAllocated || IsLiveGrabbing || IsReleasing ||
                    _isParameterReconfiguring || _isCaptureSynchronizing ||
                    !AreCamerasHwReady)
                    return false;

                var targets = _cameras
                    .Where(cam => cam != null && cam.IsConnected)
                    .ToArray();
                if (targets.Length == 0) return false;

                _captureGateOpen = false;
                ClearCaptureBoundary();
                ClearUserGrabIntents();
                string standbyReason;
                if (CanUseVerifiedStandby(targets, out standbyReason))
                {
                    SetCaptureStartPath("verified-standby");
                    FlowTrace.Log(
                        $"acquisition start path=verified-standby cams={targets.Length}");
                }
                else
                {
                    if (requireVerifiedStandby)
                    {
                        FlowTrace.Log(
                            $"acquisition start rejected path=io reason={standbyReason}");
                        return false;
                    }

                    SetCapturePhaseSynchronizing("start");
                    _isCaptureSynchronizing = true;
                    AcquisitionSyncResult sync;
                    try
                    {
                        sync = await SynchronizeAcquisitionAsync(
                            "start",
                            targets,
                            null,
                            () => ReapplyLineRatesForSynchronization("start", targets),
                            () => IsReleasing);
                    }
                    finally
                    {
                        _isCaptureSynchronizing = false;
                    }

                    if (!sync.Succeeded)
                    {
                        InvalidateCapturePhase(
                            "start-sync-" + (sync.Error ?? "failed"));
                        FlowTrace.Log(
                            $"capture synchronize failed gate=closed error={sync.Error}");
                        return false;
                    }

                    MarkCapturePhaseVerified("start-sync");
                    SetCaptureStartPath("full-sync");
                    FlowTrace.Log(
                        $"acquisition start path=full-sync cams={targets.Length} " +
                        $"reason={standbyReason}");
                }

                _display.ResetFlowFirstFrame();
                ApplyMainDisplayMode();
                _display.ResetWaterfallIfActive();
                OnCaptureSequenceReset?.Invoke();
                FlowTrace.Log("capture charts reset reason=start-grab");
                IsLiveGrabbing = true;
                foreach (var cam in targets)
                    cam.SetUserGrabIntent(true);

                FlowTrace.Log($"StartGrab（cams={_cameras.Count}）");
                _display.RefireMainViewRange("capture-start");
                if (!deferCaptureGate)
                    return OpenCaptureGate();
                return true;
            }
            finally
            {
                _allocationGate.Release();
            }
        }

        /// <summary>
        /// Opens the single product-frame acceptance boundary after the form has created the new
        /// grab id, capture plan, and duration guard. Hot standby can deliver a callback
        /// immediately, so those data owners must be ready first.
        /// </summary>
        public bool OpenCaptureGate()
        {
            if (!IsAllocated || !IsLiveGrabbing || !AreCamerasHwReady)
                return false;
            if (_captureGateOpen)
                return true;
            AniloxCamera[] targets = _cameras
                .Where(cam => cam != null && cam.IsConnected)
                .ToArray();
            if (!ArmCaptureBoundary(targets))
                return false;
            _captureGateOpen = true;
            FlowTrace.Log(
                $"capture gate open cams={targets.Length} warm={AreCamerasHwReady} " +
                $"path={_captureStartPath}");
            return true;
        }

        public void BeginCaptureOutput(string grabId, DateTime captureDate)
        {
            if (string.IsNullOrWhiteSpace(grabId))
                throw new ArgumentNullException(nameof(grabId));
            foreach (var cam in _cameras)
            {
                cam.CaptureGrabId = grabId;
                cam.CaptureDate = captureDate;
            }
            FlowTrace.Log($"capture output begin grab={grabId} date={captureDate:yyyyMMdd}");
        }

        public async Task<bool> WaitForCaptureSavesAsync(string grabId, int timeoutMs)
        {
            if (string.IsNullOrWhiteSpace(grabId)) return true;
            var stopwatch = Stopwatch.StartNew();
            while (stopwatch.ElapsedMilliseconds <= timeoutMs)
            {
                int pending = 0;
                foreach (var cam in _cameras)
                    pending += cam.GetPendingCaptureSaveCount(grabId) +
                        cam.GetActiveCaptureCallbackCount(grabId);
                if (pending == 0)
                {
                    FlowTrace.Log(
                        $"capture output drain grab={grabId} pending=0 ms={stopwatch.ElapsedMilliseconds}");
                    return true;
                }
                await Task.Delay(25).ConfigureAwait(false);
            }

            int remaining = 0;
            foreach (var cam in _cameras)
                remaining += cam.GetPendingCaptureSaveCount(grabId) +
                    cam.GetActiveCaptureCallbackCount(grabId);
            FlowTrace.Log(
                $"capture output drain timeout grab={grabId} pending={remaining} limitMs={timeoutMs}");
            return false;
        }

        public void StopGrab()
        {
            if (!IsAllocated || !IsLiveGrabbing) return;
            FlowTrace.Log("StopGrab");
            _captureGateOpen = false;
            ClearCaptureBoundary();
            IsLiveGrabbing = false;
            if (_isParameterReconfiguring)
            {
                // PauseAcquisition owns the per-camera grab lock while draining. Waiting for that
                // lock here would freeze the UI when the duration guard stops during a parameter
                // change. The global product gate is already closed, so intent cleanup can wait
                // until reconfiguration has left the camera locks.
                FlowTrace.Log("parameter stop deferred intent-clear");
            }
            else
            {
                ClearUserGrabIntents();
            }
            FlowTrace.Log("capture gate closed standby=on");
        }

        private void ClearUserGrabIntents()
        {
            foreach (var cam in _cameras)
                cam.SetUserGrabIntent(false);
        }

        private async Task<AcquisitionSyncResult> SynchronizeAcquisitionAsync(
            string reason,
            AniloxCamera[] targets,
            Action applyWhilePaused,
            Action resetTimingWhilePaused,
            Func<bool> cancellationRequested)
        {
            var result = new AcquisitionSyncResult();
            bool actionApplied = applyWhilePaused == null;

            for (int attempt = 1; attempt <= CaptureSynchronizationMaxAttempts; attempt++)
            {
                if (cancellationRequested != null && cancellationRequested())
                {
                    result.Canceled = true;
                    result.Error = "Canceled";
                    return result;
                }

                FlowTrace.Log(
                    $"acquisition sync begin reason={reason} attempt={attempt} " +
                    $"gate=closed cams={targets.Length}");

                Exception failure = null;
                try
                {
                    await Task.Run(() =>
                        System.Threading.Tasks.Parallel.ForEach(
                            targets, cam => cam.PauseAcquisition()));
                    FlowTrace.Log(
                        $"acquisition sync paused reason={reason} attempt={attempt} " +
                        $"cams={targets.Length}");

                    if (!actionApplied)
                    {
                        await Task.Run(applyWhilePaused);
                        actionApplied = true;
                    }
                    if (resetTimingWhilePaused != null)
                        await Task.Run(resetTimingWhilePaused);
                }
                catch (Exception ex)
                {
                    failure = ex;
                }
                finally
                {
                    try
                    {
                        // M_START is intentionally issued back-to-back from one worker. The
                        // measured first hardware ticks, not a fixed delay, decide readiness.
                        await Task.Run(() =>
                        {
                            foreach (var cam in targets)
                                cam.ResumeAcquisition();
                        });
                    }
                    catch (Exception ex)
                    {
                        if (failure == null) failure = ex;
                    }
                }

                if (failure != null)
                {
                    result.Error = failure.GetType().Name;
                    FlowTrace.Log(
                        $"acquisition sync failed reason={reason} attempt={attempt} " +
                        $"gate=closed error={result.Error}");
                    return result;
                }

                FlowTrace.Log(
                    $"acquisition sync resumed reason={reason} attempt={attempt} " +
                    $"cams={targets.Length}");

                AcquisitionSyncResult warm = await WaitForAcquisitionWarmAsync(
                    reason, attempt, targets, cancellationRequested);
                if (!warm.Succeeded)
                    return warm;

                result.Samples = warm.Samples;
                if (LogAndValidateCapturePhase(reason, attempt, warm.Samples))
                {
                    result.Succeeded = true;
                    result.Error = null;
                    FlowTrace.Log(
                        $"acquisition sync complete reason={reason} attempts={attempt} " +
                        $"cams={targets.Length} phase=True");
                    return result;
                }

                FlowTrace.Log(
                    $"acquisition sync retry reason={reason} attempt={attempt} " +
                    $"error=PhaseOutOfRange");
            }

            result.Error = "PhaseOutOfRange";
            return result;
        }

        private async Task<AcquisitionSyncResult> WaitForAcquisitionWarmAsync(
            string reason,
            int attempt,
            IList<AniloxCamera> targets,
            Func<bool> cancellationRequested)
        {
            var result = new AcquisitionSyncResult();
            int framePeriodMs = GetMaxFramePeriodMs(targets);
            int timeoutMs = Math.Max(5000, Math.Min(60000, framePeriodMs * 5 + 2000));
            var pending = new HashSet<int>(targets.Select(cam => cam.CameraId));
            var stopwatch = Stopwatch.StartNew();

            while (pending.Count > 0 && stopwatch.ElapsedMilliseconds <= timeoutMs)
            {
                if (IsReleasing ||
                    (cancellationRequested != null && cancellationRequested()))
                {
                    result.Canceled = true;
                    result.Error = "Canceled";
                    FlowTrace.Log(
                        $"acquisition sync canceled reason={reason} attempt={attempt} " +
                        $"gate=closed");
                    return result;
                }

                foreach (var cam in targets)
                {
                    if (!pending.Contains(cam.CameraId) || !cam.IsAcquisitionWarm)
                        continue;

                    var sample = new CapturePhaseSample
                    {
                        CameraId = cam.CameraId,
                        SystemNum = GetCameraSystemNum(cam.CameraId),
                        FrameStartTicks = cam.LastFrameStartTicks,
                        ClockFrequencyHz = cam.DataLatchClockFreqHz,
                        FrameHeight = cam.FrameHeight,
                        AppliedLineRateHz = cam.AppliedLineRateHz
                    };
                    result.Samples.Add(sample);
                    pending.Remove(cam.CameraId);
                    FlowTrace.Log(
                        $"acquisition sync ready reason={reason} attempt={attempt} " +
                        $"cam{sample.CameraId} system={sample.SystemNum} " +
                        $"tick={sample.FrameStartTicks} freq={sample.ClockFrequencyHz}");
                }

                if (pending.Count == 0)
                {
                    result.Succeeded = true;
                    return result;
                }
                await Task.Delay(20);
            }

            result.Error = "WarmTimeout";
            FlowTrace.Log(
                $"acquisition sync timeout reason={reason} attempt={attempt} " +
                $"pending={string.Join(",", pending)} limitMs={timeoutMs}");
            return result;
        }

        private bool LogAndValidateCapturePhase(
            string reason,
            int attempt,
            IList<CapturePhaseSample> samples)
        {
            if (samples == null || samples.Count == 0)
                return false;
            string phaseReason;
            return TryValidateCapturePhaseSamples(
                samples,
                out phaseReason,
                true,
                $"acquisition sync phase reason={reason} attempt={attempt}",
                "warm-snapshot");
        }

        private int GetCameraSystemNum(int cameraId)
        {
            CameraHardwareConfig config = _cameraHardwareConfigs?
                .FirstOrDefault(item => item.Id == cameraId);
            return config?.SystemNum ?? -1;
        }

        private static void ReapplyLineRatesForSynchronization(
            string reason, IEnumerable<AniloxCamera> targets)
        {
            var applied = new List<string>();
            foreach (var cam in targets)
            {
                double lineRateHz = cam.AppliedLineRateHz;
                if (lineRateHz <= 0) continue;
                cam.SetLineRateHz(lineRateHz);
                applied.Add(
                    "cam" + cam.CameraId + "=" +
                    lineRateHz.ToString(
                        "0.###", System.Globalization.CultureInfo.InvariantCulture));
            }
            FlowTrace.Log(
                $"acquisition sync timing-reset reason={reason} " +
                $"lineRates={string.Join(",", applied)}");
        }

        // ==================== Release ====================

        private void FreeCamerasCore()
        {
            FlowTrace.Log($"FreeCameras（cams={_cameras.Count}）");
            IsReleasing = true;
            _cameraStatusTimer.Stop();
            _captureGateOpen = false;
            ClearCaptureBoundary();
            InvalidateCapturePhase("free");
            IsLiveGrabbing = false;
            _hwReadyRaised = false;

            // Release is the physical stop boundary. Drain every digitizer in parallel before any
            // MIL buffer is freed; business StopGrab deliberately does not enter this path.
            var camerasToPause = _cameras.ToArray();
            try
            {
                System.Threading.Tasks.Parallel.ForEach(
                    camerasToPause, cam => cam.PauseAcquisition());
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"[FreeCameras.pause] {ex.GetType().Name}: {ex.Message}");
            }

            // 先停止 global merge（必須在 cam.Free 之前）：DisableGlobalMerge 會先清各相機 merge target
            // 再由工頭 MbufFree 合併 buffer，避免 grab hook 把幀複製進已釋放的 buffer。
            DisableGlobalMerge();

            // 相機顯示：解訂閱 + dispose。ImageCanvas 與瀑布都訂閱各 cam.OnDisplayFrame，相機即將 Free →
            // 兩者都必須 teardown。瀑布尤其重要：EnableWaterfallDisplay 冪等（_waterfallView!=null 早退），
            // 不 teardown 則重建相機後不會重新訂閱新 cam → 瀑布空白（預覽背景→開始抓取 空白的根因）。
            _display.TeardownImageDisplay();
            _display.TeardownWaterfallDisplay();

            foreach (var cam in _cameras)
                cam.Free();
            _cameras.Clear();

            foreach (var kvp in _allocatedSystems)
                CameraSystemManager.FreeSystem(kvp.Value);
            _allocatedSystems.Clear();

            CameraSystemManager.FreeApplication();

            IsAllocated = false;
            _display.UpdateCameraStatus("已釋放 (Freed)", Color.Gray);
        }

        /// <summary>
        /// 等待進行中的配置工作離開 native call 後再釋放，避免 processing allocation 與 Free 競態。
        /// Timer 必須先在呼叫端停止；耗時的 native teardown 保持在背景執行。
        /// </summary>
        public async Task ReleaseAsync()
        {
            _cameraStatusTimer.Stop();
            IsReleasing = true;
            await _allocationGate.WaitAsync();
            try
            {
                await Task.Run(() => FreeCamerasCore());
            }
            finally
            {
                IsReleasing = false;
                _allocationGate.Release();
            }
        }

        // ==================== Settings ====================

        public void SetLiveDisplayMode(bool enable, string direction)
        {
            string normalizedDirection = direction == "h" ? "h" : "v";
            _liveDisplayDirection = normalizedDirection;
            foreach (var cam in _cameras)
            {
                cam.EnableImageProcessing = enable;
                cam.LiveDisplayDirection = normalizedDirection;
            }

            WaterfallFrameLayer layer = ToWaterfallLayer(enable, normalizedDirection);
            _display.SetWaterfallDisplayLayer(layer);

            FlowTrace.Log(
                $"live enhance enabled={enable} direction={layer.ToString().ToLowerInvariant()} " +
                $"cams={_cameras.Count} scope=all-cameras waterfallHistory=preserved");
        }

        private static WaterfallFrameLayer ToWaterfallLayer(bool enable, string direction)
        {
            if (!enable) return WaterfallFrameLayer.Raw;
            return direction == "h" ? WaterfallFrameLayer.Row : WaterfallFrameLayer.Column;
        }

        public void SetScreenMmPerPixel(double mmPerPx) => _display.SetScreenMmPerPixel(mmPerPx);

        public void SetCaptureSettings(InspectionSettings settings)
        {
            if (settings == null) return;
            _inspectionSettings = settings;

            UpdateCaptureSettingsCache(settings);
            ApplyCapturePolicyToCameras();

            foreach (var cam in _cameras)
            {
                int camIdx = cam.CameraId - 1;
                cam.CameraGrabHeight     = _cameraGrabHeight[camIdx]; // 已在 UpdateCaptureSettingsCache clamp 到 MaxGrabHeightPx
                // 曝光：走 CLProtocol-aware SetExposureUs（CLProtocol 未就緒時記錄，就緒後自動重套）
                cam.SetExposureUs(_cameraExposureTimeUs[camIdx]);
                // 線掃速率：同上，CLProtocol 未就緒時記錄，就緒後自動重套
                cam.SetLineRateHz(_cameraLineRateHz[camIdx]);
            }
        }

        /// <summary>
        /// 套用執行期存檔/演算法政策，不重送曝光、線掃速率或擷取高度。
        /// PropertyGrid 設定變更走這條；完整相機 timing 只在初始化或專用相機參數流程套用。
        /// </summary>
        public void RefreshCapturePolicy(InspectionSettings settings)
        {
            if (settings == null) return;
            _inspectionSettings = settings;
            UpdateCaptureSettingsCache(settings);
            ApplyCapturePolicyToCameras();
        }

        private void ApplyCapturePolicyToCameras()
        {
            foreach (var cam in _cameras)
            {
                cam.EnableAutoCapture    = _enableAutoCapture;
                cam.SaveOriginalBmp      = _saveOriginalBmp;
                cam.CaptureRootPath      = _captureRootPath;
                cam.HessianSigma         = _ridgeSigma;
                cam.HessianFixedMax      = _hessianMaxFactor;
                cam.RidgeMode            = _ridgeMode;
                cam.SaveResizeScale      = _saveResizeScale;
                cam.SaveJpgQuality       = _saveJpgQuality;
                cam.TimestampCoordinator = _timestampCoordinator;
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
            if (IsLiveGrabbing)
            {
                FlowTrace.Log(
                    $"parameter change blocked scope=cam{camId} param=LineRate reason=GrabActive");
                return;
            }
            InvalidateCapturePhase("line-rate-cam" + camId);
            FindCamera(camId)?.SetLineRateHz(hz);
        }

        /// <summary>
        /// 對指定相機變更 Grab 高度（px）。
        /// 內部走 Stop → Free → Realloc → Restart 完整流程（由 AniloxCamera.SetGrabHeight 處理）。
        /// 同 CameraSession.SetGrabHeightForCamera()。
        /// </summary>
        public void SetGrabHeightForCamera(int camId, int height)
        {
            if (IsLiveGrabbing)
            {
                FlowTrace.Log(
                    $"parameter change blocked scope=cam{camId} param=Height reason=GrabActive");
                return;
            }
            // grab 中拉大到 ~12062 會 stall → 一律 cap 在 MaxGrabHeightPx(12000) 以下（per-camera 固定，不分台數）。
            if (height > AcquisitionDefaults.MaxGrabHeightPx) height = AcquisitionDefaults.MaxGrabHeightPx;
            var cam = FindCamera(camId);
            if (cam == null) return;
            InvalidateCapturePhase("height-cam" + camId);
            cam.PauseAcquisition();
            try { cam.SetGrabHeight(height); }
            finally { cam.ResumeAcquisition(); }
        }

        /// <summary>
        /// Applies one camera parameter, then restarts every connected digitizer as one physical
        /// generation. Restarting only the edited camera would shift its phase against the rest.
        /// </summary>
        public Task<bool> ApplyParamCoordinatedAsync(int camId, Action write)
        {
            var cam = FindCamera(camId);
            var targets = cam == null
                ? new AniloxCamera[0]
                : _cameras.ToArray();
            return ApplyParamCoordinatedCoreAsync("cam" + camId, targets, write);
        }

        /// <summary>
        /// Applies an All parameter as one reconfiguration generation. All connected digitizers
        /// are drained, written, resumed, and observed warm before product frames are accepted.
        /// </summary>
        public Task<bool> ApplyParamCoordinatedAsync(Action write)
        {
            return ApplyParamCoordinatedCoreAsync("All", _cameras.ToArray(), write);
        }

        /// <summary>
        /// Applies exposure without restarting acquisition. Exposure changes integration time but
        /// does not change line timing, so closing the capture gate and rebuilding the physical
        /// frame generation would only add several seconds of avoidable latency.
        /// </summary>
        public Task<bool> ApplyExposureFastAsync(int camId, Action write)
        {
            return ApplyExposureFastCoreAsync("cam" + camId, write);
        }

        /// <summary>Applies the all-camera exposure command while the current grab keeps flowing.</summary>
        public Task<bool> ApplyExposureFastAsync(Action write)
        {
            return ApplyExposureFastCoreAsync("All", write);
        }

        private async Task<bool> ApplyExposureFastCoreAsync(string scope, Action write)
        {
            if (write == null) return true;

            await _allocationGate.WaitAsync();
            try
            {
                if (IsReleasing) return false;

                bool live = IsLiveGrabbing && _captureGateOpen;
                var sw = Stopwatch.StartNew();
                if (live)
                    FlowTrace.Log($"exposure live apply begin scope={scope} gate=open");

                await Task.Run(write);

                if (live)
                {
                    FlowTrace.Log(
                        $"exposure live apply complete scope={scope} " +
                        $"gate={(_captureGateOpen ? "open" : "closed")} " +
                        $"elapsedMs={sw.ElapsedMilliseconds}");
                }
                return true;
            }
            catch (Exception ex)
            {
                FlowTrace.Log(
                    $"exposure live apply failed scope={scope} gate={(_captureGateOpen ? "open" : "closed")} " +
                    $"error={ex.GetType().Name}");
                Trace.TraceWarning(
                    $"[ApplyExposureFastCoreAsync.{scope}] {ex.GetType().Name}: {ex.Message}");
                return false;
            }
            finally
            {
                _allocationGate.Release();
            }
        }

        private async Task<bool> ApplyParamCoordinatedCoreAsync(
            string scope, AniloxCamera[] requestedTargets, Action write)
        {
            if (write == null) return true;

            await _allocationGate.WaitAsync();
            bool enteredReconfiguration = false;
            try
            {
                if (IsReleasing) return false;

                var targets = (requestedTargets ?? new AniloxCamera[0])
                    .Where(cam => cam != null && cam.IsConnected)
                    .Distinct()
                    .ToArray();
                bool wasCapturing = IsLiveGrabbing;
                if (!wasCapturing || targets.Length == 0)
                {
                    InvalidateCapturePhase("parameter-" + scope);
                    await Task.Run(write);
                    return true;
                }

                _isParameterReconfiguring = true;
                enteredReconfiguration = true;
                InvalidateCapturePhase("parameter-" + scope);
                SetCapturePhaseSynchronizing("parameter:" + scope);
                // This must precede every physical stop. Callbacks already inside the old
                // generation are discarded by the display/curve reset before the gate reopens.
                _captureGateOpen = false;
                FlowTrace.Log(
                    $"parameter reconfigure begin scope={scope} gate=closed targets={targets.Length}");

                AcquisitionSyncResult sync = await SynchronizeAcquisitionAsync(
                    "parameter:" + scope,
                    targets,
                    () =>
                    {
                        FlowTrace.Log(
                            $"parameter reconfigure paused scope={scope} cams={targets.Length}");
                        write();
                        FlowTrace.Log($"parameter reconfigure applied scope={scope}");
                    },
                    () => ReapplyLineRatesForSynchronization(
                        "parameter:" + scope, targets),
                    () => !IsLiveGrabbing || IsReleasing);

                if (!sync.Succeeded)
                {
                    InvalidateCapturePhase(
                        "parameter-sync-" + (sync.Error ?? "failed"));
                    if (sync.Canceled || !IsLiveGrabbing || IsReleasing)
                    {
                        if (!IsReleasing)
                        {
                            ClearUserGrabIntents();
                            FlowTrace.Log(
                                $"parameter stop intent-clear complete scope={scope}");
                        }
                        FlowTrace.Log(
                            $"parameter reconfigure canceled scope={scope} " +
                            $"gate=closed reason=StopGrab");
                        return true;
                    }
                    IsLiveGrabbing = false;
                    ClearUserGrabIntents();
                    FlowTrace.Log(
                        $"parameter reconfigure failed scope={scope} gate=closed " +
                        $"error={sync.Error}");
                    FlowTrace.Log("capture gate closed standby=on");
                    return false;
                }

                foreach (CapturePhaseSample sample in sync.Samples)
                {
                    FlowTrace.Log(
                        $"parameter warm ready scope={scope} cam{sample.CameraId} " +
                        $"tick={sample.FrameStartTicks}");
                }

                // Stop/close may run while reconfiguration is awaiting a frame. Never reopen a
                // generation that the user has already ended.
                if (!IsLiveGrabbing || IsReleasing)
                {
                    if (!IsReleasing)
                    {
                        ClearUserGrabIntents();
                        FlowTrace.Log(
                            $"parameter stop intent-clear complete scope={scope}");
                    }
                    FlowTrace.Log(
                        $"parameter reconfigure canceled scope={scope} gate=closed reason=StopGrab");
                    return true;
                }

                _display.ResetFlowFirstFrame();
                _display.ResetWaterfallIfActive();
                OnCaptureSequenceReset?.Invoke();
                _display.RefireMainViewRange("parameter-reset");
                FlowTrace.Log($"parameter sequence reset scope={scope}");

                MarkCapturePhaseVerified("parameter-sync");
                SetCaptureStartPath("parameter-sync");
                if (!ArmCaptureBoundary(targets))
                {
                    InvalidateCapturePhase("parameter-gate-no-targets");
                    IsLiveGrabbing = false;
                    ClearUserGrabIntents();
                    FlowTrace.Log(
                        $"parameter reconfigure failed scope={scope} gate=closed " +
                        $"error=no-targets");
                    return false;
                }
                _captureGateOpen = true;
                FlowTrace.Log(
                    $"parameter reconfigure complete scope={scope} gate=open warm=True");
                return true;
            }
            finally
            {
                if (enteredReconfiguration)
                {
                    _isParameterReconfiguring = false;
                    // Covers stop/release races that exit through a failure path before the
                    // normal StopGrab cancellation branch above.
                    if (!IsLiveGrabbing && !IsReleasing)
                        ClearUserGrabIntents();
                }
                _allocationGate.Release();
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
            return GetMaxFramePeriodMs(_cameras);
        }

        private static int GetMaxFramePeriodMs(IEnumerable<AniloxCamera> cameras)
        {
            int maxMs = 0;
            foreach (var cam in cameras)
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
            _ridgeSigma           = settings.RidgeSigma > 0
                ? settings.RidgeSigma
                : InspectionEngineConfig.DefaultRidgeSigma;
            _ridgeMode            = InspectionRecipe.RidgeDirectionToNative(settings.RidgeDir);
            _dcfPath              = settings.DcfPath ?? string.Empty;
        }
    }
}
