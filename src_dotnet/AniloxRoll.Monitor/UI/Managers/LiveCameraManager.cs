using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Managers
{
    public class LiveCameraManager
    {
        private readonly Form _mainForm;
        private readonly Panel _mainDisplayPanel;
        private readonly Action<string> _updatePixelInfoCallback;

        private List<AniloxCamera> _cameras = new List<AniloxCamera>();
        private List<CameraHardwareConfig> _cameraHardwareConfigs;
        private Dictionary<int, MIL_ID> _allocatedSystems = new Dictionary<int, MIL_ID>();

        private readonly Dictionary<int, Panel> _liveViewPanels  = new Dictionary<int, Panel>();
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

        /// <summary>每台相機存檔並完成 inspection 後觸發。
        /// 參數：(cameraId, fileNameWithoutExt, meanPeak_0to1, maxPeak_0to1)</summary>
        public event Action<int, string, float, float> OnInspectionResult;

        /// <summary>每幀 GPU pipeline 完成後觸發（MIL 回呼執行緒）。
        /// 參數：(cameraId, curveMean_raw255, curveMax_raw255)</summary>
        public event Action<int, float[], float[]> OnLiveCurveData;

        /// <summary>每幀 GPU pipeline 完成後觸發（MIL 回呼執行緒）。
        /// 參數：(cameraId, rowCurveMean_raw255, rowCurveMax_raw255)</summary>
        public event Action<int, float[], float[]> OnLiveRowCurveData;

        /// <summary>
        /// 正在執行釋放流程時為 true，防止 Timer Tick 在資源已釋放後繼續存取相機。
        /// 同 CameraSession.IsReleasing。
        /// </summary>
        public volatile bool IsReleasing = false;

        private int _selectedMainCameraId = 1;
        public int SelectedMainCameraId => _selectedMainCameraId;

        // --- Global merge（即時合圖）---
        private MIL_ID _mergedBuffer  = MIL.M_NULL;
        private MIL_ID _mergedDisplay = MIL.M_NULL;
        public bool IsGlobalMergeActive { get; private set; }
        private double _mergedMinStartMm;   // 合併座標系原點（mm）
        private double _mergedRefOpsMm;     // 合併像素尺寸（mm/px）
        private int    _mergedTotalW;       // 合併 buffer 寬度（px）
        private int    _mergedTotalH;       // 合併 buffer 高度（px）
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
                Font      = new Font("Segoe UI", 7.5f, FontStyle.Bold)
            };

            displayPanel.MouseClick += (s, e) => SwitchMainDisplay(cameraIndex);
            status.MouseClick       += (s, e) => SwitchMainDisplay(cameraIndex);

            parentPanel.Controls.Add(displayPanel);
            parentPanel.Controls.Add(status);
            displayPanel.BringToFront();

            _liveViewPanels[cameraIndex]     = displayPanel;
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

                var cam = new AniloxCamera(
                    currentSysId,
                    cfg.Id,
                    cfg.DevNum,
                    cfg.DcfPath,
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
                cam.Initialize();
                _cameras.Add(cam);
            }

            IsAllocated = true;
            _cameraStatusTimer.Start();
            UpdateCameraStatus("已配置", Color.White);
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

            // 先停止 global merge（必須在 cam.Free 之前，因為 Free 會清除 _mergedTargetBuffer 指向的 buffer）
            DisableGlobalMerge();

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

        public void SetLiveDisplayDirection(string dir)
        {
            foreach (var cam in _cameras)
                cam.LiveDisplayDirection = dir;
        }

        /// <summary>
        /// 套用設定至所有相機。
        /// 曝光：直接呼叫 SetExposureUs（live CLProtocol 路徑，可即時生效）。
        /// Grab Height：僅更新快取，物理變更請呼叫 SetGrabHeightForAll() 或 ReinitializeForAcquisitionSettings()。
        /// </summary>
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
        /// 對所有相機同時設定曝光時間（μs）。
        /// </summary>
        public void SetExposureForAll(double exposureUs)
        {
            for (int i = 0; i < _cameraExposureTimeUs.Length; i++) _cameraExposureTimeUs[i] = exposureUs;
            foreach (var cam in _cameras)
                cam.SetExposureUs(exposureUs);
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
        /// 對所有相機同時設定 Line Rate（Hz）。
        /// </summary>
        public void SetLineRateForAll(double hz)
        {
            for (int i = 0; i < _cameraLineRateHz.Length; i++) _cameraLineRateHz[i] = hz;
            foreach (var cam in _cameras)
                cam.SetLineRateHz(hz);
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

        /// <summary>
        /// 對所有相機同時變更 Grab 高度（px）。
        /// </summary>
        public void SetGrabHeightForAll(int height)
        {
            for (int i = 0; i < _cameraGrabHeight.Length; i++) _cameraGrabHeight[i] = height;
            foreach (var cam in _cameras)
                cam.SetGrabHeight(height);
        }

        private AniloxCamera FindCamera(int camId)
        {
            for (int i = 0; i < _cameras.Count; i++)
                if (_cameras[i].CameraId == camId) return _cameras[i];
            return null;
        }

        /// <summary>
        /// 完整重新初始化相機（適用於需要變更硬體拓樸或無法 live 套用的情境）。
        /// 單純曝光變更請用 SetExposureForAll()；Grab Height 變更請用 SetGrabHeightForAll()。
        /// </summary>
        public void ReinitializeForAcquisitionSettings(bool enableImageProcessing, InspectionSettings settings)
        {
            bool wasLive = IsLiveGrabbing;
            if (wasLive) StopGrab();

            UpdateCaptureSettingsCache(settings);

            FreeCameras();
            AllocateCameras(enableImageProcessing);
            SetCaptureSettings(settings);

            if (wasLive) StartGrab();
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
            _hessianMaxFactor     = settings.HessianMaxFactor > 0
                ? settings.HessianMaxFactor
                : InspectionEngineConfig.DefaultHessianMaxFactor;
            _ridgeMode            = InspectionRecipe.RidgeDirectionToNative(settings.RidgeDir);
        }

        // ==================== Display Switching ====================

        /// <summary>
        /// 切換主顯示到指定相機，並更新選取記錄。
        /// 用於 SetGrabHeight 完成後立即顯示該相機畫面。
        /// </summary>
        public void SwitchToCamera(int cameraId)
        {
            SwitchMainDisplay(cameraId);
        }

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

        private void SwitchMainDisplay(int cameraIndex)
        {
            if (_mainForm.InvokeRequired)
            {
                _mainForm.BeginInvoke(new Action(() => SwitchMainDisplay(cameraIndex)));
                return;
            }

            _selectedMainCameraId = cameraIndex;

            foreach (var kvp in _cameraStatusLabels)
            {
                kvp.Value.BackColor = (kvp.Key == cameraIndex)
                    ? Color.PapayaWhip
                    : Color.FromArgb(32, 32, 32);
            }

            // Global merge 時主畫面由合併 display 控制，不切換單台
            if (IsGlobalMergeActive) return;

            foreach (var cam in _cameras)
            {
                if (cam.CameraId == cameraIndex)
                    cam.SetSecondaryDisplay(_mainDisplayPanel.Handle);
                else
                    cam.SetSecondaryDisplay(IntPtr.Zero);
            }
        }

        // ==================== Global Merge ====================

        /// <summary>啟用即時全域合圖：分配合併 buffer，每幀 callback 自動 MbufCopyClip。</summary>
        public void EnableGlobalMerge(double[] opsUm, double[] startPosMm)
        {
            if (IsGlobalMergeActive || _cameras.Count == 0) return;

            // 參考像素尺寸（取第一台），計算各相機偏移與合併寬度
            double refOpsMm = opsUm[0] / 1000.0;
            double minStart = double.MaxValue, maxEnd = double.MinValue;
            int maxH = 0;

            foreach (var cam in _cameras)
            {
                int idx = cam.CameraId - 1;
                double pos = (idx < startPosMm.Length) ? startPosMm[idx] : 0;
                double ops = (idx < opsUm.Length) ? opsUm[idx] : opsUm[0];
                double widthMm = cam.FrameWidth * ops / 1000.0;
                if (pos < minStart) minStart = pos;
                if (pos + widthMm > maxEnd) maxEnd = pos + widthMm;
                if (cam.FrameHeight > maxH) maxH = cam.FrameHeight;
            }

            int totalW = (int)Math.Ceiling((maxEnd - minStart) / refOpsMm);
            if (totalW <= 0 || maxH <= 0) return;

            // 在第一台相機的 System 上分配合併 buffer + display
            MIL_ID sysId = _cameras[0].OwnerSystemId;
            if (sysId == MIL.M_NULL) return;

            MIL.MbufAlloc2d(sysId, totalW, maxH, 8 + MIL.M_UNSIGNED,
                MIL.M_IMAGE + MIL.M_DISP + MIL.M_PROC, ref _mergedBuffer);
            MIL.MbufClear(_mergedBuffer, 0);

            // 計算每台相機的偏移，按偏移排序後處理 overlap
            var entries = new List<(AniloxCamera cam, int xOffset)>();
            foreach (var cam in _cameras)
            {
                int idx = cam.CameraId - 1;
                double pos = (idx < startPosMm.Length) ? startPosMm[idx] : 0;
                int offsetX = (int)Math.Round((pos - minStart) / refOpsMm);
                entries.Add((cam, offsetX));
            }
            entries.Sort((a, b) => a.xOffset.CompareTo(b.xOffset));

            // 計算 drawLeft / drawRight（重疊區域中點分界，與 GrabImageStitcher.MergeHorizontal 一致）
            int n = entries.Count;
            var drawLeft  = new int[n];
            var drawRight = new int[n];
            for (int i = 0; i < n; i++)
            {
                drawLeft[i]  = 0;
                drawRight[i] = entries[i].cam.FrameWidth;
            }
            for (int i = 0; i < n - 1; i++)
            {
                int rightEdge = entries[i].xOffset + entries[i].cam.FrameWidth;
                int leftEdge  = entries[i + 1].xOffset;
                int overlap   = rightEdge - leftEdge;
                if (overlap > 0)
                {
                    int mid = leftEdge + overlap / 2;
                    // 前相機：drawRight = mid 在全域座標，轉為 src 座標
                    drawRight[i] = Math.Min(drawRight[i], mid - entries[i].xOffset);
                    // 後相機：drawLeft = mid 在全域座標，轉為 src 座標
                    drawLeft[i + 1] = Math.Max(drawLeft[i + 1], mid - entries[i + 1].xOffset);
                }
            }

            // 設定每台相機的合併目標（含裁切範圍）
            for (int i = 0; i < n; i++)
            {
                var cam = entries[i].cam;
                cam.SetMergeTarget(_mergedBuffer, entries[i].xOffset, drawLeft[i], drawRight[i] - drawLeft[i]);
            }

            // 解除所有相機的 secondary display，改用合併 display
            foreach (var cam in _cameras)
                cam.SetSecondaryDisplay(IntPtr.Zero);

            // 儲存座標系參數（供滑鼠回呼 + overview 計算）
            _mergedMinStartMm = minStart;
            _mergedRefOpsMm   = refOpsMm;
            _mergedTotalW     = totalW;
            _mergedTotalH     = maxH;

            MIL.MdispAlloc(sysId, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref _mergedDisplay);
            MIL.MdispSelectWindow(_mergedDisplay, _mergedBuffer, _mainDisplayPanel.Handle);
            MIL.MdispControl(_mergedDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
            MIL.MdispControl(_mergedDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
            MIL.MdispControl(_mergedDisplay, MIL.M_MOUSE_USE, MIL.M_ENABLE);

            // 關閉 MIL 自動刷新（每次 MbufCopyClip 都會觸發 repaint，多相機非同步導致閃爍）
            MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_DISABLE);

            // 改用定時器手動刷新（~30fps），確保每次顯示的是所有相機的最新合成結果
            _mergedDisplayTimer = new Timer { Interval = 33 };
            _mergedDisplayTimer.Tick += MergedDisplayTimer_Tick;
            _mergedDisplayTimer.Start();

            // Hook 滑鼠移動 → 更新 lblPixelInfo
            _mergedMouseDelegate = new MIL_DISP_HOOK_FUNCTION_PTR(MergedMouseStatusHandler);
            MIL.MdispHookFunction(_mergedDisplay, MIL.M_MOUSE_MOVE, _mergedMouseDelegate, IntPtr.Zero);

            IsGlobalMergeActive = true;
        }

        /// <summary>停用即時全域合圖：釋放合併 buffer，恢復單台 secondary display。</summary>
        public void DisableGlobalMerge()
        {
            if (!IsGlobalMergeActive) return;

            // 停止定時刷新
            if (_mergedDisplayTimer != null)
            {
                _mergedDisplayTimer.Stop();
                _mergedDisplayTimer.Dispose();
                _mergedDisplayTimer = null;
            }

            // 先清除各相機的合併目標（停止 callback 中的 MbufCopyClip）
            foreach (var cam in _cameras)
                cam.ClearMergeTarget();

            // Unhook 滑鼠 + 釋放合併 display + buffer
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
            if (_mergedBuffer != MIL.M_NULL)
            {
                MIL.MbufFree(_mergedBuffer);
                _mergedBuffer = MIL.M_NULL;
            }

            IsGlobalMergeActive = false;

            // 恢復選中相機的 secondary display
            SwitchMainDisplay(_selectedMainCameraId);
        }

        /// <summary>OPS/Start 變更時，重新計算全域合圖佈局（下一幀生效）。</summary>
        public void RefreshGlobalMergeLayout(double[] opsUm, double[] startPosMm)
        {
            if (!IsGlobalMergeActive || _cameras.Count == 0) return;

            // ① 暫停所有相機的合併複製（callback 中 _mergedTargetBuffer == M_NULL → 跳過）
            foreach (var cam in _cameras)
                cam.ClearMergeTarget();

            // ② 重算座標系
            double refOpsMm = opsUm[0] / 1000.0;
            double minStart = double.MaxValue, maxEnd = double.MinValue;
            int maxH = 0;
            foreach (var cam in _cameras)
            {
                int idx = cam.CameraId - 1;
                double pos = (idx < startPosMm.Length) ? startPosMm[idx] : 0;
                double ops = (idx < opsUm.Length) ? opsUm[idx] : opsUm[0];
                double widthMm = cam.FrameWidth * ops / 1000.0;
                if (pos < minStart) minStart = pos;
                if (pos + widthMm > maxEnd) maxEnd = pos + widthMm;
                if (cam.FrameHeight > maxH) maxH = cam.FrameHeight;
            }
            int totalW = (int)Math.Ceiling((maxEnd - minStart) / refOpsMm);
            if (totalW <= 0 || maxH <= 0) return;

            // ③ buffer 大小改變 → 重新分配
            if (totalW != _mergedTotalW || maxH != _mergedTotalH)
            {
                MIL_ID sysId = _cameras[0].OwnerSystemId;
                if (sysId == MIL.M_NULL) return;

                // 暫時解除 display 綁定
                MIL.MdispSelectWindow(_mergedDisplay, MIL.M_NULL, IntPtr.Zero);
                MIL.MbufFree(_mergedBuffer);
                _mergedBuffer = MIL.M_NULL;

                MIL.MbufAlloc2d(sysId, totalW, maxH, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_DISP + MIL.M_PROC, ref _mergedBuffer);
                MIL.MbufClear(_mergedBuffer, 0);

                // 重新綁定 display
                MIL.MdispSelectWindow(_mergedDisplay, _mergedBuffer, _mainDisplayPanel.Handle);
                MIL.MdispControl(_mergedDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
            }
            else
            {
                MIL.MbufClear(_mergedBuffer, 0);
            }

            // ④ 重算 overlap + clip（與 EnableGlobalMerge 相同邏輯）
            var entries = new List<(AniloxCamera cam, int xOffset)>();
            foreach (var cam in _cameras)
            {
                int idx = cam.CameraId - 1;
                double pos = (idx < startPosMm.Length) ? startPosMm[idx] : 0;
                int offsetX = (int)Math.Round((pos - minStart) / refOpsMm);
                entries.Add((cam, offsetX));
            }
            entries.Sort((a, b) => a.xOffset.CompareTo(b.xOffset));

            int n = entries.Count;
            var drawLeft  = new int[n];
            var drawRight = new int[n];
            for (int i = 0; i < n; i++)
            {
                drawLeft[i]  = 0;
                drawRight[i] = entries[i].cam.FrameWidth;
            }
            for (int i = 0; i < n - 1; i++)
            {
                int rightEdge = entries[i].xOffset + entries[i].cam.FrameWidth;
                int leftEdge  = entries[i + 1].xOffset;
                int overlap   = rightEdge - leftEdge;
                if (overlap > 0)
                {
                    int mid = leftEdge + overlap / 2;
                    drawRight[i]     = Math.Min(drawRight[i], mid - entries[i].xOffset);
                    drawLeft[i + 1]  = Math.Max(drawLeft[i + 1], mid - entries[i + 1].xOffset);
                }
            }

            // ⑤ 更新座標系 + 各相機 clip，然後恢復合併複製
            _mergedMinStartMm = minStart;
            _mergedRefOpsMm   = refOpsMm;
            _mergedTotalW     = totalW;
            _mergedTotalH     = maxH;

            for (int i = 0; i < n; i++)
            {
                var cam = entries[i].cam;
                cam.SetMergeTarget(_mergedBuffer, entries[i].xOffset, drawLeft[i], drawRight[i] - drawLeft[i]);
            }
        }

        // ==================== Merged Display Refresh ====================

        private void MergedDisplayTimer_Tick(object sender, EventArgs e)
        {
            if (_mergedDisplay == MIL.M_NULL) return;
            try { MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_NOW); }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.MergedDisplayTimer] {ex.GetType().Name}: {ex.Message}"); }
        }

        // ==================== Merged Display Mouse ====================

        private MIL_INT MergedMouseStatusHandler(MIL_INT HookType, MIL_ID EventId, IntPtr UserPtr)
        {
            if (_mergedBuffer == MIL.M_NULL) return MIL.M_NULL;

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
                    MIL.MbufGet2d(_mergedBuffer, x, y, 1, 1, data);
                    pixelValue = data[0];
                }
                catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.MergedMouseStatus] {ex.GetType().Name}: {ex.Message}"); }
            }

            HandleMergedMouseData(x, y, pixelValue);
            return MIL.M_NULL;
        }

        private void HandleMergedMouseData(int x, int y, int pixelValue)
        {
            if (_mainForm.InvokeRequired)
            {
                _mainForm.BeginInvoke(new Action(() => HandleMergedMouseData(x, y, pixelValue)));
                return;
            }

            string infoText;
            if (pixelValue == -1)
            {
                infoText = "即時影像 [全域合圖] | 游標超出影像範圍";
            }
            else
            {
                double physicalX = _mergedMinStartMm + x * _mergedRefOpsMm;

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
                        double physicalMag = (zoomX * _screenMmPerPx) / _mergedRefOpsMm;
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
                leftMm  = _mergedMinStartMm + pixelLeft  * _mergedRefOpsMm;
                rightMm = _mergedMinStartMm + pixelRight * _mergedRefOpsMm;
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
            if (_mainForm.InvokeRequired)
            {
                _mainForm.BeginInvoke(new Action(() => HandleMouseDataChanged(camId, x, y, pixelValue)));
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
                    double physicalX  = startPosMm + x * opsInMm;
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
                        double viewLeftMm  = startPosMm + (panOffX) * opsInMm;
                        double viewRightMm = startPosMm + (panOffX + panelW / zoomX) * opsInMm;
                        rangeStr = $"X範圍:{viewLeftMm:F1}~{viewRightMm:F1} mm | ";

                        if (rowPitchMm > 0)
                        {
                            double viewTopMm = panOffY * rowPitchMm;
                            double viewBotMm = (panOffY + panelH / zoomX) * rowPitchMm;
                            rangeStr += $"Y範圍:{viewTopMm:F1}~{viewBotMm:F1} mm | ";
                        }

                        if (_screenMmPerPx > 0 && opsInMm > 0)
                        {
                            double physicalMag = (zoomX * _screenMmPerPx) / opsInMm;
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

                string fpsText    = cam.IsLive ? $" | FPS: {cam.CurrentFps:F1}" : "";
                string statusText = isConnected
                    ? (cam.IsLive ? $"Live{fpsText}" : "Ready")
                    : "Offline";
                Color color = isConnected
                    ? (cam.IsLive ? Color.Green : Color.Yellow)
                    : Color.Red;

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
                double zoom1x = _mergedRefOpsMm / _screenMmPerPx;

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
            double zoom1xCam = opsInMm / _screenMmPerPx;

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
        }

        /// <summary>攔截 panelMainDisplay 上的 WM_MOUSEWHEEL，用 1.1x 步長取代 MIL 預設的整數倍跳躍。</summary>
        private class WheelZoomFilter : IMessageFilter
        {
            private const int WM_MOUSEWHEEL = 0x020A;
            private readonly LiveCameraManager _mgr;

            public WheelZoomFilter(LiveCameraManager mgr) => _mgr = mgr;

            public bool PreFilterMessage(ref Message m)
            {
                if (m.Msg != WM_MOUSEWHEEL) return false;
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
