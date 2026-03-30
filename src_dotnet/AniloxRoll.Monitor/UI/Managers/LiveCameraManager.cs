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

            foreach (var cam in _cameras)
            {
                if (cam.CameraId == cameraIndex)
                    cam.SetSecondaryDisplay(_mainDisplayPanel.Handle);
                else
                    cam.SetSecondaryDisplay(IntPtr.Zero);
            }
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
            catch { return; }

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
            int camIdx = _selectedMainCameraId - 1;
            var s = _inspectionSettings;
            double[] opsUmArr = s?.GetCameraOpsUmArray();
            if (opsUmArr == null || camIdx < 0 || camIdx >= opsUmArr.Length) return;

            double opsInMm = opsUmArr[camIdx] / 1000.0;
            if (opsInMm <= 0) return;

            // physicalMag = zoom * screenMmPerPx / opsInMm = 1  →  zoom = opsInMm / screenMmPerPx
            double zoom1x = opsInMm / _screenMmPerPx;

            var cam = _cameras.Find(c => c.CameraId == _selectedMainCameraId);
            if (cam == null) return;

            // 以面板中心為基準
            double cx = _mainDisplayPanel.Width / 2.0;
            double cy = _mainDisplayPanel.Height / 2.0;

            if (cam.TryGetSecondaryDisplayGeometry(out double curZoom, out _, out double curPanX, out double curPanY) && curZoom > 0)
            {
                double imgCx = curPanX + cx / curZoom;
                double imgCy = curPanY + cy / curZoom;
                double newPanX = imgCx - cx / zoom1x;
                double newPanY = imgCy - cy / zoom1x;
                cam.SetSecondaryDisplayZoom(zoom1x, newPanX, newPanY);
            }
            else
            {
                cam.SetSecondaryDisplayZoom(zoom1x, 0, 0);
            }
        }

        // ==================== Custom Wheel Zoom ====================

        internal void ApplyCustomZoom(int wheelDelta)
        {
            if (!IsLiveGrabbing) return;
            var cam = _cameras.Find(c => c.CameraId == _selectedMainCameraId);
            if (cam == null) return;
            if (!cam.TryGetSecondaryDisplayGeometry(out double zoomX, out _, out double panX, out double panY))
                return;

            double factor = wheelDelta > 0 ? 1.1 : (1.0 / 1.1);
            double newZoom = zoomX * factor;
            if (newZoom < 0.05) newZoom = 0.05;
            if (newZoom > 32.0) newZoom = 32.0;

            // 以面板中心為縮放基準點
            double cx = _mainDisplayPanel.Width / 2.0;
            double cy = _mainDisplayPanel.Height / 2.0;
            double imgX = panX + cx / zoomX;
            double imgY = panY + cy / zoomX;
            double newPanX = imgX - cx / newZoom;
            double newPanY = imgY - cy / newZoom;

            cam.SetSecondaryDisplayZoom(newZoom, newPanX, newPanY);
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
