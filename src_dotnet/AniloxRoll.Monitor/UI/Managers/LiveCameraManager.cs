using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.State;

namespace AniloxRoll.Monitor.UI.Managers
{
    public class LiveCameraManager
    {
        private readonly Form _mainForm;
        private readonly Panel _mainDisplayPanel; // 大畫面 (Panel8)
        private readonly Action<string> _updatePixelInfoCallback; // 更新座標文字的委派

        private List<AniloxCamera> _cameras = new List<AniloxCamera>();
        private List<CameraHardwareConfig> _cameraHardwareConfigs;
        private Dictionary<int, MIL_ID> _allocatedSystems = new Dictionary<int, MIL_ID>();

        private readonly Dictionary<int, Panel> _liveViewPanels = new Dictionary<int, Panel>();
        private readonly Dictionary<int, Label> _cameraStatusLabels = new Dictionary<int, Label>();

        private Timer _cameraStatusTimer;
        private bool _enableAutoCapture;
        private string _captureRootPath = string.Empty;
        private int _cameraGrabHeight;
        private double _cameraExposureTimeUs;

        public bool IsAllocated { get; private set; } = false;
        public bool IsLiveGrabbing { get; private set; } = false;

        private int _selectedMainCameraId = 1;

        public LiveCameraManager(
            Form mainForm,
            Panel[] cameraPanels,
            Panel mainDisplayPanel,
            Action<string> updatePixelInfoCallback)
        {
            if (cameraPanels == null)
            {
                throw new ArgumentNullException(nameof(cameraPanels));
            }

            if (cameraPanels.Length < 7)
            {
                throw new ArgumentException("cameraPanels must contain at least 7 panels.", nameof(cameraPanels));
            }

            _mainForm = mainForm;
            _mainDisplayPanel = mainDisplayPanel;
            _updatePixelInfoCallback = updatePixelInfoCallback;
            _mainDisplayPanel.BackColor = Color.Black;

            for (int i = 0; i < 7; i++)
            {
                SetupLivePanel(cameraPanels[i], i + 1);
            }

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
                Dock = DockStyle.Fill,
                BackColor = Color.Black
            };

            var status = new Label
            {
                Dock = DockStyle.Bottom,
                Height = 18,
                ForeColor = Color.DarkGray,
                BackColor = Color.FromArgb(32, 32, 32),
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font("Segoe UI", 7.5f, FontStyle.Bold)
            };

            displayPanel.MouseClick += (s, e) => SwitchMainDisplay(cameraIndex);
            status.MouseClick += (s, e) => SwitchMainDisplay(cameraIndex);

            parentPanel.Controls.Add(displayPanel);
            parentPanel.Controls.Add(status);
            displayPanel.BringToFront();

            _liveViewPanels[cameraIndex] = displayPanel;
            _cameraStatusLabels[cameraIndex] = status;
        }

        public void AllocateCameras(bool enableImageProcessing)
        {
            if (IsAllocated) return;

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

                if (!_liveViewPanels.TryGetValue(cfg.Id, out Panel displayPanel) || !_cameraStatusLabels.ContainsKey(cfg.Id))
                {
                    continue;
                }

                var cam = new AniloxCamera(
                    currentSysId,
                    cfg.Id,
                    cfg.DevNum,
                    cfg.DcfPath,
                    displayPanel.Handle,
                    enableImageProcessing
                );

                cam.EnableAutoCapture = _enableAutoCapture;
                cam.CaptureRootPath = _captureRootPath;
                cam.CameraGrabHeight = _cameraGrabHeight;
                cam.CameraExposureTimeUs = _cameraExposureTimeUs;
                cam.HessianSigma = InspectionEngineConfig.DefaultRidgeSigma;
                cam.HessianFixedMax =  InspectionEngineConfig.DefaultHessianMaxFactor;

                cam.OnMouseDataChanged += HandleMouseDataChanged;
                cam.OnCameraClicked += SwitchMainDisplay;
                cam.Initialize();
                _cameras.Add(cam);
            }

            IsAllocated = true;
            _cameraStatusTimer.Start();
            UpdateCameraStatus("已配置", Color.White);
            SwitchMainDisplay(_selectedMainCameraId);
        }

        public void ToggleGrab()
        {
            if (!IsAllocated) return;
            if (IsLiveGrabbing) StopGrab();
            else StartGrab();
        }

        public void EnsureAllocatedAndToggleGrab(bool enableImageProcessing)
        {
            if (!IsAllocated)
            {
                AllocateCameras(enableImageProcessing);
            }

            ToggleGrab();
        }

        public void StartGrab()
        {
            if (!IsAllocated || IsLiveGrabbing) return;

            IsLiveGrabbing = true;
            foreach (var cam in _cameras)
            {
                cam.SetUserGrabIntent(true);
            }
        }

        public void StopGrab()
        {
            if (!IsAllocated || !IsLiveGrabbing) return;

            IsLiveGrabbing = false;
            foreach (var cam in _cameras)
            {
                cam.SetUserGrabIntent(false);
            }

            // 停止抓圖後，將今日存檔目錄記入 LastDataPath，
            // 下次點「選擇資料夾」時會直接開啟剛拍攝的影像位置。
            if (_enableAutoCapture && !string.IsNullOrEmpty(_captureRootPath))
            {
                DateTime now = DateTime.Now;
                string todayDir = Path.Combine(
                    _captureRootPath,
                    now.ToString("yyyy"),
                    now.ToString("yyyyMM"),
                    now.ToString("yyyyMMdd"));

                if (Directory.Exists(todayDir))
                {
                    UserSessionState.SetLastDataPath(todayDir);
                    UserSessionState.Save();
                }
            }
        }

        public void FreeCameras()
        {
            _cameraStatusTimer.Stop();
            IsLiveGrabbing = false;

            foreach (var cam in _cameras)
            {
                cam.Free();
            }
            _cameras.Clear();

            foreach (var kvp in _allocatedSystems)
            {
                CameraSystemManager.FreeSystem(kvp.Value);
            }
            _allocatedSystems.Clear();

            CameraSystemManager.FreeApplication();

            IsAllocated = false;
            UpdateCameraStatus("已釋放 (Freed)", Color.Gray);
        }

        public void ReinitializeForAcquisitionSettings(bool enableImageProcessing, InspectionSettings settings)
        {
            bool wasLive = IsLiveGrabbing;
            if (wasLive)
            {
                StopGrab();
            }

            UpdateCaptureSettingsCache(settings);

            FreeCameras();
            AllocateCameras(enableImageProcessing);
            SetCaptureSettings(settings);

            if (wasLive)
            {
                StartGrab();
            }
        }

        public void SetImageProcessingEnabled(bool enable)
        {
            foreach (var cam in _cameras)
            {
                cam.EnableImageProcessing = enable;
            }
        }

        public void SetCaptureSettings(InspectionSettings settings)
        {
            if (settings == null) return;

            UpdateCaptureSettingsCache(settings);
            float hessianMaxFactor = settings.HessianMaxFactor > 0
                ? settings.HessianMaxFactor
                : InspectionEngineConfig.DefaultHessianMaxFactor;

            foreach (var cam in _cameras)
            {
                cam.EnableAutoCapture = _enableAutoCapture;
                cam.CaptureRootPath = _captureRootPath;
                cam.CameraGrabHeight = _cameraGrabHeight;
                cam.CameraExposureTimeUs = _cameraExposureTimeUs;
                cam.HessianSigma = InspectionEngineConfig.DefaultRidgeSigma;
                cam.HessianFixedMax = hessianMaxFactor;
                cam.ApplyAcquisitionSettings();
            }
        }

        private void UpdateCaptureSettingsCache(InspectionSettings settings)
        {
            if (settings == null) return;

            _enableAutoCapture = settings.EnableAutoCapture;
            _captureRootPath = settings.CaptureRootPath ?? string.Empty;
            _cameraGrabHeight = settings.CameraGrabHeight;
            _cameraExposureTimeUs = settings.CameraExposureTimeUs;
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
                if (kvp.Key == cameraIndex)
                    kvp.Value.BackColor = Color.PapayaWhip;
                else
                    kvp.Value.BackColor = Color.FromArgb(32, 32, 32);
            }

            foreach (var cam in _cameras)
            {
                if (cam.CameraId == cameraIndex)
                {
                    cam.SetSecondaryDisplay(_mainDisplayPanel.Handle);
                }
                else
                {
                    cam.SetSecondaryDisplay(IntPtr.Zero);
                }
            }
        }

        private void HandleMouseDataChanged(int camId, int x, int y, int pixelValue)
        {
            if (_mainForm.InvokeRequired)
            {
                _mainForm.BeginInvoke(new Action(() => HandleMouseDataChanged(camId, x, y, pixelValue)));
                return;
            }

            string infoText = pixelValue == -1
                ? $"即時影像 [CAM {camId}] | 游標超出影像範圍"
                : $"即時影像 [CAM {camId}] | X: {x}, Y: {y} | 灰階值: {pixelValue}";

            _updatePixelInfoCallback?.Invoke(infoText);
        }

        private void CameraStatusTimer_Tick(object sender, EventArgs e)
        {
            foreach (var cam in _cameras)
            {
                bool isConnected = cam.CheckPresence();
                if (isConnected && cam.UserWantsGrab && !cam.IsLive)
                {
                    cam.ApplyGrabState();
                }

                string fpsText = cam.IsLive ? $" | FPS: {cam.CurrentFps:F1}" : "";

                string statusText = isConnected
                    ? (cam.IsLive ? $"Live{fpsText}" : "Ready")
                    : "Offline";

                Color color = isConnected
                    ? (cam.IsLive ? Color.Green : Color.Yellow)
                    : Color.Red;

                UpdateSingleCameraStatus(cam.CameraId, statusText, color);
            }
        }

        private void UpdateCameraStatus(string statusText, Color color)
        {
            foreach (var pair in _cameraStatusLabels)
            {
                pair.Value.Text = $"{pair.Key}: {statusText}";
                pair.Value.ForeColor = color;
            }
        }

        private void UpdateSingleCameraStatus(int cameraIndex, string statusText, Color color)
        {
            if (_cameraStatusLabels.TryGetValue(cameraIndex, out var label))
            {
                label.Text = $"{cameraIndex}: {statusText}";
                label.ForeColor = color;
            }
        }
    }
}
