using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Interop;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Widgets;
using TanukiCv.Core;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.Managers
{
    /// <summary>
    /// 即時監控顯示協調者：擁有主畫面/縮圖/Waterfall/SmartCanvas/狀態標籤/滾輪縮放的 UI 狀態。
    /// 相機生命週期與 grab 控制仍留 <see cref="LiveCameraManager"/>；本類別只做顯示編排。
    /// </summary>
    internal sealed class LiveDisplayCoordinator
    {
        private readonly Form _mainForm;
        private readonly Panel _mainDisplayPanel;
        private readonly Panel[] _cameraPanels;
        private readonly Action<string> _updatePixelInfoCallback;
        private readonly GlobalMergeCoordinator _globalMerge;
        private readonly GpuGrayResizeProvider _gpuResizeProvider;
        private readonly Func<IReadOnlyList<AniloxCamera>> _getCameras;
        private readonly Func<InspectionSettings> _getSettings;
        private readonly Func<double[]> _getLineRates;
        private readonly Func<bool> _isLiveGrabbing;
        private readonly WheelZoomFilter _wheelFilter;

        private readonly Dictionary<int, Panel> _liveViewPanels = new Dictionary<int, Panel>();
        private readonly Dictionary<int, Panel> _liveParentPanels = new Dictionary<int, Panel>();
        private readonly Dictionary<int, Label> _cameraStatusLabels = new Dictionary<int, Label>();

        private LiveDisplayView _smartDisplay;
        private WaterfallView _waterfallView;
        private int _selectedMainCameraId = 1;
        private int _userSelectedMainCameraId = 1;
        private double _screenMmPerPx;

        public event Action<double, double, double, double> OnLiveViewRange;
        public Action OnAfterVerticalZoom { get; set; }

        public int SelectedMainCameraId => _selectedMainCameraId;
        public int UserSelectedMainCameraId => _userSelectedMainCameraId;
        public double ScreenMmPerPx => _screenMmPerPx;
        public double RowPitchMm { get; set; }

        public bool SmartCanvasMode
        {
            get
            {
                var settings = _getSettings();
                return settings != null && settings.he_MainDisplay == MainDisplayMode.SmartCanvas;
            }
        }

        public bool WaterfallMode
        {
            get
            {
                var settings = _getSettings();
                return settings != null && settings.he_MainDisplay == MainDisplayMode.Waterfall;
            }
        }

        public LiveDisplayCoordinator(
            Form mainForm,
            Panel[] cameraPanels,
            Panel mainDisplayPanel,
            Action<string> updatePixelInfoCallback,
            GlobalMergeCoordinator globalMerge,
            Func<IReadOnlyList<AniloxCamera>> getCameras,
            Func<InspectionSettings> getSettings,
            Func<double[]> getLineRates,
            Func<bool> isLiveGrabbing)
        {
            _mainForm = mainForm;
            _cameraPanels = cameraPanels ?? throw new ArgumentNullException(nameof(cameraPanels));
            _mainDisplayPanel = mainDisplayPanel ?? throw new ArgumentNullException(nameof(mainDisplayPanel));
            _updatePixelInfoCallback = updatePixelInfoCallback;
            _globalMerge = globalMerge ?? throw new ArgumentNullException(nameof(globalMerge));
            _getCameras = getCameras ?? throw new ArgumentNullException(nameof(getCameras));
            _getSettings = getSettings ?? throw new ArgumentNullException(nameof(getSettings));
            _getLineRates = getLineRates ?? throw new ArgumentNullException(nameof(getLineRates));
            _isLiveGrabbing = isLiveGrabbing ?? throw new ArgumentNullException(nameof(isLiveGrabbing));
            _gpuResizeProvider = new GpuGrayResizeProvider(
                NativeMethods.TanukiCv_AllocPinned,
                NativeMethods.TanukiCv_FreePinned,
                NativeMethods.TanukiCv_Resize_GPU);

            _mainDisplayPanel.BackColor = Color.Black;
            for (int i = 0; i < 7; i++)
                SetupLivePanel(_cameraPanels[i], i + 1);

            _wheelFilter = new WheelZoomFilter(this);
            Application.AddMessageFilter(_wheelFilter);
        }

        public void SetScreenMmPerPixel(double mmPerPx) => _screenMmPerPx = mmPerPx;

        public bool TryGetDisplayPanel(int cameraId, out Panel displayPanel)
            => _liveViewPanels.TryGetValue(cameraId, out displayPanel);

        public bool HasCameraStatusLabel(int cameraId)
            => _cameraStatusLabels.ContainsKey(cameraId);

        private void SetupLivePanel(Panel parentPanel, int cameraIndex)
        {
            parentPanel.BackColor = Color.Black;
            parentPanel.Padding = new Padding(2);
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
                Font = new Font("Segoe UI", 8.5f, FontStyle.Regular)
            };

            displayPanel.MouseClick += (s, e) => SwitchMainDisplay(cameraIndex);
            status.MouseClick += (s, e) => SwitchMainDisplay(cameraIndex);
            parentPanel.Paint += (s, e) => OnLivePanelPaint(s, e, cameraIndex);

            parentPanel.Controls.Add(displayPanel);
            parentPanel.Controls.Add(status);
            displayPanel.BringToFront();

            _liveViewPanels[cameraIndex] = displayPanel;
            _liveParentPanels[cameraIndex] = parentPanel;
            _cameraStatusLabels[cameraIndex] = status;
        }

        private void OnLivePanelPaint(object sender, PaintEventArgs e, int cameraIndex)
        {
            if (!(sender is Panel panel)) return;
            bool isSelected = cameraIndex == _selectedMainCameraId && !SmartCanvasMode;
            Color borderColor = isSelected ? Color.Orange : Color.FromArgb(60, 60, 60);
            int borderWidth = isSelected ? 3 : 1;
            ControlPaint.DrawBorder(e.Graphics, panel.ClientRectangle,
                borderColor, borderWidth, ButtonBorderStyle.Solid,
                borderColor, borderWidth, ButtonBorderStyle.Solid,
                borderColor, borderWidth, ButtonBorderStyle.Solid,
                borderColor, borderWidth, ButtonBorderStyle.Solid);
        }

        public void UpdateCameraStatus(string statusText, Color color)
        {
            foreach (var pair in _cameraStatusLabels)
            {
                pair.Value.Text = $"{pair.Key}: {statusText}";
                pair.Value.ForeColor = color;
            }
        }

        public void UpdateSingleCameraStatus(int cameraIndex, string statusText, Color color)
        {
            if (_cameraStatusLabels.TryGetValue(cameraIndex, out var label))
            {
                label.Text = $"{cameraIndex}: {statusText}";
                label.ForeColor = color;
            }
        }

        public void ApplyMainDisplayMode()
        {
            if (WaterfallMode)
            {
                TeardownSmartDisplay();
                EnableWaterfallDisplay();
            }
            else
            {
                DisableWaterfallDisplay();
                if (SmartCanvasMode) EnsureSmartDisplay();
                else TeardownSmartDisplay();
            }
        }

        public void ResetWaterfallIfActive()
        {
            if (WaterfallMode) _waterfallView?.Reset();
        }

        private void EnableWaterfallDisplay()
        {
            if (_waterfallView != null) return;
            if (_mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) return;

            var settings = _getSettings();
            int wfH = settings?.ImageView?.WaterfallTotalHeight ?? 30000;
            var wfMode = settings?.ImageView?.WaterfallFullMode ?? WaterfallFullMode.Restart;
            int slotCount = settings?.GetCameraStartPositionMmArray()?.Length ?? Cameras.Count;
            _waterfallView = new WaterfallView(_mainDisplayPanel, slotCount, wfH, wfMode, _screenMmPerPx);
            FeedWaterfallLayout();
            foreach (var cam in Cameras) cam.OnDisplayFrame += OnCameraWaterfallFrame;
        }

        public void FeedWaterfallLayout()
        {
            if (_waterfallView == null) return;
            if (_globalMerge.IsActive && _globalMerge.Merger != null && _globalMerge.Merger.SlotStartsMm != null)
            {
                _waterfallView.SetLayout(_globalMerge.Merger.SlotStartsMm, null, _globalMerge.Merger.RefOpsMm);
                return;
            }

            var settings = _getSettings();
            if (settings == null) return;
            var startMm = settings.GetCameraStartPositionMmArray();
            var opsUm = settings.GetCameraOpsUmArray();
            double refOps = (opsUm != null && opsUm.Length > 0 && opsUm[0] > 0) ? opsUm[0] / 1000.0 : 0.024;
            _waterfallView.SetLayout(startMm, null, refOps);
        }

        private void DisableWaterfallDisplay()
        {
            if (_waterfallView == null) return;
            foreach (var cam in Cameras) cam.OnDisplayFrame -= OnCameraWaterfallFrame;
            _waterfallView.Dispose();
            _waterfallView = null;
        }

        private void OnCameraWaterfallFrame(int camId, byte[] bytes, int w, int h, long tick)
            => _waterfallView?.PushFrame(camId, bytes, w, h, tick);

        public void RefreshWaterfallDisplay()
        {
            if (!WaterfallMode || _waterfallView == null) return;
            DisableWaterfallDisplay();
            EnableWaterfallDisplay();
        }

        private void EnsureSmartDisplay()
        {
            if (!SmartCanvasMode || _smartDisplay != null) return;
            if (_mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) return;

            _smartDisplay = new LiveDisplayView(_mainDisplayPanel, _cameraPanels, _screenMmPerPx);
            _smartDisplay.ApplyOptions(new LiveDisplayOptions
            {
                ThumbSelectedColor = Color.Orange,
                MergeAll = _globalMerge.IsActive,
                MergeMode = _globalMerge.IsActive
            });
            _smartDisplay.SelectRequested += SmartSelectCamera;
            _smartDisplay.SelectedCamChanged += camId => _selectedMainCameraId = camId;
            _smartDisplay.ViewRangeMmChanged += OnSmartViewRange;
            _smartDisplay.CursorStatusChanged += OnSmartCursorStatus;
            _smartDisplay.SetSelected(_selectedMainCameraId);

            if (_globalMerge.IsActive && _globalMerge.Merger != null)
            {
                var merger = _globalMerge.Merger;
                var ops = new double[merger.SlotStartsMm?.Length ?? 0];
                for (int i = 0; i < ops.Length; i++) ops[i] = merger.RefOpsMm * 1000.0;
                _smartDisplay.SetLayout(merger.SlotStartsMm, ops, 1, RowPitchMm);
            }
            foreach (var cam in Cameras) cam.OnDisplayFrame += OnCameraDisplayFrame;
            var settings = _getSettings();
            if (settings != null) SetLodMode(settings.LiveLod);
        }

        public void SetLodMode(LiveLodMode mode)
        {
            if (_smartDisplay == null) return;
            switch (mode)
            {
                case LiveLodMode.GPU: _gpuResizeProvider.Arm(); _smartDisplay.EnableLod(_gpuResizeProvider.Resize); break;
                case LiveLodMode.CPU: _smartDisplay.EnableLod(GrayResizeCpu.Resize); break;
                default: _smartDisplay.DisableLod(); break;
            }
        }

        public void TeardownSmartDisplay()
        {
            if (_smartDisplay == null) return;
            foreach (var cam in Cameras) cam.OnDisplayFrame -= OnCameraDisplayFrame;
            _smartDisplay.Dispose();
            _smartDisplay = null;
            _gpuResizeProvider.Release();
        }

        private void OnCameraDisplayFrame(int camId, byte[] bytes, int w, int h, long tick)
            => _smartDisplay?.PushFrame(camId, bytes, w, h);

        private void SmartSelectCamera(int camId) => SwitchMainDisplay(camId);

        private void OnSmartViewRange(double leftMm, double rightMm, double topMm, double botMm)
            => OnLiveViewRange?.Invoke(leftMm, rightMm, topMm, botMm);

        private void OnSmartCursorStatus(LiveDisplayView.CursorStatus s)
        {
            if (_updatePixelInfoCallback == null) return;
            string tag = _globalMerge.IsActive ? "全域合圖" : $"CAM {s.SelectedCamId}";
            _updatePixelInfoCallback.Invoke(
                $"即時影像 [{tag}] | " +
                $"位置:({s.CurMmX:F2}, {s.CurMmY:F2}) mm | " +
                $"X範圍:{s.ViewLeftMm:F1}~{s.ViewRightMm:F1} mm | " +
                $"Y範圍:{s.ViewTopMm:F1}~{s.ViewBotMm:F1} mm | " +
                $"座標: ({s.CursorX}, {s.CursorY}) | " +
                $"亮度: {s.Brightness} | " +
                $"實體倍率:{(s.PhysMag > 0 ? $"{s.PhysMag:F2}x" : "-")}");
        }

        public void SwitchMainDisplay(int cameraIndex)
        {
            if (_mainForm == null || _mainForm.IsDisposed
                || _mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) return;

            if (_mainForm.InvokeRequired)
            {
                try { _mainForm.BeginInvoke(new Action(() => SwitchMainDisplay(cameraIndex))); }
                catch (InvalidOperationException) { }
                return;
            }

            _selectedMainCameraId = cameraIndex;
            _userSelectedMainCameraId = cameraIndex;
            _smartDisplay?.SetSelected(cameraIndex);

            foreach (var kvp in _liveParentPanels)
                kvp.Value.Invalidate();

            if (_globalMerge.IsActive)
            {
                _globalMerge.PanToCameraCenter(cameraIndex);
                return;
            }

            foreach (var cam in Cameras)
            {
                if (!SmartCanvasMode && cam.CameraId == cameraIndex)
                    cam.SetSecondaryDisplay(_mainDisplayPanel.Handle);
                else
                    cam.SetSecondaryDisplay(IntPtr.Zero);
            }
        }

        public void OnMergedViewCenterCam(int newId)
        {
            if (newId == _selectedMainCameraId) return;
            _selectedMainCameraId = newId;
            foreach (var kvp in _liveParentPanels)
                kvp.Value.Invalidate();
        }

        public void ResetMainDisplayView()
        {
            if (_globalMerge.IsActive && _globalMerge.HasMilDisplay)
            {
                _globalMerge.ResetView();
                return;
            }
            FindCamera(_selectedMainCameraId)?.ResetSecondaryDisplayView();
        }

        public void SetPhysicalMagnification1x()
        {
            if (!_isLiveGrabbing() || _screenMmPerPx <= 0) return;

            if (_globalMerge.IsActive && _globalMerge.HasMilDisplay)
            {
                _globalMerge.SetPhysical1x();
                return;
            }

            int camIdx = _selectedMainCameraId - 1;
            var settings = _getSettings();
            double[] opsUmArr = settings?.GetCameraOpsUmArray();
            if (opsUmArr == null || camIdx < 0 || camIdx >= opsUmArr.Length) return;

            double opsInMm = opsUmArr[camIdx] / 1000.0;
            if (opsInMm <= 0) return;

            double zoom1xCam = PixelMmMapper.OneToOneZoom(opsInMm, _screenMmPerPx);
            var cam = FindCamera(_selectedMainCameraId);
            if (cam == null) return;

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

        internal void ApplyCustomZoom(int wheelDelta)
        {
            if (!_isLiveGrabbing()) return;

            if (_globalMerge.IsActive && _globalMerge.HasMilDisplay)
            {
                _globalMerge.ApplyZoom(wheelDelta);
                return;
            }

            var cam = FindCamera(_selectedMainCameraId);
            if (cam == null) return;
            if (!cam.TryGetSecondaryDisplayGeometry(out double zoomX, out _, out double panX, out double panY))
                return;

            double factor2 = wheelDelta > 0 ? 1.1 : (1.0 / 1.1);
            double newZoom2 = zoomX * factor2;
            if (newZoom2 < 0.05) newZoom2 = 0.05;
            if (newZoom2 > 32.0) newZoom2 = 32.0;

            double cx2 = _mainDisplayPanel.Width / 2.0;
            double cy2 = _mainDisplayPanel.Height / 2.0;
            double imgX2 = panX + cx2 / zoomX;
            double imgY2 = panY + cy2 / zoomX;
            double newPanX2 = imgX2 - cx2 / newZoom2;
            double newPanY2 = imgY2 - cy2 / newZoom2;

            cam.SetSecondaryDisplayZoom(newZoom2, newPanX2, newPanY2);
            OnAfterVerticalZoom?.Invoke();
        }

        public void HandleMouseDataChanged(int camId, int x, int y, int pixelValue, bool isReleasing)
        {
            if (isReleasing || _mainForm == null || _mainForm.IsDisposed || !_mainForm.IsHandleCreated) return;
            if (_mainForm.InvokeRequired)
            {
                try { _mainForm.BeginInvoke(new Action(() => HandleMouseDataChanged(camId, x, y, pixelValue, isReleasing))); }
                catch (InvalidOperationException) { }
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
                var settings = _getSettings();
                double[] opsUmArr = settings?.GetCameraOpsUmArray();
                double[] startMmArr = settings?.GetCameraStartPositionMmArray();

                if (opsUmArr == null || camIdx < 0 || camIdx >= opsUmArr.Length)
                {
                    infoText = $"即時影像 [CAM {camId}] | 座標: ({x}, {y}) | 亮度: {pixelValue}";
                }
                else
                {
                    double opsInMm = opsUmArr[camIdx] / 1000.0;
                    double startPosMm = startMmArr[camIdx];
                    double physicalX = PixelMmMapper.PixelToMm(x, startPosMm, opsInMm);
                    double[] lineRates = _getLineRates();
                    double lineRateHz = (lineRates != null && camIdx < lineRates.Length) ? lineRates[camIdx] : 0;
                    double speedMPerMin = settings.AniloxRollSpeedMPerMin;
                    double rowPitchMm = (speedMPerMin > 0 && lineRateHz > 0)
                        ? (speedMPerMin / 60.0 * 1000.0) / lineRateHz : 0;
                    double physicalY = y * rowPitchMm;

                    string rangeStr = "";
                    string magStr = "-";
                    var cam = FindCamera(camId);
                    if (cam != null && cam.TryGetSecondaryDisplayGeometry(
                            out double zoomX, out _, out double panOffX, out double panOffY))
                    {
                        double panelW = _mainDisplayPanel.Width;
                        double panelH = _mainDisplayPanel.Height;
                        double viewLeftMm = PixelMmMapper.PixelToMm(panOffX, startPosMm, opsInMm);
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

        public void OnGlobalMergeEnabled(double[] opsUm, double[] startPosMm)
        {
            if (SmartCanvasMode && _smartDisplay != null)
            {
                _smartDisplay.SetLayout(startPosMm, opsUm, 1, RowPitchMm);
                _smartDisplay.ApplyOptions(new LiveDisplayOptions
                {
                    ThumbSelectedColor = Color.Orange,
                    MergeAll = true,
                    MergeMode = true
                });
            }
        }

        public void OnGlobalMergeDisabled()
        {
            _smartDisplay?.ApplyOptions(new LiveDisplayOptions
            {
                ThumbSelectedColor = Color.Orange
            });
            SwitchMainDisplay(_userSelectedMainCameraId);
        }

        public void RefreshGlobalMergeLayout(double[] opsUm, double[] startPosMm, double refOpsMm)
        {
            if (SmartCanvasMode && _smartDisplay != null)
                _smartDisplay.SetLayout(startPosMm, opsUm, 1, RowPitchMm);
            if (_waterfallView != null && _globalMerge.Merger != null)
                _waterfallView.SetLayout(startPosMm, opsUm, refOpsMm);
        }

        private IReadOnlyList<AniloxCamera> Cameras => _getCameras() ?? Array.Empty<AniloxCamera>();

        private AniloxCamera FindCamera(int camId)
        {
            var cameras = Cameras;
            for (int i = 0; i < cameras.Count; i++)
                if (cameras[i].CameraId == camId) return cameras[i];
            return null;
        }

        private sealed class WheelZoomFilter : IMessageFilter
        {
            private const int WM_MOUSEWHEEL = 0x020A;
            private readonly LiveDisplayCoordinator _display;

            public WheelZoomFilter(LiveDisplayCoordinator display) => _display = display;

            public bool PreFilterMessage(ref Message m)
            {
                if (m.Msg != WM_MOUSEWHEEL) return false;
                if (_display.SmartCanvasMode || _display.WaterfallMode) return false;
                if (!_display._isLiveGrabbing()) return false;

                var panel = _display._mainDisplayPanel;
                var screenPt = Cursor.Position;
                if (!panel.RectangleToScreen(panel.ClientRectangle).Contains(screenPt))
                    return false;

                int delta = (short)(m.WParam.ToInt64() >> 16);
                _display.ApplyCustomZoom(delta);
                return true;
            }
        }
    }
}
