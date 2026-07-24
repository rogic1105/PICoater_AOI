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
    /// 即時監控顯示協調者：擁有主畫面/縮圖/Waterfall/ImageCanvas/狀態標籤/滾輪縮放的 UI 狀態。
    /// 相機生命週期與 grab 控制仍留 <see cref="LiveCameraManager"/>；本類別只做顯示編排。
    /// </summary>
    internal sealed class LiveDisplayCoordinator
    {
        private readonly Form _mainForm;
        private readonly Panel _mainDisplayPanel;
        private readonly Panel[] _cameraPanels;
        private readonly GlobalMergeCoordinator _globalMerge;
        private readonly GpuGrayResizeProvider _gpuResizeProvider;
        private readonly Func<IReadOnlyList<AniloxCamera>> _getCameras;
        private readonly Func<InspectionSettings> _getSettings;
        private readonly Func<double[]> _getLineRates;
        private readonly Func<bool> _isLiveGrabbing;

        private readonly Dictionary<int, Panel> _liveViewPanels = new Dictionary<int, Panel>();
        private readonly Dictionary<int, Panel> _liveParentPanels = new Dictionary<int, Panel>();
        private readonly Dictionary<int, Label> _cameraStatusLabels = new Dictionary<int, Label>();

        private ImageDisplayView _imageDisplay;
        private WaterfallView _waterfallView;
        private ThumbStrip _waterfallThumbs;   // 顯示鐵則1：縮圖在瀑布模式也一律即時影像（與即時模式同源 CPU ThumbStrip）
        private WaterfallFrameLayer _waterfallDisplayLayer = WaterfallFrameLayer.Raw;
        private CanvasOverlayMode _overlayMode = CanvasOverlayMode.Coordinates;
        private bool _applyingOverlayMode;

        // ── [Flow] 顯示資料流跡：咽喉點各一行（按鈕/設定→接線→首幀→顯示），落 Logs\trace-*.log。
        //    驗證＝跑一輪操作後把 log 與「預期步驟（DVT）」比對；headless 可驗、不用肉眼盯 UI。
        //    時間戳+執行緒 ID 由 FlowTrace 統一附加（多執行緒交錯時按執行緒拆序列+時間戳看先後）。
        private static void Flow(string msg) => FlowTrace.Log(msg);
        private readonly HashSet<int> _flowFirstFrame = new HashSet<int>();   // 每次 StartGrab/重建後，各 cam 首幀記一行

        /// <summary>重置首幀 log 追蹤（每次 StartGrab 呼叫）→ 每輪 grab 都能驗「幀有流到 view」。</summary>
        public void ResetFlowFirstFrame() { lock (_flowFirstFrame) _flowFirstFrame.Clear(); }

        /// <summary>各 cam 首幀記一行（MIL 回呼執行緒，多 cam 並發 → 鎖）。target=IC/WF。</summary>
        private void FlowFirstFrame(int camId, int w, int h, string target)
        {
            bool first;
            lock (_flowFirstFrame) first = _flowFirstFrame.Add(camId);
            if (first) Flow($"firstFrame cam{camId} {w}x{h} → {target}");
        }
        private int _selectedMainCameraId = 1;
        private int _userSelectedMainCameraId = 1;
        private double _screenMmPerPx;

        public event Action<double, double, double, double> OnLiveViewRange;
        public event Action MainContentPresented;
        public event Action<CanvasOverlayMode> OverlayModeChanged;

        public int SelectedMainCameraId => _selectedMainCameraId;
        public int UserSelectedMainCameraId => _userSelectedMainCameraId;
        public double ScreenMmPerPx => _screenMmPerPx;
        public CanvasOverlayMode OverlayMode
        {
            get => _overlayMode;
            set
            {
                _overlayMode = value;
                ApplyOverlayModeToViews();
            }
        }
        public double RowPitchMm
        {
            get => _rowPitchMm;
            set
            {
                _rowPitchMm = value;
                _waterfallView?.SetRowPitch(value);
            }
        }
        private double _rowPitchMm;

        public bool ImageCanvasMode
        {
            get
            {
                var settings = _getSettings();
                return settings != null && settings.he_MainDisplay == MainDisplayMode.ImageCanvas;
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

        private bool ShouldFlipVertical
            => (_getSettings()?.ImageView?.VerticalDirection ?? InspectionDefaults.VerticalDirection)
               == VerticalDisplayDirection.BottomToTop;

        public LiveDisplayCoordinator(
            Form mainForm,
            Panel[] cameraPanels,
            Panel mainDisplayPanel,
            GlobalMergeCoordinator globalMerge,
            Func<IReadOnlyList<AniloxCamera>> getCameras,
            Func<InspectionSettings> getSettings,
            Func<double[]> getLineRates,
            Func<bool> isLiveGrabbing)
        {
            _mainForm = mainForm;
            _cameraPanels = cameraPanels ?? throw new ArgumentNullException(nameof(cameraPanels));
            _mainDisplayPanel = mainDisplayPanel ?? throw new ArgumentNullException(nameof(mainDisplayPanel));
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

        }

        public void SetScreenMmPerPixel(double mmPerPx) => _screenMmPerPx = mmPerPx;

        public bool TryGetDisplayPanel(int cameraId, out Panel displayPanel)
            => _liveViewPanels.TryGetValue(cameraId, out displayPanel);

        public bool HasCameraStatusLabel(int cameraId)
            => _cameraStatusLabels.ContainsKey(cameraId);

        private void SetupLivePanel(Panel parentPanel, int cameraIndex)
        {
            parentPanel.BackColor = Color.Black;
            parentPanel.Padding = Padding.Empty;   // 縮圖貼邊（與回顧同款：橘框緊貼 cell 邊緣）
            parentPanel.Controls.Clear();

            var displayPanel = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = Color.Black
            };

            // 狀態列改「浮在縮圖上、內縮於橘框內側」（非 Dock.Bottom 佔空間）：ThumbView 吃滿整個 panel
            // → 橘框貼邊與回顧同款；label 內縮 3px（框線畫在 1,1,w-3,h-3、2px 粗）→ 不蓋掉橘框。
            var status = new Label
            {
                Height = 18,
                Width = Math.Max(1, parentPanel.ClientSize.Width - 6),
                Top = Math.Max(0, parentPanel.ClientSize.Height - 18 - 3),
                Left = 3,
                Anchor = AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right,
                ForeColor = Color.DarkGray,
                BackColor = Color.FromArgb(32, 32, 32),
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font("Segoe UI", 8.5f, FontStyle.Regular)
            };

            displayPanel.MouseClick += (s, e) => SwitchMainDisplay(cameraIndex, centerView: true);
            status.MouseClick += (s, e) => SwitchMainDisplay(cameraIndex, centerView: true);
            parentPanel.Paint += (s, e) => OnLivePanelPaint(s, e, cameraIndex);

            parentPanel.Controls.Add(displayPanel);
            parentPanel.Controls.Add(status);
            status.BringToFront();   // 狀態字浮在縮圖上（縮圖 ThumbView 建立後由 BringStatusLabelsToFront 再抬一次）

            _liveViewPanels[cameraIndex] = displayPanel;
            _liveParentPanels[cameraIndex] = parentPanel;
            _cameraStatusLabels[cameraIndex] = status;
        }

        /// <summary>ThumbStrip 建立 ThumbView 時會 BringToFront → 蓋掉浮動狀態字；建立後呼叫本方法把狀態字抬回最上。</summary>
        private void BringStatusLabelsToFront()
        {
            foreach (var kvp in _cameraStatusLabels)
                if (!kvp.Value.IsDisposed) kvp.Value.BringToFront();
        }

        private void OnLivePanelPaint(object sender, PaintEventArgs e, int cameraIndex)
        {
            // 縮圖選中框由 ThumbStrip 橘色高亮唯一負責（顯示鐵則：兩模式縮圖同源同行為）；此處不畫任何框。
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

        // 模式錯掛主動偵測（S 系列 view 互斥不變量的執行期自檢）：幀流進「不屬於當前模式」的路徑
        // ＝訂閱錯掛，立即自報違規（每個 view 週期警一次，避免每幀洗版）。
        private bool _warnFrameToIcInWf, _warnFrameToWfInIc;

        public void ApplyMainDisplayMode()
        {
            _warnFrameToIcInWf = _warnFrameToWfInIc = false;
            // 靜音鍵＝閘門輸入之一：預覽中任何人呼叫閘門（改設定/相機數/合圖重建）都得到預覽畫面
            // ＝「存活」policy（設定記著、退出預覽才生效）。修「平行通道被閘門踩壞」整類病（2026-07-09）。
            if (_bgPreviewOverride)
            {
                Flow("ApplyMainDisplayMode → BgPreview（靜音鍵，設定不動）");
                DisableWaterfallDisplay();
                EnsureImageDisplay();
                RefreshEnhanceColorMap();
                ApplyBgPreviewLayout();
                return;
            }
            Flow($"ApplyMainDisplayMode → {(WaterfallMode ? "Waterfall" : "ImageCanvas")}");
            if (WaterfallMode)
            {
                TeardownImageDisplay();
                EnableWaterfallDisplay();
            }
            else
            {
                DisableWaterfallDisplay();
                if (ImageCanvasMode) EnsureImageDisplay();
                else TeardownImageDisplay();
            }
            RefreshEnhanceColorMap();
        }

        public void ResetWaterfallIfActive()
        {
            if (!WaterfallMode || _waterfallView == null) return;
            SeedWaterfallFramePeriod();
            _waterfallView.Reset();
        }

        public void RefireMainViewRange(string reason)
        {
            if (WaterfallMode && _waterfallView != null)
            {
                _waterfallView.RefireViewRange();
                Flow($"viewRange refire reason={reason} mode=WF");
                return;
            }

            if (_imageDisplay != null)
            {
                _imageDisplay.RefireViewRange();
                Flow($"viewRange refire reason={reason} mode=IC");
            }
        }

        private void SeedWaterfallFramePeriod()
        {
            if (_waterfallView == null) return;

            foreach (var cam in Cameras)
            {
                if (cam == null || !cam.IsConnected || cam.FrameHeight <= 0 ||
                    cam.AppliedLineRateHz <= 0 || cam.DataLatchClockFreqHz <= 0)
                    continue;

                double periodMs = cam.FrameHeight * 1000.0 / cam.AppliedLineRateHz;
                long periodTicks = (long)Math.Round(
                    cam.DataLatchClockFreqHz * cam.FrameHeight / cam.AppliedLineRateHz);
                if (periodTicks <= 0 || periodMs <= 0) continue;

                _waterfallView.SetExpectedFramePeriod(periodTicks, periodMs);
                Flow($"WF bootstrap period cam{cam.CameraId} periodMs={periodMs:F3} source=applied-hardware");
                return;
            }

            Flow("WF bootstrap period unavailable; learn from runtime frames");
        }

        public void SetWaterfallDisplayLayer(WaterfallFrameLayer layer)
        {
            _waterfallDisplayLayer = layer;
            _waterfallView?.SetDisplayLayer(layer);
            RefreshEnhanceColorMap();
        }

        /// <summary>
        /// 強化熱力圖只改主畫面調色盤；縮圖、背景預覽與原圖恒保持灰階。
        /// 現有 8-bit 幀與瀑布歷史直接重畫，不重讀檔、不重算 Curve。
        /// </summary>
        public void RefreshEnhanceColorMap()
        {
            var settings = _getSettings();
            IntensityColorMap imageMap = _bgPreviewOverride || settings?.ImageView == null
                ? IntensityColorMap.Grayscale
                : settings.ImageView.ResolveColorMap(settings.EnableMuraEnhance);
            IntensityColorMap waterfallMap = _bgPreviewOverride || settings?.ImageView == null
                ? IntensityColorMap.Grayscale
                : settings.ImageView.ResolveColorMap(_waterfallDisplayLayer != WaterfallFrameLayer.Raw);

            if (_imageDisplay != null)
            {
                _imageDisplay.MainColorMap = imageMap;
                _imageDisplay.RefreshNow();
            }
            if (_waterfallView != null)
                _waterfallView.ColorMap = waterfallMap;
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
            _waterfallView.InformationTextProvider = () =>
                CanvasParameterTextBuilder.FromCurrentSettings(_getSettings());
            _waterfallView.SetDisplayLayer(_waterfallDisplayLayer);
            _waterfallView.ColorMap = ResolveWaterfallColorMap();
            _waterfallView.FlipVertical = ShouldFlipVertical;
            _waterfallView.SetRowPitch(RowPitchMm);
            _waterfallView.SelectRequested += OnWaterfallSelectRequested;
            _waterfallView.ViewRangeMmChanged += OnImageViewRange;
            _waterfallView.ContentPresented += OnMainContentPresented;
            _waterfallView.CenterCamChanged += OnWaterfallCenterCam;   // 主畫面 pan → 縮圖橘框跟隨（反向連動）
            _waterfallView.FlowLog = s => FlowTrace.Display("WF", s);  // 互動 intent與診斷快照分級
            _waterfallView.Canvas.OverlayMode = _overlayMode;
            _waterfallView.Canvas.OverlayModeChanged += OnCanvasOverlayModeChanged;

            // 顯示鐵則1：瀑布模式的 7 台縮圖也一律即時影像（CPU ThumbStrip，與即時模式同源；點選/高亮一致）。
            _waterfallThumbs = new ThumbStrip(_cameraPanels);
            _waterfallThumbs.FlipVertical = ShouldFlipVertical;
            _waterfallThumbs.SetSelectedColor(Color.Orange);   // 與即時模式一致（選中框唯一樣式＝橘色）
            _waterfallThumbs.SelectRequested += idx => SwitchMainDisplay(idx + 1, centerView: true);
            _waterfallThumbs.SetSelected(_selectedMainCameraId - 1);
            BringStatusLabelsToFront();

            FeedWaterfallLayout();
            foreach (var cam in Cameras) cam.OnWaterfallFrame += OnCameraWaterfallFrame;
            _flowFirstFrame.Clear();
            Flow($"EnableWaterfall create（view+thumbs）+ subscribe {Cameras.Count} cams");
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

        /// <summary>放掉相機時呼叫：瀑布訂閱的是各 cam.OnWaterfallFrame，相機被 Free 後這些訂閱指向死物件，
        /// 且 EnableWaterfallDisplay 冪等（_waterfallView!=null 早退）→ 不 teardown 就不會重新訂閱新相機 → 空白。
        /// 故 FreeCameras 必須連瀑布一起 teardown（與 TeardownImageDisplay 對稱）。</summary>
        public void TeardownWaterfallDisplay() => DisableWaterfallDisplay();

        private void DisableWaterfallDisplay()
        {
            if (_waterfallView == null) return;
            Flow($"TeardownWaterfall（unsubscribe {Cameras.Count} cams）");
            foreach (var cam in Cameras) cam.OnWaterfallFrame -= OnCameraWaterfallFrame;
            _waterfallView.SelectRequested -= OnWaterfallSelectRequested;
            _waterfallView.ViewRangeMmChanged -= OnImageViewRange;
            _waterfallView.ContentPresented -= OnMainContentPresented;
            _waterfallView.CenterCamChanged -= OnWaterfallCenterCam;
            _waterfallView.Canvas.OverlayModeChanged -= OnCanvasOverlayModeChanged;
            _waterfallView.Dispose();
            _waterfallView = null;
            _waterfallThumbs?.Dispose();
            _waterfallThumbs = null;
        }

        private void OnCameraWaterfallFrame(
            int camId, byte[] raw, byte[] column, byte[] row, int w, int h, long tick)
        {
            if (ImageCanvasMode && !_warnFrameToWfInIc)
            {
                _warnFrameToWfInIc = true;
                Flow("⚠ 契約違規：即時模式下幀流入瀑布路徑（訂閱錯掛/殘留）");
            }
            FlowFirstFrame(camId, w, h, "Waterfall");
            _waterfallView?.PushFrameVariants(camId, raw, column, row, w, h, tick);
            byte[] thumb = _waterfallDisplayLayer == WaterfallFrameLayer.Column ? column
                : _waterfallDisplayLayer == WaterfallFrameLayer.Row ? row
                : raw;
            _waterfallThumbs?.PushFrame(camId - 1, thumb ?? raw, w, h);
        }

        private void OnWaterfallSelectRequested(int camId)
        {
            if (camId > 0) SwitchMainDisplay(camId);
        }

        /// <summary>瀑布主畫面 pan/zoom → 視野中心相機變更 → 縮圖橘框跟隨（程式化來源，不再回頭置中防遞迴）。</summary>
        private void OnWaterfallCenterCam(int camId)
        {
            _selectedMainCameraId = camId;
            _waterfallThumbs?.SetSelected(camId - 1);
            Flow($"centerCam → cam{camId}（WF）");   // F6：拖主畫面→橘框跟隨（每跨一台一行,可驗）
        }

        public void RefreshWaterfallDisplay()
        {
            if (!WaterfallMode || _waterfallView == null) return;
            DisableWaterfallDisplay();
            EnableWaterfallDisplay();
        }

        private void EnsureImageDisplay()
        {
            if ((!ImageCanvasMode && !_bgPreviewOverride) || _imageDisplay != null) return;
            if (_mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) return;

            _imageDisplay = new ImageDisplayView(_mainDisplayPanel, _cameraPanels, _screenMmPerPx);
            _imageDisplay.Canvas.InformationTextProvider = () =>
                CanvasParameterTextBuilder.FromCurrentSettings(_getSettings());
            ApplyImageDisplayOptions(_globalMerge.IsActive);
            _imageDisplay.SelectRequested += ImageSelectCamera;
            _imageDisplay.SelectedCamChanged += camId =>
            {
                _selectedMainCameraId = camId;
                Flow($"centerCam → cam{camId}（IC）");   // F6：拖主畫面→橘框跟隨（每跨一台一行,可驗）
            };
            _imageDisplay.FlowLog = s => FlowTrace.Display("IC", s);   // 互動 intent與診斷快照分級
            _imageDisplay.Canvas.OverlayMode = _overlayMode;
            _imageDisplay.Canvas.OverlayModeChanged += OnCanvasOverlayModeChanged;
            _imageDisplay.ViewRangeMmChanged += OnImageViewRange;
            _imageDisplay.ContentPresented += OnMainContentPresented;
            _imageDisplay.SetSelected(_selectedMainCameraId);

            if (_globalMerge.IsActive && _globalMerge.Merger != null)
            {
                var merger = _globalMerge.Merger;
                var ops = new double[merger.SlotStartsMm?.Length ?? 0];
                for (int i = 0; i < ops.Length; i++) ops[i] = merger.RefOpsMm * 1000.0;
                _imageDisplay.SetLayout(merger.SlotStartsMm, ops, 1, RowPitchMm);
            }
            foreach (var cam in Cameras) cam.OnDisplayFrame += OnCameraDisplayFrame;
            ClearMissingCameraFrames();
            BringStatusLabelsToFront();   // ImageDisplayView 內建 ThumbStrip 建立後，狀態字抬回縮圖之上
            var settings = _getSettings();
            if (settings != null) SetLodMode(settings.LiveLod);
            _flowFirstFrame.Clear();
            Flow($"EnsureImageDisplay create + subscribe {Cameras.Count} cams（merge={_globalMerge.IsActive}）");
        }

        // ==================== 背景預覽（顯示鐵則0：預覽主畫面＝7 台背景合圖，走同一個 ImageDisplayView）====================
        /// <summary>預覽期間暫用 IC view 的「view 層靜音鍵」——不動 he_MainDisplay 設定（SSoT 不被暫時態污染）。</summary>
        private bool _bgPreviewOverride;

        /// <summary>預覽狀態唯一真相（form 唯讀轉發，不再自存一份）。</summary>
        public bool IsBgPreviewActive => _bgPreviewOverride;

        /// <summary>進入背景預覽：只改狀態→呼閘門（建拆一律閘門做，本方法不再自己動手）。
        /// 之後由呼叫端 PushStaticFrame 餵各台背景灰階 → 合圖/縮圖/縮放/overlay 全走共用路。</summary>
        public void EnterBackgroundPreview()
        {
            _bgPreviewOverride = true;
            ApplyMainDisplayMode();
            Flow($"EnterBackgroundPreview（view={(_imageDisplay != null)} merge={_globalMerge.IsActive} mode={(WaterfallMode ? "WF設定" : "IC")}）");
        }

        /// <summary>預覽佈局：合圖未啟用（未 grab）時用設定值餵佈局（合圖啟用時 EnsureImageDisplay 已設）。</summary>
        private void ApplyBgPreviewLayout()
        {
            if (_imageDisplay == null) return;
            if (_globalMerge.IsActive && _globalMerge.Merger != null) return;
            var st = _getSettings();
            if (st != null)
                _imageDisplay.SetLayout(st.GetCameraStartPositionMmArray(), st.GetCameraOpsUmArray(), 1, RowPitchMm);
        }

        /// <summary>離開背景預覽：清幀 + 回設定模式（WF 會重建）。呼叫端負責 grab 前先呼叫本方法。</summary>
        public void ExitBackgroundPreview(int camCount)
        {
            if (!_bgPreviewOverride) return;
            _bgPreviewOverride = false;
            Flow("ExitBackgroundPreview");
            if (_imageDisplay != null)
                for (int camId = 1; camId <= camCount; camId++) _imageDisplay.ClearFrame(camId);
            ApplyMainDisplayMode();
        }

        /// <summary>餵「靜態幀」（背景預覽等非相機來源）進共用顯示；與 grab 幀同一條路（PushFrame）。</summary>
        public void PushStaticFrame(int camId, byte[] gray, int w, int h)
        {
            Flow($"bgPreview push cam{camId} {w}x{h}（view={(_imageDisplay != null)}）");
            _imageDisplay?.PushFrame(camId, gray, w, h);
        }

        public void SetLodMode(LiveLodMode mode)
        {
            if (_imageDisplay == null) return;
            switch (mode)
            {
                case LiveLodMode.GPU: _gpuResizeProvider.Arm(); _imageDisplay.EnableLod(_gpuResizeProvider.Resize); break;
                case LiveLodMode.CPU: _imageDisplay.EnableLod(GrayResizeCpu.Resize); break;
                default: _imageDisplay.DisableLod(); break;
            }
        }

        public void ApplyDisplayDirection()
        {
            if (_imageDisplay != null)
            {
                ApplyImageDisplayOptions(_globalMerge.IsActive);
                _imageDisplay.RefireViewRange();
            }
            if (_waterfallView != null)
            {
                _waterfallView.FlipVertical = ShouldFlipVertical;
                _waterfallView.RefireViewRange();
            }
            if (_waterfallThumbs != null)
                _waterfallThumbs.FlipVertical = ShouldFlipVertical;   // 縮圖翻轉跟上方向變更（原只在建立時設＝中途改方向縮圖殘舊）
        }

        private void ApplyImageDisplayOptions(bool mergeMode)
        {
            _imageDisplay?.ApplyOptions(new LiveDisplayOptions
            {
                ThumbSelectedColor = Color.Orange,
                MergeAll = mergeMode,
                MergeMode = mergeMode,
                // 影像翻轉與瀑布同一條規則（由下而上=翻）。曾為修「切換方向反」而取反 → 物理朝向顛倒
                // （2026-07-08 使用者抓包），已回正；chart 資料規則同步去 XOR（見 RowCurveDisplayAdapter）。
                FlipVertical = ShouldFlipVertical
            });
            // 垂直座標約定與影像翻轉解耦（live 影像不翻但座標要照「上下方向」）：由下而上＝0 錨定畫面底
            if (_imageDisplay != null)
            {
                _imageDisplay.VerticalZeroAtBottom = ShouldFlipVertical;
                _imageDisplay.MainColorMap = ResolveImageColorMap();
            }
        }

        private IntensityColorMap ResolveImageColorMap()
        {
            var settings = _getSettings();
            return _bgPreviewOverride || settings?.ImageView == null
                ? IntensityColorMap.Grayscale
                : settings.ImageView.ResolveColorMap(settings.EnableMuraEnhance);
        }

        private IntensityColorMap ResolveWaterfallColorMap()
        {
            var settings = _getSettings();
            return _bgPreviewOverride || settings?.ImageView == null
                ? IntensityColorMap.Grayscale
                : settings.ImageView.ResolveColorMap(_waterfallDisplayLayer != WaterfallFrameLayer.Raw);
        }

        public void TeardownImageDisplay()
        {
            if (_imageDisplay == null) return;
            Flow($"TeardownImageDisplay（unsubscribe {Cameras.Count} cams）");
            foreach (var cam in Cameras) cam.OnDisplayFrame -= OnCameraDisplayFrame;
            _imageDisplay.ContentPresented -= OnMainContentPresented;
            _imageDisplay.Canvas.OverlayModeChanged -= OnCanvasOverlayModeChanged;
            _imageDisplay.Dispose();
            _imageDisplay = null;
            _gpuResizeProvider.Release();
        }

        private void OnCameraDisplayFrame(int camId, byte[] bytes, int w, int h, long tick)
        {
            // 背景預覽期間相機幀不進 view（預覽=靜態幀來源）：取得背景自動停後的「排水殘幀」
            // 晚到 ~1s 會蓋掉剛推的背景（2026-07-09 log 定罪 29.585 firstFrame→IC）。Exit 即恢復。
            if (_bgPreviewOverride) return;
            if (WaterfallMode && !_warnFrameToIcInWf)
            {
                _warnFrameToIcInWf = true;
                Flow("⚠ 契約違規：瀑布模式下幀流入 IC 路徑（訂閱錯掛/殘留）");
            }
            FlowFirstFrame(camId, w, h, "ImageDisplayView");
            _imageDisplay?.PushFrame(camId, bytes, w, h);
        }

        public void ClearCameraFrame(int camId)
        {
            _imageDisplay?.ClearFrame(camId);
        }

        public void ClearMissingCameraFrames()
        {
            if (_imageDisplay == null) return;
            var present = new bool[_cameraPanels.Length + 1];
            foreach (var cam in Cameras)
                if (cam != null && cam.CameraId > 0 && cam.CameraId < present.Length)
                    present[cam.CameraId] = true;
            _imageDisplay.ClearFramesExcept(present);
        }

        private void ImageSelectCamera(int camId) => SwitchMainDisplay(camId);

        private void OnImageViewRange(double leftMm, double rightMm, double topMm, double botMm)
            => OnLiveViewRange?.Invoke(leftMm, rightMm, topMm, botMm);

        private void OnMainContentPresented()
        {
            try { MainContentPresented?.Invoke(); }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"[LiveDisplayCoordinator.MainContentPresented] {ex.GetType().Name}: {ex.Message}");
            }
        }

        private void ApplyOverlayModeToViews()
        {
            if (_applyingOverlayMode) return;
            _applyingOverlayMode = true;
            try
            {
                if (_imageDisplay != null) _imageDisplay.Canvas.OverlayMode = _overlayMode;
                if (_waterfallView != null) _waterfallView.Canvas.OverlayMode = _overlayMode;
            }
            finally
            {
                _applyingOverlayMode = false;
            }
        }

        private void OnCanvasOverlayModeChanged(CanvasOverlayMode mode)
        {
            if (_applyingOverlayMode) return;
            _overlayMode = mode;
            ApplyOverlayModeToViews();
            OverlayModeChanged?.Invoke(mode);
        }

        public void SwitchMainDisplay(int cameraIndex) => SwitchMainDisplay(cameraIndex, centerView: false);

        /// <summary>centerView=true 僅限「明確點縮圖/狀態字」手勢：主畫面置中該相機。
        /// 其他來源（畫布點擊選取、程式化 refresh、配置後還原）一律 false——置中會蓋掉使用者拖出的視野（回彈）。</summary>
        public void SwitchMainDisplay(int cameraIndex, bool centerView)
        {
            if (_mainForm == null || _mainForm.IsDisposed
                || _mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) return;

            if (_mainForm.InvokeRequired)
            {
                try { _mainForm.BeginInvoke(new Action(() => SwitchMainDisplay(cameraIndex, centerView))); }
                catch (InvalidOperationException) { }
                return;
            }

            Flow($"SwitchMainDisplay cam={cameraIndex} center={centerView} mode={(WaterfallMode ? "WF" : "IC")}");
            _selectedMainCameraId = cameraIndex;
            _userSelectedMainCameraId = cameraIndex;
            _imageDisplay?.SetSelected(cameraIndex);
            _waterfallThumbs?.SetSelected(cameraIndex - 1);

            foreach (var kvp in _liveParentPanels)
                kvp.Value.Invalidate();

            // 縮圖→主畫面連動：明確點縮圖才置中（即時=ImageDisplayView、瀑布=WaterfallView）。
            if (centerView)
            {
                if (ImageCanvasMode) _imageDisplay?.CenterOnCamera(cameraIndex);
                else if (WaterfallMode) _waterfallView?.CenterOnCamera(cameraIndex);
            }
        }

        public void OnGlobalMergeEnabled(double[] opsUm, double[] startPosMm)
        {
            if (ImageCanvasMode && _imageDisplay != null)
            {
                _imageDisplay.SetLayout(startPosMm, opsUm, 1, RowPitchMm);
                ApplyImageDisplayOptions(true);
            }
        }

        public void OnGlobalMergeDisabled()
        {
            ApplyImageDisplayOptions(false);
            SwitchMainDisplay(_userSelectedMainCameraId);
        }

        public void RefreshGlobalMergeLayout(double[] opsUm, double[] startPosMm, double refOpsMm)
        {
            if (ImageCanvasMode && _imageDisplay != null)
                _imageDisplay.SetLayout(startPosMm, opsUm, 1, RowPitchMm);
            if (_waterfallView != null && _globalMerge.Merger != null)
                _waterfallView.SetLayout(startPosMm, opsUm, refOpsMm);
        }

        private IReadOnlyList<AniloxCamera> Cameras => _getCameras() ?? Array.Empty<AniloxCamera>();
    }
}
