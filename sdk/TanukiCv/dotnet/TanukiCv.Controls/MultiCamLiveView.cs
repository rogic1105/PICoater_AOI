using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Windows.Forms;
using TanukiCv.Core; // PixelMmMapper

namespace TanukiCv.Controls
{
    /// <summary>
    /// 多相機即時監控顯示元件（純 CPU、吃 8bpp 灰階 bytes，0 依賴 MIL / app）：
    /// 在主 panel 疊一個 <see cref="SmartCanvas"/>（zoom/pan/雙三擊/mm overlay）、各 cam panel 疊雙緩衝
    /// 縮圖（<see cref="ThumbView"/>，自繪不閃）。單相機 / CPU 合圖（<see cref="MergeLayout"/> 佈局 + 重疊分界）。
    ///
    /// 兩專案共用：主程式 LiveCameraManager（餵 AniloxCamera.OnDisplayFrame，feedScale=1）、
    /// 範例 MilGrabberPbForm（餵 MilCamera.FrameReady 縮圖，feedScale=resizeScale）。
    /// 取代原各自的 LiveSmartDisplay / PictureBox 顯示路徑。
    /// </summary>
    public sealed class MultiCamLiveView : IDisposable
    {
        private readonly Panel _mainPanel;
        private readonly Panel[] _camPanels;
        private readonly int _camCount;
        private SmartCanvas _canvas;
        private ThumbStrip _thumbStrip;            // 多相機縮圖條（唯一來源；批量 CPU 建圖不閃）
        private readonly double _screenMmPerPx;

        private sealed class Frame { public readonly byte[] Bytes; public readonly int W, H; public Frame(byte[] b, int w, int h) { Bytes = b; W = w; H = h; } }
        private readonly Frame[] _latest;
        private readonly bool[] _readySinceMerge;  // 合圖：自上次重建後該台是否已出新幀（湊齊所有在線相機才重建，仿範例 → 不每幀重建巨圖）
        private int _lastMergeTick;                // 上次合圖重建時間（200ms fallback：某台停了也別凍住）

        private volatile int _selectedCamId = 1;
        private volatile bool _mergeMode;
        private volatile bool _disposed;

        // 合圖佈局（caller 推入）
        private volatile bool _mergeReady;
        private double[] _startPosMm, _opsUm; private int _feedScale = 1; private double _rowPitchMm;
        private volatile int _mergeCapK = 1; // 合圖巨圖再降採樣倍率（mm overlay 用）

        private int _mainW = -1, _mainH = -1;
        private volatile bool _mainDirty = true;
        private System.Windows.Forms.Timer _timer;

        private const int MergeMaxW = 30000;   // 合圖點陣寬上限（GDI 16-bit 座標 ~32767，超過 wrap → 內容錯位）

        /// <summary>縮圖被點 → 要求切換選中相機（1-based camId）。</summary>
        public event Action<int> SelectRequested;
        /// <summary>視野左右範圍（mm）→ 上層 overview 聯動。</summary>
        public event Action<double, double> ViewRangeMmChanged;
        /// <summary>合圖重疊分界策略（預設中線）。</summary>
        public MergeOverlap MergeStrategy { get; set; } = MergeOverlap.Midline;

        /// <summary>內部主畫面 SmartCanvas（供上層接 LOD / FOV 校正 / 計時等專屬測試知識；一般顯示不需碰）。</summary>
        public SmartCanvas Canvas => _canvas;

        /// <summary>選中相機的最新全解析度灰階快照（供 LOD provider 等需要原圖的場景；無則 null）。</summary>
        public byte[] GetSelectedRaw(out int w, out int h)
        {
            int idx = _selectedCamId - 1;
            Frame f = (idx >= 0 && idx < _latest.Length) ? _latest[idx] : null;
            if (f == null) { w = h = 0; return null; }
            w = f.W; h = f.H; return f.Bytes;
        }

        public MultiCamLiveView(Panel mainPanel, Panel[] camPanels, double screenMmPerPx)
        {
            _mainPanel = mainPanel;
            _camPanels = camPanels ?? new Panel[0];
            _camCount = _camPanels.Length;
            _screenMmPerPx = screenMmPerPx;
            _latest = new Frame[_camCount];
            _readySinceMerge = new bool[_camCount];

            _canvas = new SmartCanvas { Dock = DockStyle.Fill };
            _canvas.FitRelativeZoom = false;          // 可放大也可縮小到 fit 以下（同 camReviewMain）
            _canvas.DoubleClickFitToScreen = true;
            _canvas.TripleClickPhysical1x = true;
            _canvas.ClampPan = false;                 // 監控自由拖看
            _canvas.StatusChanged += OnCanvasStatus;
            _mainPanel.Controls.Add(_canvas);
            _canvas.BringToFront();

            _thumbStrip = new ThumbStrip(_camPanels);                 // 縮圖唯一來源（批量 CPU 建圖不閃）
            _thumbStrip.SelectRequested += idx => SelectRequested?.Invoke(idx + 1); // 0-based idx → 1-based camId

            _timer = new System.Windows.Forms.Timer { Interval = 33 };
            _timer.Tick += (s, e) => RefreshMain();
            _timer.Start();
        }

        public void SetSelected(int camId) { _selectedCamId = camId; _mainDirty = true; }
        public void SetMergeMode(bool on) { _mergeMode = on; _mainDirty = true; }

        /// <summary>合圖佈局：各台 start(mm) + ops(µm，全解析度) + feedScale（餵入幀相對全解析度的降採樣；
        /// 主程式餵全解析度=1、範例餵縮圖=resizeScale）+ rowPitchMm（Y 方向 mm/px，0 則退回 ops）。</summary>
        public void SetMergeLayout(double[] startPosMm, double[] opsUm, int feedScale, double rowPitchMm)
        {
            _startPosMm = startPosMm; _opsUm = opsUm;
            _feedScale = feedScale > 0 ? feedScale : 1;
            _rowPitchMm = rowPitchMm;
            _mergeReady = startPosMm != null && opsUm != null && opsUm.Length > 0 && opsUm[0] > 0;
        }

        /// <summary>相機每幀（可能在背景執行緒）：存全解析度快照（供主畫面/合圖）+ 餵縮圖條。**不在此重建主畫面**
        /// （避免每幀每台 BeginInvoke 灌爆 UI → 閃爍/卡死）；主畫面改由 33ms timer 在「選中/合圖」才重建。
        /// 縮圖由 <see cref="ThumbStrip"/> 自己批量刷。</summary>
        public void PushFrame(int camId, byte[] gray, int w, int h)
        {
            if (_disposed || camId < 1 || camId > _camCount || gray == null || w <= 0 || h <= 0) return;
            int n = w * h;
            var copy = new byte[n];
            Array.Copy(gray, copy, Math.Min(gray.Length, n));
            _latest[camId - 1] = new Frame(copy, w, h);
            _thumbStrip?.PushFrame(camId - 1, gray, w, h);   // 縮圖唯一來源（ThumbStrip 自己存快照 + 批量建圖）

            if (_mergeMode)
            {
                // 合圖：等所有在線相機都出新幀（湊齊一輪）才重建 → 不每幀重建巨圖（範例順的關鍵）。
                _readySinceMerge[camId - 1] = true;
                if (AllActiveReadySinceMerge()) { ClearReadyFlags(); _mainDirty = true; _lastMergeTick = Environment.TickCount; }
            }
            else if (camId == _selectedCamId)
            {
                // 單張：只在選中那台有新幀才重建（不被其他相機拖去重建）。
                _mainDirty = true;
            }
        }

        private bool AllActiveReadySinceMerge()
        {
            bool any = false;
            for (int i = 0; i < _camCount; i++)
            {
                if (_latest[i] == null) continue;   // 非在線（沒出過幀）
                any = true;
                if (!_readySinceMerge[i]) return false;
            }
            return any;
        }
        private bool AnyReadySinceMerge() { for (int i = 0; i < _camCount; i++) if (_readySinceMerge[i]) return true; return false; }
        private void ClearReadyFlags() { for (int i = 0; i < _camCount; i++) _readySinceMerge[i] = false; }

        // ── UI timer（33ms）：主畫面（僅 dirty 才重建；合圖湊不齊但逾時也補一次）。縮圖由 ThumbStrip 自管 ──
        private void RefreshMain()
        {
            if (_disposed || _canvas == null || _canvas.IsDisposed) return;

            // 合圖湊不齊（某台停了）但有新幀且逾 200ms → 強制補一次，避免凍住
            if (_mergeMode && !_mainDirty && AnyReadySinceMerge() && Environment.TickCount - _lastMergeTick > 200)
            { ClearReadyFlags(); _mainDirty = true; _lastMergeTick = Environment.TickCount; }

            if (!_mainDirty) return;
            _mainDirty = false;
            Bitmap bmp;
            try { bmp = _mergeMode ? BuildMerge() : BuildSingle(); }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[MultiCamLiveView.RefreshMain] {ex.GetType().Name}: {ex.Message}"); return; }
            if (bmp == null) return;
            var old = _canvas.Image;
            _canvas.Image = bmp;
            old?.Dispose();
            if (bmp.Width != _mainW || bmp.Height != _mainH)
            {
                _mainW = bmp.Width; _mainH = bmp.Height;
                _canvas.FitToScreen();
            }
            if (GetDisplayCoords(out _, out double opsInMm, out double sf))
                _canvas.SetPhysicalCalibration(opsInMm * sf, _screenMmPerPx);
        }

        /// <summary>當前顯示座標基準：合圖用 minStart/ops0、sf=feedScale×capK；單相機用各台 ops/start、sf=feedScale。</summary>
        private bool GetDisplayCoords(out double startMm, out double opsInMm, out double sf)
        {
            // sf = 顯示像素 → 全解析度像素：feedScale（餵入相對全解析度）×（合圖再 cap k）。主畫面用全解析度 _latest。
            if (_mergeMode && _mergeReady)
            {
                startMm = MinStart(); opsInMm = _opsUm[0] / 1000.0; sf = _feedScale * _mergeCapK;
                return opsInMm > 0;
            }
            int idx = _selectedCamId - 1;
            if (_opsUm != null && _startPosMm != null && idx >= 0 && idx < _opsUm.Length && _opsUm[idx] > 0)
            { opsInMm = _opsUm[idx] / 1000.0; startMm = _startPosMm[idx]; sf = _feedScale; return true; }
            startMm = 0; opsInMm = 0; sf = _feedScale; return false;
        }

        private double MinStart()
        {
            double m = double.MaxValue;
            if (_startPosMm != null) for (int i = 0; i < _startPosMm.Length; i++) if (_startPosMm[i] < m) m = _startPosMm[i];
            return m == double.MaxValue ? 0 : m;
        }

        private Bitmap BuildSingle()
        {
            int idx = _selectedCamId - 1;
            Frame f = (idx >= 0 && idx < _latest.Length) ? _latest[idx] : null;
            return f != null ? GrayBitmap.From(f.Bytes, f.W, f.H) : null;
        }

        /// <summary>CPU 合圖：佈局/重疊分界委派 <see cref="MergeLayout"/>（含 8 槽全納入：無畫面相機留黑占空間）；
        /// 巨圖超 MergeMaxW 再降採樣 k 倍（防 GDI 座標 wrap），k&gt;1 用平滑內插避免馬賽克。</summary>
        private Bitmap BuildMerge()
        {
            if (!_mergeReady) return BuildSingle();
            double refOpsMm = _opsUm[0] / 1000.0;
            if (refOpsMm <= 0) return BuildSingle();

            // 代表尺寸（無畫面槽占位用）：任一有畫面相機的尺寸
            int defW = 0, defH = 0;
            for (int i = 0; i < _camCount; i++)
                if (_latest[i] != null && defW == 0) { defW = _latest[i].W; defH = _latest[i].H; }
            if (defW == 0) return null;

            double minStart = MinStart();
            var geoms = new System.Collections.Generic.List<MergeLayout.CamGeom>();
            int maxH = 0;
            for (int i = 0; i < _camCount; i++)
            {
                double st = (_startPosMm != null && i < _startPosMm.Length) ? _startPosMm[i] : 0;
                bool present = _latest[i] != null;
                int w = present ? _latest[i].W : defW;
                int h = present ? _latest[i].H : defH;
                if (h > maxH) maxH = h;
                geoms.Add(new MergeLayout.CamGeom { CameraId = i + 1, StartMm = st, WidthPx = w });
            }
            if (geoms.Count == 0 || maxH <= 0) return null;

            var placements = MergeLayout.Compute(geoms, minStart, refOpsMm, _feedScale, MergeStrategy, out int totalW);
            if (totalW <= 0) return null;

            int k = Math.Max(1, (totalW + MergeMaxW - 1) / MergeMaxW);
            _mergeCapK = k;
            int mw = Math.Max(1, totalW / k), mh = Math.Max(1, maxH / k);

            var merged = new Bitmap(mw, mh, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(merged))
            {
                g.Clear(Color.Black);
                g.InterpolationMode = InterpolationMode.NearestNeighbor; // 即時合圖每幀重建 → 用快的 Nearest（HighQuality 大圖會卡取像）
                g.PixelOffsetMode = PixelOffsetMode.Half;
                foreach (var p in placements)
                {
                    Frame f = _latest[p.CameraId - 1];
                    if (f == null || p.SrcWidth <= 0) continue;
                    int dx = (int)Math.Round(p.DestX / (double)k);
                    int dw = Math.Max(1, (int)Math.Round(p.SrcWidth / (double)k));
                    int dh = Math.Max(1, (int)Math.Round(f.H / (double)k));
                    using (var cam = GrayBitmap.From(f.Bytes, f.W, f.H))
                        g.DrawImage(cam, new Rectangle(dx, 0, dw, dh),
                            new Rectangle(p.SrcLeft, 0, p.SrcWidth, f.H), GraphicsUnit.Pixel);
                }
            }
            return merged;
        }

        // ── mm overlay + 實體校正（StatusChanged）──
        private void OnCanvasStatus(CanvasInfo info)
        {
            if (_disposed || _canvas == null || info.Zoom <= 0) return;
            if (!GetDisplayCoords(out double startMm, out double opsInMm, out double sf))
            { _canvas.SetRangeOverlay("", "", "", "", ""); return; }

            double leftMm  = PixelMmMapper.PixelToMm((0 - info.PanOffset.X) / info.Zoom * sf, startMm, opsInMm);
            double rightMm = PixelMmMapper.PixelToMm((_canvas.Width - info.PanOffset.X) / info.Zoom * sf, startMm, opsInMm);
            double yPitch = _rowPitchMm > 0 ? _rowPitchMm : opsInMm;
            double topMm = (0 - info.PanOffset.Y) / info.Zoom * sf * yPitch;
            double botMm = (_canvas.Height - info.PanOffset.Y) / info.Zoom * sf * yPitch;

            _canvas.SetPhysicalCalibration(opsInMm * sf, _screenMmPerPx);
            double physMag = _canvas.PhysicalMagnification;
            _canvas.SetRangeOverlay(physMag > 0 ? $"{physMag:F2}x" : "",
                $"{leftMm:F1}", $"{rightMm:F1}", $"{topMm:F1}", $"{botMm:F1}");

            double curMmX = PixelMmMapper.PixelToMm(info.ImageX * sf, startMm, opsInMm);
            double curMmY = info.ImageY * sf * yPitch;
            _canvas.SetCursorMm($"({curMmX:F2}, {curMmY:F2})");

            ViewRangeMmChanged?.Invoke(leftMm, rightMm);
        }

        public void Dispose()
        {
            _disposed = true;
            if (_timer != null) { _timer.Stop(); _timer.Dispose(); _timer = null; }
            if (_thumbStrip != null) { _thumbStrip.Dispose(); _thumbStrip = null; }
            if (_canvas != null)
            {
                _canvas.StatusChanged -= OnCanvasStatus;
                var old = _canvas.Image; _canvas.Image = null; old?.Dispose();
                _mainPanel?.Controls.Remove(_canvas);
                _canvas.Dispose(); _canvas = null;
            }
        }
    }
}
