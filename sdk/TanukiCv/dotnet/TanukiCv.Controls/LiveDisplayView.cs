using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Windows.Forms;
using TanukiCv.Core; // PixelMmMapper

namespace TanukiCv.Controls
{
    /// <summary>灰階影像縮放委派（src wxh → dst dwxdh 的灰階 bytes）。LOD provider 的可插拔縮放步驟：
    /// GPU 版由呼叫端餵（如 core_cv 的 TanukiCv_Resize_GPU），CPU 版 TanukiCv 內建（見 GrayResizeCpu，階段 2）。
    /// 把「LOD 要不要 GPU」的差異收斂成這一個委派——LOD 機制本身（SmartCanvas）純 CPU。</summary>
    public delegate byte[] GrayResize(byte[] src, int srcW, int srcH, int dstW, int dstH);

    /// <summary>
    /// 多相機即時監控顯示元件（**絞殺榕重寫版**，取 app MultiCamLiveView + 範例 MilGrabberPbForm 主畫面兩者成功部分）。
    /// 純 CPU 骨架、吃 8bpp 灰階 bytes、0 依賴 MIL/app：主 panel 疊 <see cref="SmartCanvas"/>（zoom/pan/雙三擊/mm overlay/LOD）、
    /// 各 cam panel 疊 <see cref="ThumbStrip"/>（批量縮圖不閃）。合圖/合圖全部委派 <see cref="MergeLayout"/>。
    ///
    /// 統一介面（收斂 app/sample 4 條接線差異）：
    ///   ① 餵入 <see cref="PushFrame"/>（灰階 bytes；feedScale 由 <see cref="SetLayout"/> 給）
    ///   ② ops/mm 來源統一 <see cref="SetLayout"/>（不管 FOV 還是設定，都換算成 start mm + ops µm 餵進來）
    ///   ③ LOD provider 插槽 <see cref="EnableLod"/>（GPU/CPU 皆可，呼叫端給 <see cref="GrayResize"/>）
    ///   ④ <see cref="FlipVertical"/>（主畫面 + 縮圖一起翻）
    ///
    /// 階段 1（骨架）：display/thumb/merge/flip/mm-overlay 已實作；LOD 已接線（需呼叫端給 GrayResize）。
    /// app/sample 之後分別接入（階段 3/4），舊路退場（階段 5）。
    /// </summary>
    public sealed class LiveDisplayView : IDisposable
    {
        private readonly Panel _mainPanel;
        private readonly int _camCount;
        private SmartCanvas _canvas;
        private ThumbStrip _thumbStrip;
        private readonly double _screenMmPerPx;

        private sealed class Frame { public readonly byte[] Bytes; public readonly int W, H; public Frame(byte[] b, int w, int h) { Bytes = b; W = w; H = h; } }
        private readonly Frame[] _latest;            // 各台最新「全解析度（相對餵入）」灰階快照（原子 ref 換）
        private readonly bool[] _readySinceMerge;
        private int _lastMergeTick;

        private volatile int _selectedCamId = 1;
        private volatile bool _mergeMode;
        private volatile bool _disposed;
        private volatile bool _mainDirty = true;
        private bool _cursorProfileCleared;   // 剖面已歸零旗標（游標出界/離開畫布；防重複 fire null + Invalidate）
        private int _mainW = -1, _mainH = -1;
        private System.Windows.Forms.Timer _timer;

        // 佈局（ops/start 座標來源，單一介面）
        private volatile bool _mergeReady;
        private double[] _startPosMm, _opsUm;
        private int _feedScale = 1;
        private double _rowPitchMm;
        private volatile int _mergeCapK = 1;

        // 合圖幾何快取（BuildMerge 與「合圖 LOD」合成 provider 共用單一來源）
        private List<CameraPlacement> _mergePlacements;
        private int _mergeTotalW, _mergeMaxH;

        // LOD（單張或合圖；可插拔 GrayResize）
        private GrayResize _lodResize;
        private volatile bool _lodWanted;
        private int _lodCamId = -1;           // 單張：目前 LOD 綁的相機（換相機要重綁虛擬尺寸）
        private int _lodMergeW = -1, _lodMergeH = -1; // 合圖：目前 LOD 綁的虛擬尺寸（變了要重綁）

        private const int MergeMaxW = 30000; // 合圖點陣寬上限（GDI 16-bit 座標 wrap 防護）

        /// <summary>縮圖被點 → 要求切換選中相機（1-based camId）。</summary>
        public event Action<int> SelectRequested;

        /// <summary>選中相機因「視野移動」自動變更（合圖模式反向連動：視野中心最近的相機 → 自動高亮縮圖）。
        /// 1-based camId。程式化來源（非使用者點擊），上層通常只需更新狀態、勿再呼 CenterOnCamera（防遞迴）。</summary>
        public event Action<int> SelectedCamChanged;
        /// <summary>視野可見範圍（mm）：leftX, rightX, topY, botY → 上層曲線圖（切向用 X、法向用 Y、overview 用 X）zoom 連動。</summary>
        public event Action<double, double, double, double> ViewRangeMmChanged;

        /// <summary>游標十字剖面（L0 通用：游標那列/行的原始像素值）+ 對齊資訊（曲線圖畫點 + zoom 同步用）。
        /// 單張＝選定相機全幀；合圖＝游標列橫跨整張合圖（用 BuildMerge 同份 placements 拼，與畫面對齊）、
        /// 游標行取所屬相機。純像素、0 依賴檢測（Hessian）。app 之後可在 L1 換成自己的曲線資料（同對齊路）。</summary>
        public event Action<CursorProfile> CursorProfileChanged;

        /// <summary>游標剖面資料 + 對齊座標（曲線圖：X 點 mm = StartXmm + i×OpsXmm；Y 點 mm = i×OpsYmm；軸 zoom 用 View*Mm）。</summary>
        public sealed class CursorProfile
        {
            public byte[] RowProfile;   // 游標那「列」沿影像 X 的像素（長度=影像寬）
            public byte[] ColProfile;   // 游標那「行」沿影像 Y 的像素（長度=影像高）
            public double StartXmm, OpsXmm;   // X 軸 mm 映射
            public double OpsYmm;             // Y 軸 mm 映射（top=0）
            public double ViewLeftMm, ViewRightMm, ViewTopMm, ViewBotMm; // 目前可見範圍 → 曲線圖軸 zoom（跟影像對齊）
            public int CursorX, CursorY;      // 游標影像座標
        }

        /// <summary>游標狀態（mm 位置 + 可見範圍 + 實體倍率 + 原始座標/亮度）→ 上層更新狀態列等。
        /// 與 <see cref="OnCanvasStatus"/> 同源計算（mm 換算只在這裡做一次），上層只負責格式化（文字屬上層政策）。</summary>
        public event Action<CursorStatus> CursorStatusChanged;

        /// <summary>游標狀態快照（單一來源＝LiveDisplayView 內部 mm 換算；上層不重算）。</summary>
        public sealed class CursorStatus
        {
            public double CurMmX, CurMmY;                                   // 游標位置 mm（X=沿料寬，Y=沿走料方向）
            public double ViewLeftMm, ViewRightMm, ViewTopMm, ViewBotMm;    // 可見範圍 mm
            public double PhysMag;                                          // 實體倍率（<=0＝無校正）
            public int CursorX, CursorY;                                    // 游標影像座標（像素）
            public int Brightness;                                          // 該點灰階值 0~255
            public int SelectedCamId;                                       // 1-based 當前選中相機
        }

        /// <summary>合圖重疊分界策略（預設中線）。</summary>
        public MergeOverlap MergeStrategy { get; set; } = MergeOverlap.Midline;

        /// <summary>合圖全部：含無畫面相機（黑色占空間）。false=只合有畫面的。</summary>
        public bool MergeAll { get; set; }

        /// <summary>上下翻轉（線掃由下往上拍 / GPU 輸出 bottom-up）。主畫面 + 縮圖一起翻。</summary>
        public bool FlipVertical
        {
            get => _flip;
            set { _flip = value; if (_thumbStrip != null) _thumbStrip.FlipVertical = value; _mainDirty = true; }
        }
        private bool _flip;

        /// <summary>內部主畫面 SmartCanvas（供上層接計時 / app 專屬事件等；一般顯示不需碰）。</summary>
        public SmartCanvas Canvas => _canvas;

        /// <summary>縮圖選取框色（雙向連動高亮的唯一視覺來源；上層若原本自畫選取框請移除，避免雙框）。</summary>
        public System.Drawing.Color ThumbSelectedColor { set => _thumbStrip?.SetSelectedColor(value); }

        /// <summary>選中相機最新全解析度灰階快照（LOD provider / 進階用；無則 null）。</summary>
        public byte[] GetSelectedRaw(out int w, out int h)
        {
            int idx = _selectedCamId - 1;
            Frame f = (idx >= 0 && idx < _latest.Length) ? _latest[idx] : null;
            if (f == null) { w = h = 0; return null; }
            w = f.W; h = f.H; return f.Bytes;
        }

        public LiveDisplayView(Panel mainPanel, Panel[] camPanels, double screenMmPerPx)
        {
            _mainPanel = mainPanel ?? throw new ArgumentNullException(nameof(mainPanel));
            camPanels = camPanels ?? new Panel[0];
            _camCount = camPanels.Length;
            _screenMmPerPx = screenMmPerPx;
            _latest = new Frame[_camCount];
            _readySinceMerge = new bool[_camCount];

            _canvas = new SmartCanvas { Dock = DockStyle.Fill };
            _canvas.FitRelativeZoom = false;        // 可放大也可縮小到 fit 以下（同 camReviewMain / 範例 panelMain）
            _canvas.DoubleClickFitToScreen = true;
            _canvas.TripleClickPhysical1x = true;
            _canvas.ClampPan = false;
            _canvas.StatusChanged += OnCanvasStatus;
            _canvas.MouseLeave += (s, e) => ClearCursorProfile(); // 游標離開畫布 → 剖面歸零
            _mainPanel.Controls.Add(_canvas);
            _canvas.BringToFront();

            _thumbStrip = new ThumbStrip(camPanels);
            _thumbStrip.SelectRequested += idx =>
            {
                int camId = idx + 1; // 0-based idx → 1-based camId
                // 縮圖↔主畫面雙向連動（正向）：合圖模式點縮圖 → 主畫面 pan 定位到該相機（保 zoom）。
                // 單張模式交給上層（SelectRequested → 上層 SetSelected 換顯示相機）。
                if (_mergeMode) CenterOnCamera(camId);
                SetSelected(camId);
                SelectRequested?.Invoke(camId);
            };
            _thumbStrip.SetSelected(_selectedCamId - 1); // 初始高亮

            _timer = new System.Windows.Forms.Timer { Interval = 33 };
            _timer.Tick += (s, e) => RefreshMain();
            _timer.Start();
        }

        // ==================== 介面 ====================

        public void SetSelected(int camId)
        {
            _selectedCamId = camId; _mainDirty = true;
            _thumbStrip?.SetSelected(camId - 1);   // 縮圖高亮跟著走（雙向連動的「秀」）
        }
        public void SetMergeMode(bool on) { _mergeMode = on; _mainDirty = true; }

        /// <summary>合圖模式：把主畫面 pan 定位到指定相機的槽中心（保 zoom）。
        /// 縮圖↔主畫面雙向連動的「正向」；非合圖 / 佈局未備則無動作。</summary>
        public void CenterOnCamera(int camId)
        {
            if (!_mergeMode || _canvas == null) return;
            var placements = _mergePlacements;
            if (placements == null && !TryComputeMergeGeometry()) return;
            placements = _mergePlacements;
            if (placements == null) return;
            int k = Math.Max(1, _mergeCapK);
            foreach (var p in placements)
            {
                if (p.CameraId != camId) continue;
                double centerPx = (p.DestX + p.SrcWidth / 2.0) / k;   // 顯示（capped）座標
                float zoom = _canvas.Zoom;
                if (zoom <= 0) return;
                float newPanX = _canvas.Width / 2.0f - (float)(centerPx * zoom);
                _canvas.SetView(zoom, new PointF(newPanX, _canvas.PanOffset.Y));
                return;
            }
        }

        /// <summary>座標來源（單一介面）：各台 start(mm) + ops(µm，相對全解析度) + feedScale（餵入幀相對全解析度的降採樣；
        /// app 餵全解析度=1、範例餵縮圖=resizeScale）+ rowPitchMm（Y 方向 mm/px，0 則退回 ops 方形像素）。
        /// FOV 來源的呼叫端自己把 fovMm/frameWidth 換算成 ops 再餵進來 → 單張/合圖/overlay 同一條路。</summary>
        public void SetLayout(double[] startPosMm, double[] opsUm, int feedScale, double rowPitchMm)
        {
            _startPosMm = startPosMm; _opsUm = opsUm;
            _feedScale = feedScale > 0 ? feedScale : 1;
            _rowPitchMm = rowPitchMm;
            _mergeReady = startPosMm != null && opsUm != null && opsUm.Length > 0 && opsUm[0] > 0;
        }

        /// <summary>啟用動態 LOD（單張模式）：傳入可插拔的灰階縮放委派（GPU 或 CPU）。
        /// LOD 機制（裁可見區/拉伸/背景重算）在 SmartCanvas（純 CPU）；resize 那一步用此委派。</summary>
        public void EnableLod(GrayResize resize)
        {
            _lodResize = resize;
            _lodWanted = resize != null;
            _lodCamId = -1;     // 下次 RefreshMain 對選中相機重新綁定虛擬尺寸
            _mainDirty = true;
        }

        /// <summary>停用 LOD → 退回一般 .Image 顯示。</summary>
        public void DisableLod()
        {
            _lodWanted = false; _lodResize = null; _lodCamId = -1;
            if (_canvas != null && _canvas.LodActive) _canvas.DisableLod();
            _mainW = _mainH = -1; // 下次 fit
            _mainDirty = true;
        }

        /// <summary>相機每幀（可能背景執行緒）：存全解析度快照 + 餵縮圖條 + 標記主畫面 dirty。</summary>
        public void PushFrame(int camId, byte[] gray, int w, int h)
        {
            if (_disposed || camId < 1 || camId > _camCount || gray == null || w <= 0 || h <= 0) return;
            int n = w * h;
            var copy = new byte[n];
            Array.Copy(gray, copy, Math.Min(gray.Length, n));
            _latest[camId - 1] = new Frame(copy, w, h);
            _thumbStrip?.PushFrame(camId - 1, gray, w, h);

            if (_mergeMode)
            {
                _readySinceMerge[camId - 1] = true;
                if (AllActiveReadySinceMerge()) { ClearReadyFlags(); _mainDirty = true; _lastMergeTick = Environment.TickCount; }
            }
            else if (camId == _selectedCamId)
            {
                _mainDirty = true;
            }
        }

        // ==================== UI timer（33ms）：主畫面更新（縮圖由 ThumbStrip 自管）====================

        /// <summary>用「當前」zoom/pan 重發 ViewRangeMmChanged（不需滑鼠互動）。
        /// 用途：上層 chart 重建（重載/強化切換）會重設軸範圍 → 載入完呼此補發，曲線立即恢復跟隨視野。</summary>
        public void RefireViewRange()
        {
            if (_canvas == null || ViewRangeMmChanged == null) return;
            if (!GetDisplayCoords(out double startMm, out double opsInMm, out double sf)) return;
            float zoom = _canvas.Zoom;
            if (zoom <= 0) return;
            var pan = _canvas.PanOffset;
            double leftMm = PixelMmMapper.PixelToMm((0 - pan.X) / zoom * sf, startMm, opsInMm);
            double rightMm = PixelMmMapper.PixelToMm((_canvas.Width - pan.X) / zoom * sf, startMm, opsInMm);
            double yPitch = _rowPitchMm > 0 ? _rowPitchMm : opsInMm;
            double topMm = (0 - pan.Y) / zoom * sf * yPitch;
            double botMm = (_canvas.Height - pan.Y) / zoom * sf * yPitch;
            ViewRangeMmChanged?.Invoke(leftMm, rightMm, topMm, botMm);
        }

        /// <summary>縮圖↔主畫面雙向連動（反向）：合圖模式視野中心最近的相機 → 自動高亮縮圖
        /// （不觸發 SelectRequested 防遞迴）。OnCanvasStatus（互動）+ 33ms timer（快拖事件合併時補刷，
        /// 中間相機不被跳過）兩處呼叫；計算極便宜（找最近 placement 中心）。</summary>
        private void UpdateReverseThumbSync()
        {
            if (!_mergeMode || _mergePlacements == null || _canvas == null) return;
            float zoom = _canvas.Zoom;
            if (zoom <= 0) return;
            int k = Math.Max(1, _mergeCapK);
            double viewCenterPx = (_canvas.Width / 2.0 - _canvas.PanOffset.X) / zoom; // capped 顯示座標
            int bestCam = 0; double bestDist = double.MaxValue;
            foreach (var p in _mergePlacements)
            {
                double c = (p.DestX + p.SrcWidth / 2.0) / k;
                double d = Math.Abs(viewCenterPx - c);
                if (d < bestDist) { bestDist = d; bestCam = p.CameraId; }
            }
            if (bestCam > 0 && bestCam != _selectedCamId)
            {
                _selectedCamId = bestCam;                  // 只更新欄位+高亮，不設 _mainDirty（合圖畫面不需重建）
                _thumbStrip?.SetSelected(bestCam - 1);
                SelectedCamChanged?.Invoke(bestCam);
            }
        }

        private void RefreshMain()
        {
            UpdateReverseThumbSync();   // 快拖補刷（33ms；StatusChanged 限流時的保險）
            if (_disposed || _canvas == null || _canvas.IsDisposed) return;

            // 合圖湊不齊（某台停了）但有新幀且逾 200ms → 強制補一次，避免凍住
            if (_mergeMode && !_mainDirty && AnyReadySinceMerge() && Environment.TickCount - _lastMergeTick > 200)
            { ClearReadyFlags(); _mainDirty = true; _lastMergeTick = Environment.TickCount; }

            // LOD 路徑（單張 or 合圖 + 有 provider）：畫布用 provider tile，不走 .Image
            if (_lodWanted && _lodResize != null)
            {
                if (_mergeMode)
                {
                    // 合圖 LOD：虛擬圖=完整合圖佈局（未 cap），provider 從各相機合成可見區
                    if (TryComputeMergeGeometry())
                    {
                        if (!_canvas.LodActive || _lodCamId != 0 || _lodMergeW != _mergeTotalW || _lodMergeH != _mergeMaxH)
                        {
                            _lodCamId = 0; _lodMergeW = _mergeTotalW; _lodMergeH = _mergeMaxH; _mergeCapK = 1;
                            _canvas.EnableLod(_mergeTotalW, _mergeMaxH, MergeLodProvide);
                            ApplyCalibration();
                        }
                        else if (_mainDirty) _canvas.RefreshLod();
                        _mainDirty = false;
                        return;
                    }
                }
                else
                {
                    int idx = _selectedCamId - 1;
                    Frame f = (idx >= 0 && idx < _latest.Length) ? _latest[idx] : null;
                    if (f != null)
                    {
                        if (!_canvas.LodActive || _lodCamId != _selectedCamId)
                        {
                            _lodCamId = _selectedCamId;
                            _canvas.EnableLod(f.W, f.H, LodProvide); // 虛擬尺寸=全解析度；停住才請 provider 裁+縮
                            ApplyCalibration();
                        }
                        else if (_mainDirty) _canvas.RefreshLod();
                        _mainDirty = false;
                        return;
                    }
                }
            }
            if ((!_lodWanted || _lodResize == null) && _canvas.LodActive)
            {
                _canvas.DisableLod(); _lodCamId = -1; _lodMergeW = _lodMergeH = -1; _mainW = _mainH = -1;
            }

            if (!_mainDirty) return;
            _mainDirty = false;
            Bitmap bmp;
            try { bmp = _mergeMode ? BuildMerge() : BuildSingle(); }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveDisplayView.RefreshMain] {ex.GetType().Name}: {ex.Message}"); return; }
            if (bmp == null) return;
            var old = _canvas.Image;
            _canvas.Image = bmp;
            old?.Dispose();
            if (bmp.Width != _mainW || bmp.Height != _mainH)
            {
                bool firstFrame = _mainW < 0;
                _mainW = bmp.Width; _mainH = bmp.Height;
                // 首幀一定 fit；之後尺寸變動只在「使用者仍在 fit 視角」才 fit（live 新幀不把手動縮放拉回 fit）。
                if (firstFrame || _canvas.IsAtFitView()) _canvas.FitToScreen();
            }
            ApplyCalibration();
        }

        private void ApplyCalibration()
        {
            if (GetDisplayCoords(out _, out double opsInMm, out double sf))
                _canvas.SetPhysicalCalibration(opsInMm * sf, _screenMmPerPx);
        }

        /// <summary>LOD provider（SmartCanvas 在背景執行緒呼叫）：從選中相機全解析度快照裁可見虛擬區 → GrayResize（GPU/CPU）→ 灰階 bitmap。</summary>
        private Bitmap LodProvide(Rectangle srcRect, Size target)
        {
            GrayResize resize = _lodResize;
            int idx = _selectedCamId - 1;
            Frame f = (idx >= 0 && idx < _latest.Length) ? _latest[idx] : null;
            if (resize == null || f == null) return null;
            byte[] full = f.Bytes; int fw = f.W, fh = f.H;

            int sx = Math.Max(0, Math.Min(srcRect.X, fw - 1));
            int sy = Math.Max(0, Math.Min(srcRect.Y, fh - 1));
            int sw = Math.Max(1, Math.Min(srcRect.Width, fw - sx));
            int sh = Math.Max(1, Math.Min(srcRect.Height, fh - sy));
            int tw = Math.Max(1, target.Width), th = Math.Max(1, target.Height);

            // 裁可見區到連續緩衝（逐列複製）
            var crop = new byte[sw * sh];
            for (int y = 0; y < sh; y++) Array.Copy(full, (sy + y) * fw + sx, crop, y * sw, sw);

            byte[] dst = resize(crop, sw, sh, tw, th);
            if (dst == null) return null;
            return GrayBitmap.From(dst, tw, th, _flip);
        }

        /// <summary>合圖 LOD provider（背景執行緒）：從完整合圖虛擬座標裁可見區，逐欄找對應相機合成
        /// （重疊分界已在 placements 的 SrcLeft/SrcWidth；無相機覆蓋處留黑）→ GrayResize → 灰階 bitmap。
        /// stride 把合成緩衝壓到 ~target 大小（縮太多時不配巨緩衝）；resize 做最終縮放。</summary>
        private Bitmap MergeLodProvide(Rectangle srcRect, Size target)
        {
            GrayResize resize = _lodResize;
            var placements = _mergePlacements;
            int vw = _mergeTotalW, vh = _mergeMaxH;
            if (resize == null || placements == null || vw <= 0 || vh <= 0) return null;

            int sx = Math.Max(0, Math.Min(srcRect.X, vw - 1));
            int sy = Math.Max(0, Math.Min(srcRect.Y, vh - 1));
            int sw = Math.Max(1, Math.Min(srcRect.Width, vw - sx));
            int sh = Math.Max(1, Math.Min(srcRect.Height, vh - sy));
            int tw = Math.Max(1, target.Width), th = Math.Max(1, target.Height);

            int strideX = Math.Max(1, sw / tw), strideY = Math.Max(1, sh / th);
            int cw = Math.Max(1, sw / strideX), ch = Math.Max(1, sh / strideY);
            var comp = new byte[cw * ch]; // 預設黑（無相機覆蓋處）

            for (int cx = 0; cx < cw; cx++)
            {
                int vx = sx + cx * strideX;
                Frame f = null; int srcX = 0;
                for (int pi = 0; pi < placements.Count; pi++)
                {
                    var p = placements[pi];
                    if (vx >= p.DestX && vx < p.DestX + p.SrcWidth)
                    { f = _latest[p.CameraId - 1]; srcX = p.SrcLeft + (vx - p.DestX); break; }
                }
                if (f == null || srcX < 0 || srcX >= f.W) continue; // 無畫面/越界 → 留黑
                byte[] fb = f.Bytes; int fwid = f.W, fhei = f.H;
                for (int cy = 0; cy < ch; cy++)
                {
                    int vy = sy + cy * strideY;
                    if (vy < fhei) comp[cy * cw + cx] = fb[vy * fwid + srcX];
                }
            }

            byte[] dst = resize(comp, cw, ch, tw, th);
            if (dst == null) return null;
            return GrayBitmap.From(dst, tw, th, _flip);
        }

        // ==================== 建圖（單張 / 合圖）====================

        private Bitmap BuildSingle()
        {
            int idx = _selectedCamId - 1;
            Frame f = (idx >= 0 && idx < _latest.Length) ? _latest[idx] : null;
            return f != null ? GrayBitmap.From(f.Bytes, f.W, f.H, _flip) : null;
        }

        /// <summary>CPU 合圖：佈局/重疊分界委派 <see cref="MergeLayout"/>（含 8 槽全納入：無畫面相機留黑占位）；
        /// 巨圖超 MergeMaxW 再降採樣 k 倍（防 GDI 座標 wrap）。</summary>
        /// <summary>算合圖幾何（placements / totalW / maxH，全解析度未 cap）→ 存快取，供 BuildMerge（cap+畫）
        /// 與合圖 LOD provider（合成可見區）共用同一份。回傳 false=無法合圖（沒畫面/ops 未設）。</summary>
        private bool TryComputeMergeGeometry()
        {
            _mergePlacements = null; _mergeTotalW = 0; _mergeMaxH = 0;
            if (!_mergeReady || _opsUm == null || _opsUm.Length == 0) return false;
            double refOpsMm = _opsUm[0] / 1000.0;
            if (refOpsMm <= 0) return false;

            int defW = 0, defH = 0;
            for (int i = 0; i < _camCount; i++)
                if (_latest[i] != null) { defW = _latest[i].W; defH = _latest[i].H; break; }
            if (defW == 0) return false;

            double minStart = MinStart();
            var geoms = new List<MergeLayout.CamGeom>();
            int maxH = 0;
            for (int i = 0; i < _camCount; i++)
            {
                bool present = _latest[i] != null;
                if (!MergeAll && !present) continue;   // 一般合圖只納入有畫面的；合圖全部 8 槽全納入（無畫面占黑）
                double st = (_startPosMm != null && i < _startPosMm.Length) ? _startPosMm[i] : 0;
                int wpx = present ? _latest[i].W : defW;
                int hpx = present ? _latest[i].H : defH;
                if (hpx > maxH) maxH = hpx;
                geoms.Add(new MergeLayout.CamGeom { CameraId = i + 1, StartMm = st, WidthPx = wpx });
            }
            if (geoms.Count == 0 || maxH <= 0) return false;

            var placements = MergeLayout.Compute(geoms, minStart, refOpsMm, _feedScale, MergeStrategy, out int totalW);
            if (totalW <= 0) return false;
            _mergePlacements = placements; _mergeTotalW = totalW; _mergeMaxH = maxH;
            return true;
        }

        private Bitmap BuildMerge()
        {
            if (!_mergeReady) return BuildSingle();
            if (!TryComputeMergeGeometry()) return BuildSingle();
            var placements = _mergePlacements;
            int totalW = _mergeTotalW, maxH = _mergeMaxH;

            int k = Math.Max(1, (totalW + MergeMaxW - 1) / MergeMaxW);
            _mergeCapK = k;
            int mw = Math.Max(1, totalW / k), mh = Math.Max(1, maxH / k);

            var merged = new Bitmap(mw, mh, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(merged))
            {
                g.Clear(Color.Black);
                g.InterpolationMode = InterpolationMode.NearestNeighbor; // 每幀重建 → 用快的 Nearest
                g.PixelOffsetMode = PixelOffsetMode.Half;
                foreach (var p in placements)
                {
                    Frame f = _latest[p.CameraId - 1];
                    if (f == null || p.SrcWidth <= 0) continue;
                    int dx = (int)Math.Round(p.DestX / (double)k);
                    int dw = Math.Max(1, (int)Math.Round(p.SrcWidth / (double)k));
                    int dh = Math.Max(1, (int)Math.Round(f.H / (double)k));
                    using (var cam = GrayBitmap.From(f.Bytes, f.W, f.H, _flip))
                        g.DrawImage(cam, new Rectangle(dx, 0, dw, dh),
                            new Rectangle(p.SrcLeft, 0, p.SrcWidth, f.H), GraphicsUnit.Pixel);
                }
            }
            return merged;
        }

        // ==================== 座標 / mm overlay ====================

        /// <summary>當前顯示座標基準：合圖用 minStart/ops0、sf=feedScale×capK；單相機用各台 ops/start、sf=feedScale。</summary>
        private bool GetDisplayCoords(out double startMm, out double opsInMm, out double sf)
        {
            if (_mergeMode && _mergeReady)
            {
                startMm = MinStart(); opsInMm = _opsUm[0] / 1000.0; sf = _feedScale * _mergeCapK;
                return opsInMm > 0;
            }
            int idx = _selectedCamId - 1;
            if (_opsUm != null && _startPosMm != null && idx >= 0 && idx < _opsUm.Length && _opsUm[idx] > 0)
            { opsInMm = _opsUm[idx] / 1000.0; startMm = _startPosMm[idx]; sf = _lodWanted && _lodResize != null && !_mergeMode ? 1 : _feedScale; return true; }
            startMm = 0; opsInMm = 0; sf = _feedScale; return false;
        }

        private double MinStart()
        {
            double m = double.MaxValue;
            if (_startPosMm != null) for (int i = 0; i < _startPosMm.Length; i++) if (_startPosMm[i] < m) m = _startPosMm[i];
            return m == double.MaxValue ? 0 : m;
        }

        private void OnCanvasStatus(CanvasInfo info)
        {
            if (_disposed || _canvas == null || info.Zoom <= 0) return;
            if (!GetDisplayCoords(out double startMm, out double opsInMm, out double sf))
            { _canvas.SetRangeOverlay("", "", "", "", ""); return; }

            double leftMm = PixelMmMapper.PixelToMm((0 - info.PanOffset.X) / info.Zoom * sf, startMm, opsInMm);
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

            ViewRangeMmChanged?.Invoke(leftMm, rightMm, topMm, botMm);

            // 游標狀態（含位置/亮度/倍率）→ 上層狀態列；mm 換算同源、不在上層重算。
            CursorStatusChanged?.Invoke(new CursorStatus
            {
                CurMmX = curMmX, CurMmY = curMmY,
                ViewLeftMm = leftMm, ViewRightMm = rightMm, ViewTopMm = topMm, ViewBotMm = botMm,
                PhysMag = physMag,
                CursorX = info.ImageX, CursorY = info.ImageY,
                Brightness = info.PixelColor.R,
                SelectedCamId = _selectedCamId,
            });

            UpdateReverseThumbSync();

            // L0 游標剖面（游標那列/行的原始像素 → 曲線圖。座標跟影像同源 → 自動對齊）。
            // 單張：取選定相機全幀那列/行。合圖：游標列橫跨整張合圖（用 BuildMerge 同一份 _mergePlacements
            // 拼，故與畫面 pixel 對齊）、游標行取所屬相機那行；座標基準走合圖（MinStart/ops0/sf=feedScale×capK）。
            // 游標出界（在畫布內但出影像 / 合圖）→ row 維持 null → 下面歸零（不停留在最後一點）。
            if (CursorProfileChanged != null)
            {
                byte[] row = null, col = null; int cx = 0, cy = 0;
                if (!_mergeMode)
                {
                    int idx = _selectedCamId - 1;
                    Frame f = (idx >= 0 && idx < _latest.Length) ? _latest[idx] : null;
                    if (f != null && info.ImageX >= 0 && info.ImageX < f.W && info.ImageY >= 0 && info.ImageY < f.H)
                    {
                        cx = info.ImageX; cy = info.ImageY; // 游標在影像座標（單張：=_latest 座標）
                        row = new byte[f.W];
                        Array.Copy(f.Bytes, cy * f.W, row, 0, f.W);
                        col = new byte[f.H];
                        for (int y = 0; y < f.H; y++) col[y] = f.Bytes[y * f.W + cx];
                    }
                }
                else if (_mergeReady && _mergePlacements != null)
                {
                    // 合圖剖面（capped 合圖座標）：mw×mh = BuildMerge 的 cap 後尺寸。
                    int k = Math.Max(1, _mergeCapK);
                    int mw = Math.Max(1, _mergeTotalW / k);
                    int mh = Math.Max(1, _mergeMaxH / k);
                    if (info.ImageX >= 0 && info.ImageX < mw && info.ImageY >= 0 && info.ImageY < mh)
                    {
                        cx = info.ImageX; cy = info.ImageY;
                        row = new byte[mw];
                        col = new byte[mh];
                        foreach (var p in _mergePlacements)
                        {
                            Frame f = (p.CameraId - 1 >= 0 && p.CameraId - 1 < _latest.Length) ? _latest[p.CameraId - 1] : null;
                            if (f == null || p.SrcWidth <= 0) continue;
                            int dx0 = (int)Math.Round(p.DestX / (double)k);
                            int dw  = Math.Max(1, (int)Math.Round(p.SrcWidth / (double)k));
                            // 游標 Y 對應本相機 source 列（含上下翻轉，與 BuildMerge 的 _flip 一致）
                            int fy = cy * k; if (fy >= f.H) fy = f.H - 1;
                            int rowBase = (_flip ? (f.H - 1 - fy) : fy) * f.W;
                            for (int j = 0; j < dw; j++)
                            {
                                int mxi = dx0 + j;
                                if (mxi < 0 || mxi >= mw) continue;
                                int sx = p.SrcLeft + (int)((long)j * p.SrcWidth / dw);
                                if (sx < 0) sx = 0; else if (sx >= f.W) sx = f.W - 1;
                                row[mxi] = f.Bytes[rowBase + sx];
                            }
                            // 縱切面：游標 X 落在本相機合圖範圍 → 取本相機該行（capped）
                            if (cx >= dx0 && cx < dx0 + dw)
                            {
                                int sx = p.SrcLeft + (int)((long)(cx - dx0) * p.SrcWidth / dw);
                                if (sx < 0) sx = 0; else if (sx >= f.W) sx = f.W - 1;
                                int ch = Math.Min(mh, Math.Max(1, f.H / k));
                                for (int y = 0; y < ch; y++)
                                {
                                    int fyy = y * k; if (fyy >= f.H) fyy = f.H - 1;
                                    col[y] = f.Bytes[(_flip ? (f.H - 1 - fyy) : fyy) * f.W + sx];
                                }
                            }
                        }
                    }
                }

                if (row != null)
                {
                    _cursorProfileCleared = false;
                    CursorProfileChanged(new CursorProfile
                    {
                        RowProfile = row, ColProfile = col,
                        StartXmm = startMm, OpsXmm = opsInMm * sf, OpsYmm = yPitch * sf,
                        ViewLeftMm = leftMm, ViewRightMm = rightMm, ViewTopMm = topMm, ViewBotMm = botMm,
                        CursorX = cx, CursorY = cy
                    });
                }
                else ClearCursorProfile(); // 出界 → 歸零（防重複 fire）
            }
        }

        /// <summary>剖面歸零（游標出界 / 離開畫布）：fire null 給訂閱者清 chart；旗標防重複 Invalidate。</summary>
        private void ClearCursorProfile()
        {
            if (_cursorProfileCleared) return;
            _cursorProfileCleared = true;
            CursorProfileChanged?.Invoke(null);
        }

        private bool AllActiveReadySinceMerge()
        {
            bool any = false;
            for (int i = 0; i < _camCount; i++)
            {
                if (_latest[i] == null) continue;
                any = true;
                if (!_readySinceMerge[i]) return false;
            }
            return any;
        }
        private bool AnyReadySinceMerge() { for (int i = 0; i < _camCount; i++) if (_readySinceMerge[i]) return true; return false; }
        private void ClearReadyFlags() { for (int i = 0; i < _camCount; i++) _readySinceMerge[i] = false; }

        public void Dispose()
        {
            _disposed = true;
            if (_timer != null) { _timer.Stop(); _timer.Dispose(); _timer = null; }
            if (_thumbStrip != null) { _thumbStrip.Dispose(); _thumbStrip = null; }
            if (_canvas != null)
            {
                _canvas.StatusChanged -= OnCanvasStatus;
                if (_canvas.LodActive) _canvas.DisableLod();
                var old = _canvas.Image; _canvas.Image = null; old?.Dispose();
                _mainPanel?.Controls.Remove(_canvas);
                _canvas.Dispose(); _canvas = null;
            }
        }
    }
}
