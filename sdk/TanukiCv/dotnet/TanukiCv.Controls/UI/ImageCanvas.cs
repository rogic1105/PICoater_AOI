// TanukiCv.Controls\UI\ImageCanvas.cs

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Threading.Tasks; // LOD tile GPU 重算丟背景執行緒
using System.Windows.Forms;
using TanukiCv.Core; // PixelMmMapper（實體 1:1 zoom 公式）

namespace TanukiCv.Controls
{
    public enum CanvasOverlayMode
    {
        Coordinates = 0,
        CoordinateFrames = 1,
        CoordinateFramesParameters = 2,
        Hidden = 3
    }

    public class CanvasInfo
    {
        public int ImageX { get; set; }
        public int ImageY { get; set; }
        public Color PixelColor { get; set; }
        public int Brightness { get; set; }
        public float Zoom { get; set; }
        public PointF PanOffset { get; set; }
    }

    public class ImageCanvas : PictureBox
    {
        [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
        public new Cursor Cursor
        {
            get => base.Cursor;
            set => base.Cursor = value;
        }

        private float _zoom = 1.0f;
        private PointF _panOffset = PointF.Empty;
        private bool _isDragging = false;
        private Point _lastMousePos;

        private int _lastImgX = 0;
        private int _lastImgY = 0;
        private Color _lastColor = Color.Black;
        private bool _hasLastColor;
        private string _cursorMm = ""; // 上層推入的「位置 mm」字串；空=fallback 像素座標

        // ── 畫布資訊 overlay（游標座標/亮度跟滑鼠、四邊 mm 範圍、右下實體倍率；右鍵開關）──
        private CanvasOverlayMode _overlayMode = CanvasOverlayMode.Coordinates;
        private string _ovMag = "", _ovXLeft = "", _ovXRight = "", _ovYTop = "", _ovYBottom = "";
        private Point _cursorPos;
        private bool _cursorInside;
        private Rectangle _cursorDirty = Rectangle.Empty; // 上次游標 overlay 重畫區（用於只失效小塊）

        // ── 顯示快取：整張圖在「當前 zoom」下的點陣（不含 pan）。pan 時只改貼圖偏移、不重建
        //    → FitToScreen 拖曳不再每幀重縮整張大圖。只在 zoom/Image 變時重建。 ──
        private Bitmap _viewCache;
        private float _cacheZoom = float.NaN;
        private Image _cacheImg;
        private static readonly Font  _ovFont      = new Font("Segoe UI", 9f);
        private static readonly Font  _infoFont    = new Font("Consolas", 7.5f);
        private static readonly Brush _ovBackBrush = new SolidBrush(Color.FromArgb(150, 0, 0, 0));
        private static readonly Brush _ovTextBrush = new SolidBrush(Color.White);

        public event Action<CanvasInfo> StatusChanged;
        public event Action<CanvasOverlayMode> OverlayModeChanged;

        /// <summary>從顯示色取回原始灰階強度；預設灰階讀 R，偽彩色可指定保留強度的通道。</summary>
        public Func<Color, int> BrightnessSelector { get; set; } = color => color.R;

        /// <summary>
        /// Raised on the UI thread after a requested LOD content generation has installed a tile.
        /// View-only recomputes retain the current generation.
        /// </summary>
        public event Action<long> LodTileApplied;

        // [新增] 邊緣觸發事件 (int direction: -1=上一張, 1=下一張)
        public event Action<int> EdgeReached;

        // [新增] 避免重複觸發的冷卻旗標
        private bool _edgeTriggeredInDrag = false;
        private int  _lastStatusTickMs = 0;
        private const int StatusThrottleMs = 32; // 拖曳中 chart/statusbar 最高 ~30 fps
        private int  _lastPanPaintMs = 0;        // 拖曳/hover 重繪限流 ~120fps（防高輪詢滑鼠 paint 風暴餓死 timer）

        public float Zoom => _zoom;
        public PointF PanOffset => _panOffset;
        /// <summary>相對 fit 的放大倍率（fit=1.0×）。滾 N 格 = 1.1^N，與餵入 bitmap 解析度無關 → 跨 resize 一致。</summary>
        public float ZoomRelativeToFit => _fitZoom > 0 ? _zoom / _fitZoom : _zoom;

        // ── 動態 LOD（可選，預設關）：zoom/pan 導覽一張「虛擬全解析度圖」，停住才請 provider
        //    裁可見區 + 縮到 ~panel 解析度產 tile（互動中先用舊 tile 拉伸頂著）。非 LOD 用法完全不受影響。 ──
        private bool _lodActive;
        private int _lodVirtualW, _lodVirtualH;                 // 虛擬全解析度圖尺寸
        private Func<Rectangle, Size, Bitmap> _lodProvider;     // (可見虛擬區 srcRect, 目標像素) → tile（caller 做 GPU 裁+縮）
        private Bitmap _lodTile;                                // 當前 tile（~panel 大小，canvas 持有並 Dispose）
        private Rectangle _lodTileRect;                         // _lodTile 代表的虛擬區
        private volatile bool _lodRecomputeInFlight;            // provider(GPU) 背景進行中
        private volatile bool _lodRecomputePending;            // 進行中又有新請求 → 完成後再算一次（用最新視角）
        private long _lodContentGeneration;
        private bool _visibleRefreshPending;                    // 隱藏宿主轉可見後，等待第一張可畫 LOD tile
        private Timer _lodSettleTimer;                          // 停住才重算 crisp tile
        private const int LodSettleMs = 150;

        // 滾輪 zoom 防抖：滾動中只「拉伸現有畫面」(便宜)，停下才做昂貴重建(viewCache 重建 / LOD tile 重算)
        private bool _zoomingActive;
        private Timer _zoomSettleTimer;
        private const int ZoomSettleMs = 150;

        /// <summary>LOD overscan：tile 多裁「可見區 × 此倍率」的邊（每側）。1.0=各側多一個視窗(3×3)→ 拖曳不易破圖。</summary>
        public float LodMargin { get; set; } = 1.0f;

        // ── 滾輪相對 fit 縮放（opt-in；預設 false=可縮到內容仍至少保留 1px 的實用下限）──
        //    開啟後：fit=最小(看全圖)，滾 N 格不管 resize 放大程度一致；上限=bitmap 1:1 的 N 倍
        //    → resize 多(bitmap 小)放不了那麼大、resize 少可放更大看真細節。
        private float _fitZoom = 1f;                            // FitToScreen 當下的 _zoom（=「1×」基準）
        public bool FitRelativeZoom { get; set; } = false;
        public float MaxZoomOverBitmap { get; set; } = 8f;     // 上限 = 餵入 bitmap 的 1:1 再放大幾倍（像素級檢視）

        private float MinimumUsefulZoom()
        {
            int width = ContentW;
            int height = ContentH;
            if (width <= 0 || height <= 0) return 0.000001f;

            // 固定 0.01 對超寬合圖會變成可見的縮小牆。依內容計算下限，直到較短邊仍有 1px；
            // 再縮已無可辨識資訊，且可能讓繪圖尺寸退化成 0。
            return Math.Max(0.000001f, Math.Max(1f / width, 1f / height));
        }

        /// <summary>縮小（fit/overview）時整圖快取的內插模式。預設 <see cref="InterpolationMode.NearestNeighbor"/>（快，
        /// 適合每幀重建的即時顯示）；靜態畫面（如回顧 camReviewMain）可設 HighQualityBilinear 換平滑（成本只在
        /// Image/zoom 變時付一次）。即時路徑切勿設 HighQuality —— 全解析度大圖每幀 HighQuality 縮放會卡死取像。</summary>
        public InterpolationMode DownscaleInterpolation { get; set; } = InterpolationMode.NearestNeighbor;

        // 多擊手勢（opt-in）：
        //   雙擊 = FitToScreen；三擊 = 實體 1:1（需先 SetPhysicalCalibration）。
        public bool DoubleClickFitToScreen { get; set; } = false;
        public bool TripleClickPhysical1x  { get; set; } = false;

        // 手勢事件（給上層做 app 專屬的事，如 UiActionLogger 記錄；手勢偵測/動作本身由 ImageCanvas 做）
        public event EventHandler FitPerformed;        // 雙擊 fit 完成

        /// <summary>互動流跡（診斷用，可為 null）：wheel 縮放等「使用者手勢」各記一行，
        /// 供上層流程契約驗證（誰在動視野：使用者手勢 vs 系統自動 fit）。100ms 節流防洗版。</summary>
        public Action<string> FlowLog;
        private int _flowWheelLastTick;
        public event EventHandler Physical1xPerformed; // 三擊實體 1:1 完成
        public event EventHandler DragStarted;         // 實際拖曳開始（第一次帶鍵移動）
        private bool _dragMoved;

        private readonly MultiClickDetector _multiClickDetector = new MultiClickDetector();

        /// <summary>是否在 fit-to-screen 視角（zoom≈fit + pan≈置中）。供多擊「已 fit→不歸零讓三擊接手」用，
        /// 也可給上層當「是否已 fit」單一來源（取代 app 自己重算 fitZoom）。</summary>
        public bool IsAtFitView()
        {
            if (ContentW <= 0 || ContentH <= 0 || _fitZoom <= 0) return false;
            if (Math.Abs(_zoom - _fitZoom) > _fitZoom * 0.01f) return false;
            float drawW = ContentW * _fitZoom, drawH = ContentH * _fitZoom;
            float fitPanX = (Width - drawW) / 2f, fitPanY = (Height - drawH) / 2f;
            return Math.Abs(_panOffset.X - fitPanX) < 1.5f && Math.Abs(_panOffset.Y - fitPanY) < 1.5f;
        }

        // 實體校正（mm/影像像素、mm/螢幕像素）→ 三擊跳實體 1:1。caller 依顯示模式算好 mmPerImagePx 餵進來
        // （LOD=每虛擬全解析度像素的 mm；非 LOD=每縮圖像素的 mm=opsInMm×scale）。
        private double _mmPerImagePx, _physScreenMmPerPx;
        private bool _physCalibrated;
        public void SetPhysicalCalibration(double mmPerImagePx, double screenMmPerPx)
        {
            _mmPerImagePx = mmPerImagePx; _physScreenMmPerPx = screenMmPerPx;
            _physCalibrated = mmPerImagePx > 0 && screenMmPerPx > 0;
        }

        /// <summary>實體放大倍率（螢幕上 1mm = 實際 1mm 時為 1.0×）。需先 SetPhysicalCalibration，否則回 0。
        /// 這是「實體倍率」≠ ZoomRelativeToFit「螢幕縮放」，兩者互不可取代。</summary>
        public double PhysicalMagnification =>
            _physCalibrated && _mmPerImagePx > 0 ? PixelMmMapper.PhysicalMagnification(_zoom, _physScreenMmPerPx, _mmPerImagePx) : 0;

        /// <summary>跳到實體 1:1（螢幕上 1mm = 實際 1mm），anchor 所指的影像點移到畫布中央。需先 SetPhysicalCalibration。</summary>
        public void ZoomToOneToOne(Point anchor)
        {
            if (!_physCalibrated || _zoom <= 0) return;
            float z1 = (float)PixelMmMapper.OneToOneZoom(_mmPerImagePx, _physScreenMmPerPx);
            if (z1 <= 0) return;
            float imgX = (anchor.X - _panOffset.X) / _zoom;   // 滑鼠下的影像座標
            float imgY = (anchor.Y - _panOffset.Y) / _zoom;
            _zoom = z1;
            _panOffset.X = Width  / 2f - imgX * _zoom;          // 該點移到畫布中央
            _panOffset.Y = Height / 2f - imgY * _zoom;
            if (ClampPan) ApplyPanClamp();
            this.Invalidate();
            TriggerStatusChange();
            if (_lodActive) RecomputeLodTile();
        }

        public bool LodActive => _lodActive;
        public long LodContentGeneration => _lodContentGeneration;
        public Size ContentSize => new Size(ContentW, ContentH);

        // 內容尺寸：LOD 模式用虛擬尺寸，否則用實際 Image 尺寸（座標/fit/clamp/overlay 共用）
        internal int ContentW => _lodActive ? _lodVirtualW : (this.Image?.Width  ?? 0);
        internal int ContentH => _lodActive ? _lodVirtualH : (this.Image?.Height ?? 0);   // ImageDisplayView 垂直物理座標總高用（同幾何公式基準）

        /// <summary>是否在畫布上疊顯資訊（游標座標/亮度、四邊 mm 範圍、右下實體倍率）。滑鼠右鍵切換。</summary>
        public bool ShowOverlay
        {
            get => _overlayMode != CanvasOverlayMode.Hidden;
            set { OverlayMode = value ? CanvasOverlayMode.Coordinates : CanvasOverlayMode.Hidden; }
        }

        /// <summary>
        /// Right-click cumulative display cycle: coordinates, coordinates with camera frames,
        /// coordinates with frames and parameters, then hidden.
        /// </summary>
        public CanvasOverlayMode OverlayMode
        {
            get => _overlayMode;
            set
            {
                if (_overlayMode == value) return;
                _overlayMode = value;
                Invalidate();
                OverlayModeChanged?.Invoke(value);
            }
        }

        /// <summary>Optional caller-owned text. The SDK only renders it and has no product-setting dependency.</summary>
        public Func<string> InformationTextProvider { get; set; }

        /// <summary>
        /// Optional image-space camera regions. The SDK transforms and draws the supplied rectangles;
        /// camera layout policy remains owned by the caller.
        /// </summary>
        public Func<IReadOnlyList<RectangleF>> CameraFrameRegionsProvider { get; set; }

        /// <summary>由上層（CanvasInteractionHelper）推入算好的「四邊範圍 + 倍率」字串。
        /// 座標與亮度由 canvas 自繪（已知 _lastImgX/Y、_lastColor，不必繞上層、確保跟手）。</summary>
        public void SetRangeOverlay(string magnification, string xLeft, string xRight, string yTop, string yBottom)
        {
            magnification = magnification ?? "";
            xLeft = xLeft ?? ""; xRight = xRight ?? "";
            yTop  = yTop  ?? ""; yBottom = yBottom ?? "";

            // 值沒變（hover 不動 viewport，四邊範圍/倍率相同）→ 不重畫，避免每次滑鼠移動整張重繪
            if (magnification == _ovMag && xLeft == _ovXLeft && xRight == _ovXRight &&
                yTop == _ovYTop && yBottom == _ovYBottom)
                return;

            _ovMag   = magnification;
            _ovXLeft = xLeft; _ovXRight  = xRight;
            _ovYTop  = yTop;  _ovYBottom = yBottom;
            if (ShowOverlay) Invalidate(); // 範圍真的變了（zoom/pan）才整張重畫
        }

        /// <summary>由上層（CanvasInteractionHelper）推入算好的「游標位置 mm」字串。
        /// 只儲存字串，不呼叫 Invalidate（OnMouseMove 既有的區域失效會重畫游標區，
        /// 且此方法在 paint 前同步呼叫，字串已就緒不會延遲）。空字串時 DrawOverlays
        /// fallback 回像素座標（讓未推 mm 的畫布如背景預覽優雅退化）。</summary>
        public void SetCursorMm(string text)
        {
            _cursorMm = text ?? "";
        }

        /// <summary>設定游標實體座標；即時、瀑布等呼叫端共用固定的 `(x, y)mm` 顯示格式。</summary>
        public void SetCursorMm(double x, double y)
        {
            _cursorMm = $"({x:F2}, {y:F2})mm";
        }

        /// <summary>
        /// 啟用後 pan 限制在控制項邊界內（影像不會拖出可見區域），
        /// 行為與 MIL M_CENTER_DISPLAY 一致。預設 false（自由拖曳）。
        /// </summary>
        public bool ClampPan { get; set; } = false;

        public ImageCanvas()
        {
            this.DoubleBuffered = true;
            this.SizeMode = PictureBoxSizeMode.Normal;
            this.Cursor = CreateCrosshairCursor(31, 2, Color.White);
            this.BackColor = Color.Black;

            _lodSettleTimer = new Timer { Interval = LodSettleMs };
            _lodSettleTimer.Tick += (s, e) => { _lodSettleTimer.Stop(); RecomputeLodTile(); };

            _zoomSettleTimer = new Timer { Interval = ZoomSettleMs };
            _zoomSettleTimer.Tick += (s, e) => { _zoomSettleTimer.Stop(); _zoomingActive = false; Invalidate(); };
        }

        // ── 動態 LOD API ──────────────────────────────────────────────────────
        /// <summary>啟用動態 LOD：宣告虛擬全解析度圖尺寸 + provider（裁可見區+縮到目標像素）。
        /// 啟用後 canvas 不用 Image 當主內容，改用 provider 產的 tile；zoom/pan 以虛擬座標導覽。</summary>
        public void EnableLod(int virtualW, int virtualH, Func<Rectangle, Size, Bitmap> provider)
        {
            _lodVirtualW = Math.Max(1, virtualW);
            _lodVirtualH = Math.Max(1, virtualH);
            _lodProvider = provider;
            _lodActive   = provider != null;
            if (_lodActive)
            {
                System.Threading.Interlocked.Increment(ref _lodContentGeneration);
                FitToScreen();
                RecomputeLodTile();
            }
        }

        /// <summary>虛擬尺寸變了（如換相機/換合圖大小）但仍要 LOD → 更新尺寸並重 fit。</summary>
        public void UpdateLodVirtualSize(int virtualW, int virtualH)
        {
            if (!_lodActive) return;
            _lodVirtualW = Math.Max(1, virtualW);
            _lodVirtualH = Math.Max(1, virtualH);
            System.Threading.Interlocked.Increment(ref _lodContentGeneration);
            FitToScreen();
            RecomputeLodTile();
        }

        public void DisableLod()
        {
            _lodActive = false;
            _lodSettleTimer.Stop();
            if (_lodTile != null) { _lodTile.Dispose(); _lodTile = null; }
            _lodProvider = null;
            // 必須重繪：LOD 模式下 .Image 為 null，不 Invalidate 則螢幕殘留最後一次 tile 像素
            // 且之後無人觸發 paint（「清空內容」路徑的殘影根因，2026-07-09 預覽第二按定罪）
            Invalidate();
        }

        /// <summary>新幀進來（內容變、視角沒變）→ 立刻用當前視角重算 tile（不延遲）。</summary>
        public void RefreshLod() => RefreshLod(false);

        /// <summary>
        /// 重新計算 LOD 內容。新一輪資料開始時可先清除目前 tile，避免舊內容在非同步重算完成前
        /// 繼續顯示；一般串流更新則保留目前 tile，避免每幀閃黑。
        /// </summary>
        public void RefreshLod(bool clearCurrentTile)
        {
            if (!_lodActive) return;
            System.Threading.Interlocked.Increment(ref _lodContentGeneration);
            if (clearCurrentTile && _lodTile != null)
            {
                _lodTile.Dispose();
                _lodTile = null;
                _lodTileRect = Rectangle.Empty;
                Invalidate();
            }
            RecomputeLodTile();
        }

        /// <summary>
        /// 宿主由隱藏轉為可見時，以完成 layout 後的控制項尺寸補畫內容。
        /// LOD 尚無 tile 時主動重算；已有 tile 或一般 Image 時同步 paint，避免必須先有滑鼠互動才顯示。
        /// </summary>
        public void RefreshVisibleContent()
        {
            if (IsDisposed) return;

            bool ready = _lodActive ? _lodTile != null : Image != null;
            if (_lodActive)
            {
                _visibleRefreshPending = !ready;
                RefreshLod();
            }

            Invalidate();
            Update();
            if (ready)
                FlowLog?.Invoke($"visiblePaint ready=True lod={_lodActive} size={ClientSize.Width}x{ClientSize.Height}");
            else if (!_lodActive)
                FlowLog?.Invoke($"visiblePaint ready=False lod=False size={ClientSize.Width}x{ClientSize.Height}");
        }

        private void RestartLodSettle()
        {
            if (!_lodActive) return;
            _lodSettleTimer.Stop();
            _lodSettleTimer.Start();
        }

        private int _lodLastRecomputeMs;

        /// <summary>拖曳中視角變了：排程停住重算；若已拖出 overscan tile 範圍 → 立刻補（節流 120ms，避免每像素 GPU）。</summary>
        private void LodOnDrag()
        {
            if (!_lodActive) return;
            RestartLodSettle();
            int now = Environment.TickCount;
            if (now - _lodLastRecomputeMs >= 120 && !VisibleInsideTile())
                RecomputeLodTile(); // 內含更新 _lodLastRecomputeMs
        }

        private bool VisibleInsideTile()
        {
            if (_lodTile == null || _zoom <= 0) return false;
            float vx0 = (0 - _panOffset.X) / _zoom;
            float vy0 = (0 - _panOffset.Y) / _zoom;
            float vx1 = (Width  - _panOffset.X) / _zoom;
            float vy1 = (Height - _panOffset.Y) / _zoom;
            return vx0 >= _lodTileRect.X && vy0 >= _lodTileRect.Y &&
                   vx1 <= _lodTileRect.X + _lodTileRect.Width &&
                   vy1 <= _lodTileRect.Y + _lodTileRect.Height;
        }

        /// <summary>算當前視角的可見虛擬區 → 請 provider 裁+縮到 ~螢幕投影大小 → 存為 tile。</summary>
        private void RecomputeLodTile()
        {
            if (!_lodActive || _lodProvider == null || _zoom <= 0 || Width <= 0 || Height <= 0) return;

            // 可見虛擬區（虛擬座標）= 螢幕視窗反投影
            float vx0 = (0 - _panOffset.X) / _zoom;
            float vy0 = (0 - _panOffset.Y) / _zoom;
            float vx1 = (Width  - _panOffset.X) / _zoom;
            float vy1 = (Height - _panOffset.Y) / _zoom;

            // overscan：往外擴「可見寬高 × LodMargin」（各側）→ 拖曳在 tile 內、不破圖；再 clamp 進 [0, virtual]
            float mx = (vx1 - vx0) * LodMargin;
            float my = (vy1 - vy0) * LodMargin;
            int rx = (int)Math.Floor(Math.Max(0, vx0 - mx));
            int ry = (int)Math.Floor(Math.Max(0, vy0 - my));
            int rr = (int)Math.Ceiling(Math.Min(_lodVirtualW, vx1 + mx));
            int rb = (int)Math.Ceiling(Math.Min(_lodVirtualH, vy1 + my));
            int rw = Math.Max(1, rr - rx);
            int rh = Math.Max(1, rb - ry);
            var srcRect = new Rectangle(rx, ry, rw, rh);

            // 目標像素 = 該區投影到螢幕的大小 → 即「縮到 ~panel 解析度（含 overscan）」；放大時 rw 小→縮得少→清晰。
            // 上限約 (1+2×margin)×視窗，避免 tile 過大。
            int capW = (int)((1f + 2f * LodMargin) * Width)  + 2;
            int capH = (int)((1f + 2f * LodMargin) * Height) + 2;
            int tw = Math.Max(1, Math.Min(capW, (int)Math.Round(rw * _zoom)));
            int th = Math.Max(1, Math.Min(capH, (int)Math.Round(rh * _zoom)));

            // provider(GPU 裁+縮) 丟背景執行緒 → 停下/換幀不凍 UI；同時間只一個（in-flight），
            // 進行中又被請求則記 pending，完成後用「最新視角」再算一次。
            if (_lodRecomputeInFlight) { _lodRecomputePending = true; return; }
            _lodRecomputeInFlight = true;
            var provider = _lodProvider;
            var size = new Size(tw, th);
            long contentGeneration = _lodContentGeneration;
            Task.Run(() =>
            {
                Bitmap tile = null;
                try { tile = provider(srcRect, size); } catch { tile = null; }
                try
                {
                    if (IsHandleCreated && !IsDisposed)
                        BeginInvoke((Action)(() => ApplyLodTile(tile, srcRect, contentGeneration)));
                    else { tile?.Dispose(); _lodRecomputeInFlight = false; }
                }
                catch { tile?.Dispose(); _lodRecomputeInFlight = false; }
            });
        }

        /// <summary>背景 provider 完成 → UI 執行緒套 tile + 若有 pending 再算一次（最新視角）。</summary>
        private void ApplyLodTile(Bitmap tile, Rectangle rect, long contentGeneration)
        {
            _lodRecomputeInFlight = false;
            if (!_lodActive) { tile?.Dispose(); return; }
            if (contentGeneration != System.Threading.Interlocked.Read(ref _lodContentGeneration))
            {
                tile?.Dispose();
                if (_lodRecomputePending)
                {
                    _lodRecomputePending = false;
                    RecomputeLodTile();
                }
                return;
            }
            if (tile != null)
            {
                if (_lodTile != null && !ReferenceEquals(_lodTile, tile)) _lodTile.Dispose();
                _lodTile = tile;
                _lodTileRect = rect;
                _lodLastRecomputeMs = Environment.TickCount;
                this.Invalidate();
                LodTileApplied?.Invoke(contentGeneration);
            }
            if (_visibleRefreshPending)
            {
                FlowLog?.Invoke($"visiblePaint ready={tile != null} lod=True size={ClientSize.Width}x{ClientSize.Height}");
                if (tile != null) _visibleRefreshPending = false;
            }
            if (_lodRecomputePending) { _lodRecomputePending = false; RecomputeLodTile(); }
        }

        private static Cursor CreateCrosshairCursor(int size, int lineWidth, Color foreColor)
        {
            if (size % 2 == 0) size++;          // 確保奇數，十字正中央
            int half = size / 2;
            int outlineWidth = lineWidth + 2;   // 黑邊比白線各多 1px

            using (var bmp = new Bitmap(size, size))
            {
                bmp.SetPixel(0, 0, Color.FromArgb(1, 0, 0, 0)); // 強制非全透明，防止 Windows icon 反色

                using (var g = Graphics.FromImage(bmp))
                {
                    // 先畫黑色描邊（較粗）
                    using (var outline = new Pen(Color.Black, outlineWidth))
                    {
                        g.DrawLine(outline, half, 0, half, size - 1);
                        g.DrawLine(outline, 0, half, size - 1, half);
                    }
                    // 再畫白色前景（較細，疊在上面）
                    using (var fore = new Pen(foreColor, lineWidth))
                    {
                        g.DrawLine(fore, half, 0, half, size - 1);
                        g.DrawLine(fore, 0, half, size - 1, half);
                    }
                }
                return new Cursor(bmp.GetHicon());
            }
        }

        private void TriggerStatusChange()
        {
            _statStatusCount++;   // 飽和組成計數（stats 行用）
            StatusChanged?.Invoke(new CanvasInfo
            {
                ImageX = _lastImgX,
                ImageY = _lastImgY,
                PixelColor = _lastColor,
                Brightness = _hasLastColor ? ResolveBrightness(_lastColor) : 0,
                Zoom = _zoom,
                PanOffset = _panOffset
            });
        }

        public void SetView(float zoom, PointF panOffset)
        {
            _zoom = zoom;
            _panOffset = panOffset;
            this.Invalidate();
            TriggerStatusChange();
        }

        public void FitToScreen()
        {
            int iw = ContentW, ih = ContentH;
            if (!AspectFitCalculator.TryCompute(
                iw, ih, Width, Height, out AspectFitTransform fit)) return;

            _zoom = fit.Zoom;
            _panOffset = new PointF(fit.PanX, fit.PanY);
            _fitZoom = _zoom;                 // 記下「1×」基準（FitRelativeZoom 用）

            this.Invalidate();
            TriggerStatusChange();
            if (_lodActive) RecomputeLodTile(); // 視角變了 → LOD 立即重產 tile
        }

        protected override void OnMouseDown(MouseEventArgs e)
        {
            base.OnMouseDown(e);
            if (e.Button == MouseButtons.Left)
            {
                _isDragging = true;
                _dragMoved = false;
                _lastMousePos = e.Location;
                _edgeTriggeredInDrag = false; // 重置觸發旗標

                // 多擊手勢：雙擊 fit、三擊實體 1:1（opt-in；歸零防下一下誤觸更高擊數）
                int clicks = _multiClickDetector.RegisterClick(e.Location);
                if (clicks == 2 && DoubleClickFitToScreen && (this.Image != null || _lodActive) && !IsAtFitView())
                {
                    // 只有「非 fit」才 fit + 歸零；已在 fit → 不做事也不歸零，讓第三下能觸發三擊。
                    _isDragging = false; _multiClickDetector.Consume();
                    FitToScreen();
                    FitPerformed?.Invoke(this, EventArgs.Empty);
                }
                else if (clicks >= 3 && TripleClickPhysical1x && _physCalibrated)
                {
                    _isDragging = false; _multiClickDetector.Consume();
                    ZoomToOneToOne(e.Location);
                    Physical1xPerformed?.Invoke(this, EventArgs.Empty);
                }
            }
            else if (e.Button == MouseButtons.Right)
            {
                OverlayMode = (CanvasOverlayMode)(((int)OverlayMode + 1) % 4);
                FlowLog?.Invoke($"overlay mode={OverlayMode}");
            }
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            base.OnMouseUp(e);
            bool wasDragging = _isDragging && _dragMoved;
            _isDragging = false;
            this.Invalidate();     // 重繪限流的尾緣保證：最終 pan 位置必上畫（中間跳過的只是超過螢幕更新率的浪費幀）
            TriggerStatusChange(); // 拖曳結束後補一次，更新 chart range 與 status bar
            // 拖曳放開＝一行四邊座標（上層 SetRangeOverlay 推入的 mm 字串；遠端驗「畫面上下左右範圍」用）
            if (wasDragging && FlowLog != null && (_ovXLeft != "" || _ovYTop != ""))
                FlowLog($"viewEdges X {_ovXLeft}~{_ovXRight}｜Y {_ovYTop}~{_ovYBottom}");
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);

            _cursorPos = e.Location;   // overlay 游標座標/亮度跟手用
            _cursorInside = true;
            bool firstDragMove = false;

            if (_isDragging)
            {
                if (!_dragMoved)
                {
                    firstDragMove = true;
                    _dragMoved = true;
                    DragStarted?.Invoke(this, EventArgs.Empty);
                    FlowLog?.Invoke("drag(start)");
                }
                _panOffset.X += e.X - _lastMousePos.X;
                _panOffset.Y += e.Y - _lastMousePos.Y;
                _lastMousePos = e.Location;
                if (ClampPan) ApplyPanClamp();
                // 重繪限流 ~120fps：高輪詢率滑鼠（500~1000Hz）每則 move 都 Invalidate → 「input→paint」
                // 死循環每秒畫 300~400 次（螢幕才 60Hz＝白畫 6 倍）、佇列永遠不空 → WM_TIMER（最低優先權）
                // 全體餓死＝縮圖橘框/33ms 補刷卡死（2026-07-07 IC stats paints=386/s 定罪）。
                // pan 值照每則 move 累積（最終位置不失真），只有「畫」限流；MouseUp 補一次最終重繪。
                int nowPan = Environment.TickCount;
                if (nowPan - _lastPanPaintMs >= 8)
                {
                    _lastPanPaintMs = nowPan;
                    this.Invalidate();
                }
                LodOnDrag(); // LOD：停住重算 + 拖出 overscan 範圍即時補

                // [新增] 檢查是否拉到邊界
                if (!ClampPan) CheckEdgeTrigger();
            }

            if (this.Image != null || _lodActive)
            {
                float imgXf = (e.X - _panOffset.X) / _zoom;
                float imgYf = (e.Y - _panOffset.Y) / _zoom;

                _lastImgX = (int)imgXf;
                _lastImgY = (int)imgYf;

                if (_isDragging)
            {
                // 拖曳中：跳過 GetPixel（慢），chart/statusbar 限流到 ~30 fps
                // 讓 canvas Invalidate() 能以最快速度進入 OnPaint
                int now = Environment.TickCount;
                // 首個位移不能吃到 hover 共用的 32ms 節流窗；否則短拖曳只剩 MouseUp 尾緣，
                // 圖片雖跟手，curve 卻看起來完全不動。首發後才回到既有 ~30fps 上限。
                if (firstDragMove || now - _lastStatusTickMs >= StatusThrottleMs)
                {
                    _lastStatusTickMs = now;
                    TriggerStatusChange();
                    // StatusChanged 是同步事件；這行出現代表所有 view-range subscriber 已返回，
                    // 可作 DVT 的「首個 curve 視野已落地」證據，而不只是事件已送出。
                    if (firstDragMove) FlowLog?.Invoke("drag(view-published)");
                }
            }
            else
            {
                if (_lodActive)
                {
                    _hasLastColor = TrySampleLodColor(_lastImgX, _lastImgY, out _lastColor);
                }
                else if (this.Image is Bitmap bmp &&
                         _lastImgX >= 0 && _lastImgX < bmp.Width &&
                         _lastImgY >= 0 && _lastImgY < bmp.Height)
                {
                    _lastColor = bmp.GetPixel(_lastImgX, _lastImgY);
                    _hasLastColor = true;
                }
                else
                {
                    _lastColor = Color.Black;
                    _hasLastColor = false;
                }

                // 重的 status/chart 同步限流 ~30fps；游標 overlay 區域失效限流 ~120fps
                //（高輪詢滑鼠 hover 也會 paint 風暴，同拖曳 pan 的病）
                int now = Environment.TickCount;
                if (now - _lastStatusTickMs >= StatusThrottleMs)
                {
                    _lastStatusTickMs = now;
                    TriggerStatusChange();
                }
                if (ShowOverlay && now - _lastPanPaintMs >= 8)
                {
                    _lastPanPaintMs = now;
                    InvalidateCursorOverlay();
                }
            }
            }
        }

        private bool TrySampleLodColor(int imageX, int imageY, out Color color)
        {
            color = Color.Black;
            if (_lodTile == null || _lodTileRect.Width <= 0 || _lodTileRect.Height <= 0 ||
                !_lodTileRect.Contains(imageX, imageY))
                return false;

            int tileX = (int)((long)(imageX - _lodTileRect.X) * _lodTile.Width / _lodTileRect.Width);
            int tileY = (int)((long)(imageY - _lodTileRect.Y) * _lodTile.Height / _lodTileRect.Height);
            tileX = Math.Max(0, Math.Min(_lodTile.Width - 1, tileX));
            tileY = Math.Max(0, Math.Min(_lodTile.Height - 1, tileY));
            color = _lodTile.GetPixel(tileX, tileY);
            return true;
        }

        private string GetCursorOverlayText()
        {
            string head = string.IsNullOrEmpty(_cursorMm)
                ? $"({_lastImgX}, {_lastImgY})"
                : _cursorMm;
            string tail = _hasLastColor ? $" | 亮度:{ResolveBrightness(_lastColor)}" : " | 亮度:--";
            return head + tail;
        }

        private int ResolveBrightness(Color color)
        {
            Func<Color, int> selector = BrightnessSelector;
            if (selector == null) return color.R;
            return Math.Max(0, Math.Min(255, selector(color)));
        }

        private Rectangle GetCursorLabelBounds(Point cursorPosition)
        {
            string text = GetCursorOverlayText();
            Size size = TextRenderer.MeasureText(text, _ovFont, Size.Empty,
                TextFormatFlags.NoPadding | TextFormatFlags.SingleLine);

            int x = cursorPosition.X + 14;
            int y = cursorPosition.Y + 14;
            x = Math.Max(0, Math.Min(x, Math.Max(0, Width - size.Width)));
            y = Math.Max(0, Math.Min(y, Math.Max(0, Height - size.Height)));
            return new Rectangle(x, y, size.Width, size.Height);
        }

        private Rectangle GetCursorOverlayBounds(Point cursorPosition)
        {
            var bounds = Rectangle.Inflate(GetCursorLabelBounds(cursorPosition), 3, 2);
            return Rectangle.Intersect(ClientRectangle, bounds);
        }

        private void DrawCursorLabel(Graphics g)
        {
            string text = GetCursorOverlayText();
            Rectangle bounds = GetCursorLabelBounds(_cursorPos);
            Rectangle background = Rectangle.Inflate(bounds, 2, 1);
            g.FillRectangle(_ovBackBrush, background);
            TextRenderer.DrawText(g, text, _ovFont, bounds.Location, Color.White,
                TextFormatFlags.NoPadding | TextFormatFlags.SingleLine);
        }

        /// <summary>只失效游標 overlay 的實際區域（上次 + 這次兩塊分離矩形）。
        /// 文字包含座標/亮度且會因數值改變寬度，不得用固定框估算，否則框外尾字會留下殘影。</summary>
        private void InvalidateCursorOverlay()
        {
            var box = GetCursorOverlayBounds(_cursorPos);
            using (var region = new Region(box))
            {
                if (_cursorDirty != Rectangle.Empty)
                    region.Union(_cursorDirty);
                Invalidate(region);
            }
            _cursorDirty = box;
        }

        protected override void OnMouseLeave(EventArgs e)
        {
            base.OnMouseLeave(e);
            _cursorInside = false;
            if (ShowOverlay && _cursorDirty != Rectangle.Empty)
            {
                Invalidate(_cursorDirty); // 只清掉游標 overlay 那一小塊
                _cursorDirty = Rectangle.Empty;
            }
        }

        // [新增] 檢查邊界邏輯
        private void CheckEdgeTrigger()
        {
            if (this.Image == null || _edgeTriggeredInDrag) return;

            float drawW = this.Image.Width * _zoom;
            float imageRightEdgeX = _panOffset.X + drawW;

            // 觸發門檻值 (例如 50 pixel)
            float threshold = 50.0f;

            // 1. 往左拉 (想看右邊/下一張)
            // 當圖片的「右邊緣」已經非常接近畫布的左邊界 (甚至進入負值)
            // 代表使用者已經快要把這張圖拉完了
            if (imageRightEdgeX < threshold)
            {
                _edgeTriggeredInDrag = true;
                _isDragging = false; // 強制停止拖曳，避免連續觸發
                EdgeReached?.Invoke(1); // 1 = Next
            }

            // 2. 往右拉 (想看左邊/上一張)
            // 當圖片的「左邊緣」已經非常接近畫布的右邊界
            else if (_panOffset.X > (this.Width - threshold))
            {
                _edgeTriggeredInDrag = true;
                _isDragging = false;
                EdgeReached?.Invoke(-1); // -1 = Prev
            }
        }

        protected override void OnMouseWheel(MouseEventArgs e)
        {
            float oldZoom = _zoom;
            // 縮放正比於滾輪「實際轉動量」：一格 ±120 → ×1.1。事件合併（UI 卡頓時 Windows 把多格併成一個
            // 大 e.Delta）也按比例 → 同樣的物理滾動量得到同樣縮放，與卡頓/餵入 bitmap 解析度無關。
            float notches = e.Delta / 120f;
            _zoom *= (float)Math.Pow(1.1, notches);

            float minimumZoom;
            if (FitRelativeZoom)
            {
                // fit=最小（看全圖），上限=bitmap 1:1 的 N 倍。滾 N 格相對 fit 放大一致；
                // resize 多→fit 小→到上限的「相對倍率」小（放不了那麼大），resize 少→可放更大。
                float lo = _fitZoom;
                float hi = Math.Max(_fitZoom, MaxZoomOverBitmap);
                if (_zoom < lo) _zoom = lo;
                if (_zoom > hi) _zoom = hi;
                minimumZoom = lo;
            }
            else
            {
                minimumZoom = MinimumUsefulZoom();
                if (_zoom < minimumZoom) _zoom = minimumZoom;
                if (_zoom > 100.0f) _zoom = 100.0f;
            }

            float scaleChange = _zoom / oldZoom;

            if (FlowLog != null && Environment.TickCount - _flowWheelLastTick > 100)
            {
                _flowWheelLastTick = Environment.TickCount;
                FlowLog(
                    $"wheelZoom {(notches > 0 ? "in" : "out")} → " +
                    $"zoom={_zoom:0.########}（fit={_fitZoom:0.########} min={minimumZoom:0.########} " +
                    $"content={ContentW}x{ContentH}）");
            }

            _panOffset.X = e.X - (e.X - _panOffset.X) * scaleChange;
            _panOffset.Y = e.Y - (e.Y - _panOffset.Y) * scaleChange;
            if (ClampPan) ApplyPanClamp();

            // zoom 防抖：滾動中只拉伸現有畫面（非 LOD 拉伸舊 viewCache、LOD 拉伸舊 tile），停下才重建
            _zoomingActive = true;
            _zoomSettleTimer.Stop(); _zoomSettleTimer.Start();

            this.Invalidate();
            TriggerStatusChange();
            RestartLodSettle(); // LOD：停住才重算 crisp tile（互動中先用舊 tile 拉伸）
        }

        /// <summary>
        /// 限制 pan 使影像不會被拖出控制項邊界。
        /// 影像比控制項小時置中；比控制項大時限制邊緣。
        /// </summary>
        private void ApplyPanClamp()
        {
            if (ContentW <= 0 || ContentH <= 0) return;
            float drawW = ContentW * _zoom;
            float drawH = ContentH * _zoom;

            // X 軸
            if (drawW <= this.Width)
                _panOffset.X = (this.Width - drawW) / 2;   // 置中
            else
            {
                if (_panOffset.X > 0) _panOffset.X = 0;                          // 左邊緣
                if (_panOffset.X + drawW < this.Width) _panOffset.X = this.Width - drawW;  // 右邊緣
            }

            // Y 軸
            if (drawH <= this.Height)
                _panOffset.Y = (this.Height - drawH) / 2;  // 置中
            else
            {
                if (_panOffset.Y > 0) _panOffset.Y = 0;
                if (_panOffset.Y + drawH < this.Height) _panOffset.Y = this.Height - drawH;
            }
        }

        /// <summary>上次 OnPaint 實際繪製耗時（ms，含小數）。供效能量測/比較（如 MIL 直繪 vs PictureBox）。</summary>
        public double LastPaintMs => _paintTimer.LastMs;
        /// <summary>自上次 <see cref="ResetMaxPaintMs"/> 以來的最大 OnPaint 耗時（worst-case，判斷卡頓用）。</summary>
        public double MaxPaintMs => _paintTimer.MaxMs;
        /// <summary>讀取 MaxPaintMs 後呼叫以重置視窗（取「每段時間內最差」）。</summary>
        public double ResetMaxPaintMs() => _paintTimer.ResetMax();
        private readonly PerfTimer _paintTimer = new PerfTimer(); // 繪製計時（TanukiCv.Core 通用計時器，與範例縮圖計時同一來源）

        // 飽和組成計數（FlowLog 有接才啟用）：每秒一行 paint 次數/總耗時/狀態事件數 → 看拖曳時 UI 在忙什麼比例
        private int _statWinStartMs, _statPaintCount, _statPaintMs, _statStatusCount;

        protected override void OnPaint(PaintEventArgs pe)
        {
            // 卡頓歸因儀器：畫一次超過 50ms 即經 FlowLog 留痕（呼叫端決定 log 去向；null=零成本）
            var swPaint = FlowLog != null ? System.Diagnostics.Stopwatch.StartNew() : null;
            try { OnPaintBody(pe); }
            finally
            {
                if (swPaint != null)
                {
                    int ms = (int)swPaint.ElapsedMilliseconds;
                    if (ms > 50)
                        FlowLog($"paint {ms}ms（zoom={_zoom:F2} lod={_lodActive} cache={_viewCache != null}）");
                    _statPaintCount++; _statPaintMs += ms;
                    int now = Environment.TickCount;
                    if (_statWinStartMs == 0) _statWinStartMs = now;
                    else if (now - _statWinStartMs >= 1000)
                    {
                        if (_statPaintCount > 5)   // 靜態畫面偶發重繪不洗版；拖曳/串流時每秒一行
                            FlowLog($"stats paints={_statPaintCount}/s paintMs={_statPaintMs} statusEv={_statStatusCount}/s");
                        _statWinStartMs = now; _statPaintCount = 0; _statPaintMs = 0; _statStatusCount = 0;
                    }
                }
            }
        }

        private void OnPaintBody(PaintEventArgs pe)
        {
            if (_lodActive)
            {
                _paintTimer.Start();
                pe.Graphics.Clear(this.BackColor);
                if (_lodTile != null)
                {
                    // tile 代表 _lodTileRect（虛擬區）→ 畫到該區投影到螢幕的位置/大小。
                    // 互動中視角已變但 tile 未重算 → 同一 tile 被拉伸（便宜，停住才換 crisp）。
                    float dx = _lodTileRect.X * _zoom + _panOffset.X;
                    float dy = _lodTileRect.Y * _zoom + _panOffset.Y;
                    float dw = _lodTileRect.Width  * _zoom;
                    float dh = _lodTileRect.Height * _zoom;
                    pe.Graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
                    pe.Graphics.PixelOffsetMode   = PixelOffsetMode.Half;
                    pe.Graphics.DrawImage(_lodTile, dx, dy, dw, dh);
                }
                if (ShowOverlay) DrawOverlays(pe.Graphics);
                _paintTimer.Stop();
                return;
            }

            if (this.Image == null) { base.OnPaint(pe); return; }

            _paintTimer.Start();
            EnsureViewCache();
            pe.Graphics.Clear(this.BackColor); // 整圖快取只蓋影像範圍，邊緣/失效區先填底色

            if (_viewCache != null)
            {
                if (_zoomingActive && _cacheZoom > 0 && _zoom != _cacheZoom)
                {
                    // 滾動中：把舊 cache（在 _cacheZoom 下）依 _zoom/_cacheZoom 拉伸（便宜）→ 停下才重建一次
                    float s = _zoom / _cacheZoom;
                    pe.Graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
                    pe.Graphics.PixelOffsetMode   = PixelOffsetMode.Half;
                    pe.Graphics.DrawImage(_viewCache, _panOffset.X, _panOffset.Y,
                        _viewCache.Width * s, _viewCache.Height * s);
                }
                else
                {
                    // 整圖快取：pan 只是把同一張縮好的圖「貼到不同位置」→ 不重縮 → FitToScreen 拖曳超順
                    pe.Graphics.DrawImageUnscaled(_viewCache,
                        (int)Math.Round(_panOffset.X), (int)Math.Round(_panOffset.Y));
                }
            }
            else
            {
                // 放大太多、整圖超過快取預算 → per-frame 只畫可見區（放大時取樣小、便宜）
                pe.Graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
                pe.Graphics.PixelOffsetMode   = PixelOffsetMode.Half;
                pe.Graphics.DrawImage(this.Image, _panOffset.X, _panOffset.Y,
                    this.Image.Width * _zoom, this.Image.Height * _zoom);
            }

            if (ShowOverlay) DrawOverlays(pe.Graphics);

            _paintTimer.Stop();
        }

        /// <summary>_viewCache =「整張圖在當前 zoom 下」的點陣（不含 pan）。pan 直接以偏移貼上、不重建
        /// → FitToScreen 拖曳不再每幀重縮整張。只在 zoom/Image/超預算狀態改變時重建。
        /// 放大太多致整圖超過記憶體預算（~6× 控制項面積）→ _viewCache=null，OnPaint 改 per-frame（放大時便宜）。</summary>
        private void EnsureViewCache()
        {
            // 滾輪 zoom 進行中且有現成 cache → 不重建，OnPaint 拉伸舊 cache（便宜）；停下 settle 才重建一次。
            if (_zoomingActive && _viewCache != null) return;

            int  sw = Math.Max(1, (int)Math.Round(this.Image.Width  * _zoom));
            int  sh = Math.Max(1, (int)Math.Round(this.Image.Height * _zoom));
            long budget = Math.Max(6L * Width * Height, 8_000_000L);

            if ((long)sw * sh > budget) // 整圖太大 → 不快取，走 per-frame
            {
                if (_viewCache != null) { _viewCache.Dispose(); _viewCache = null; }
                _cacheZoom = float.NaN;
                return;
            }

            bool valid = _viewCache != null && _viewCache.Width == sw && _viewCache.Height == sh
                         && _zoom == _cacheZoom && ReferenceEquals(this.Image, _cacheImg);
            if (valid) return; // zoom/Image 沒變 → pan 直接沿用，不重建

            if (_viewCache == null || _viewCache.Width != sw || _viewCache.Height != sh)
            {
                _viewCache?.Dispose();
                _viewCache = new Bitmap(sw, sh, System.Drawing.Imaging.PixelFormat.Format32bppPArgb);
            }
            using (var g = Graphics.FromImage(_viewCache))
            {
                g.Clear(this.BackColor);
                // 縮小（sw<原圖寬，如 fit/overview）→ 用 DownscaleInterpolation（預設 Nearest，快；
                // 靜態畫面可設 HighQualityBilinear 換平滑）。放大 → NearestNeighbor 保持像素邊界清晰、不糊。
                // ⚠ 即時顯示每幀重建快取 → 縮小務必用快的 Nearest，否則全解析度大圖 HighQuality 縮放會卡死取像。
                bool shrinking = sw < this.Image.Width;
                g.InterpolationMode = shrinking ? DownscaleInterpolation : InterpolationMode.NearestNeighbor;
                g.PixelOffsetMode   = PixelOffsetMode.Half;
                g.DrawImage(this.Image, 0, 0, sw, sh); // 整張縮到快取（不含 pan）
            }
            _cacheZoom = _zoom;
            _cacheImg  = this.Image;
        }

        // ── Overlay 繪製 ──────────────────────────────────────────────────────
        private void DrawOverlays(Graphics g)
        {
            g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;
            const int pad = 4;

            bool drawFrames = _overlayMode == CanvasOverlayMode.CoordinateFrames ||
                _overlayMode == CanvasOverlayMode.CoordinateFramesParameters;
            if (drawFrames)
                DrawCameraFrames(g);

            // All visible modes retain coordinates; later modes add camera regions and CFG.
            if (!string.IsNullOrEmpty(_ovYTop))    DrawLabel(g, _ovYTop,    Width / 2,    pad,           0.5f, 0f);
            if (!string.IsNullOrEmpty(_ovYBottom)) DrawLabel(g, _ovYBottom, Width / 2,    Height - pad,  0.5f, 1f);
            if (!string.IsNullOrEmpty(_ovXLeft))   DrawRotatedLabel(g, _ovXLeft,  pad,          Height / 2, true);
            if (!string.IsNullOrEmpty(_ovXRight))  DrawRotatedLabel(g, _ovXRight, Width - pad,  Height / 2, false);
            if (!string.IsNullOrEmpty(_ovMag)) DrawLabel(g, _ovMag, Width - pad, Height - pad, 1f, 1f);

            // 游標位置 mm + 亮度（跟滑鼠）；無 mm 字串時 fallback 像素座標
            if (_cursorInside && ContentW > 0 && ContentH > 0 &&
                _lastImgX >= 0 && _lastImgY >= 0 &&
                _lastImgX < ContentW && _lastImgY < ContentH)
            {
                DrawCursorLabel(g);
            }

            if (_overlayMode == CanvasOverlayMode.CoordinateFramesParameters)
                DrawInformationPanel(g);
        }

        private void DrawCameraFrames(Graphics g)
        {
            IReadOnlyList<RectangleF> regions = null;
            try { regions = CameraFrameRegionsProvider?.Invoke(); }
            catch { regions = null; }
            if (regions == null || regions.Count == 0) return;

            using (var pen = new Pen(Color.FromArgb(249, 168, 37), 2f))
            {
                for (int i = 0; i < regions.Count; i++)
                {
                    RectangleF region = regions[i];
                    if (region.Width <= 0 || region.Height <= 0) continue;
                    float x = _panOffset.X + region.X * _zoom;
                    float y = _panOffset.Y + region.Y * _zoom;
                    float w = region.Width * _zoom;
                    float h = region.Height * _zoom;
                    g.DrawRectangle(pen, x, y, w, h);
                }
            }
        }

        private void DrawInformationPanel(Graphics g)
        {
            string text = null;
            try { text = InformationTextProvider?.Invoke(); }
            catch { text = null; }
            if (string.IsNullOrWhiteSpace(text)) return;

            int maxWidth = Math.Max(160, Math.Min(520, Width / 2));
            Size proposed = new Size(maxWidth, Math.Max(40, Height - 16));
            TextFormatFlags flags = TextFormatFlags.Left | TextFormatFlags.Top |
                TextFormatFlags.WordBreak | TextFormatFlags.NoPrefix;
            Size measured = TextRenderer.MeasureText(g, text, _infoFont, proposed, flags);
            var bounds = new Rectangle(8, 8,
                Math.Min(maxWidth, measured.Width + 10),
                Math.Min(Math.Max(24, Height - 16), measured.Height + 8));
            g.FillRectangle(_ovBackBrush, bounds);
            TextRenderer.DrawText(g, text, _infoFont,
                Rectangle.Inflate(bounds, -5, -4), Color.White, flags);
        }

        /// <summary>畫帶半透明底的文字。(ax, ay) 為錨點對齊比例（0=左/上，0.5=中，1=右/下），再 clamp 進畫布。</summary>
        private void DrawLabel(Graphics g, string text, int x, int y, float ax, float ay)
        {
            SizeF sz = g.MeasureString(text, _ovFont);
            float bx = x - sz.Width  * ax;
            float by = y - sz.Height * ay;
            bx = Math.Max(0, Math.Min(bx, Width  - sz.Width));
            by = Math.Max(0, Math.Min(by, Height - sz.Height));
            g.FillRectangle(_ovBackBrush, bx - 2, by - 1, sz.Width + 4, sz.Height + 2);
            g.DrawString(text, _ovFont, _ovTextBrush, bx, by);
        }

        /// <summary>沿左/右邊畫 90° 旋轉的垂直文字（讀向由下往上），垂直置中於 yCenter。</summary>
        private void DrawRotatedLabel(Graphics g, string text, int x, int yCenter, bool leftEdge)
        {
            SizeF sz = g.MeasureString(text, _ovFont);
            var state = g.Save();
            g.TranslateTransform(x, yCenter);
            g.RotateTransform(-90);                       // 逆時針 90°：文字由下往上
            float bx = -sz.Width / 2f;                    // 沿（旋轉後的）垂直方向置中
            float by = leftEdge ? 0 : -sz.Height;         // 左邊文字往內（右），右邊往內（左）
            g.FillRectangle(_ovBackBrush, bx - 2, by - 1, sz.Width + 4, sz.Height + 2);
            g.DrawString(text, _ovFont, _ovTextBrush, bx, by);
            g.Restore(state);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _viewCache?.Dispose(); _viewCache = null;
                _lodSettleTimer?.Dispose(); _lodSettleTimer = null;
                _zoomSettleTimer?.Dispose(); _zoomSettleTimer = null;
                _lodTile?.Dispose(); _lodTile = null;
            }
            base.Dispose(disposing);
        }
    }
}
