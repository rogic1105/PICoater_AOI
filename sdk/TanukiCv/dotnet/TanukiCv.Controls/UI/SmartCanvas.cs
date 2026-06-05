// TanukiCv.Controls\UI\SmartCanvas.cs

using System;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace TanukiCv.Controls
{
    public class CanvasInfo
    {
        public int ImageX { get; set; }
        public int ImageY { get; set; }
        public Color PixelColor { get; set; }
        public float Zoom { get; set; }
        public PointF PanOffset { get; set; }
    }

    public class SmartCanvas : PictureBox
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

        // ── 畫布資訊 overlay（游標座標/亮度跟滑鼠、四邊 mm 範圍、右下實體倍率；右鍵開關）──
        private bool _showOverlay = true;
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
        private static readonly Brush _ovBackBrush = new SolidBrush(Color.FromArgb(150, 0, 0, 0));
        private static readonly Brush _ovTextBrush = new SolidBrush(Color.White);

        public event Action<CanvasInfo> StatusChanged;

        // [新增] 邊緣觸發事件 (int direction: -1=上一張, 1=下一張)
        public event Action<int> EdgeReached;

        // [新增] 避免重複觸發的冷卻旗標
        private bool _edgeTriggeredInDrag = false;
        private int  _lastStatusTickMs = 0;
        private const int StatusThrottleMs = 32; // 拖曳中 chart/statusbar 最高 ~30 fps

        public float Zoom => _zoom;
        public PointF PanOffset => _panOffset;

        /// <summary>是否在畫布上疊顯資訊（游標座標/亮度、四邊 mm 範圍、右下實體倍率）。滑鼠右鍵切換。</summary>
        public bool ShowOverlay
        {
            get => _showOverlay;
            set { _showOverlay = value; Invalidate(); }
        }

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
            if (_showOverlay) Invalidate(); // 範圍真的變了（zoom/pan）才整張重畫
        }

        /// <summary>
        /// 啟用後 pan 限制在控制項邊界內（影像不會拖出可見區域），
        /// 行為與 MIL M_CENTER_DISPLAY 一致。預設 false（自由拖曳）。
        /// </summary>
        public bool ClampPan { get; set; } = false;

        public SmartCanvas()
        {
            this.DoubleBuffered = true;
            this.SizeMode = PictureBoxSizeMode.Normal;
            this.Cursor = CreateCrosshairCursor(31, 2, Color.White);
            this.BackColor = Color.Black;
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
            StatusChanged?.Invoke(new CanvasInfo
            {
                ImageX = _lastImgX,
                ImageY = _lastImgY,
                PixelColor = _lastColor,
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
            if (this.Image == null) return;

            float ratioW = (float)this.Width / this.Image.Width;
            float ratioH = (float)this.Height / this.Image.Height;
            _zoom = Math.Min(ratioW, ratioH) * 0.95f;

            float drawW = this.Image.Width * _zoom;
            float drawH = this.Image.Height * _zoom;
            _panOffset = new PointF((this.Width - drawW) / 2, (this.Height - drawH) / 2);

            this.Invalidate();
            TriggerStatusChange();
        }

        protected override void OnMouseDown(MouseEventArgs e)
        {
            base.OnMouseDown(e);
            if (e.Button == MouseButtons.Left)
            {
                _isDragging = true;
                _lastMousePos = e.Location;
                _edgeTriggeredInDrag = false; // 重置觸發旗標
            }
            else if (e.Button == MouseButtons.Right)
            {
                ShowOverlay = !ShowOverlay; // 右鍵開關畫布資訊
            }
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            base.OnMouseUp(e);
            _isDragging = false;
            TriggerStatusChange(); // 拖曳結束後補一次，更新 chart range 與 status bar
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);

            _cursorPos = e.Location;   // overlay 游標座標/亮度跟手用
            _cursorInside = true;

            if (_isDragging)
            {
                _panOffset.X += e.X - _lastMousePos.X;
                _panOffset.Y += e.Y - _lastMousePos.Y;
                _lastMousePos = e.Location;
                if (ClampPan) ApplyPanClamp();
                this.Invalidate();

                // [新增] 檢查是否拉到邊界
                if (!ClampPan) CheckEdgeTrigger();
            }

            if (this.Image != null)
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
                if (now - _lastStatusTickMs >= StatusThrottleMs)
                {
                    _lastStatusTickMs = now;
                    TriggerStatusChange();
                }
            }
            else
            {
                if (this.Image is Bitmap bmp &&
                    _lastImgX >= 0 && _lastImgX < bmp.Width &&
                    _lastImgY >= 0 && _lastImgY < bmp.Height)
                {
                    _lastColor = bmp.GetPixel(_lastImgX, _lastImgY);
                }
                else
                {
                    _lastColor = Color.Black;
                }

                // 重的 status/chart 同步限流 ~30fps；游標 overlay（便宜）每次都更新跟手
                int now = Environment.TickCount;
                if (now - _lastStatusTickMs >= StatusThrottleMs)
                {
                    _lastStatusTickMs = now;
                    TriggerStatusChange();
                }
                if (_showOverlay) InvalidateCursorOverlay();
            }
            }
        }

        /// <summary>只失效游標 overlay 的區域（上次 + 這次當「兩塊分離小矩形」，非外接框）。
        /// 用 Region 而非 Rectangle.Union：快速移動時舊/新位置離很遠，外接框會幾乎=整張 → 失去意義。
        /// GDI 把 OnPaint 的 DrawImage 裁切到此區 → 成本只剩這兩小塊。</summary>
        private void InvalidateCursorOverlay()
        {
            // 框需完整蓋住標籤實際繪製範圍（標籤在游標右下 +14；邊界 clamp 時可能移到左/上）。
            // 不足會殘留黑底殘影 → 兩側都給足（最寬座標文字約 170px）。
            var box = new Rectangle(_cursorPos.X - 190, _cursorPos.Y - 28, 380, 74);
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
            if (_showOverlay && _cursorDirty != Rectangle.Empty)
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
            float factor = 1.1f;

            if (e.Delta > 0) _zoom *= factor;
            else _zoom /= factor;

            if (_zoom < 0.01f) _zoom = 0.01f;
            if (_zoom > 100.0f) _zoom = 100.0f;

            float scaleChange = _zoom / oldZoom;

            _panOffset.X = e.X - (e.X - _panOffset.X) * scaleChange;
            _panOffset.Y = e.Y - (e.Y - _panOffset.Y) * scaleChange;
            if (ClampPan) ApplyPanClamp();

            this.Invalidate();
            TriggerStatusChange();
        }

        /// <summary>
        /// 限制 pan 使影像不會被拖出控制項邊界。
        /// 影像比控制項小時置中；比控制項大時限制邊緣。
        /// </summary>
        private void ApplyPanClamp()
        {
            if (this.Image == null) return;
            float drawW = this.Image.Width * _zoom;
            float drawH = this.Image.Height * _zoom;

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

        protected override void OnPaint(PaintEventArgs pe)
        {
            if (this.Image == null) { base.OnPaint(pe); return; }

            EnsureViewCache();
            pe.Graphics.Clear(this.BackColor); // 整圖快取只蓋影像範圍，邊緣/失效區先填底色

            if (_viewCache != null)
            {
                // 整圖快取：pan 只是把同一張縮好的圖「貼到不同位置」→ 不重縮 → FitToScreen 拖曳超順
                pe.Graphics.DrawImageUnscaled(_viewCache,
                    (int)Math.Round(_panOffset.X), (int)Math.Round(_panOffset.Y));
            }
            else
            {
                // 放大太多、整圖超過快取預算 → per-frame 只畫可見區（放大時取樣小、便宜）
                pe.Graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
                pe.Graphics.PixelOffsetMode   = PixelOffsetMode.Half;
                pe.Graphics.DrawImage(this.Image, _panOffset.X, _panOffset.Y,
                    this.Image.Width * _zoom, this.Image.Height * _zoom);
            }

            if (_showOverlay) DrawOverlays(pe.Graphics);
        }

        /// <summary>_viewCache =「整張圖在當前 zoom 下」的點陣（不含 pan）。pan 直接以偏移貼上、不重建
        /// → FitToScreen 拖曳不再每幀重縮整張。只在 zoom/Image/超預算狀態改變時重建。
        /// 放大太多致整圖超過記憶體預算（~6× 控制項面積）→ _viewCache=null，OnPaint 改 per-frame（放大時便宜）。</summary>
        private void EnsureViewCache()
        {
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
                g.InterpolationMode = InterpolationMode.NearestNeighbor;
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

            // 四邊範圍：Y 上/下邊中央、X 左/右邊垂直（90° 旋轉）
            if (!string.IsNullOrEmpty(_ovYTop))    DrawLabel(g, _ovYTop,    Width / 2,    pad,           0.5f, 0f);
            if (!string.IsNullOrEmpty(_ovYBottom)) DrawLabel(g, _ovYBottom, Width / 2,    Height - pad,  0.5f, 1f);
            if (!string.IsNullOrEmpty(_ovXLeft))   DrawRotatedLabel(g, _ovXLeft,  pad,          Height / 2, true);
            if (!string.IsNullOrEmpty(_ovXRight))  DrawRotatedLabel(g, _ovXRight, Width - pad,  Height / 2, false);

            // 右下角：實體倍率
            if (!string.IsNullOrEmpty(_ovMag)) DrawLabel(g, _ovMag, Width - pad, Height - pad, 1f, 1f);

            // 游標座標 + 亮度（跟滑鼠）
            if (_cursorInside && this.Image != null &&
                _lastImgX >= 0 && _lastImgY >= 0 &&
                _lastImgX < this.Image.Width && _lastImgY < this.Image.Height)
            {
                DrawLabel(g, $"({_lastImgX}, {_lastImgY})  {_lastColor.R}",
                          _cursorPos.X + 14, _cursorPos.Y + 14, 0f, 0f);
            }
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
            if (disposing) { _viewCache?.Dispose(); _viewCache = null; }
            base.Dispose(disposing);
        }
    }
}