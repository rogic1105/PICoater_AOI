// AOI_SDK\src_dotnet\AOI.SDK.UI\SmartCanvas.cs

using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace AOI.SDK.UI
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
        private float _zoom = 1.0f;
        private PointF _panOffset = PointF.Empty;
        private bool _isDragging = false;
        private Point _lastMousePos;

        private int _lastImgX = 0;
        private int _lastImgY = 0;
        private Color _lastColor = Color.Black;

        public event Action<CanvasInfo> StatusChanged;

        // [新增] 邊緣觸發事件 (int direction: -1=上一張, 1=下一張)
        public event Action<int> EdgeReached;

        // [新增] 避免重複觸發的冷卻旗標
        private bool _edgeTriggeredInDrag = false;
        private int  _lastStatusTickMs = 0;
        private const int StatusThrottleMs = 32; // 拖曳中 chart/statusbar 最高 ~30 fps

        public float Zoom => _zoom;
        public PointF PanOffset => _panOffset;

        public SmartCanvas()
        {
            this.DoubleBuffered = true;
            this.SizeMode = PictureBoxSizeMode.Normal;
            this.Cursor = Cursors.Cross;
            this.BackColor = Color.Black;
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

            if (_isDragging)
            {
                _panOffset.X += e.X - _lastMousePos.X;
                _panOffset.Y += e.Y - _lastMousePos.Y;
                _lastMousePos = e.Location;
                this.Invalidate();

                // [新增] 檢查是否拉到邊界
                CheckEdgeTrigger();
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

                TriggerStatusChange();
            }
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

            this.Invalidate();
            TriggerStatusChange();
        }

        protected override void OnPaint(PaintEventArgs pe)
        {
            if (this.Image == null) { base.OnPaint(pe); return; }

            pe.Graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
            pe.Graphics.PixelOffsetMode = PixelOffsetMode.Half;

            float drawW = this.Image.Width * _zoom;
            float drawH = this.Image.Height * _zoom;

            pe.Graphics.DrawImage(this.Image, _panOffset.X, _panOffset.Y, drawW, drawH);
        }
    }
}