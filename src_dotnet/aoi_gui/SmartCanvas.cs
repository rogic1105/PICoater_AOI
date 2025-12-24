// SmartCanvas.cs

using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace AOI_GUI
{
    public class SmartCanvas : PictureBox
    {
        private float _zoom = 1.0f;
        private PointF _panOffset = PointF.Empty; // 改用 PointF 提高運算精度
        private bool _isDragging = false;
        private Point _lastMousePos;

        public event Action<int, int, Color> PixelHovered;

        public SmartCanvas()
        {
            this.DoubleBuffered = true;
            this.SizeMode = PictureBoxSizeMode.Normal;
            this.Cursor = Cursors.Cross;
            this.BackColor = Color.Black;
        }

        // 1. [新增] 自動縮放至視窗大小
        public void FitToScreen()
        {
            if (this.Image == null) return;

            // 計算寬高比，取較小的那個作為縮放比，確保整張圖都看得到
            float ratioW = (float)this.Width / this.Image.Width;
            float ratioH = (float)this.Height / this.Image.Height;
            _zoom = Math.Min(ratioW, ratioH);

            // 稍微留一點邊距 (95%)
            _zoom *= 0.95f;

            // 計算置中偏移量
            float drawW = this.Image.Width * _zoom;
            float drawH = this.Image.Height * _zoom;
            _panOffset = new PointF((this.Width - drawW) / 2, (this.Height - drawH) / 2);

            this.Invalidate();
        }

        // 重置視野 (保留原本的方法，但通常用 FitToScreen 比較多)
        public void ResetView()
        {
            FitToScreen();
        }

        protected override void OnMouseDown(MouseEventArgs e)
        {
            base.OnMouseDown(e);
            if (e.Button == MouseButtons.Left)
            {
                _isDragging = true;
                _lastMousePos = e.Location;
            }
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            base.OnMouseUp(e);
            _isDragging = false;
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);

            if (_isDragging)
            {
                // 拖曳平移
                _panOffset.X += e.X - _lastMousePos.X;
                _panOffset.Y += e.Y - _lastMousePos.Y;
                _lastMousePos = e.Location;
                this.Invalidate();
            }

            // 顯示數值
            if (this.Image != null && this.Image is Bitmap bmp)
            {
                // 反算座標: (Mouse - Offset) / Zoom
                float imgXf = (e.X - _panOffset.X) / _zoom;
                float imgYf = (e.Y - _panOffset.Y) / _zoom;
                int imgX = (int)imgXf;
                int imgY = (int)imgYf;

                if (imgX >= 0 && imgX < bmp.Width && imgY >= 0 && imgY < bmp.Height)
                {
                    Color c = bmp.GetPixel(imgX, imgY);
                    PixelHovered?.Invoke(imgX, imgY, c);
                }
            }
        }

        // 2. [修改] 滾輪縮放：以滑鼠為中心
        protected override void OnMouseWheel(MouseEventArgs e)
        {
            float oldZoom = _zoom;
            float factor = 1.1f;

            if (e.Delta > 0) _zoom *= factor;
            else _zoom /= factor;

            // 限制縮放範圍
            if (_zoom < 0.01f) _zoom = 0.01f;
            if (_zoom > 100.0f) _zoom = 100.0f;

            // === 關鍵數學：保持滑鼠指向的圖片點不動 ===
            // 公式：NewOffset = Mouse - (Mouse - OldOffset) * (NewZoom / OldZoom)
            float scaleChange = _zoom / oldZoom;

            _panOffset.X = e.X - (e.X - _panOffset.X) * scaleChange;
            _panOffset.Y = e.Y - (e.Y - _panOffset.Y) * scaleChange;

            this.Invalidate();
        }

        protected override void OnPaint(PaintEventArgs pe)
        {
            if (this.Image == null) { base.OnPaint(pe); return; }

            pe.Graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
            pe.Graphics.PixelOffsetMode = PixelOffsetMode.Half;

            float drawW = this.Image.Width * _zoom;
            float drawH = this.Image.Height * _zoom;

            // 繪製圖片
            pe.Graphics.DrawImage(this.Image, _panOffset.X, _panOffset.Y, drawW, drawH);
        }
    }
}