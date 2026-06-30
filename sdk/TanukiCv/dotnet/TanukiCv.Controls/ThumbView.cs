using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace TanukiCv.Controls
{
    /// <summary>
    /// 雙緩衝縮圖控制項：自繪 bitmap（zoom-to-fit 黑底 letterbox），換圖只 <see cref="SetImage"/> → Invalidate，
    /// **不閃黑**。取代 PictureBox.Image swap —— PictureBox 重繪會「先填底色再畫圖」，且雙緩衝對它未必生效。
    /// 多相機監控縮圖（<see cref="ImageDisplayView"/>）與範例縮圖共用。執行緒：SetImage 須在 UI 執行緒呼叫。
    /// </summary>
    public sealed class ThumbView : Control
    {
        private Bitmap _img;
        private bool _selected;

        /// <summary>選取高亮（主畫面↔縮圖雙向連動的視覺）：true 畫邊框。</summary>
        public bool Selected
        {
            get => _selected;
            set { if (_selected != value) { _selected = value; Invalidate(); } }
        }

        /// <summary>選取邊框色（預設亮藍；上層可改）。</summary>
        public Color SelectedBorderColor { get; set; } = Color.DeepSkyBlue;

        public ThumbView()
        {
            DoubleBuffered = true;
            SetStyle(ControlStyles.ResizeRedraw | ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint, true);
            BackColor = Color.Black;
        }

        /// <summary>換顯示圖（接管 ownership，dispose 舊圖）；控制項已釋放則 dispose 傳入圖。</summary>
        public void SetImage(Bitmap img)
        {
            if (IsDisposed) { img?.Dispose(); return; }
            var old = _img; _img = img; old?.Dispose();
            Invalidate();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            e.Graphics.Clear(BackColor);
            var img = _img;
            var cs = ClientSize;
            if (img != null && cs.Width > 0 && cs.Height > 0 && img.Width > 0 && img.Height > 0)
            {
                double s = Math.Min(cs.Width / (double)img.Width, cs.Height / (double)img.Height);
                int dw = Math.Max(1, (int)(img.Width * s)), dh = Math.Max(1, (int)(img.Height * s));
                int dx = (cs.Width - dw) / 2, dy = (cs.Height - dh) / 2;
                // 即時縮圖每幀重繪 → 一律 NearestNeighbor（快）；HighQuality 對大圖每幀縮放會卡 → 反而閃黑更久。
                e.Graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
                e.Graphics.PixelOffsetMode = PixelOffsetMode.Half;
                e.Graphics.DrawImage(img, dx, dy, dw, dh);
            }
            if (_selected && cs.Width > 2 && cs.Height > 2)
            {
                using (var pen = new Pen(SelectedBorderColor, 2f))
                    e.Graphics.DrawRectangle(pen, 1, 1, cs.Width - 3, cs.Height - 3);
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) { _img?.Dispose(); _img = null; }
            base.Dispose(disposing);
        }
    }
}
