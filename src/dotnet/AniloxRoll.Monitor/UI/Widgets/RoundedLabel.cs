using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>圓角晶片 Label：以 <see cref="Control.BackColor"/> 當晶片填色，**反鋸齒**繪製圓角矩形底，
    /// 文字交回 <c>base.OnPaint</c>（Label 原生繪製）→ 與設計時/其他 label 一致、清晰（不自畫文字避免模糊）。
    /// 取代 Region 裁切（Region 硬邊無法反鋸齒）。強制無 BorderStyle 方框（否則圓角外露出細黑框）。
    /// 現有設 BackColor 的程式（SetIoLed/UpdateIoStateLabel）不需改，照舊運作。</summary>
    public class RoundedLabel : Label
    {
        private int _cornerRadius = 10;

        /// <summary>角半徑（px）。夠大時近似膠囊/橢圓。</summary>
        public int CornerRadius
        {
            get => _cornerRadius;
            set { _cornerRadius = Math.Max(0, value); Invalidate(); }
        }

        public RoundedLabel()
        {
            SetStyle(ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint
                   | ControlStyles.OptimizedDoubleBuffer | ControlStyles.ResizeRedraw, true);
            base.BorderStyle = BorderStyle.None;
        }

        /// <summary>強制無外框（自訂圓角繪製，不要方形 BorderStyle 露出細黑框）。設計檔設 FixedSingle 也會被忽略。</summary>
        public new BorderStyle BorderStyle
        {
            get => BorderStyle.None;
            set { base.BorderStyle = BorderStyle.None; }
        }

        protected override void OnPaintBackground(PaintEventArgs e) { /* 改由 OnPaint 畫圓角底 */ }

        protected override void OnPaint(PaintEventArgs e)
        {
            var g = e.Graphics;

            // 角落填父容器底色（圓角外的方角透出背景）
            g.Clear(Parent?.BackColor ?? SystemColors.Control);

            // 反鋸齒圓角底（填滿整塊，不留 1px 邊）
            g.SmoothingMode = SmoothingMode.AntiAlias;
            int r = Math.Max(1, Math.Min(_cornerRadius, Math.Min(Width, Height) / 2));
            using (var path = RoundedRect(new Rectangle(0, 0, Width, Height), r))
            using (var brush = new SolidBrush(BackColor))
                g.FillPath(brush, path);
            g.SmoothingMode = SmoothingMode.Default;

            // 文字交給 Label 原生繪製（與設計時一致、清晰，不自畫避免模糊）
            base.OnPaint(e);
        }

        private static GraphicsPath RoundedRect(Rectangle r, int radius)
        {
            int d = radius * 2;
            var path = new GraphicsPath();
            path.AddArc(r.X, r.Y, d, d, 180, 90);
            path.AddArc(r.Right - d, r.Y, d, d, 270, 90);
            path.AddArc(r.Right - d, r.Bottom - d, d, d, 0, 90);
            path.AddArc(r.X, r.Bottom - d, d, d, 90, 90);
            path.CloseFigure();
            return path;
        }
    }
}
