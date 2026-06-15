using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace TanukiCv.Controls
{
    /// <summary>
    /// 圓角晶片 Label（通用控制項，0 依賴 app/MIL）：以 <see cref="Control.BackColor"/> 當晶片填色，
    /// 反鋸齒繪製圓角矩形底，文字交回 <c>base.OnPaint</c>（Label 原生繪製）→ 與其他 label 一致且清晰。
    /// 取代 Region 裁切（Region 硬邊無法反鋸齒）。強制無 <see cref="BorderStyle"/> 方框（否則圓角外露細黑框）。
    /// 用法：當一般 Label 用，設 <see cref="Control.BackColor"/> 即變晶片色；<see cref="CornerRadius"/> 調圓角、
    /// <see cref="Raised"/> 開關立體漸層。狀態燈 / 標籤晶片皆適用。
    /// </summary>
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

        /// <summary>立體感（上亮下暗垂直漸層 + 邊緣高光 rim），預設開。</summary>
        public bool Raised { get; set; } = true;

        protected override void OnPaint(PaintEventArgs e)
        {
            var g = e.Graphics;

            // 角落填父容器底色（圓角外的方角透出背景）
            g.Clear(Parent?.BackColor ?? SystemColors.Control);

            // 反鋸齒圓角底（填滿整塊，不留 1px 邊）
            g.SmoothingMode = SmoothingMode.AntiAlias;
            int r = Math.Max(1, Math.Min(_cornerRadius, Math.Min(Width, Height) / 2));
            var rect = new Rectangle(0, 0, Width, Height);
            using (var path = RoundedRect(rect, r))
            {
                if (Raised && Width > 2 && Height > 2)
                {
                    // 上亮下暗的垂直漸層 → 立體浮起感（依 BackColor 自動衍生，任何狀態色都適用）
                    using (var brush = new LinearGradientBrush(
                        new Rectangle(0, -1, Width, Height + 2),
                        Lighten(BackColor, 0.30f), Darken(BackColor, 0.22f),
                        LinearGradientMode.Vertical))
                        g.FillPath(brush, path);

                    // 邊緣高光 rim（半透明白內框），加強浮起立體感
                    using (var hi = new Pen(Color.FromArgb(70, 255, 255, 255), 1.4f))
                    using (var topPath = RoundedRect(new Rectangle(1, 1, Width - 2, Height), r))
                        g.DrawPath(hi, topPath);
                }
                else
                {
                    using (var brush = new SolidBrush(BackColor))
                        g.FillPath(brush, path);
                }
            }
            g.SmoothingMode = SmoothingMode.Default;

            // 文字交給 Label 原生繪製（與設計時一致、清晰，不自畫避免模糊）
            base.OnPaint(e);
        }

        private static Color Lighten(Color c, float f) => Color.FromArgb(c.A,
            (int)Math.Min(255, c.R + (255 - c.R) * f),
            (int)Math.Min(255, c.G + (255 - c.G) * f),
            (int)Math.Min(255, c.B + (255 - c.B) * f));

        private static Color Darken(Color c, float f) => Color.FromArgb(c.A,
            (int)(c.R * (1 - f)), (int)(c.G * (1 - f)), (int)(c.B * (1 - f)));

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
