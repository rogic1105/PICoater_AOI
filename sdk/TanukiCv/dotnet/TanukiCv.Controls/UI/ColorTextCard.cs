using System;
using System.Drawing;
using System.Windows.Forms;

namespace TanukiCv.Controls
{
    /// <summary>
    /// 彩色三行字卡片（通用自繪控制項，0 依賴 app/MIL）：以 <see cref="Control.BackColor"/> 當整塊底色，
    /// 上 / 中 / 下三段文字（上下小字帶、中間大字填滿剩餘高度）全部在單一 <see cref="OnPaint"/> 一次畫完。
    /// <para>
    /// 換色 = 一次 <see cref="SetContent"/> = 一次 Invalidate = <b>原子重繪</b>，不會出現「上半舊色下半新色」的分區
    /// flicker。取代「Panel + 3 個 Dock 的 Label」的多控制項疊法（那會各自非同步重繪 → 帶狀變色）。雙緩衝。
    /// </para>
    /// 用途：狀態色卡 / KPI 卡 / 良率卡 等「一塊底色 + 三行字、換色要一次到位」的需求。
    /// 顏色門檻、文字內容等業務規則由呼叫端算好，透過 <see cref="SetContent"/> 餵進來。
    /// </summary>
    public sealed class ColorTextCard : Control
    {
        private string _top = string.Empty;
        private string _center = string.Empty;
        private string _bottom = string.Empty;

        /// <summary>上方小字帶高度（px）。預設 22。</summary>
        public int TopBandHeight { get; set; } = 22;

        /// <summary>下方小字帶高度（px）。預設 18。中段自動填滿 <c>Height - TopBandHeight - BottomBandHeight</c>。</summary>
        public int BottomBandHeight { get; set; } = 18;

        /// <summary>上段字型（控制項擁有並負責 dispose）。</summary>
        public Font TopFont { get; set; }

        /// <summary>中段字型（控制項擁有並負責 dispose）。</summary>
        public Font CenterFont { get; set; }

        /// <summary>下段字型（控制項擁有並負責 dispose）。</summary>
        public Font BottomFont { get; set; }

        public ColorTextCard()
        {
            ForeColor  = Color.White;
            TopFont    = new Font("Segoe UI", 8.5f, FontStyle.Bold);
            CenterFont = new Font("Segoe UI", 18f,  FontStyle.Bold);
            BottomFont = new Font("Segoe UI", 7.5f);
            SetStyle(ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint
                   | ControlStyles.OptimizedDoubleBuffer | ControlStyles.ResizeRedraw, true);
        }

        /// <summary>一次設定底色 + 三行字並重畫（唯一原子更新入口）。</summary>
        public void SetContent(Color back, string top, string center, string bottom)
        {
            _top    = top    ?? string.Empty;
            _center = center ?? string.Empty;
            _bottom = bottom ?? string.Empty;
            if (BackColor != back)
                BackColor = back;   // OnBackColorChanged 觸發 Invalidate
            Invalidate();           // 文字變化也要重畫
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            var g = e.Graphics;
            g.Clear(BackColor);

            const TextFormatFlags flags = TextFormatFlags.HorizontalCenter
                                        | TextFormatFlags.VerticalCenter
                                        | TextFormatFlags.NoPrefix
                                        | TextFormatFlags.EndEllipsis;
            int w = ClientSize.Width;
            int h = ClientSize.Height;
            int midH = Math.Max(0, h - TopBandHeight - BottomBandHeight);

            TextRenderer.DrawText(g, _top, TopFont,
                new Rectangle(0, 0, w, TopBandHeight), ForeColor, flags);
            TextRenderer.DrawText(g, _center, CenterFont,
                new Rectangle(0, TopBandHeight, w, midH), ForeColor, flags);
            TextRenderer.DrawText(g, _bottom, BottomFont,
                new Rectangle(0, h - BottomBandHeight, w, BottomBandHeight), ForeColor, flags);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                TopFont?.Dispose();
                CenterFont?.Dispose();
                BottomFont?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
