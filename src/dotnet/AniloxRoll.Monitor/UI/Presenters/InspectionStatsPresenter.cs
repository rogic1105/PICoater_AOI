using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Presenters
{
    /// <summary>
    /// 管理「檢測報表」Tab 上方 7 個 Panel 卡片（良率色卡）。
    /// 卡片顏色：良率 ≥ 95% → 綠；80–95% → 橙；< 80% → 紅；無資料 → 灰。
    /// </summary>
    public class InspectionStatsPresenter
    {
        private readonly Panel[] _panels;   // index 0 = CAM1 … index 6 = CAM7

        private static readonly Color ColorGood    = Color.FromArgb(102, 187, 106);
        private static readonly Color ColorWarning = Color.FromArgb(255, 167,  38);
        private static readonly Color ColorBad     = Color.FromArgb(239,  83,  80);
        private static readonly Color ColorEmpty   = Color.FromArgb(158, 158, 158);

        private readonly YieldCardView[] _cards = new YieldCardView[7];

        public InspectionStatsPresenter(Panel[] panels)
        {
            if (panels == null || panels.Length < 7)
                throw new ArgumentException("panels must contain 7 entries.");
            _panels = panels;
        }

        public void Initialize()
        {
            InitCards();
        }

        private void InitCards()
        {
            var fontName  = new Font("Segoe UI", 8.5f, FontStyle.Bold);
            var fontRate  = new Font("Segoe UI", 18f,  FontStyle.Bold);
            var fontSmall = new Font("Segoe UI", 7.5f);

            for (int i = 0; i < 7; i++)
            {
                var panel = _panels[i];
                panel.Controls.Clear();
                panel.BackColor = ColorEmpty;

                // 一張卡片 = 一個雙緩衝自繪控制項（背景 + 三行字同一 OnPaint 一次畫完）。
                // 舊版用 3 個 Dock 的透明 Label，換色時各自非同步重繪 → 半紅半綠俄羅斯方塊。
                var card = new YieldCardView(fontName, fontRate, fontSmall)
                {
                    Dock = DockStyle.Fill,
                };
                card.SetContent(ColorEmpty, $"CAM{i + 1}", "—", "0 / 0");
                panel.Controls.Add(card);
                _cards[i] = card;
            }
        }

        /// <summary>更新 7 個色卡（Pass/Fail/Rate）。</summary>
        public void Update(Dictionary<int, CameraStats> stats)
        {
            for (int i = 1; i <= 7; i++)
            {
                CameraStats s = stats.TryGetValue(i, out var v) ? v : null;
                UpdateCard(i - 1, s);
            }
        }

        private void UpdateCard(int idx, CameraStats s)
        {
            if (s == null || s.Total == 0)
            {
                _cards[idx].SetContent(ColorEmpty, $"CAM{idx + 1}", "—", "0 / 0");
                return;
            }

            float rate = s.PassRate;
            Color back = rate >= 0.95f ? ColorGood
                       : rate >= 0.80f ? ColorWarning
                                       : ColorBad;
            _cards[idx].SetContent(back, $"CAM{idx + 1}", $"{rate:P1}", $"{s.Pass} / {s.Total}");
        }

        /// <summary>
        /// 良率色卡自繪控制項：雙緩衝，背景 + CAM 名（上）+ 良率（中大字）+ Pass/Fail（下）
        /// 全部在單一 OnPaint 內畫完。換色 = 一次 Invalidate = 一個原子重繪，不會半紅半綠。
        /// </summary>
        private sealed class YieldCardView : Control
        {
            private readonly Font _fontName;
            private readonly Font _fontRate;
            private readonly Font _fontSmall;
            private string _camName = string.Empty;
            private string _rate = "—";
            private string _passFail = "0 / 0";

            private const int TopBand    = 22;
            private const int BottomBand = 18;

            public YieldCardView(Font fontName, Font fontRate, Font fontSmall)
            {
                _fontName  = fontName;
                _fontRate  = fontRate;
                _fontSmall = fontSmall;
                ForeColor  = Color.White;
                SetStyle(ControlStyles.OptimizedDoubleBuffer
                       | ControlStyles.AllPaintingInWmPaint
                       | ControlStyles.UserPaint
                       | ControlStyles.ResizeRedraw, true);
            }

            public void SetContent(Color back, string camName, string rate, string passFail)
            {
                _camName = camName ?? string.Empty;
                _rate = rate ?? string.Empty;
                _passFail = passFail ?? string.Empty;
                if (BackColor != back)
                    BackColor = back;   // 觸發 Invalidate（OnBackColorChanged）
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
                int midH = Math.Max(0, h - TopBand - BottomBand);

                TextRenderer.DrawText(g, _camName, _fontName,
                    new Rectangle(0, 0, w, TopBand), ForeColor, flags);
                TextRenderer.DrawText(g, _rate, _fontRate,
                    new Rectangle(0, TopBand, w, midH), ForeColor, flags);
                TextRenderer.DrawText(g, _passFail, _fontSmall,
                    new Rectangle(0, h - BottomBand, w, BottomBand), ForeColor, flags);
            }
        }
    }
}
