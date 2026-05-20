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

        private readonly Label[] _lblCamName  = new Label[7];
        private readonly Label[] _lblPassFail = new Label[7];
        private readonly Label[] _lblRate     = new Label[7];

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
            var font      = new Font("Segoe UI", 8.5f, FontStyle.Bold);
            var fontBig   = new Font("Segoe UI", 18f,  FontStyle.Bold);
            var fontSmall = new Font("Segoe UI", 7.5f);

            for (int i = 0; i < 7; i++)
            {
                var panel = _panels[i];
                panel.Controls.Clear();
                panel.BackColor = ColorEmpty;

                _lblCamName[i] = new Label
                {
                    Text      = $"CAM{i + 1}",
                    Font      = font,
                    ForeColor = Color.White,
                    Dock      = DockStyle.Top,
                    Height    = 22,
                    TextAlign = ContentAlignment.MiddleCenter,
                    BackColor = Color.Transparent
                };
                _lblRate[i] = new Label
                {
                    Text      = "—",
                    Font      = fontBig,
                    ForeColor = Color.White,
                    Dock      = DockStyle.Fill,
                    TextAlign = ContentAlignment.MiddleCenter,
                    BackColor = Color.Transparent
                };
                _lblPassFail[i] = new Label
                {
                    Text      = "0 / 0",
                    Font      = fontSmall,
                    ForeColor = Color.White,
                    Dock      = DockStyle.Bottom,
                    Height    = 18,
                    TextAlign = ContentAlignment.MiddleCenter,
                    BackColor = Color.Transparent
                };

                panel.Controls.Add(_lblRate[i]);
                panel.Controls.Add(_lblCamName[i]);
                panel.Controls.Add(_lblPassFail[i]);
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
                _panels[idx].BackColor = ColorEmpty;
                _lblRate[idx].Text     = "—";
                _lblPassFail[idx].Text = "0 / 0";
                return;
            }

            float rate = s.PassRate;
            _panels[idx].BackColor = rate >= 0.95f ? ColorGood
                                   : rate >= 0.80f ? ColorWarning
                                                   : ColorBad;
            _lblRate[idx].Text     = $"{rate:P1}";
            _lblPassFail[idx].Text = $"{s.Pass} / {s.Total}";
        }
    }
}
