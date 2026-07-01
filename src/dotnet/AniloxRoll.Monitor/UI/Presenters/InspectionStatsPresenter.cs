using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Services;
using TanukiCv.Controls;

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

        private readonly ColorTextCard[] _cards = new ColorTextCard[7];

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
            for (int i = 0; i < 7; i++)
            {
                var panel = _panels[i];
                panel.Controls.Clear();
                panel.BackColor = ColorEmpty;

                // 一張卡片 = 一個 sdk ColorTextCard（雙緩衝自繪，背景 + 三行字同一 OnPaint 一次畫完）。
                // 舊版用 3 個 Dock 的 Label 疊在 Panel 上，換色時各自非同步重繪 → 半紅半綠俄羅斯方塊。
                var card = new ColorTextCard { Dock = DockStyle.Fill };
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
    }
}
