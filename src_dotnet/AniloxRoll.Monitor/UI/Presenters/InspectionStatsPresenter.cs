using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Presenters
{
    /// <summary>
    /// 管理「檢測數據」Tab 的 ListView（表格）與 7 個 Panel（卡片）顯示。
    /// 卡片顏色：良率 ≥ 95% → 綠；80–95% → 橙；< 80% → 紅；無資料 → 灰。
    /// </summary>
    public class InspectionStatsPresenter
    {
        private readonly ListView   _listView;
        private readonly Panel[]    _panels;   // index 0 = CAM1 … index 6 = CAM7

        private static readonly Color ColorGood    = Color.FromArgb(102, 187, 106);  // 綠
        private static readonly Color ColorWarning = Color.FromArgb(255, 167,  38);  // 橙
        private static readonly Color ColorBad     = Color.FromArgb(239,  83,  80);  // 紅
        private static readonly Color ColorEmpty   = Color.FromArgb(158, 158, 158);  // 灰

        // 每個 panel 內的子 Label（動態建立）
        private readonly Label[] _lblCamName  = new Label[7];
        private readonly Label[] _lblPassFail = new Label[7];
        private readonly Label[] _lblRate     = new Label[7];

        public InspectionStatsPresenter(ListView listView, Panel[] panels)
        {
            if (panels == null || panels.Length < 7)
                throw new ArgumentException("panels must contain 7 entries.");
            _listView = listView;
            _panels   = panels;
        }

        // ── 初始化 ────────────────────────────────────────────────────────

        public void Initialize()
        {
            InitListView();
            InitCards();
        }

        private void InitListView()
        {
            _listView.View          = View.Details;
            _listView.FullRowSelect = true;
            _listView.GridLines     = true;
            _listView.Columns.Clear();
            _listView.Items.Clear();

            _listView.Columns.Add("相機", -1, HorizontalAlignment.Center);
            _listView.Columns.Add("Pass", -1, HorizontalAlignment.Center);
            _listView.Columns.Add("Fail", -1, HorizontalAlignment.Center);
            _listView.Columns.Add("Total", -1, HorizontalAlignment.Center);
            _listView.Columns.Add("合格率", -1, HorizontalAlignment.Center);

            for (int i = 1; i <= 7; i++)
            {
                var item = new ListViewItem($"{i}");
                item.SubItems.Add("—");
                item.SubItems.Add("—");
                item.SubItems.Add("—");
                item.SubItems.Add("—");
                _listView.Items.Add(item);
            }

            // 依欄位標題文字長度按比例分配寬度，填滿控制項（不出現水平捲軸）
            FitColumnsProportional();
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

        // ── 更新 ─────────────────────────────────────────────────────────

        public void Update(Dictionary<int, CameraStats> stats)
        {
            for (int i = 1; i <= 7; i++)
            {
                CameraStats s = stats.TryGetValue(i, out var v) ? v : null;
                UpdateCard(i - 1, s);
                UpdateRow(i - 1, s);
            }
        }

        private void UpdateCard(int idx, CameraStats s)
        {
            if (s == null || s.Total == 0)
            {
                _panels[idx].BackColor  = ColorEmpty;
                _lblRate[idx].Text      = "—";
                _lblPassFail[idx].Text  = "0 / 0";
                return;
            }

            float rate = s.PassRate;
            _panels[idx].BackColor  = rate >= 0.95f ? ColorGood
                                    : rate >= 0.80f ? ColorWarning
                                                    : ColorBad;
            _lblRate[idx].Text      = $"{rate:P1}";
            _lblPassFail[idx].Text  = $"{s.Pass} / {s.Total}";
        }

        private void UpdateRow(int idx, CameraStats s)
        {
            var item = _listView.Items[idx];
            if (s == null || s.Total == 0)
            {
                item.SubItems[1].Text = "—";
                item.SubItems[2].Text = "—";
                item.SubItems[3].Text = "—";
                item.SubItems[4].Text = "—";
                item.BackColor = SystemColors.Window;
                return;
            }

            item.SubItems[1].Text = s.Pass.ToString();
            item.SubItems[2].Text = s.Fail.ToString();
            item.SubItems[3].Text = s.Total.ToString();
            item.SubItems[4].Text = $"{s.PassRate:P1}";
            item.BackColor = s.PassRate >= 0.95f ? Color.FromArgb(232, 245, 233)
                           : s.PassRate >= 0.80f ? Color.FromArgb(255, 243, 224)
                                                 : Color.FromArgb(255, 235, 238);
        }

        private void FitColumnsProportional()
        {
            if (_listView.Columns.Count == 0) return;
            int available = _listView.ClientSize.Width - SystemInformation.VerticalScrollBarWidth;
            if (available <= 0) return;

            using (var g = _listView.CreateGraphics())
            {
                var weights = new float[_listView.Columns.Count];
                float totalWeight = 0;
                for (int i = 0; i < _listView.Columns.Count; i++)
                {
                    float w = g.MeasureString(_listView.Columns[i].Text + "WW", _listView.Font).Width;
                    weights[i] = w;
                    totalWeight += w;
                }
                if (totalWeight <= 0) return;

                int assigned = 0;
                for (int i = 0; i < _listView.Columns.Count; i++)
                {
                    int colW = (i < _listView.Columns.Count - 1)
                        ? (int)(available * weights[i] / totalWeight)
                        : available - assigned;
                    _listView.Columns[i].Width = Math.Max(20, colW);
                    assigned += _listView.Columns[i].Width;
                }
            }
        }
    }
}
