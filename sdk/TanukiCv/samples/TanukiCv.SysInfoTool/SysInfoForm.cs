using System;
using System.Drawing;
using System.Windows.Forms;
using TanukiCv.Core; // SystemInfo

namespace TanukiCv.SysInfoTool
{
    /// <summary>
    /// 系統資訊工具（code-built，無 Designer）：列出 TanukiCv.Core.SystemInfo 的通用硬體 + 螢幕。
    /// 展示「系統硬體資訊的資料來源已收進 sdk 唯一來源」——主程式、MIL 範例三擊計算同源。
    /// </summary>
    public sealed class SysInfoForm : Form
    {
        private readonly ListView _list;

        public SysInfoForm()
        {
            Text = "TanukiCv 系統資訊（SystemInfo）";
            ClientSize = new Size(420, 460);
            StartPosition = FormStartPosition.CenterScreen;
            Font = new Font("Segoe UI", 9f);

            _list = new ListView
            {
                Dock = DockStyle.Fill,
                View = View.Details,
                FullRowSelect = true,
                GridLines = true,
                HeaderStyle = ColumnHeaderStyle.Nonclickable
            };
            _list.Columns.Add("參數", 150);
            _list.Columns.Add("值", 250);

            var btnRefresh = new Button { Text = "重新整理", Dock = DockStyle.Bottom, Height = 32 };
            btnRefresh.Click += (s, e) => Reload();

            Controls.Add(_list);
            Controls.Add(btnRefresh);

            Reload();
        }

        private void Reload()
        {
            _list.BeginUpdate();
            try
            {
                _list.Items.Clear();
                foreach (var kv in SystemInfo.GetGenericHardwareRows()) // CPU / RAM / GPU
                    _list.Items.Add(new ListViewItem(new[] { kv.Key, kv.Value }));
                foreach (var kv in SystemInfo.GetScreenRows())          // 螢幕（含 mm/px）
                    _list.Items.Add(new ListViewItem(new[] { kv.Key, kv.Value }));

                var s = SystemInfo.GetScreenMetrics();
                _list.Items.Add(new ListViewItem(new[] { "—", "── 三擊 1:1 用 ──" }));
                _list.Items.Add(new ListViewItem(new[] { "screen mm/px", s.MmPerPx.ToString("F4") }));
            }
            finally { _list.EndUpdate(); }
        }
    }
}
