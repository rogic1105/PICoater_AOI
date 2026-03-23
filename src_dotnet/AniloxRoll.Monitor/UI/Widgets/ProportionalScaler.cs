using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// Form 等比例縮放 Helper。
    /// Load 時記錄所有控制項的相對位置/大小，Resize 時按比例還原。
    /// </summary>
    public class ProportionalScaler
    {
        private struct ControlRecord
        {
            public float XRatio;   // Left   / ParentW
            public float YRatio;   // Top    / ParentH
            public float WRatio;   // Width  / ParentW
            public float HRatio;   // Height / ParentH
            public float FontRatio; // FontSize / FormH
        }

        private readonly Form _form;
        private readonly Dictionary<Control, ControlRecord> _records = new Dictionary<Control, ControlRecord>();
        private Size _baseSize;
        private bool _initialized;
        private bool _scaling;

        public ProportionalScaler(Form form)
        {
            _form = form ?? throw new ArgumentNullException(nameof(form));
        }

        /// <summary>
        /// 記錄所有控制項初始比例。在 Form.Load 或 InitializeComponent 之後呼叫。
        /// </summary>
        public void Initialize()
        {
            _baseSize = _form.ClientSize;
            if (_baseSize.Width <= 0 || _baseSize.Height <= 0) return;

            _records.Clear();
            RecordRecursive(_form);
            _initialized = true;

            _form.Resize += OnFormResize;

            // TabControl：切換頁籤時補記錄未掃描過的控制項
            HookTabControls(_form);
        }

        /// <summary>
        /// 動態新增控制項後呼叫，補記錄該控制項及其子控制項。
        /// </summary>
        public void RegisterControl(Control ctrl)
        {
            if (!_initialized || ctrl == null) return;
            RecordRecursive(ctrl);
        }

        private void RecordRecursive(Control parent)
        {
            foreach (Control c in parent.Controls)
            {
                if (_records.ContainsKey(c)) continue;

                Control p = c.Parent;
                if (p == null) continue;
                float pw = p.ClientSize.Width;
                float ph = p.ClientSize.Height;
                if (pw <= 0 || ph <= 0) continue;

                _records[c] = new ControlRecord
                {
                    XRatio    = c.Left   / pw,
                    YRatio    = c.Top    / ph,
                    WRatio    = c.Width  / pw,
                    HRatio    = c.Height / ph,
                    FontRatio = c.Font.Size / _baseSize.Height
                };

                // 記錄後立即移除 Anchor，由 Scaler 全權接管定位
                c.Anchor = AnchorStyles.Left | AnchorStyles.Top;

                RecordRecursive(c);
            }
        }

        private void OnFormResize(object sender, EventArgs e)
        {
            if (!_initialized || _scaling) return;
            if (_form.WindowState == FormWindowState.Minimized) return;

            Size newSize = _form.ClientSize;
            if (newSize.Width <= 0 || newSize.Height <= 0) return;

            _scaling = true;
            _form.SuspendLayout();

            try
            {
                ScaleRecursive(_form, newSize);
            }
            finally
            {
                _form.ResumeLayout(true);
                _scaling = false;
            }
        }

        private void ScaleRecursive(Control parent, Size formSize)
        {
            foreach (Control c in parent.Controls)
            {
                if (!_records.TryGetValue(c, out ControlRecord rec)) continue;

                Control p = c.Parent;
                if (p == null) continue;
                float pw = p.ClientSize.Width;
                float ph = p.ClientSize.Height;
                if (pw <= 0 || ph <= 0) continue;

                // Scaler 全權接管定位，永久移除 Anchor 避免 Layout 引擎覆蓋
                c.Anchor = AnchorStyles.Left | AnchorStyles.Top;

                int x = (int)(rec.XRatio * pw);
                int y = (int)(rec.YRatio * ph);
                int w = Math.Max(1, (int)(rec.WRatio * pw));
                int h = Math.Max(1, (int)(rec.HRatio * ph));
                c.SetBounds(x, y, w, h);

                // 等比例字體
                float newFontSize = rec.FontRatio * formSize.Height;
                if (newFontSize >= 4f && newFontSize <= 72f &&
                    Math.Abs(newFontSize - c.Font.Size) > 0.5f)
                {
                    try { c.Font = new Font(c.Font.FontFamily, newFontSize, c.Font.Style); }
                    catch { /* 字體大小不合法時忽略 */ }
                }

                ScaleRecursive(c, formSize);
            }
        }

        private void HookTabControls(Control parent)
        {
            foreach (Control c in parent.Controls)
            {
                if (c is TabControl tc)
                {
                    tc.SelectedIndexChanged += (s, e) =>
                    {
                        if (!_initialized) return;
                        var tab = tc.SelectedTab;
                        if (tab != null) RecordRecursive(tab);
                    };
                }
                HookTabControls(c);
            }
        }
    }
}
