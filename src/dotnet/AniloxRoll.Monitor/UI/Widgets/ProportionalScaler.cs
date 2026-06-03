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

        /// <summary>全域字體縮放係數（DPI 感知後微調用；1=原樣，&lt;1=縮小）。
        /// DPI 感知下點數字體會隨 DPI 放大、但固定像素版面不變 → 用此係數把字體收小以配合版面。</summary>
        public float FontScale { get; set; } = 1f;

        /// <summary>立即依目前 Form 尺寸重套一次（套用 FontScale，不必等使用者 resize）。</summary>
        public void Reapply()
        {
            if (_initialized) OnFormResize(_form, EventArgs.Empty);
        }

        /// <summary>重新縮放所有 TabControl「目前作用中」的 tab。
        /// 初始最大化時，作用中 tab（如 tabPageLiveView）的子控制項因 WinForms TabControl lazy-layout
        /// 沒被 ScaleRecursive 套到（要切到別 tab 再切回才放大）→ 開窗後主動補一次。</summary>
        public void RescaleActiveTabs()
        {
            if (!_initialized) return;
            RescaleActiveTabsRecursive(_form);
        }

        private void RescaleActiveTabsRecursive(Control parent)
        {
            foreach (Control c in parent.Controls)
            {
                if (c is TabControl tc && tc.SelectedTab != null)
                {
                    RecordRecursive(tc.SelectedTab);
                    ScaleRecursive(tc.SelectedTab, _form.ClientSize);
                }
                RescaleActiveTabsRecursive(c);
            }
        }

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

        private void RecordRecursive(Control parent)
        {
            foreach (Control c in parent.Controls)
            {
                if (_records.ContainsKey(c)) continue;

                // Dock != None 或 TabPage：由 Layout 引擎/TabControl 全權管理，Scaler 不介入。
                // 容器型控制項（TabControl、Panel、TabPage 等）仍遞迴找出子控制項；
                // 複雜控制項（PropertyGrid 等）自行管理內部佈局，不可遞迴否則破壞 internal controls。
                if (c.Dock != DockStyle.None || c is TabPage)
                {
                    bool isLayoutContainer = (c is TabControl || c is Panel || c is GroupBox || c is SplitContainer || c is TabPage);
                    if (isLayoutContainer)
                        RecordRecursive(c);
                    continue;
                }

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

                // 在 Initialize 階段（視窗顯示前）就套用 FontScale，避免改到 Shown 才整批重建字體凍結 UI。
                // FontRatio 已用原始字級記錄，後續 resize 公式（×FontScale）與此一致、不會疊乘。
                if (Math.Abs(FontScale - 1f) > 0.001f)
                {
                    float fs = c.Font.Size * FontScale;
                    if (fs >= 4f && fs <= 72f)
                        try { c.Font = new Font(c.Font.FontFamily, fs, c.Font.Style); } catch { }
                }

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
                if (!_records.TryGetValue(c, out ControlRecord rec))
                {
                    // Dock/TabPage 控制項未記錄；只遞迴容器型控制項，
                    // 避免進入 PropertyGrid 等複雜控制項的 internal controls。
                    if (c is TabControl || c is Panel || c is GroupBox || c is SplitContainer || c is TabPage)
                        ScaleRecursive(c, formSize);
                    continue;
                }

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

                // 等比例字體（乘上全域 FontScale 微調）
                float newFontSize = rec.FontRatio * formSize.Height * FontScale;
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
                        if (tab == null) return;
                        // 補記錄（動態新增 control 用）+ 重新 scale：
                        // WinForms TabControl 對 inactive TabPage 有 lazy layout，maximize 時
                        // ScaleRecursive 寫入的 Bounds 可能在 TabPage 變 active 時被 layout 引擎
                        // reset 回 Anchor (Top|Left) 預設位置 → 切 tab 看不到放大。
                        // 切 tab 時主動重 scale 該 tab，依當前 Form 尺寸恢復正確比例。
                        RecordRecursive(tab);
                        ScaleRecursive(tab, _form.ClientSize);
                    };
                }
                HookTabControls(c);
            }
        }
    }
}
