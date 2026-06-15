using System;
using System.Windows.Forms;

namespace TanukiCv.Controls
{
    /// <summary>
    /// 反轉 ComboBox 滾輪方向：上滾 (delta &gt; 0) → SelectedIndex 增加（數值變大）。
    /// 預設 ComboBox 行為是上滾減少 index，此攔截器直接處理後 return，略過原生訊息。
    /// </summary>
    public sealed class ComboBoxWheelReverser : NativeWindow
    {
        private const int WM_MOUSEWHEEL = 0x020A;
        private readonly ComboBox _cb;

        public ComboBoxWheelReverser(ComboBox cb)
        {
            _cb = cb;
            AssignHandle(cb.Handle);
            cb.HandleCreated   += (s, e) => AssignHandle(_cb.Handle);
            cb.HandleDestroyed += (s, e) => ReleaseHandle();
        }

        protected override void WndProc(ref Message m)
        {
            if (m.Msg == WM_MOUSEWHEEL)
            {
                int delta  = (short)(((long)m.WParam >> 16) & 0xFFFF);
                int newIdx = _cb.SelectedIndex + Math.Sign(delta); // 正 delta = 上滾 = index++
                if (newIdx >= 0 && newIdx < _cb.Items.Count)
                    _cb.SelectedIndex = newIdx;
                return; // 跳過原生行為（原生是上滾 index--）
            }
            base.WndProc(ref m);
        }
    }
}
