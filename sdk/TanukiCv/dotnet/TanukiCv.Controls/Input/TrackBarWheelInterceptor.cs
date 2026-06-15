using System;
using System.Windows.Forms;

namespace TanukiCv.Controls
{
    /// <summary>
    /// 攔截原生 WM_MOUSEWHEEL：Windows TRACKBAR 每個滾輪 notch 會送出 3 個
    /// TB_LINEUP/TB_LINEDOWN（等同 3 × SmallChange），此攔截器改為每格僅移動 1。
    /// </summary>
    public sealed class TrackBarWheelInterceptor : NativeWindow
    {
        private const int WM_MOUSEWHEEL = 0x020A;
        private readonly TrackBar _bar;

        public TrackBarWheelInterceptor(TrackBar bar)
        {
            _bar = bar;
            AssignHandle(bar.Handle);
            bar.HandleCreated   += (s, e) => AssignHandle(_bar.Handle);
            bar.HandleDestroyed += (s, e) => ReleaseHandle();
        }

        protected override void WndProc(ref Message m)
        {
            if (m.Msg == WM_MOUSEWHEEL)
            {
                int delta = (short)(((long)m.WParam >> 16) & 0xFFFF);
                _bar.Value = Math.Max(_bar.Minimum, Math.Min(_bar.Maximum, _bar.Value + Math.Sign(delta)));
                return; // 跳過原生 3 格行為
            }
            base.WndProc(ref m);
        }
    }
}
