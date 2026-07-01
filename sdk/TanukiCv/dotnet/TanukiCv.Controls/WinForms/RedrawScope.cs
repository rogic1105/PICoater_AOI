using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace TanukiCv.Controls.WinForms
{
    public sealed class RedrawScope : IDisposable
    {
        private const int WM_SETREDRAW = 0x000B;

        private readonly Control[] _controls;
        private bool _disposed;

        public RedrawScope(params Control[] controls)
        {
            _controls = controls ?? new Control[0];

            foreach (var control in _controls)
            {
                if (!CanUseHandle(control)) continue;
                SendMessage(control.Handle, WM_SETREDRAW, IntPtr.Zero, IntPtr.Zero);
                control.SuspendLayout();
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            for (int i = _controls.Length - 1; i >= 0; i--)
            {
                var control = _controls[i];
                if (!CanUseHandle(control)) continue;

                control.ResumeLayout(false);
                SendMessage(control.Handle, WM_SETREDRAW, new IntPtr(1), IntPtr.Zero);
                control.Invalidate(true);
            }
        }

        private static bool CanUseHandle(Control control)
            => control != null && !control.IsDisposed && control.IsHandleCreated;

        [DllImport("user32.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr SendMessage(IntPtr hWnd, int msg, IntPtr wParam, IntPtr lParam);
    }
}
