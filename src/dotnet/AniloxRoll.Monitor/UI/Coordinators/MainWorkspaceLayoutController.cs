using System;
using System.Windows.Forms;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    /// <summary>
    /// Owns the temporary full-width workspace layout. The Designer remains the source of the
    /// normal layout; this controller only applies and removes the right-panel override.
    /// </summary>
    internal sealed class MainWorkspaceLayoutController : IDisposable
    {
        private readonly Form _form;
        private readonly Control _toggleTarget;
        private readonly Control _workspace;
        private readonly Control _rightPanel;
        private readonly Action _rescaleActiveTabs;
        private readonly Action<bool> _persistFullWidth;
        private readonly Action<string> _flowLog;
        private readonly int _normalGap;
        private int _clickCount;
        private int _lastClickTick;
        private bool _fullWidth;
        private bool _disposed;

        private const int ClickSequenceGapMs = 1200;

        public MainWorkspaceLayoutController(
            Form form,
            Control toggleTarget,
            Control workspace,
            Control rightPanel,
            bool initialFullWidth,
            Action rescaleActiveTabs,
            Action<bool> persistFullWidth,
            Action<string> flowLog)
        {
            _form = form ?? throw new ArgumentNullException(nameof(form));
            _toggleTarget = toggleTarget ?? throw new ArgumentNullException(nameof(toggleTarget));
            _workspace = workspace ?? throw new ArgumentNullException(nameof(workspace));
            _rightPanel = rightPanel ?? throw new ArgumentNullException(nameof(rightPanel));
            _rescaleActiveTabs = rescaleActiveTabs;
            _persistFullWidth = persistFullWidth;
            _flowLog = flowLog;
            _fullWidth = initialFullWidth;
            _normalGap = Math.Max(0, _rightPanel.Left - _workspace.Right);

            // MouseDown is raised for every physical press, including clicks that WinForms folds
            // into DoubleClick. MouseClick can therefore under-count a fast five-click gesture.
            _toggleTarget.MouseDown += ToggleTarget_MouseDown;
            _form.Resize += Form_Resize;
        }

        private void ToggleTarget_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left)
            {
                ResetClickSequence();
                return;
            }

            int now = Environment.TickCount;
            int elapsed = unchecked(now - _lastClickTick);
            _clickCount = _clickCount == 0 || elapsed < 0 || elapsed > ClickSequenceGapMs
                ? 1
                : _clickCount + 1;
            _lastClickTick = now;
            if (_clickCount < 5) return;

            ResetClickSequence();
            _fullWidth = !_fullWidth;
            ApplyLayout();
            _persistFullWidth?.Invoke(_fullWidth);
            _flowLog?.Invoke($"ui:IO state five-click rightPanel={(_fullWidth ? "hidden" : "visible")}");
        }

        public void ApplyPersistedLayout()
        {
            ApplyLayout();
            _flowLog?.Invoke($"workspace restore rightPanel={(_fullWidth ? "hidden" : "visible")}");
        }

        private void ResetClickSequence()
        {
            _clickCount = 0;
            _lastClickTick = 0;
        }

        private void Form_Resize(object sender, EventArgs e)
        {
            if (_fullWidth) ApplyLayout();
        }

        private void ApplyLayout()
        {
            if (_disposed || _form.WindowState == FormWindowState.Minimized) return;

            _form.SuspendLayout();
            try
            {
                if (_fullWidth)
                {
                    int right = _rightPanel.Right;
                    _rightPanel.Visible = false;
                    _workspace.Width = Math.Max(1, right - _workspace.Left);
                }
                else
                {
                    _rightPanel.Visible = true;
                    _workspace.Width = Math.Max(1,
                        _rightPanel.Left - _normalGap - _workspace.Left);
                }

                _rescaleActiveTabs?.Invoke();
                _workspace.Invalidate(true);
            }
            finally
            {
                _form.ResumeLayout(true);
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _toggleTarget.MouseDown -= ToggleTarget_MouseDown;
            _form.Resize -= Form_Resize;
        }
    }
}
