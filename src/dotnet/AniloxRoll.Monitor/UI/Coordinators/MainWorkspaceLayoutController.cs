using System;
using System.Windows.Forms;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    /// <summary>
    /// Owns the main workspace/right-panel split and the temporary full-width override.
    /// </summary>
    internal sealed class MainWorkspaceLayoutController : IDisposable
    {
        private readonly Form _form;
        private readonly Control _toggleTarget;
        private readonly Func<MouseEventArgs, bool> _toggleHitTest;
        private readonly Control _workspace;
        private readonly Control _rightPanel;
        private readonly Action _rescaleActiveTabs;
        private readonly Action<bool> _persistFullWidth;
        private readonly Action<string> _flowLog;
        private readonly int _normalGap;
        private readonly int _normalRightMargin;
        private int _clickCount;
        private int _lastClickTick;
        private bool _fullWidth;
        private bool _disposed;

        private const int ClickSequenceGapMs = 1200;
        private const int RightPanelWidthDivisor = 5;

        public MainWorkspaceLayoutController(
            Form form,
            Control toggleTarget,
            Func<MouseEventArgs, bool> toggleHitTest,
            Control workspace,
            Control rightPanel,
            bool initialFullWidth,
            Action rescaleActiveTabs,
            Action<bool> persistFullWidth,
            Action<string> flowLog)
        {
            _form = form ?? throw new ArgumentNullException(nameof(form));
            _toggleTarget = toggleTarget ?? throw new ArgumentNullException(nameof(toggleTarget));
            _toggleHitTest = toggleHitTest;
            _workspace = workspace ?? throw new ArgumentNullException(nameof(workspace));
            _rightPanel = rightPanel ?? throw new ArgumentNullException(nameof(rightPanel));
            _rescaleActiveTabs = rescaleActiveTabs;
            _persistFullWidth = persistFullWidth;
            _flowLog = flowLog;
            _fullWidth = initialFullWidth;
            _normalGap = Math.Max(0, _rightPanel.Left - _workspace.Right);
            _normalRightMargin = Math.Max(0, _form.ClientSize.Width - _rightPanel.Right);

            // MouseDown is raised for every physical press, including clicks that WinForms folds
            // into DoubleClick. MouseClick can therefore under-count a fast five-click gesture.
            _toggleTarget.MouseDown += ToggleTarget_MouseDown;
            _form.Resize += Form_Resize;
        }

        private void ToggleTarget_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left ||
                (_toggleHitTest != null && !_toggleHitTest(e)))
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
            LogLayout("ui:monitor tab five-click");
        }

        public void ApplyPersistedLayout()
        {
            ApplyLayout();
            LogLayout("workspace restore");
        }

        private void ResetClickSequence()
        {
            _clickCount = 0;
            _lastClickTick = 0;
        }

        private void Form_Resize(object sender, EventArgs e)
        {
            ApplyLayout();
        }

        private void ApplyLayout()
        {
            if (_disposed || _form.WindowState == FormWindowState.Minimized) return;

            _form.SuspendLayout();
            try
            {
                int right = Math.Max(
                    _workspace.Left + 1,
                    _form.ClientSize.Width - _normalRightMargin);

                if (_fullWidth)
                {
                    _rightPanel.Visible = false;
                    _workspace.Width = Math.Max(1, right - _workspace.Left);
                }
                else
                {
                    int availableWidth = Math.Max(1,
                        right - _workspace.Left - _normalGap);
                    int rightPanelWidth = Math.Max(1,
                        availableWidth / RightPanelWidthDivisor);
                    int workspaceWidth = Math.Max(1,
                        availableWidth - rightPanelWidth);

                    _rightPanel.SetBounds(
                        _workspace.Left + workspaceWidth + _normalGap,
                        _rightPanel.Top,
                        rightPanelWidth,
                        _rightPanel.Height);
                    _rightPanel.Visible = true;
                    _workspace.Width = workspaceWidth;
                }

                _rescaleActiveTabs?.Invoke();
                _workspace.Invalidate(true);
            }
            finally
            {
                _form.ResumeLayout(true);
            }
        }

        private void LogLayout(string intent)
        {
            _flowLog?.Invoke(
                $"{intent} rightPanel={(_fullWidth ? "hidden" : "visible")} " +
                $"workspaceW={_workspace.Width} rightPanelW={_rightPanel.Width}");
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
