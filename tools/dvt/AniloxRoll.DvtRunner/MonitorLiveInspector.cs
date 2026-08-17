using System;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Windows.Automation;
using System.Windows.Forms;

namespace AniloxRoll.DvtRunner
{
    internal sealed class MonitorLiveSelection
    {
        public string ReferenceKey { get; set; }
        public string DisplayName { get; set; }
        public Rectangle ScreenBounds { get; set; }
    }

    internal sealed class MonitorElementPicker : IDisposable
    {
        private const int WhMouseLl = 14;
        private const int WmLButtonDown = 0x0201;
        private const int WmLButtonUp = 0x0202;

        private readonly Control _dispatcher;
        private readonly Timer _hoverTimer = new Timer();
        private readonly Func<Point, MonitorLiveSelection> _inspect;
        private readonly HighlightOverlay _highlight = new HighlightOverlay();
        private readonly LowLevelMouseProc _mouseProc;
        private IntPtr _mouseHook;
        private MonitorLiveSelection _hovered;
        private MonitorLiveSelection _pendingSelection;
        private bool _capturingLeftButton;
        private bool _disposed;

        public event Action<MonitorLiveSelection> SelectionCompleted;

        public event Action Canceled;

        public bool IsActive => _mouseHook != IntPtr.Zero;

        public MonitorElementPicker(
            Control dispatcher,
            Func<Point, MonitorLiveSelection> inspect)
        {
            _dispatcher = dispatcher ??
                throw new ArgumentNullException(nameof(dispatcher));
            _inspect = inspect ?? throw new ArgumentNullException(nameof(inspect));
            _mouseProc = OnLowLevelMouse;
            _hoverTimer.Interval = 80;
            _hoverTimer.Tick += OnHoverTimer;
        }

        public void Start()
        {
            ThrowIfDisposed();
            if (IsActive) return;

            using (Process process = Process.GetCurrentProcess())
            using (ProcessModule module = process.MainModule)
            {
                _mouseHook = SetWindowsHookEx(
                    WhMouseLl,
                    _mouseProc,
                    GetModuleHandle(module.ModuleName),
                    0);
            }
            if (_mouseHook == IntPtr.Zero)
                throw new InvalidOperationException(
                    "無法啟動真實元件選取器。Win32=" +
                    Marshal.GetLastWin32Error());

            _hoverTimer.Start();
        }

        public void Stop()
        {
            _hoverTimer.Stop();
            _hovered = null;
            _pendingSelection = null;
            _capturingLeftButton = false;
            _highlight.HideHighlight();
            if (_mouseHook == IntPtr.Zero) return;
            UnhookWindowsHookEx(_mouseHook);
            _mouseHook = IntPtr.Zero;
        }

        private void OnHoverTimer(object sender, EventArgs e)
        {
            Point cursor = Cursor.Position;
            MonitorLiveSelection selection;
            try
            {
                selection = _inspect(cursor);
            }
            catch (ElementNotAvailableException)
            {
                selection = null;
            }
            catch (InvalidOperationException)
            {
                selection = null;
            }
            catch (COMException)
            {
                selection = null;
            }

            _hovered = selection;
            if (selection == null)
            {
                _highlight.HideHighlight();
                return;
            }
            _highlight.ShowHighlight(selection.ScreenBounds);
        }

        private IntPtr OnLowLevelMouse(
            int code,
            IntPtr message,
            IntPtr data)
        {
            if (code >= 0 && IsActive)
            {
                int kind = message.ToInt32();
                if (kind == WmLButtonDown && _capturingLeftButton)
                    return new IntPtr(1);
                if (kind == WmLButtonDown &&
                    _hovered != null &&
                    _hovered.ScreenBounds.Contains(Cursor.Position))
                {
                    _capturingLeftButton = true;
                    _pendingSelection = _hovered;
                    _hoverTimer.Stop();
                    _highlight.HideHighlight();
                    return new IntPtr(1);
                }
                if (kind == WmLButtonUp && _capturingLeftButton)
                {
                    _capturingLeftButton = false;
                    MonitorLiveSelection selected = _pendingSelection;
                    _dispatcher.BeginInvoke(new Action(() =>
                    {
                        Stop();
                        SelectionCompleted?.Invoke(selected);
                    }));
                    return new IntPtr(1);
                }
            }
            return CallNextHookEx(_mouseHook, code, message, data);
        }

        public void Cancel()
        {
            if (!IsActive) return;
            Stop();
            Canceled?.Invoke();
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            Stop();
            _hoverTimer.Dispose();
            _highlight.Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(GetType().Name);
        }

        private delegate IntPtr LowLevelMouseProc(
            int code,
            IntPtr message,
            IntPtr data);

        [DllImport("user32.dll", SetLastError = true)]
        private static extern IntPtr SetWindowsHookEx(
            int hookId,
            LowLevelMouseProc callback,
            IntPtr module,
            uint threadId);

        [DllImport("user32.dll", SetLastError = true)]
        private static extern bool UnhookWindowsHookEx(IntPtr hook);

        [DllImport("user32.dll")]
        private static extern IntPtr CallNextHookEx(
            IntPtr hook,
            int code,
            IntPtr message,
            IntPtr data);

        [DllImport("kernel32.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr GetModuleHandle(string moduleName);
    }

    internal sealed class HighlightOverlay : Form
    {
        private const int WsExTransparent = 0x00000020;
        private const int WsExToolWindow = 0x00000080;
        private const int WsExNoActivate = 0x08000000;

        public HighlightOverlay()
        {
            FormBorderStyle = FormBorderStyle.None;
            ShowInTaskbar = false;
            StartPosition = FormStartPosition.Manual;
            TopMost = true;
            BackColor = Color.Fuchsia;
            TransparencyKey = Color.Fuchsia;
        }

        protected override bool ShowWithoutActivation => true;

        protected override CreateParams CreateParams
        {
            get
            {
                CreateParams value = base.CreateParams;
                value.ExStyle |=
                    WsExTransparent | WsExToolWindow | WsExNoActivate;
                return value;
            }
        }

        public void ShowHighlight(Rectangle screenBounds)
        {
            if (screenBounds.Width <= 1 || screenBounds.Height <= 1)
            {
                HideHighlight();
                return;
            }
            Rectangle target = screenBounds;
            target.Inflate(3, 3);
            Bounds = target;
            if (!Visible) Show();
            Invalidate();
        }

        public void HideHighlight()
        {
            if (Visible) Hide();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            using (var pen = new Pen(Color.FromArgb(255, 140, 0), 4F))
            {
                e.Graphics.DrawRectangle(
                    pen,
                    2,
                    2,
                    Math.Max(1, ClientSize.Width - 5),
                    Math.Max(1, ClientSize.Height - 5));
            }
        }
    }

    internal static class MonitorUiReference
    {
        private const string ControlPrefix = "control:";
        private const string PropertyPrefix = "property:";

        public static string ForControl(string controlId) =>
            ControlPrefix + controlId;

        public static string ForProperty(string propertyName) =>
            PropertyPrefix + propertyName;

        public static bool TryGetControl(string referenceKey, out string controlId)
        {
            return TryStripPrefix(referenceKey, ControlPrefix, out controlId);
        }

        public static bool TryGetProperty(string referenceKey, out string propertyName)
        {
            return TryStripPrefix(referenceKey, PropertyPrefix, out propertyName);
        }

        private static bool TryStripPrefix(
            string referenceKey,
            string prefix,
            out string value)
        {
            value = null;
            if (string.IsNullOrWhiteSpace(referenceKey) ||
                !referenceKey.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                return false;
            value = referenceKey.Substring(prefix.Length);
            return !string.IsNullOrWhiteSpace(value);
        }
    }
}
