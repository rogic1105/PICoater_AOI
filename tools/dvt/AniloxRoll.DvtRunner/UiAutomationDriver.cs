using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Automation;

namespace AniloxRoll.DvtRunner
{
    internal sealed class UiAutomationDriver
    {
        private Process _process;
        private AutomationElement _root;

        public bool IsAttached =>
            _process != null && !_process.HasExited && _root != null;

        public async Task AttachOrLaunchAsync(
            string exePath,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            if (IsAttached) return;
            if (!File.Exists(exePath))
                throw new FileNotFoundException("Monitor executable not found.", exePath);

            string processName = Path.GetFileNameWithoutExtension(exePath);
            _process = Process.GetProcessesByName(processName)
                .FirstOrDefault(p => p.MainWindowHandle != IntPtr.Zero);
            if (_process == null)
            {
                _process = Process.Start(new ProcessStartInfo
                {
                    FileName = exePath,
                    WorkingDirectory = Path.GetDirectoryName(exePath),
                    UseShellExecute = true
                });
            }

            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (_process.HasExited)
                    throw new InvalidOperationException(
                        "AniloxRoll.Monitor exited before its main window became available.");

                _process.Refresh();
                if (_process.MainWindowHandle != IntPtr.Zero)
                {
                    _root = AutomationElement.FromHandle(_process.MainWindowHandle);
                    if (_root != null &&
                        await WaitForStableWindowAsync(deadline, cancellationToken))
                        return;
                }
                await Task.Delay(200, cancellationToken);
            }
            throw new TimeoutException("Timed out waiting for the monitor main window.");
        }

        public async Task<string> WaitForElementAsync(
            string name,
            string expectedValue,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            DateTime? deadline = timeoutSeconds > 0
                ? DateTime.UtcNow.AddSeconds(timeoutSeconds)
                : (DateTime?)null;
            while (!deadline.HasValue || DateTime.UtcNow < deadline.Value)
            {
                cancellationToken.ThrowIfCancellationRequested();
                AutomationElement element = FindUnique(name, throwIfMissing: false);
                if (element != null && element.Current.IsEnabled)
                {
                    string value = TryReadValue(element);
                    if (string.IsNullOrEmpty(expectedValue) ||
                        string.Equals(value, expectedValue, StringComparison.Ordinal))
                        return string.IsNullOrEmpty(value)
                            ? name + " is enabled"
                            : name + "=" + value;
                }
                await Task.Delay(150, cancellationToken);
            }
            throw new TimeoutException(
                "Timed out waiting for enabled UI element: " + name +
                (string.IsNullOrEmpty(expectedValue) ? "" : "=" + expectedValue));
        }

        public async Task CloseAppAsync(
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            if (!IsAttached) return;
            if (!_process.CloseMainWindow())
                throw new InvalidOperationException(
                    "AniloxRoll.Monitor did not accept the close request.");

            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (_process.HasExited)
                {
                    _root = null;
                    return;
                }
                await Task.Delay(150, cancellationToken);
            }
            throw new TimeoutException(
                "Timed out waiting for AniloxRoll.Monitor to close.");
        }

        public string GetPropertyValue(string displayName)
        {
            AutomationElement element = FindUniqueDataItem(displayName);
            return ReadRequiredValue(element);
        }

        public async Task SetPropertyValueAsync(
            string displayName,
            string value,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            if (string.Equals(
                GetPropertyValue(displayName), value, StringComparison.Ordinal))
                return;

            AutomationElement element = await BringDataItemIntoViewAsync(
                displayName, timeoutSeconds, cancellationToken);
            if (!element.Current.IsEnabled)
                throw new InvalidOperationException("Property is disabled: " + displayName);

            NativeMethods.SetForegroundWindow(_process.MainWindowHandle);
            System.Windows.Rect bounds = element.Current.BoundingRectangle;
            NativeMethods.ClickScreenPoint(
                (int)Math.Round(bounds.Left + bounds.Width * 0.75),
                (int)Math.Round(bounds.Top + bounds.Height / 2.0));
            await Task.Delay(150, cancellationToken);

            AutomationElement editor = AutomationElement.FocusedElement;
            object valuePattern;
            if (editor == null ||
                !editor.TryGetCurrentPattern(ValuePattern.Pattern, out valuePattern))
                throw new InvalidOperationException(
                    "Property editor does not expose ValuePattern: " + displayName);

            ((ValuePattern)valuePattern).SetValue(value);
            IntPtr editorHandle = new IntPtr(editor.Current.NativeWindowHandle);

            // PropertyGrid accessibility SetValue only updates the edit box. Enter is the
            // actual commit boundary that raises PropertyValueChanged and SettingsHub routing.
            if (editorHandle != IntPtr.Zero)
            {
                NativeMethods.SendMessage(
                    editorHandle, NativeMethods.WmKeyDown,
                    new IntPtr(NativeMethods.VkReturn), IntPtr.Zero);
                NativeMethods.SendMessage(
                    editorHandle, NativeMethods.WmKeyUp,
                    new IntPtr(NativeMethods.VkReturn), IntPtr.Zero);
            }
            else
            {
                NativeMethods.PressKey((byte)NativeMethods.VkReturn);
            }

            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                string current = GetPropertyValue(displayName);
                if (string.Equals(current, value, StringComparison.Ordinal))
                    return;
                await Task.Delay(150, cancellationToken);
            }
            throw new TimeoutException(
                $"Property value did not settle: {displayName} expected={value}");
        }

        public async Task ClickAsync(
            string name,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            AutomationElement element = null;
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                element = FindUnique(name, throwIfMissing: false);
                if (element != null && element.Current.IsEnabled) break;
                await Task.Delay(150, cancellationToken);
            }
            if (element == null || !element.Current.IsEnabled)
                throw new TimeoutException("Timed out waiting to click: " + name);

            IntPtr handle = new IntPtr(element.Current.NativeWindowHandle);
            NativeMethods.SetForegroundWindow(_process.MainWindowHandle);
            if (handle != IntPtr.Zero)
            {
                NativeMethods.SendMessage(
                    handle, NativeMethods.BmClick, IntPtr.Zero, IntPtr.Zero);
                return;
            }

            System.Windows.Rect bounds = element.Current.BoundingRectangle;
            if (bounds.IsEmpty)
                throw new InvalidOperationException(
                    "UI element has no native handle or clickable bounds: " + name);
            NativeMethods.ClickScreenPoint(
                (int)Math.Round(bounds.Left + bounds.Width / 2.0),
                (int)Math.Round(bounds.Top + bounds.Height / 2.0));
        }

        public async Task TryStopCaptureAsync(CancellationToken cancellationToken)
        {
            if (!IsAttached) return;
            var buttonNames = new OrCondition(
                new PropertyCondition(
                    AutomationElement.NameProperty, "開始抓取"),
                new PropertyCondition(
                    AutomationElement.NameProperty, "停止抓取"));
            AutomationElement grabButton = _root.FindFirst(
                TreeScope.Descendants, buttonNames);
            if (grabButton == null ||
                !grabButton.Current.IsEnabled ||
                !string.Equals(
                    grabButton.Current.Name,
                    "停止抓取",
                    StringComparison.Ordinal))
                return;

            IntPtr handle = new IntPtr(grabButton.Current.NativeWindowHandle);
            NativeMethods.SetForegroundWindow(_process.MainWindowHandle);
            if (handle != IntPtr.Zero)
            {
                NativeMethods.SendMessage(
                    handle, NativeMethods.BmClick, IntPtr.Zero, IntPtr.Zero);
            }
            else
            {
                System.Windows.Rect bounds =
                    grabButton.Current.BoundingRectangle;
                NativeMethods.ClickScreenPoint(
                    (int)Math.Round(bounds.Left + bounds.Width / 2.0),
                    (int)Math.Round(bounds.Top + bounds.Height / 2.0));
            }
            await Task.Delay(150, cancellationToken);
        }

        private AutomationElement FindUniqueDataItem(string name)
        {
            EnsureAttached();
            var condition = new AndCondition(
                new PropertyCondition(AutomationElement.NameProperty, name),
                new PropertyCondition(
                    AutomationElement.ControlTypeProperty,
                    ControlType.DataItem));
            AutomationElement element = _root.FindFirst(
                TreeScope.Descendants, condition);
            if (element == null)
                throw new InvalidOperationException(
                    "PropertyGrid item not found: " + name);
            return element;
        }

        private async Task<AutomationElement> BringDataItemIntoViewAsync(
            string name,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                AutomationElement item = FindUniqueDataItem(name);
                AutomationElement category =
                    TreeWalker.RawViewWalker.GetParent(item);
                AutomationElement table = category == null
                    ? null
                    : TreeWalker.RawViewWalker.GetParent(category);
                if (table == null || table.Current.ControlType != ControlType.Table)
                    throw new InvalidOperationException(
                        "Cannot locate PropertyGrid table for " + name);

                System.Windows.Rect itemBounds = item.Current.BoundingRectangle;
                System.Windows.Rect tableBounds = table.Current.BoundingRectangle;
                bool visible =
                    itemBounds.Top >= tableBounds.Top &&
                    itemBounds.Bottom <= tableBounds.Bottom &&
                    itemBounds.Height > 0;
                if (visible) return item;

                int delta = itemBounds.Top > tableBounds.Bottom ? -1200 : 1200;
                NativeMethods.SetForegroundWindow(_process.MainWindowHandle);
                NativeMethods.WheelAt(
                    (int)Math.Round(tableBounds.Left + tableBounds.Width / 2.0),
                    (int)Math.Round(tableBounds.Top + tableBounds.Height / 2.0),
                    delta);
                await Task.Delay(150, cancellationToken);
            }
            throw new TimeoutException(
                "Timed out scrolling PropertyGrid item into view: " + name);
        }

        private AutomationElement FindUnique(string name, bool throwIfMissing)
        {
            EnsureAttached();
            var condition = new PropertyCondition(
                AutomationElement.NameProperty, name);
            AutomationElement match = _root.FindFirst(
                TreeScope.Descendants, condition);
            if (match == null)
            {
                if (throwIfMissing)
                    throw new InvalidOperationException("UI element not found: " + name);
                return null;
            }
            return match;
        }

        private async Task<bool> WaitForStableWindowAsync(
            DateTime deadline,
            CancellationToken cancellationToken)
        {
            int consecutiveResponsiveChecks = 0;
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (_process.HasExited) return false;

                if (NativeMethods.IsWindowResponsive(
                    _process.MainWindowHandle, 500))
                {
                    consecutiveResponsiveChecks++;
                    if (consecutiveResponsiveChecks >= 5)
                        return true;
                }
                else
                {
                    consecutiveResponsiveChecks = 0;
                }

                await Task.Delay(250, cancellationToken);
            }
            return false;
        }

        private static string ReadRequiredValue(AutomationElement element)
        {
            string value = TryReadValue(element);
            if (value == null)
                throw new InvalidOperationException(
                    "UI element does not expose ValuePattern: " + element.Current.Name);
            return value;
        }

        private static string TryReadValue(AutomationElement element)
        {
            object pattern;
            return element.TryGetCurrentPattern(ValuePattern.Pattern, out pattern)
                ? ((ValuePattern)pattern).Current.Value
                : null;
        }

        private void EnsureAttached()
        {
            if (!IsAttached)
                throw new InvalidOperationException("The runner is not attached to AniloxRoll.Monitor.");
        }
    }
}
