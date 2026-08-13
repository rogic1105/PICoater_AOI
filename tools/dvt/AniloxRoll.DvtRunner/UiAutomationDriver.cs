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
        private AutomationElement _propertyGridTable;
        private Dictionary<string, List<AutomationElement>> _propertyGridItems;
        private List<AutomationElement> _propertyGridItemOrder;
        private string _selectedPropertyGridKey;

        public bool IsAttached =>
            _process != null && !_process.HasExited && _root != null;

        public int ProcessId =>
            _process != null && !_process.HasExited ? _process.Id : 0;

        public void RefreshPropertyGridIndex()
        {
            EnsureAttached();
            _propertyGridItems = null;
            _propertyGridItemOrder = null;
            _selectedPropertyGridKey = null;
        }

        public void RefreshRoot()
        {
            EnsureAttached();
            _process.Refresh();
            if (_process.MainWindowHandle == IntPtr.Zero)
                throw new InvalidOperationException(
                    "Monitor main window is unavailable while refreshing UI Automation.");
            _root = AutomationElement.FromHandle(
                _process.MainWindowHandle);
            _propertyGridTable = null;
            _propertyGridItems = null;
            _propertyGridItemOrder = null;
            _selectedPropertyGridKey = null;
        }

        public async Task AttachOrLaunchAsync(
            string exePath,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            if (IsAttached) return;
            if (_process != null)
            {
                _process.Dispose();
                _process = null;
                _root = null;
                _propertyGridTable = null;
                _propertyGridItems = null;
                _propertyGridItemOrder = null;
                _selectedPropertyGridKey = null;
            }
            if (!File.Exists(exePath))
                throw new FileNotFoundException("Monitor executable not found.", exePath);

            string processName = Path.GetFileNameWithoutExtension(exePath);
            string expectedPath = Path.GetFullPath(exePath);
            foreach (Process candidate in
                Process.GetProcessesByName(processName))
            {
                bool selected = false;
                try
                {
                    candidate.Refresh();
                    string candidatePath =
                        candidate.MainModule?.FileName;
                    selected =
                        candidate.MainWindowHandle != IntPtr.Zero &&
                        !string.IsNullOrWhiteSpace(candidatePath) &&
                        string.Equals(
                            Path.GetFullPath(candidatePath),
                            expectedPath,
                            StringComparison.OrdinalIgnoreCase);
                    if (selected)
                    {
                        _process = candidate;
                        break;
                    }
                }
                catch
                {
                    // Protected or exiting same-name process: it cannot be
                    // proven to be this runner's executable, so ignore it.
                }
                finally
                {
                    if (!selected)
                        candidate.Dispose();
                }
            }
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
                    // Every scenario action has its own enabled/value/log wait. Requiring five
                    // consecutive WM_NULL replies here made the runner and the product's DVT
                    // stall sampler amplify each other before the first action could run.
                    if (NativeMethods.IsWindowResponsive(
                        _process.MainWindowHandle, 2000))
                    {
                        _root = AutomationElement.FromHandle(_process.MainWindowHandle);
                        _propertyGridTable = null;
                        _propertyGridItems = null;
                        _propertyGridItemOrder = null;
                        _selectedPropertyGridKey = null;
                    }
                    if (_root != null)
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

        public async Task ObserveElementsAsync(
            string targetList,
            int durationSeconds,
            Action<string> progress,
            CancellationToken cancellationToken)
        {
            string[] names = (targetList ?? string.Empty)
                .Split(new[] { '|' }, StringSplitOptions.RemoveEmptyEntries);
            if (names.Length == 0)
                throw new InvalidOperationException(
                    "Soak step requires pipe-delimited UI targets.");

            var observed = new List<KeyValuePair<string, IntPtr>>();
            foreach (string rawName in names)
            {
                string name = rawName.Trim();
                AutomationElement element =
                    FindUnique(name, throwIfMissing: false);
                if (element == null || !element.Current.IsEnabled)
                    throw new InvalidOperationException(
                        "Required soak state was not present at start: " + name);
                IntPtr handle =
                    new IntPtr(element.Current.NativeWindowHandle);
                if (handle == IntPtr.Zero)
                    throw new InvalidOperationException(
                        "Required soak state has no native window handle: " + name);
                observed.Add(
                    new KeyValuePair<string, IntPtr>(name, handle));
            }

            DateTime started = DateTime.UtcNow;
            DateTime deadline = started.AddSeconds(durationSeconds);
            DateTime nextProgress = started;
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (_process == null || _process.HasExited)
                    throw new InvalidOperationException(
                        "AniloxRoll.Monitor exited during soak.");

                _process.Refresh();
                if (!_process.Responding)
                    throw new InvalidOperationException(
                        "AniloxRoll.Monitor stopped responding during soak.");

                foreach (KeyValuePair<string, IntPtr> pair in observed)
                {
                    bool enabled =
                        NativeMethods.IsWindow(pair.Value) &&
                        NativeMethods.IsWindowEnabled(pair.Value);
                    if (!enabled)
                        throw new InvalidOperationException(
                            "Required soak state was lost: " + pair.Key);

                    string current =
                        NativeMethods.ReadWindowText(pair.Value, 500);
                    if (!string.Equals(
                        current,
                        pair.Key,
                        StringComparison.Ordinal))
                    {
                        throw new InvalidOperationException(
                            "Required soak state changed: expected=" +
                            pair.Key + " actual=" + current);
                    }
                }

                if (DateTime.UtcNow >= nextProgress)
                {
                    int elapsedSeconds = (int)Math.Round(
                        (DateTime.UtcNow - started).TotalSeconds);
                    progress?.Invoke(
                        $"[Soak] elapsed={elapsedSeconds}s states=healthy");
                    nextProgress = DateTime.UtcNow.AddMinutes(1);
                }

                TimeSpan remaining = deadline - DateTime.UtcNow;
                if (remaining <= TimeSpan.Zero) break;
                await Task.Delay(
                    remaining < TimeSpan.FromSeconds(2)
                        ? remaining
                        : TimeSpan.FromSeconds(2),
                    cancellationToken);
            }
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
                    _propertyGridTable = null;
                    _propertyGridItems = null;
                    _propertyGridItemOrder = null;
                    _selectedPropertyGridKey = null;
                    return;
                }
                await Task.Delay(150, cancellationToken);
            }
            throw new TimeoutException(
                "Timed out waiting for AniloxRoll.Monitor to close.");
        }

        public bool ForceTerminateApp(int timeoutSeconds)
        {
            if (_process == null || _process.HasExited)
            {
                _root = null;
                _propertyGridTable = null;
                _propertyGridItems = null;
                _propertyGridItemOrder = null;
                _selectedPropertyGridKey = null;
                return true;
            }

            _process.Kill();
            bool exited = _process.WaitForExit(
                Math.Max(1, timeoutSeconds) * 1000);
            if (exited)
            {
                _root = null;
                _propertyGridTable = null;
                _propertyGridItems = null;
                _propertyGridItemOrder = null;
                _selectedPropertyGridKey = null;
            }
            return exited;
        }

        public string GetPropertyValue(string displayName, int occurrence = 0)
        {
            AutomationElement element = FindDataItem(displayName, occurrence);
            return ReadRequiredValue(element);
        }

        public async Task<string> GetPropertyValueAsync(
            string displayName,
            int occurrence,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            AutomationElement element = await BringDataItemIntoViewAsync(
                displayName,
                occurrence,
                timeoutSeconds,
                cancellationToken);
            string value = ReadRequiredValue(element);
            return value;
        }

        public async Task SetPropertyValueAsync(
            string displayName,
            string value,
            int timeoutSeconds,
            CancellationToken cancellationToken,
            int occurrence = 0)
        {
            LastPropertySelectionMode = "unknown";
            if (string.Equals(
                await GetPropertyValueAsync(
                    displayName,
                    occurrence,
                    timeoutSeconds,
                    cancellationToken),
                value,
                StringComparison.Ordinal))
            {
                LastPropertySelectionMode = "already-current";
                return;
            }

            AutomationElement element = await BringDataItemIntoViewAsync(
                displayName, occurrence, timeoutSeconds, cancellationToken);
            if (!element.Current.IsEnabled)
                throw new InvalidOperationException("Property is disabled: " + displayName);

            DateTime editorDeadline =
                DateTime.UtcNow.AddSeconds(timeoutSeconds);
            AutomationElement editor = null;
            object valuePattern = null;
            while (DateTime.UtcNow < editorDeadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                element = await BringDataItemIntoViewAsync(
                    displayName, occurrence, timeoutSeconds, cancellationToken);
                NativeMethods.SetForegroundWindow(_process.MainWindowHandle);
                System.Windows.Rect bounds = element.Current.BoundingRectangle;
                NativeMethods.ClickScreenPoint(
                    (int)Math.Round(bounds.Left + bounds.Width * 0.75),
                    (int)Math.Round(bounds.Top + bounds.Height / 2.0));
                await Task.Delay(100, cancellationToken);

                editor = AutomationElement.FocusedElement;
                if (editor != null &&
                    editor.TryGetCurrentPattern(
                        ValuePattern.Pattern, out valuePattern))
                {
                    break;
                }

                // A first click can select the row without opening its editor.
                // Retry against the current row instead of assuming a fixed delay.
                editor = null;
                valuePattern = null;
                await Task.Delay(100, cancellationToken);
            }

            if (editor == null || valuePattern == null)
                throw new InvalidOperationException(
                    "Property editor does not expose ValuePattern: " + displayName);

            var editorValue = (ValuePattern)valuePattern;
            bool editorIsReadOnly = editorValue.Current.IsReadOnly;
            if (!editorIsReadOnly)
            {
                editorValue.SetValue(value);

                // PropertyGrid accessibility SetValue only updates the edit
                // box. Enter commits it and raises PropertyValueChanged. A
                // cross-process SendMessage waits for the whole value-change
                // route (including chart/display refresh) and can deadlock the
                // runner. Keyboard input is asynchronous; the read-back loop
                // below remains the actual commit boundary.
                NativeMethods.PressKey((byte)NativeMethods.VkReturn);

                // Do not accept the editor text as proof before WinForms has
                // dispatched PropertyValueChanged and persisted the setting.
                await Task.Delay(300, cancellationToken);

                DateTime directDeadline = DateTime.UtcNow.AddSeconds(
                    Math.Min(1.0, timeoutSeconds));
                while (DateTime.UtcNow < directDeadline)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    string current = await GetPropertyValueAsync(
                        displayName,
                        occurrence,
                        timeoutSeconds,
                        cancellationToken);
                    if (string.Equals(
                        current, value, StringComparison.Ordinal))
                    {
                        // The value and route are the commit boundary. A
                        // setting such as AppRole can rebuild the layout, so
                        // do not touch the pre-change AutomationElement.
                        LastPropertySelectionMode = "value-pattern";
                        return;
                    }
                    await Task.Delay(150, cancellationToken);
                }
            }

            // Some WinForms PropertyGrid enum editors expose ValuePattern but
            // mark it read-only, or do not commit text that was not chosen
            // from the standard-value list. Prefer the exact visible option
            // text; retain keyboard enumeration for accessibility hosts that
            // do not expose the native drop-down items.
            element = await BringDataItemIntoViewAsync(
                displayName, occurrence, timeoutSeconds, cancellationToken);
            NativeMethods.SetForegroundWindow(_process.MainWindowHandle);
            System.Windows.Rect directBounds =
                element.Current.BoundingRectangle;
            NativeMethods.ClickScreenPoint(
                (int)Math.Round(directBounds.Left + directBounds.Width * 0.75),
                (int)Math.Round(directBounds.Top + directBounds.Height / 2.0));
            await Task.Delay(100, cancellationToken);
            NativeMethods.PressKey((byte)NativeMethods.VkF4);
            if (await TrySelectVisibleStandardValueAsync(
                value, timeoutSeconds, cancellationToken))
            {
                string selected = await GetPropertyValueAsync(
                    displayName,
                    occurrence,
                    timeoutSeconds,
                    cancellationToken);
                if (string.Equals(selected, value, StringComparison.Ordinal))
                {
                    LastPropertySelectionMode = "option-text";
                    return;
                }
            }
            NativeMethods.PressKey((byte)NativeMethods.VkEscape);

            DateTime selectionDeadline =
                DateTime.UtcNow.AddSeconds(timeoutSeconds);
            for (int choiceIndex = 0;
                 choiceIndex < 32 && DateTime.UtcNow < selectionDeadline;
                 choiceIndex++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                element = await BringDataItemIntoViewAsync(
                    displayName, occurrence, timeoutSeconds, cancellationToken);
                NativeMethods.SetForegroundWindow(_process.MainWindowHandle);
                System.Windows.Rect bounds =
                    element.Current.BoundingRectangle;
                NativeMethods.ClickScreenPoint(
                    (int)Math.Round(bounds.Left + bounds.Width * 0.75),
                    (int)Math.Round(bounds.Top + bounds.Height / 2.0));
                await Task.Delay(100, cancellationToken);

                // F4 opens the PropertyGrid standard-value drop-down. Without
                // opening it, Home/Down only edit the current text and some
                // three-or-more-value enums oscillate between two entries.
                NativeMethods.PressKey((byte)NativeMethods.VkF4);
                await Task.Delay(100, cancellationToken);
                NativeMethods.PressKey((byte)NativeMethods.VkHome);
                for (int i = 0; i < choiceIndex; i++)
                    NativeMethods.PressKey((byte)NativeMethods.VkDown);
                NativeMethods.PressKey((byte)NativeMethods.VkReturn);
                await Task.Delay(200, cancellationToken);

                string current = await GetPropertyValueAsync(
                    displayName,
                    occurrence,
                    timeoutSeconds,
                    cancellationToken);
                if (!string.Equals(
                    current, value, StringComparison.Ordinal))
                    continue;
                LastPropertySelectionMode = "keyboard-fallback";
                return;
            }
            throw new TimeoutException(
                $"Property value did not settle: {displayName} expected={value}");
        }

        public string LastPropertySelectionMode { get; private set; }

        private async Task<bool> TrySelectVisibleStandardValueAsync(
            string value,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            DateTime deadline = DateTime.UtcNow.AddSeconds(
                Math.Min(1.5, Math.Max(0.5, timeoutSeconds)));
            var condition = new AndCondition(
                new PropertyCondition(
                    AutomationElement.ProcessIdProperty,
                    _process.Id),
                new PropertyCondition(
                    AutomationElement.ControlTypeProperty,
                    ControlType.ListItem),
                new PropertyCondition(
                    AutomationElement.NameProperty,
                    value));

            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                AutomationElementCollection options =
                    AutomationElement.RootElement.FindAll(
                        TreeScope.Descendants,
                        condition);
                for (int i = 0; i < options.Count; i++)
                {
                    AutomationElement option = options[i];
                    try
                    {
                        System.Windows.Rect bounds =
                            option.Current.BoundingRectangle;
                        if (option.Current.IsOffscreen ||
                            !option.Current.IsEnabled ||
                            bounds.Width <= 0 ||
                            bounds.Height <= 0)
                            continue;

                        object selectionPattern;
                        if (option.TryGetCurrentPattern(
                            SelectionItemPattern.Pattern,
                            out selectionPattern))
                        {
                            ((SelectionItemPattern)selectionPattern).Select();
                            NativeMethods.PressKey(
                                (byte)NativeMethods.VkReturn);
                        }
                        else
                        {
                            NativeMethods.ClickScreenPoint(
                                (int)Math.Round(bounds.Left + bounds.Width / 2.0),
                                (int)Math.Round(bounds.Top + bounds.Height / 2.0));
                        }
                        await Task.Delay(250, cancellationToken);
                        return true;
                    }
                    catch (ElementNotAvailableException)
                    {
                        // The drop-down was rebuilt while UI Automation read it.
                    }
                    catch (InvalidOperationException)
                    {
                        // This accessibility provider exposes the item but
                        // rejects SelectionItemPattern. Let the caller use
                        // the verified keyboard fallback for this property.
                        return false;
                    }
                }
                await Task.Delay(50, cancellationToken);
            }
            return false;
        }

        public async Task ClickAsync(
            string name,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                IntPtr handle = NativeMethods.FindDescendantButtonByText(
                    _process.MainWindowHandle, name);
                if (handle != IntPtr.Zero &&
                    NativeMethods.IsWindowEnabled(handle) &&
                    NativeMethods.IsWindowVisible(handle) &&
                    NativeMethods.ClickWindowCenter(handle))
                    return;

                // ToolStrip items such as output-health labels do not own an
                // HWND. UI Automation still exposes their screen bounds.
                AutomationElement element =
                    FindUnique(name, throwIfMissing: false);
                if (element != null &&
                    element.Current.IsEnabled &&
                    !element.Current.IsOffscreen)
                {
                    System.Windows.Rect bounds =
                        element.Current.BoundingRectangle;
                    if (!bounds.IsEmpty &&
                        bounds.Width > 1 &&
                        bounds.Height > 1)
                    {
                        NativeMethods.SetForegroundWindow(
                            _process.MainWindowHandle);
                        NativeMethods.ClickScreenPoint(
                            (int)Math.Round(bounds.Left + bounds.Width / 2.0),
                            (int)Math.Round(bounds.Top + bounds.Height / 2.0));
                        return;
                    }
                }
                await Task.Delay(150, cancellationToken);
            }
            throw new TimeoutException("Timed out waiting to click: " + name);
        }

        public async Task ClickButtonAsync(
            string name,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                IntPtr handle = NativeMethods.FindDescendantButtonByText(
                    _process.MainWindowHandle, name);
                if (handle != IntPtr.Zero &&
                    NativeMethods.IsWindowEnabled(handle) &&
                    NativeMethods.IsWindowVisible(handle) &&
                    NativeMethods.ClickWindowCenter(handle))
                    return;
                await Task.Delay(100, cancellationToken);
            }
            throw new TimeoutException(
                "Timed out waiting to click native button: " + name);
        }

        public async Task ClickAutomationIdAsync(
            string automationId,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                IntPtr handle =
                    NativeMethods.FindDescendantWindowByAccessibleName(
                        _process.MainWindowHandle,
                        automationId,
                        string.Empty);
                if (handle != IntPtr.Zero &&
                    NativeMethods.IsWindowEnabled(handle) &&
                    NativeMethods.IsWindowVisible(handle))
                {
                    NativeMethods.SetForegroundWindow(
                        _process.MainWindowHandle);
                    if (NativeMethods.ClickWindowCenter(handle))
                        return;
                }
                await Task.Delay(150, cancellationToken);
            }
            throw new TimeoutException(
                "Timed out waiting to click control id: " + automationId);
        }

        public async Task<string> WheelAsync(
            string name,
            string value,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            int delta;
            if (!int.TryParse(value, out delta) || delta == 0)
                throw new InvalidOperationException(
                    "Wheel delta must be a non-zero integer.");

            IntPtr handle = await WaitForAccessibleWindowAsync(
                name, timeoutSeconds, cancellationToken);
            NativeMethods.SetForegroundWindow(_process.MainWindowHandle);
            if (!NativeMethods.WheelWindowCenter(handle, delta))
                throw new InvalidOperationException(
                    "Failed to wheel UI target: " + name);
            await Task.Delay(250, cancellationToken);
            return $"{name} wheel={delta}";
        }

        public async Task<string> DragAsync(
            string name,
            string value,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            string[] parts = (value ?? string.Empty).Split(',');
            int offsetX;
            int offsetY;
            if (parts.Length != 2 ||
                !int.TryParse(parts[0], out offsetX) ||
                !int.TryParse(parts[1], out offsetY))
            {
                throw new InvalidOperationException(
                    "Drag offset must use the form X,Y.");
            }

            IntPtr handle = await WaitForAccessibleWindowAsync(
                name, timeoutSeconds, cancellationToken);
            NativeMethods.SetForegroundWindow(_process.MainWindowHandle);
            if (!NativeMethods.DragWindowCenter(
                    handle, offsetX, offsetY))
            {
                throw new InvalidOperationException(
                    "Failed to drag UI target: " + name);
            }
            await Task.Delay(250, cancellationToken);
            return $"{name} drag={offsetX},{offsetY}";
        }

        public async Task<string> SelectTabAsync(
            string name,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            int index;
            if (string.Equals(name, "監控", StringComparison.Ordinal))
                index = 0;
            else if (string.Equals(name, "回顧", StringComparison.Ordinal))
                index = 1;
            else if (string.Equals(name, "報表", StringComparison.Ordinal))
                index = 2;
            else
                throw new InvalidOperationException(
                    "Unknown main tab: " + name);

            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (NativeMethods.IsMainTabSelected(
                    _process.MainWindowHandle, index))
                    return "tab=" + name;
                if (NativeMethods.SelectMainTab(
                    _process.MainWindowHandle, index))
                {
                    await Task.Delay(150, cancellationToken);
                    if (NativeMethods.IsMainTabSelected(
                        _process.MainWindowHandle, index))
                        return "tab=" + name;
                }
                await Task.Delay(150, cancellationToken);
            }
            throw new TimeoutException(
                "Timed out selecting tab: " + name);
        }

        public async Task<string> SelectComboAsync(
            string name,
            string movement,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            IntPtr comboHandle = IntPtr.Zero;
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                comboHandle =
                    NativeMethods.FindDescendantWindowByAccessibleName(
                        _process.MainWindowHandle,
                        name,
                        "COMBOBOX");
                if (comboHandle != IntPtr.Zero) break;
                await Task.Delay(100, cancellationToken);
            }
            if (comboHandle == IntPtr.Zero)
                throw new TimeoutException(
                    "Timed out locating ComboBox: " + name);

            string command = (movement ?? string.Empty).Trim();
            int virtualKey;
            int count;
            int delayMilliseconds = 0;
            if (string.Equals(command, "first", StringComparison.OrdinalIgnoreCase))
            {
                virtualKey = NativeMethods.VkHome;
                count = 1;
            }
            else if (string.Equals(command, "last", StringComparison.OrdinalIgnoreCase))
            {
                virtualKey = NativeMethods.VkEnd;
                count = 1;
            }
            else
            {
                string[] parts = command.Split(':');
                if ((parts.Length != 2 && parts.Length != 3) ||
                    !int.TryParse(parts[1], out count) ||
                    count < 1)
                {
                    throw new InvalidOperationException(
                        "Combo movement must be first, last, next:N[:delayMs], or previous:N[:delayMs].");
                }
                if (parts.Length == 3 &&
                    (!int.TryParse(parts[2], out delayMilliseconds) ||
                     delayMilliseconds < 0 || delayMilliseconds > 1000))
                    throw new InvalidOperationException(
                        "Combo movement delay must be between 0 and 1000 ms.");
                if (string.Equals(
                    parts[0], "next", StringComparison.OrdinalIgnoreCase))
                    virtualKey = NativeMethods.VkDown;
                else if (string.Equals(
                    parts[0], "previous", StringComparison.OrdinalIgnoreCase))
                    virtualKey = NativeMethods.VkUp;
                else
                    throw new InvalidOperationException(
                        "Combo movement must be first, last, next:N[:delayMs], or previous:N[:delayMs].");
            }

            for (int i = 0; i < count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (DateTime.UtcNow >= deadline)
                    throw new TimeoutException(
                        "Timed out changing ComboBox selection: " + name);
                if (!NativeMethods.PostKeyToWindow(
                    comboHandle, virtualKey))
                {
                    throw new InvalidOperationException(
                        "Failed to post selection key to ComboBox: " + name);
                }
                if (delayMilliseconds > 0)
                {
                    await Task.Delay(delayMilliseconds, cancellationToken);
                }
                else if ((i + 1) % 20 == 0)
                {
                    await Task.Delay(1, cancellationToken);
                }
            }
            await Task.Delay(250, cancellationToken);
            string result = NativeMethods.ReadWindowText(
                comboHandle, 1000);
            if (string.IsNullOrWhiteSpace(result))
                result = command;
            return name + "=" + result;
        }

        public async Task<string> ReadComboValueAsync(
            string name,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                IntPtr comboHandle =
                    NativeMethods.FindDescendantWindowByAccessibleName(
                        _process.MainWindowHandle,
                        name,
                        "COMBOBOX");
                if (comboHandle != IntPtr.Zero)
                {
                    string value = NativeMethods.ReadWindowText(
                        comboHandle, 1000);
                    if (!string.IsNullOrWhiteSpace(value))
                        return value;
                }
                await Task.Delay(100, cancellationToken);
            }
            throw new TimeoutException(
                "Timed out reading ComboBox value: " + name);
        }

        public async Task<string> ConfirmFolderAsync(
            string buttonName,
            string path,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            if (!Directory.Exists(path))
                throw new DirectoryNotFoundException(
                    "DVT data directory not found: " + path);

            await ClickAsync(
                buttonName, timeoutSeconds, cancellationToken);
            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            IntPtr dialogHandle = IntPtr.Zero;
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                dialogHandle = NativeMethods.FindDescendantWindowByText(
                    _process.MainWindowHandle, "瀏覽資料夾");
                if (dialogHandle == IntPtr.Zero)
                    dialogHandle =
                        NativeMethods.FindTopLevelWindowByProcessAndText(
                            _process.Id, "瀏覽資料夾");
                if (dialogHandle == IntPtr.Zero)
                    dialogHandle = NativeMethods.FindDescendantWindowByText(
                        _process.MainWindowHandle, "Browse For Folder");
                if (dialogHandle == IntPtr.Zero)
                    dialogHandle =
                        NativeMethods.FindTopLevelWindowByProcessAndText(
                            _process.Id, "Browse For Folder");
                if (dialogHandle != IntPtr.Zero) break;
                await Task.Delay(150, cancellationToken);
            }
            if (dialogHandle == IntPtr.Zero)
                throw new TimeoutException(
                    "Timed out waiting for folder selection dialog.");

            IntPtr ok = NativeMethods.FindDescendantWindowByText(
                dialogHandle, "確定");
            if (ok == IntPtr.Zero)
                ok = NativeMethods.FindDescendantWindowByText(
                    dialogHandle, "OK");
            if (ok == IntPtr.Zero)
                throw new InvalidOperationException(
                    "Folder selection dialog has no OK button.");
            NativeMethods.SendMessage(
                ok,
                NativeMethods.BmClick,
                IntPtr.Zero,
                IntPtr.Zero);

            DateTime closeDeadline = DateTime.UtcNow.AddSeconds(
                Math.Min(5, Math.Max(1, timeoutSeconds)));
            while (NativeMethods.IsWindow(dialogHandle) &&
                   NativeMethods.IsWindowVisible(dialogHandle) &&
                   DateTime.UtcNow < closeDeadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                await Task.Delay(100, cancellationToken);
            }
            if (NativeMethods.IsWindow(dialogHandle) &&
                NativeMethods.IsWindowVisible(dialogHandle))
                throw new InvalidOperationException(
                    "Folder selection dialog did not close after OK.");
            return "folder=" + path;
        }

        public async Task TryStopCaptureAsync(CancellationToken cancellationToken)
        {
            if (!IsAttached) return;
            IntPtr handle = NativeMethods.FindDescendantWindowByText(
                _process.MainWindowHandle, "停止抓取");
            if (handle == IntPtr.Zero ||
                !NativeMethods.IsWindowEnabled(handle))
                return;

            NativeMethods.ClickWindowCenter(handle);
            await Task.Delay(150, cancellationToken);
        }

        private AutomationElement FindDataItem(string name, int occurrence)
        {
            EnsureAttached();
            if (occurrence < 0)
                throw new ArgumentOutOfRangeException(nameof(occurrence));
            Dictionary<string, List<AutomationElement>> items =
                GetPropertyGridItems();
            List<AutomationElement> matches;
            if (!items.TryGetValue(name, out matches) ||
                occurrence >= matches.Count)
                throw new InvalidOperationException(
                    "PropertyGrid item not found: " + name +
                    " occurrence=" + occurrence);
            return matches[occurrence];
        }

        private async Task<AutomationElement> BringDataItemIntoViewAsync(
            string name,
            int occurrence,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                AutomationElement item = FindDataItem(name, occurrence);
                AutomationElement table = GetPropertyGridTable();

                AutomationElement selected = FindFocusedDataItem(name);
                if (selected != null)
                    item = selected;

                System.Windows.Rect itemBounds = item.Current.BoundingRectangle;
                System.Windows.Rect tableBounds = table.Current.BoundingRectangle;
                double itemCenterY = itemBounds.Top + itemBounds.Height / 2.0;
                bool visible =
                    itemBounds.Height > 0 &&
                    itemCenterY >= tableBounds.Top &&
                    itemCenterY <= tableBounds.Bottom;
                if (visible)
                {
                    _selectedPropertyGridKey = name + "\u001f" + occurrence;
                    return item;
                }

                object scrollItemPattern;
                if (item.TryGetCurrentPattern(
                        ScrollItemPattern.Pattern,
                        out scrollItemPattern))
                {
                    ((ScrollItemPattern)scrollItemPattern).ScrollIntoView();
                    await Task.Delay(150, cancellationToken);

                    // WinForms sometimes reports ScrollItemPattern as
                    // supported but treats ScrollIntoView as a no-op. Only
                    // accept it after the row actually enters the viewport;
                    // otherwise continue to the native scrollbar fallback.
                    item = FindDataItem(name, occurrence);
                    itemBounds = item.Current.BoundingRectangle;
                    itemCenterY = itemBounds.Top + itemBounds.Height / 2.0;
                    visible = itemBounds.Height > 0 &&
                        itemCenterY >= tableBounds.Top &&
                        itemCenterY <= tableBounds.Bottom;
                    if (visible)
                    {
                        _selectedPropertyGridKey =
                            name + "\u001f" + occurrence;
                        return item;
                    }
                }

                bool down = itemBounds.Top > tableBounds.Bottom;

                AutomationElement scrollOwner = table;
                IntPtr scrollHandle = IntPtr.Zero;
                while (scrollOwner != null && scrollHandle == IntPtr.Zero)
                {
                    scrollHandle = new IntPtr(
                        scrollOwner.Current.NativeWindowHandle);
                    scrollOwner = TreeWalker.RawViewWalker.GetParent(scrollOwner);
                }
                if (!NativeMethods.ScrollVerticalPage(scrollHandle, down))
                {
                    // Some accessibility hosts expose the table without its
                    // native scrollbar handle. Keyboard paging is the final
                    // fallback and is verified again at the top of the loop.
                    NativeMethods.SetForegroundWindow(_process.MainWindowHandle);
                    table.SetFocus();
                    System.Windows.Forms.SendKeys.SendWait(
                        down ? "{PGDN}" : "{PGUP}");
                }
                await Task.Delay(150, cancellationToken);
            }
            throw new TimeoutException(
                "Timed out scrolling PropertyGrid item into view: " + name);
        }

        private AutomationElement FindFocusedDataItem(string expectedName)
        {
            AutomationElement element = AutomationElement.FocusedElement;
            while (element != null)
            {
                try
                {
                    if (element.Current.ProcessId != _process.Id)
                        return null;
                    if (element.Current.ControlType == ControlType.DataItem)
                    {
                        return string.Equals(
                            element.Current.Name,
                            expectedName,
                            StringComparison.Ordinal)
                            ? element
                            : null;
                    }
                    element = TreeWalker.RawViewWalker.GetParent(element);
                }
                catch (ElementNotAvailableException)
                {
                    return null;
                }
            }
            return null;
        }

        private AutomationElement GetPropertyGridTable()
        {
            EnsureAttached();
            if (_propertyGridTable != null)
            {
                try
                {
                    if (_propertyGridTable.Current.ProcessId == _process.Id)
                        return _propertyGridTable;
                }
                catch (ElementNotAvailableException)
                {
                    _propertyGridTable = null;
                    _propertyGridItems = null;
                    _propertyGridItemOrder = null;
                    _selectedPropertyGridKey = null;
                }
            }

            var condition = new AndCondition(
                new PropertyCondition(
                    AutomationElement.NameProperty,
                    "屬性視窗"),
                new PropertyCondition(
                    AutomationElement.ControlTypeProperty,
                    ControlType.Table));
            _propertyGridTable = _root.FindFirst(
                TreeScope.Descendants,
                condition);
            if (_propertyGridTable == null)
                throw new InvalidOperationException(
                    "PropertyGrid table not found.");
            return _propertyGridTable;
        }

        private Dictionary<string, List<AutomationElement>> GetPropertyGridItems()
        {
            if (_propertyGridItems != null)
                return _propertyGridItems;

            AutomationElement table = GetPropertyGridTable();
            AutomationElementCollection elements = table.FindAll(
                TreeScope.Descendants,
                new PropertyCondition(
                    AutomationElement.ControlTypeProperty,
                    ControlType.DataItem));
            var items = new Dictionary<string, List<AutomationElement>>(
                StringComparer.Ordinal);
            var itemOrder = new List<AutomationElement>(elements.Count);
            foreach (AutomationElement element in elements)
            {
                itemOrder.Add(element);
                string name = element.Current.Name;
                if (string.IsNullOrWhiteSpace(name))
                    continue;
                List<AutomationElement> matches;
                if (!items.TryGetValue(name, out matches))
                {
                    matches = new List<AutomationElement>();
                    items.Add(name, matches);
                }
                matches.Add(element);
            }
            _propertyGridItems = items;
            _propertyGridItemOrder = itemOrder;
            return _propertyGridItems;
        }

        private AutomationElement FindUnique(string name, bool throwIfMissing)
        {
            EnsureAttached();
            var nameCondition = new PropertyCondition(
                AutomationElement.NameProperty, name);
            var processCondition = new PropertyCondition(
                AutomationElement.ProcessIdProperty, _process.Id);
            AutomationElement match =
                AutomationElement.RootElement.FindFirst(
                    TreeScope.Descendants,
                    new AndCondition(nameCondition, processCondition));
            if (match == null)
            {
                if (throwIfMissing)
                    throw new InvalidOperationException("UI element not found: " + name);
                return null;
            }
            return match;
        }

        private async Task<IntPtr> WaitForAccessibleWindowAsync(
            string name,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                IntPtr handle =
                    NativeMethods.FindDescendantWindowByAccessibleName(
                        _process.MainWindowHandle,
                        name,
                        string.Empty);
                if (handle != IntPtr.Zero &&
                    NativeMethods.IsWindowVisible(handle) &&
                    NativeMethods.IsWindowEnabled(handle))
                    return handle;
                await Task.Delay(100, cancellationToken);
            }
            throw new TimeoutException(
                "Timed out locating interactive UI target: " + name);
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
