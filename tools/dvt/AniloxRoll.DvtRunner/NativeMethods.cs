using System;
using System.Runtime.InteropServices;
using System.Text;
using Accessibility;

namespace AniloxRoll.DvtRunner
{
    internal static class NativeMethods
    {
        internal const uint BmClick = 0x00F5;
        internal const uint WmNull = 0x0000;
        internal const uint WmGetText = 0x000D;
        internal const uint WmGetTextLength = 0x000E;
        internal const uint WmKeyDown = 0x0100;
        internal const uint WmKeyUp = 0x0101;
        private const uint WmVScroll = 0x0115;
        internal const uint WmLButtonDown = 0x0201;
        internal const uint WmLButtonUp = 0x0202;
        private const uint TcmGetItemCount = 0x1304;
        private const uint TcmGetCurSel = 0x130B;
        internal const int VkReturn = 0x0D;
        internal const int VkEscape = 0x1B;
        internal const int VkHome = 0x24;
        internal const int VkEnd = 0x23;
        internal const int VkUp = 0x26;
        internal const int VkDown = 0x28;
        internal const int VkF4 = 0x73;
        private const int MkLButton = 0x0001;
        private const int SbPageUp = 2;
        private const int SbPageDown = 3;
        private const uint SmtoAbortIfHung = 0x0002;
        private const uint MouseeventfLeftdown = 0x0002;
        private const uint MouseeventfLeftup = 0x0004;
        private const uint MouseeventfWheel = 0x0800;
        private const uint KeyeventfKeyup = 0x0002;
        private const uint ObjidClient = 0xFFFFFFFC;
        private const int SelflagTakeFocus = 0x1;
        private const int SelflagTakeSelection = 0x2;
        private const int SwRestore = 9;

        private static readonly Guid IAccessibleGuid =
            new Guid("618736E0-3C3D-11CF-810C-00AA00389B71");

        [DllImport("user32.dll", CharSet = CharSet.Auto)]
        internal static extern IntPtr SendMessage(
            IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);

        [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        private static extern IntPtr SendMessageTimeout(
            IntPtr hWnd,
            uint msg,
            IntPtr wParam,
            IntPtr lParam,
            uint flags,
            uint timeout,
            out IntPtr result);

        [DllImport(
            "user32.dll",
            CharSet = CharSet.Unicode,
            SetLastError = true,
            EntryPoint = "SendMessageTimeoutW")]
        private static extern IntPtr SendMessageTimeoutText(
            IntPtr hWnd,
            uint msg,
            IntPtr wParam,
            StringBuilder lParam,
            uint flags,
            uint timeout,
            out IntPtr result);

        [DllImport("user32.dll")]
        internal static extern bool IsWindow(IntPtr hWnd);

        [DllImport("user32.dll")]
        internal static extern bool IsWindowEnabled(IntPtr hWnd);

        [DllImport("user32.dll")]
        internal static extern bool IsWindowVisible(IntPtr hWnd);

        [DllImport("user32.dll")]
        internal static extern bool SetForegroundWindow(IntPtr hWnd);

        [DllImport("user32.dll")]
        private static extern bool ShowWindow(IntPtr hWnd, int command);

        [DllImport("user32.dll", SetLastError = true)]
        private static extern bool MoveWindow(
            IntPtr hWnd,
            int x,
            int y,
            int width,
            int height,
            bool repaint);

        [DllImport("user32.dll")]
        internal static extern bool SetCursorPos(int x, int y);

        [DllImport("user32.dll", SetLastError = true)]
        private static extern bool PostMessage(
            IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);

        internal static bool RestoreAndMoveWindow(
            IntPtr handle,
            int x,
            int y,
            int width,
            int height)
        {
            if (handle == IntPtr.Zero || width <= 0 || height <= 0)
                return false;
            ShowWindow(handle, SwRestore);
            return MoveWindow(handle, x, y, width, height, true);
        }

        [DllImport("user32.dll", SetLastError = true)]
        private static extern bool GetClientRect(
            IntPtr hWnd, out NativeRect rect);

        [DllImport("user32.dll", SetLastError = true)]
        private static extern bool GetWindowRect(
            IntPtr hWnd, out NativeRect rect);

        [DllImport("user32.dll")]
        private static extern bool EnumChildWindows(
            IntPtr parent,
            EnumWindowCallback callback,
            IntPtr parameter);

        [DllImport("user32.dll")]
        private static extern bool EnumWindows(
            EnumWindowCallback callback,
            IntPtr parameter);

        [DllImport("user32.dll")]
        private static extern uint GetWindowThreadProcessId(
            IntPtr hWnd,
            out uint processId);

        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        private static extern int GetWindowText(
            IntPtr hWnd,
            StringBuilder text,
            int capacity);

        [DllImport("user32.dll")]
        private static extern int GetWindowTextLength(IntPtr hWnd);

        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        private static extern int GetClassName(
            IntPtr hWnd,
            StringBuilder className,
            int capacity);

        [DllImport("user32.dll")]
        private static extern void mouse_event(
            uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);

        [DllImport("user32.dll")]
        private static extern void keybd_event(
            byte virtualKey, byte scanCode, uint flags, UIntPtr extraInfo);

        [DllImport("oleacc.dll")]
        private static extern int AccessibleObjectFromWindow(
            IntPtr hWnd,
            uint objectId,
            ref Guid interfaceId,
            [MarshalAs(UnmanagedType.Interface)] out object accessible);

        internal static void ClickScreenPoint(int x, int y)
        {
            SetCursorPos(x, y);
            mouse_event(MouseeventfLeftdown, 0, 0, 0, UIntPtr.Zero);
            mouse_event(MouseeventfLeftup, 0, 0, 0, UIntPtr.Zero);
        }

        internal static void DragScreenPoint(
            int startX,
            int startY,
            int endX,
            int endY)
        {
            const int steps = 10;
            SetCursorPos(startX, startY);
            mouse_event(MouseeventfLeftdown, 0, 0, 0, UIntPtr.Zero);
            try
            {
                for (int i = 1; i <= steps; i++)
                {
                    SetCursorPos(
                        startX + (endX - startX) * i / steps,
                        startY + (endY - startY) * i / steps);
                    System.Threading.Thread.Sleep(20);
                }
            }
            finally
            {
                mouse_event(MouseeventfLeftup, 0, 0, 0, UIntPtr.Zero);
            }
        }

        internal static bool WheelWindowCenter(IntPtr handle, int delta)
        {
            NativeRect rect;
            if (handle == IntPtr.Zero ||
                !GetWindowRect(handle, out rect))
                return false;

            WheelAt(
                rect.Left + (rect.Right - rect.Left) / 2,
                rect.Top + (rect.Bottom - rect.Top) / 2,
                delta);
            return true;
        }

        internal static bool DragWindowCenter(
            IntPtr handle,
            int offsetX,
            int offsetY)
        {
            NativeRect rect;
            if (handle == IntPtr.Zero ||
                !GetWindowRect(handle, out rect) ||
                rect.Right - rect.Left < 20 ||
                rect.Bottom - rect.Top < 20)
                return false;

            int startX = rect.Left + (rect.Right - rect.Left) / 2;
            int startY = rect.Top + (rect.Bottom - rect.Top) / 2;
            int endX = Math.Max(
                rect.Left + 10,
                Math.Min(rect.Right - 10, startX + offsetX));
            int endY = Math.Max(
                rect.Top + 10,
                Math.Min(rect.Bottom - 10, startY + offsetY));
            DragScreenPoint(startX, startY, endX, endY);
            return true;
        }

        internal static bool ClickWindowCenter(IntPtr handle)
        {
            NativeRect rect;
            if (handle == IntPtr.Zero ||
                !GetClientRect(handle, out rect))
                return false;

            int x = Math.Max(1, (rect.Right - rect.Left) / 2);
            int y = Math.Max(1, (rect.Bottom - rect.Top) / 2);
            var point = new IntPtr((y << 16) | (x & 0xFFFF));
            return
                PostMessage(
                    handle,
                    WmLButtonDown,
                    new IntPtr(MkLButton),
                    point) &&
                PostMessage(
                    handle,
                    WmLButtonUp,
                    IntPtr.Zero,
                    point);
        }

        internal static bool ScrollVerticalPage(IntPtr handle, bool down)
        {
            if (handle == IntPtr.Zero || !IsWindow(handle))
                return false;

            IntPtr scrollBar = IntPtr.Zero;
            EnumChildWindows(
                handle,
                (child, parameter) =>
                {
                    var className = new StringBuilder(128);
                    GetClassName(child, className, className.Capacity);
                    NativeRect rect;
                    if (className.ToString().IndexOf(
                            "SCROLLBAR",
                            StringComparison.OrdinalIgnoreCase) >= 0 &&
                        GetWindowRect(child, out rect) &&
                        rect.Bottom - rect.Top > rect.Right - rect.Left)
                    {
                        scrollBar = child;
                        return false;
                    }
                    return true;
                },
                IntPtr.Zero);
            if (scrollBar == IntPtr.Zero)
                return false;

            IntPtr result;
            return SendMessageTimeout(
                handle,
                WmVScroll,
                new IntPtr(down ? SbPageDown : SbPageUp),
                scrollBar,
                SmtoAbortIfHung,
                1000,
                out result) != IntPtr.Zero;
        }

        internal static bool SelectMainTab(IntPtr parent, int index)
        {
            IntPtr tab = FindMainTabControl(parent);
            if (tab == IntPtr.Zero)
                return false;
            if (IsMainTabSelected(parent, index))
                return true;
            return TryInvokeTabDefaultAction(tab, index);
        }

        internal static bool IsMainTabSelected(IntPtr parent, int index)
        {
            IntPtr tab = FindMainTabControl(parent);
            return tab != IntPtr.Zero &&
                SendMessage(
                    tab,
                    TcmGetCurSel,
                    IntPtr.Zero,
                    IntPtr.Zero).ToInt32() == index;
        }

        private static bool TryInvokeTabDefaultAction(IntPtr tab, int index)
        {
            object accessibleObject = null;
            try
            {
                Guid interfaceId = IAccessibleGuid;
                if (AccessibleObjectFromWindow(
                        tab,
                        ObjidClient,
                        ref interfaceId,
                        out accessibleObject) != 0)
                    return false;

                var accessible = accessibleObject as IAccessible;
                int childId = index + 1;
                if (accessible == null ||
                    childId > accessible.accChildCount)
                    return false;
                accessible.accSelect(
                    SelflagTakeFocus | SelflagTakeSelection,
                    childId);
                accessible.accDoDefaultAction(childId);
                return true;
            }
            catch
            {
                return false;
            }
            finally
            {
                if (accessibleObject != null &&
                    Marshal.IsComObject(accessibleObject))
                {
                    Marshal.ReleaseComObject(accessibleObject);
                }
            }
        }

        internal static IntPtr FindDescendantWindowByText(
            IntPtr parent,
            string expectedText)
        {
            return FindDescendantWindowByTextAndClass(
                parent, expectedText, string.Empty);
        }

        internal static IntPtr FindDescendantButtonByText(
            IntPtr parent,
            string expectedText)
        {
            return FindDescendantWindowByTextAndClass(
                parent, expectedText, "BUTTON");
        }

        private static IntPtr FindDescendantWindowByTextAndClass(
            IntPtr parent,
            string expectedText,
            string classNameFragment)
        {
            if (parent == IntPtr.Zero ||
                string.IsNullOrEmpty(expectedText))
                return IntPtr.Zero;

            IntPtr found = IntPtr.Zero;
            EnumChildWindows(
                parent,
                delegate(IntPtr handle, IntPtr parameter)
                {
                    if (!IsWindowVisible(handle))
                        return true;

                    if (!string.IsNullOrEmpty(classNameFragment))
                    {
                        var className = new StringBuilder(128);
                        GetClassName(handle, className, className.Capacity);
                        if (className.ToString().IndexOf(
                                classNameFragment,
                                StringComparison.OrdinalIgnoreCase) < 0)
                            return true;
                    }

                    string text;
                    if (!TryReadWindowText(handle, 100, out text) ||
                        text.Length == 0)
                        return true;
                    if (!string.Equals(
                        text,
                        expectedText,
                        StringComparison.Ordinal))
                        return true;

                    found = handle;
                    return false;
                },
                IntPtr.Zero);
            return found;
        }

        internal static IntPtr FindTopLevelWindowByProcessAndText(
            int processId,
            string expectedText)
        {
            if (processId <= 0 || string.IsNullOrEmpty(expectedText))
                return IntPtr.Zero;

            IntPtr found = IntPtr.Zero;
            EnumWindows(
                delegate(IntPtr handle, IntPtr parameter)
                {
                    uint ownerProcessId;
                    GetWindowThreadProcessId(handle, out ownerProcessId);
                    if (ownerProcessId != unchecked((uint)processId) ||
                        !IsWindowVisible(handle))
                        return true;

                    int length = GetWindowTextLength(handle);
                    if (length <= 0) return true;

                    var text = new StringBuilder(length + 1);
                    GetWindowText(handle, text, text.Capacity);
                    if (!string.Equals(
                        text.ToString(),
                        expectedText,
                        StringComparison.Ordinal))
                        return true;

                    found = handle;
                    return false;
                },
                IntPtr.Zero);
            return found;
        }

        internal static IntPtr FindDescendantWindowByAccessibleName(
            IntPtr parent,
            string expectedName,
            string classNameFragment)
        {
            if (parent == IntPtr.Zero ||
                string.IsNullOrEmpty(expectedName))
                return IntPtr.Zero;

            IntPtr found = IntPtr.Zero;
            EnumChildWindows(
                parent,
                delegate(IntPtr handle, IntPtr parameter)
                {
                    var className = new StringBuilder(128);
                    GetClassName(
                        handle, className, className.Capacity);
                    if (!string.IsNullOrEmpty(classNameFragment) &&
                        className.ToString().IndexOf(
                            classNameFragment,
                            StringComparison.OrdinalIgnoreCase) < 0)
                        return true;

                    object accessibleObject = null;
                    try
                    {
                        Guid interfaceId = IAccessibleGuid;
                        if (AccessibleObjectFromWindow(
                                handle,
                                ObjidClient,
                                ref interfaceId,
                                out accessibleObject) != 0)
                            return true;

                        var accessible =
                            accessibleObject as IAccessible;
                        string name = accessible == null
                            ? null
                            : accessible.get_accName(0);
                        if (!string.Equals(
                            name,
                            expectedName,
                            StringComparison.Ordinal))
                            return true;

                        found = handle;
                        return false;
                    }
                    catch
                    {
                        return true;
                    }
                    finally
                    {
                        if (accessibleObject != null &&
                            Marshal.IsComObject(accessibleObject))
                        {
                            Marshal.ReleaseComObject(accessibleObject);
                        }
                    }
                },
                IntPtr.Zero);
            return found;
        }

        internal static void WheelAt(int x, int y, int delta)
        {
            SetCursorPos(x, y);
            mouse_event(
                MouseeventfWheel, 0, 0, unchecked((uint)delta), UIntPtr.Zero);
        }

        private static IntPtr FindMainTabControl(IntPtr parent)
        {
            IntPtr selected = IntPtr.Zero;
            int selectedLeft = int.MaxValue;
            int selectedTop = int.MaxValue;
            NativeRect parentRect;
            if (!GetWindowRect(parent, out parentRect))
                return IntPtr.Zero;
            EnumChildWindows(
                parent,
                delegate(IntPtr handle, IntPtr parameter)
                {
                    if (!IsWindowVisible(handle))
                        return true;

                    var className = new StringBuilder(128);
                    GetClassName(handle, className, className.Capacity);
                    if (className.ToString().IndexOf(
                        "SysTabControl32",
                        StringComparison.OrdinalIgnoreCase) < 0)
                        return true;

                    int count = SendMessage(
                        handle,
                        TcmGetItemCount,
                        IntPtr.Zero,
                        IntPtr.Zero).ToInt32();
                    NativeRect rect;
                    if (count != 3 ||
                        !GetWindowRect(handle, out rect) ||
                        rect.Top > parentRect.Top + 160)
                        return true;

                    bool isHigherRow = rect.Top < selectedTop - 4;
                    bool isSameRow = Math.Abs(rect.Top - selectedTop) <= 4;
                    bool isPreferredSide = rect.Left < selectedLeft;
                    if (!isHigherRow &&
                        (!isSameRow || !isPreferredSide))
                        return true;
                    selected = handle;
                    selectedLeft = rect.Left;
                    selectedTop = rect.Top;
                    return true;
                },
                IntPtr.Zero);
            return selected;
        }

        internal static bool IsWindowResponsive(IntPtr handle, int timeoutMs)
        {
            IntPtr result;
            return SendMessageTimeout(
                handle,
                WmNull,
                IntPtr.Zero,
                IntPtr.Zero,
                SmtoAbortIfHung,
                unchecked((uint)timeoutMs),
                out result) != IntPtr.Zero;
        }

        internal static string ReadWindowText(IntPtr handle, int timeoutMs)
        {
            string text;
            if (TryReadWindowText(handle, timeoutMs, out text))
                return text;
            throw new InvalidOperationException(
                "Timed out reading control text.");
        }

        private static bool TryReadWindowText(
            IntPtr handle,
            int timeoutMs,
            out string text)
        {
            text = string.Empty;
            IntPtr lengthResult;
            if (SendMessageTimeout(
                handle,
                WmGetTextLength,
                IntPtr.Zero,
                IntPtr.Zero,
                SmtoAbortIfHung,
                unchecked((uint)timeoutMs),
                out lengthResult) == IntPtr.Zero)
                return false;

            int length = Math.Max(0, lengthResult.ToInt32());
            if (length == 0)
                return true;
            var buffer = new StringBuilder(length + 1);
            IntPtr textResult;
            if (SendMessageTimeoutText(
                handle,
                WmGetText,
                new IntPtr(buffer.Capacity),
                buffer,
                SmtoAbortIfHung,
                unchecked((uint)timeoutMs),
                out textResult) == IntPtr.Zero)
                return false;
            text = buffer.ToString();
            return true;
        }

        internal static void PressKey(byte virtualKey)
        {
            keybd_event(virtualKey, 0, 0, UIntPtr.Zero);
            keybd_event(virtualKey, 0, KeyeventfKeyup, UIntPtr.Zero);
        }

        internal static bool PostKeyToWindow(
            IntPtr handle,
            int virtualKey)
        {
            if (handle == IntPtr.Zero || !IsWindow(handle))
                return false;

            return
                PostMessage(
                    handle,
                    WmKeyDown,
                    new IntPtr(virtualKey),
                    IntPtr.Zero) &&
                PostMessage(
                    handle,
                    WmKeyUp,
                    new IntPtr(virtualKey),
                    IntPtr.Zero);
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct NativeRect
        {
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;
        }

        private delegate bool EnumWindowCallback(
            IntPtr handle,
            IntPtr parameter);
    }
}
