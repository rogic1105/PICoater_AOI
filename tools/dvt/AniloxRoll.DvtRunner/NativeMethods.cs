using System;
using System.Runtime.InteropServices;

namespace AniloxRoll.DvtRunner
{
    internal static class NativeMethods
    {
        internal const uint BmClick = 0x00F5;
        internal const uint WmNull = 0x0000;
        internal const uint WmKeyDown = 0x0100;
        internal const uint WmKeyUp = 0x0101;
        internal const int VkReturn = 0x0D;
        private const uint SmtoAbortIfHung = 0x0002;
        private const uint MouseeventfLeftdown = 0x0002;
        private const uint MouseeventfLeftup = 0x0004;
        private const uint MouseeventfWheel = 0x0800;
        private const uint KeyeventfKeyup = 0x0002;

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

        [DllImport("user32.dll")]
        internal static extern bool SetForegroundWindow(IntPtr hWnd);

        [DllImport("user32.dll")]
        internal static extern bool SetCursorPos(int x, int y);

        [DllImport("user32.dll")]
        private static extern void mouse_event(
            uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);

        [DllImport("user32.dll")]
        private static extern void keybd_event(
            byte virtualKey, byte scanCode, uint flags, UIntPtr extraInfo);

        internal static void ClickScreenPoint(int x, int y)
        {
            SetCursorPos(x, y);
            mouse_event(MouseeventfLeftdown, 0, 0, 0, UIntPtr.Zero);
            mouse_event(MouseeventfLeftup, 0, 0, 0, UIntPtr.Zero);
        }

        internal static void WheelAt(int x, int y, int delta)
        {
            SetCursorPos(x, y);
            mouse_event(
                MouseeventfWheel, 0, 0, unchecked((uint)delta), UIntPtr.Zero);
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

        internal static void PressKey(byte virtualKey)
        {
            keybd_event(virtualKey, 0, 0, UIntPtr.Zero);
            keybd_event(virtualKey, 0, KeyeventfKeyup, UIntPtr.Zero);
        }
    }
}
