using System.Runtime.InteropServices;

namespace AOIUI.Interop
{
    internal static class Win32
    {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern bool SetDllDirectory(string lpPathName);
    }
}
