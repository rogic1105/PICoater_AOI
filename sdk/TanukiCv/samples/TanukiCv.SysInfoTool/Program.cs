using System;
using System.Windows.Forms;

namespace TanukiCv.SysInfoTool
{
    /// <summary>TanukiCv 系統資訊工具入口：顯示 TanukiCv.Core.SystemInfo 查到的 CPU/GPU/RAM/螢幕。</summary>
    internal static class Program
    {
        [STAThread]
        private static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new SysInfoForm());
        }
    }
}
