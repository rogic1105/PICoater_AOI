// TanukiCv.BenchUi\Program.cs

using System;
using System.Windows.Forms;
using TanukiCv.BenchUi.Forms; // [新增] 引用 Forms namespace

namespace TanukiCv.BenchUi
{
    internal static class Program
    {
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new SdkForm());
        }
    }
}