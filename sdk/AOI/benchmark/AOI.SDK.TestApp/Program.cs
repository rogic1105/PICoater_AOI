// AOI_SDK\src\dotnet\AOI.SDK.TestApp\Program.cs

using System;
using System.Windows.Forms;
using AOI.SDK.TestApp.Forms; // [新增] 引用 Forms namespace

namespace AOI.SDK.TestApp
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